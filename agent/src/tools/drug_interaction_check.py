"""Drug interaction check tool using free public APIs (RxNorm + openFDA).

This module provides a LangGraph-compatible tool that checks for dangerous
interactions between two or more drugs.  It uses:

- **NIH RxNorm API** to normalise drug names (free, no auth).
- **openFDA Drug Label API** to retrieve FDA-published drug-interaction
  sections from approved labelling (free, no auth).

Note: The NIH Drug-Drug Interaction API was discontinued on 2024-01-02.
This implementation uses openFDA label data as the interaction source instead.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

# Ensure the agent package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

_RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
_OPENFDA_BASE = "https://api.fda.gov/drug/label.json"


# ── RxNorm helpers ───────────────────────────────────────────────────────────

async def _resolve_drug_name(client: httpx.AsyncClient, drug_name: str) -> str | None:
    """Normalise a drug name via RxNorm.  Returns the canonical name or *None*."""
    try:
        resp = await client.get(
            f"{_RXNORM_BASE}/rxcui.json",
            params={"name": drug_name},
            timeout=10.0,
        )
        resp.raise_for_status()
        body = resp.json()
        ids = body.get("idGroup", {}).get("rxnormId")
        if not ids:
            return None
        # Fetch the canonical name for the resolved RxCUI.
        prop_resp = await client.get(
            f"{_RXNORM_BASE}/rxcui/{ids[0]}/properties.json",
            timeout=10.0,
        )
        prop_resp.raise_for_status()
        return prop_resp.json().get("properties", {}).get("name") or drug_name
    except (httpx.HTTPError, KeyError):
        return None


# ── openFDA helpers ──────────────────────────────────────────────────────────

def _clean_label_text(raw: str) -> str:
    """Strip common label noise (section numbers, bullet chars, etc.)."""
    # Remove leading section numbers like "7 DRUG INTERACTIONS" or "7.1 ..."
    text = re.sub(r"^\d+(?:\.\d+)?\s+(?:DRUG INTERACTIONS\s*)?", "", raw)
    return text.strip()


async def _search_label_interactions(
    client: httpx.AsyncClient,
    drug_a: str,
    drug_b: str,
) -> str | None:
    """Query openFDA for drug_a's label and look for drug_b in the interaction section."""
    try:
        # Build the URL manually — openFDA expects literal '+AND+' in the
        # search parameter but httpx's param encoding would percent-encode
        # the '+' characters, which breaks the query.
        search = (
            f'openfda.generic_name:"{quote(drug_a)}"'
            f'+AND+drug_interactions:"{quote(drug_b)}"'
        )
        url = f"{_OPENFDA_BASE}?search={search}&limit=1"
        resp = await client.get(url, timeout=15.0)
        if resp.status_code == 404:
            # openFDA returns 404 when no results match.
            return None
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    results = body.get("results", [])
    if not results:
        return None

    sections = results[0].get("drug_interactions", [])
    if not sections:
        return None

    # The section is one big blob.  Extract the paragraph that mentions drug_b.
    full_text = sections[0]
    paragraphs = re.split(r"\n{2,}|\r\n{2,}", full_text)
    # If the text isn't really multi-paragraph, split on sentence-ish boundaries.
    if len(paragraphs) <= 1:
        paragraphs = re.split(r"(?<=\.)\s+(?=[A-Z])", full_text)

    drug_b_lower = drug_b.lower()
    relevant = [
        _clean_label_text(p)
        for p in paragraphs
        if drug_b_lower in p.lower()
    ]

    if relevant:
        return " ".join(relevant)[:600]

    # Fallback: return a truncated version of the whole section.
    return _clean_label_text(full_text)[:400]


# ── Formatting ───────────────────────────────────────────────────────────────

def _format_results(
    interactions: list[dict],
    unresolved: list[str],
) -> str:
    """Format parsed interaction data into a readable string."""
    lines: list[str] = []

    if unresolved:
        for name in unresolved:
            lines.append(
                f"Could not find drug: {name}. Please verify spelling."
            )
        lines.append("")

    if interactions:
        lines.append(f"Found {len(interactions)} interaction(s):\n")
        for ix in interactions:
            lines.append(f"- {ix['drug_pair']}")
            lines.append(f"  Severity: {ix['severity']}")
            lines.append(f"  {ix['description']}")
            lines.append("")
    elif not unresolved:
        lines.append(
            "No known interactions found between these medications."
        )

    return "\n".join(lines).strip()


# ── Tool ─────────────────────────────────────────────────────────────────────

@tool
async def drug_interaction_check(drug_names: list[str]) -> str:
    """Check for dangerous interactions between two or more drugs.

    Use this tool when you need to verify whether a set of medications can be
    safely taken together.  Provide a list of drug names (e.g., ["aspirin",
    "warfarin"]).  The tool checks FDA-approved drug labelling for documented
    interactions between every pair of drugs in the list.

    Examples:
        - drug_interaction_check(drug_names=["aspirin", "warfarin"])
        - drug_interaction_check(drug_names=["lisinopril", "potassium", "spironolactone"])
    """
    if not drug_names or len(drug_names) < 2:
        return "Error: At least two drug names are required to check interactions."

    canonical: dict[str, str] = {}  # original -> canonical name
    unresolved: list[str] = []

    async with httpx.AsyncClient(verify=True) as client:
        # 1. Normalise drug names via RxNorm (concurrently).
        resolve_tasks = {
            name: _resolve_drug_name(client, name) for name in drug_names
        }
        results = await asyncio.gather(
            *resolve_tasks.values(), return_exceptions=True,
        )
        for name, result in zip(resolve_tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error("RxNorm lookup failed for %s: %s", name, result)
                unresolved.append(name)
            elif result is None:
                unresolved.append(name)
            else:
                canonical[name] = result

        if len(canonical) < 2:
            return _format_results([], unresolved)

        # 2. Check every ordered pair via openFDA labels (concurrently).
        #    We check both directions (A's label for B, and B's label for A)
        #    because labelling isn't always symmetric.
        resolved_names = list(canonical.values())
        pair_tasks: dict[tuple[str, str], asyncio.Task] = {}
        for a, b in itertools.combinations(resolved_names, 2):
            pair_tasks[(a, b)] = _search_label_interactions(client, a, b)
            pair_tasks[(b, a)] = _search_label_interactions(client, b, a)

        pair_results = await asyncio.gather(
            *pair_tasks.values(), return_exceptions=True,
        )

    # 3. Collate results — deduplicate per unordered pair.
    seen_pairs: set[tuple[str, str]] = set()
    interactions: list[dict] = []

    for (a, b), result in zip(pair_tasks.keys(), pair_results):
        if isinstance(result, Exception) or result is None:
            continue
        pair_key = tuple(sorted((a, b)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        interactions.append({
            "drug_pair": f"{pair_key[0]} + {pair_key[1]}",
            "severity": "see description",
            "description": result,
        })

    return _format_results(interactions, unresolved)


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    async def _test():
        print("=== Drug Interaction Check: aspirin + warfarin ===\n")
        result = await drug_interaction_check.ainvoke(
            {"drug_names": ["aspirin", "warfarin"]},
        )
        print(result)

    asyncio.run(_test())
