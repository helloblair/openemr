"""Allergy check tool for retrieving a patient's documented allergies from OpenEMR.

This module provides a LangGraph-compatible tool that queries the OpenEMR
FHIR API to fetch AllergyIntolerance resources for a given patient.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path

import httpx
from langchain_core.tools import tool

# Ensure the agent package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.auth.oauth2 import OpenEMRAuth

logger = logging.getLogger(__name__)

_FHIR_PREFIX = "/apis/default/fhir"


def _strip_html(html: str) -> str:
    """Remove HTML tags and return plain text."""
    return re.sub(r"<[^>]+>", "", html).strip()


def _is_absent_reason(codings: list[dict]) -> bool:
    """Check if the coding uses a data-absent-reason system (i.e. no real code)."""
    return any(
        "data-absent-reason" in (c.get("system") or "") for c in codings
    )


def _parse_allergy(resource: dict) -> dict:
    """Extract relevant fields from a FHIR AllergyIntolerance resource."""
    # Substance name — OpenEMR often puts a data-absent-reason in code.coding
    # and stores the actual name in text.div as HTML.
    code = resource.get("code", {})
    codings = code.get("coding", [])

    # Prefer code.text or code.coding[0].display when it's a real code
    substance = None
    if code.get("text"):
        substance = code["text"]
    elif codings and not _is_absent_reason(codings):
        substance = codings[0].get("display")

    # Fall back to the narrative text.div (strip HTML tags)
    if not substance:
        text_div = resource.get("text", {}).get("div", "")
        if text_div:
            substance = _strip_html(text_div)

    substance = substance or "Unknown substance"

    # Category (medication, food, environment)
    categories = resource.get("category", [])
    category = categories[0] if categories else "unknown"

    # Criticality
    criticality = resource.get("criticality", "unknown")

    # Reactions
    reactions: list[str] = []
    for reaction_entry in resource.get("reaction", []):
        manifestations = reaction_entry.get("manifestation", [])
        for manifestation in manifestations:
            m_codings = manifestation.get("coding", [])
            text = (
                m_codings[0].get("display") if m_codings else None
            ) or manifestation.get("text")
            if text:
                reactions.append(text)
        # Also check the description field
        description = reaction_entry.get("description")
        if description and description not in reactions:
            reactions.append(description)

    return {
        "substance": substance,
        "category": category,
        "criticality": criticality,
        "reactions": reactions if reactions else None,
    }


def _format_allergies(allergies: list[dict]) -> str:
    """Format a list of parsed allergy dicts into a readable string."""
    lines: list[str] = []
    lines.append(f"Found {len(allergies)} documented allergy(ies):\n")

    for a in allergies:
        lines.append(f"- {a['substance']}")
        lines.append(f"  Category: {a['category']}  |  Criticality: {a['criticality']}")
        if a["reactions"]:
            lines.append(f"  Reactions: {', '.join(a['reactions'])}")
        lines.append("")

    return "\n".join(lines).strip()


@tool
async def allergy_check(patient_uuid: str) -> str:
    """Retrieve a patient's documented allergies from the medical records system.

    Use this tool when you need to check what allergies a patient has on file.
    Provide the patient's UUID (obtainable from the patient_lookup tool).

    Returns a list of allergies with substance name, category, criticality,
    and any documented reactions. Returns a message if no allergies are found.

    Examples:
        - allergy_check(patient_uuid="95f2f211-3580-4b58-9a15-2d9b29f49c72")
    """
    if not patient_uuid or not patient_uuid.strip():
        return "Error: patient_uuid is required."

    auth = OpenEMRAuth()

    try:
        async with auth.get_client() as client:
            resp = await client.get(
                f"{_FHIR_PREFIX}/AllergyIntolerance",
                params={"patient": patient_uuid},
                timeout=10.0,
            )

            if resp.status_code == 401:
                # Force a fresh token and retry once.
                auth._access_token = None
                auth._refresh_token = None
                async with auth.get_client() as retry_client:
                    resp = await retry_client.get(
                        f"{_FHIR_PREFIX}/AllergyIntolerance",
                        params={"patient": patient_uuid},
                        timeout=10.0,
                    )

            resp.raise_for_status()

    except httpx.TimeoutException:
        return "Unable to reach medical records system. Please try again."
    except httpx.HTTPStatusError as exc:
        logger.error("Allergy check API error: %s", exc)
        return "Unable to reach medical records system. Please try again."

    body = resp.json()

    # FHIR Bundle: entries live under "entry"; an empty bundle may omit the key.
    entries = body.get("entry", [])

    if not entries:
        return "No allergies documented for this patient."

    allergies = [_parse_allergy(entry.get("resource", entry)) for entry in entries]

    return _format_allergies(allergies)


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    async def _test():
        # First, look up a test patient to get their UUID.
        from src.tools.patient_lookup import patient_lookup

        print("=== Step 1: Look up John Smith ===\n")
        lookup_result = await patient_lookup.ainvoke({"last_name": "Smith", "first_name": "John"})
        print(lookup_result)

        # Extract UUID from the lookup result.
        uuid = None
        for line in lookup_result.splitlines():
            if "UUID:" in line:
                uuid = line.split("UUID:")[1].strip().rstrip(")")
                break

        if not uuid:
            print("\nCould not extract UUID from lookup result. Exiting.")
            return

        print(f"\n=== Step 2: Check allergies for UUID {uuid} ===\n")
        result = await allergy_check.ainvoke({"patient_uuid": uuid})
        print(result)

    asyncio.run(_test())
