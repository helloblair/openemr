"""Patient lookup tool for searching OpenEMR patients by name or DOB.

This module provides a LangGraph-compatible tool that queries the OpenEMR
REST API to find patient records.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.tools import tool

# Ensure the agent package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.auth.oauth2 import OpenEMRAuth

logger = logging.getLogger(__name__)

_API_PREFIX = "/apis/default/api"
_MAX_RESULTS = 5


def _format_patient(patient: dict) -> dict:
    """Extract the fields we care about from a raw API patient record."""
    street = patient.get("street", "")
    city = patient.get("city", "")
    state = patient.get("state", "")
    postal = patient.get("postal_code", "")
    address_parts = [p for p in (street, city, state, postal) if p]

    return {
        "uuid": patient.get("uuid", ""),
        "name": f"{patient.get('fname', '')} {patient.get('lname', '')}".strip(),
        "dob": patient.get("DOB", ""),
        "sex": patient.get("sex", ""),
        "phone": patient.get("phone_home", "") or patient.get("phone_cell", ""),
        "address": ", ".join(address_parts),
    }


@tool
async def patient_lookup(
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    dob: Optional[str] = None,
) -> str:
    """Search for a patient in the medical records system by name or date of birth.

    Use this tool when you need to find a patient's record. You can search by
    first name, last name, date of birth (YYYY-MM-DD format), or any combination.
    At least one search field must be provided.

    Returns patient details including UUID, full name, DOB, sex, phone, and address.

    Examples:
        - patient_lookup(last_name="Smith")
        - patient_lookup(first_name="Jane", last_name="Doe")
        - patient_lookup(dob="1980-01-15")
    """
    if not any([first_name, last_name, dob]):
        return "Error: At least one search field (first_name, last_name, or dob) must be provided."

    params: dict[str, str] = {}
    if first_name:
        params["fname"] = first_name
    if last_name:
        params["lname"] = last_name
    if dob:
        params["DOB"] = dob

    auth = OpenEMRAuth()

    try:
        async with auth.get_client() as client:
            resp = await client.get(
                f"{_API_PREFIX}/patient",
                params=params,
                timeout=10.0,
            )

            if resp.status_code == 401:
                # Force a fresh token and retry once.
                auth._access_token = None
                auth._refresh_token = None
                async with auth.get_client() as retry_client:
                    resp = await retry_client.get(
                        f"{_API_PREFIX}/patient",
                        params=params,
                        timeout=10.0,
                    )

            resp.raise_for_status()

    except httpx.TimeoutException:
        return "Unable to reach medical records system. Please try again."
    except httpx.HTTPStatusError as exc:
        logger.error("Patient lookup API error: %s", exc)
        return "Unable to reach medical records system. Please try again."

    body = resp.json()

    # The API wraps results in a "data" key.
    patients = body if isinstance(body, list) else body.get("data", [])

    if not patients:
        return "No patients found matching criteria."

    formatted = [_format_patient(p) for p in patients[:_MAX_RESULTS]]

    lines: list[str] = []
    if len(patients) > _MAX_RESULTS:
        lines.append(
            f"Multiple patients found ({len(patients)} total, showing first {_MAX_RESULTS}):\n"
        )
    elif len(patients) > 1:
        lines.append(f"Multiple patients found ({len(patients)}):\n")

    for p in formatted:
        lines.append(f"- {p['name']} (UUID: {p['uuid']})")
        lines.append(f"  DOB: {p['dob']}  Sex: {p['sex']}")
        if p["phone"]:
            lines.append(f"  Phone: {p['phone']}")
        if p["address"]:
            lines.append(f"  Address: {p['address']}")
        lines.append("")

    return "\n".join(lines).strip()


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    async def _test():
        print("=== Patient Lookup Test ===\n")
        result = await patient_lookup.ainvoke({"last_name": "Smith"})
        print(result)

    asyncio.run(_test())
