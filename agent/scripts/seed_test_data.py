"""Seed OpenEMR with test patients and allergies via the REST API.

Run from the agent/ directory:
    python -m scripts.seed_test_data
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure the agent package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auth.oauth2 import OpenEMRAuth

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Test data ────────────────────────────────────────────────────────────────

PATIENTS = [
    {"fname": "John", "lname": "Smith", "DOB": "1980-01-15", "sex": "Male"},
    {"fname": "Jane", "lname": "Doe", "DOB": "1975-06-20", "sex": "Female"},
    {"fname": "Bob", "lname": "Wilson", "DOB": "1990-03-10", "sex": "Male"},
]

# Allergies keyed by (fname, lname) → list of allergy dicts
ALLERGIES: dict[tuple[str, str], list[dict]] = {
    ("John", "Smith"): [
        {"title": "Penicillin", "comments": "Medication allergy – category: medication"},
    ],
    ("Jane", "Doe"): [
        {"title": "Sulfa drugs", "comments": "Medication allergy – category: medication"},
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

API_PREFIX = "/apis/default/api"


async def create_patient(client, patient: dict) -> dict:
    """POST a new patient and return the response JSON."""
    resp = await client.post(f"{API_PREFIX}/patient", json=patient)
    if resp.status_code == 201:
        data = resp.json().get("data", resp.json())
        logger.info(
            "Created patient %s %s  (pid=%s, uuid=%s)",
            patient["fname"],
            patient["lname"],
            data.get("pid", "?"),
            data.get("uuid", "?"),
        )
        return data

    logger.error(
        "Failed to create %s %s — %s: %s",
        patient["fname"],
        patient["lname"],
        resp.status_code,
        resp.text,
    )
    return {}


async def add_allergy(client, patient_uuid: str, allergy: dict) -> dict:
    """POST an allergy for the given patient UUID."""
    resp = await client.post(
        f"{API_PREFIX}/patient/{patient_uuid}/allergy",
        json=allergy,
    )
    # OpenEMR returns 200 (not 201) for allergy creation.
    if resp.status_code in (200, 201):
        body = resp.json()
        data = body.get("data", body)
        if body.get("validationErrors") or body.get("internalErrors"):
            logger.error(
                "  Allergy '%s' returned errors: %s",
                allergy["title"],
                resp.text,
            )
            return {}
        logger.info(
            "  Added allergy '%s' (uuid=%s)",
            allergy["title"],
            data.get("uuid", "?"),
        )
        return data

    logger.error(
        "  Failed to add allergy '%s' — %s: %s",
        allergy["title"],
        resp.status_code,
        resp.text,
    )
    return {}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    auth = OpenEMRAuth()

    created: dict[tuple[str, str], dict] = {}

    async with auth.get_client() as client:
        # 1. Create patients
        print("\n=== Creating test patients ===\n")
        for patient in PATIENTS:
            key = (patient["fname"], patient["lname"])
            result = await create_patient(client, patient)
            if result:
                created[key] = result

        # 2. Add allergies via the standard REST allergy endpoint
        print("\n=== Adding allergies ===\n")
        for (fname, lname), allergy_list in ALLERGIES.items():
            patient_data = created.get((fname, lname))
            if not patient_data:
                logger.warning("Skipping allergies for %s %s — patient not created", fname, lname)
                continue

            puuid = patient_data.get("uuid")
            if not puuid:
                logger.warning("No UUID for %s %s — skipping allergies", fname, lname)
                continue

            logger.info("Adding allergies for %s %s (uuid=%s):", fname, lname, puuid)
            for allergy in allergy_list:
                await add_allergy(client, puuid, allergy)

    # 3. Summary
    print("\n=== Summary ===\n")
    if not created:
        print("No patients were created. Check errors above.")
        return

    for (fname, lname), data in created.items():
        allergies = ALLERGIES.get((fname, lname), [])
        allergy_names = ", ".join(a["title"] for a in allergies) if allergies else "(none)"
        print(f"  {fname} {lname}")
        print(f"    pid  = {data.get('pid', '?')}")
        print(f"    uuid = {data.get('uuid', '?')}")
        print(f"    allergies = {allergy_names}")
        print()

    # Patients without allergies — manual instructions
    patients_without = [
        (f, l) for (f, l) in created if (f, l) not in ALLERGIES
    ]
    if patients_without:
        print("Patients without allergies added via API:")
        for fname, lname in patients_without:
            print(f"  - {fname} {lname}: add allergies via the UI at")
            print(f"    Patient > Medical Issues > Allergies")
        print()


if __name__ == "__main__":
    asyncio.run(main())
