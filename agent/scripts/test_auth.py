"""Test script for the OpenEMR OAuth2 authentication flow.

Run from the agent/ directory:

    python -m scripts.test_auth
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

# Ensure config loads .env before anything else
import src.config as cfg  # noqa: F401
from src.auth.oauth2 import OpenEMRAuth

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    auth = OpenEMRAuth()

    # Step 1 — Register a client if credentials are missing
    if not auth.client_id or not auth.client_secret:
        logger.info("No client credentials found — registering a new client …")
        client_id, client_secret = await auth.register_client()
        print("\n=== Save these in your .env file ===")
        print(f"OPENEMR_CLIENT_ID={client_id}")
        print(f"OPENEMR_CLIENT_SECRET={client_secret}")
        print("====================================\n")

        # Enable the client in the Docker database (required before token request)
        logger.info("Enabling the client in Docker …")
        OpenEMRAuth.enable_client_via_docker(client_id)

    # Step 2 — Obtain an access token (password grant)
    token = await auth.ensure_token()
    logger.info("Access token acquired (first 12 chars): %s…", token[:12])

    # Step 3 — Make an authenticated API request
    async with auth.get_client() as client:
        resp = await client.get("/apis/default/api/patient")
        resp.raise_for_status()
        data = resp.json()
        print("\n=== GET /apis/default/api/patient ===")
        print(json.dumps(data, indent=2)[:2000])
        print("=====================================\n")

    logger.info("Auth flow completed successfully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.error("Auth test failed: %s", exc)
        sys.exit(1)
