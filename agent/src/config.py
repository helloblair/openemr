"""Load environment variables from .env for the OpenEMR agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the agent/ directory (one level up from src/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENEMR_BASE_URL: str = os.environ.get("OPENEMR_BASE_URL", "http://localhost:8300")
OPENEMR_CLIENT_ID: str = os.environ.get("OPENEMR_CLIENT_ID", "")
OPENEMR_CLIENT_SECRET: str = os.environ.get("OPENEMR_CLIENT_SECRET", "")
OPENEMR_USERNAME: str = os.environ.get("OPENEMR_USERNAME", "admin")
OPENEMR_PASSWORD: str = os.environ.get("OPENEMR_PASSWORD", "pass")
OPENEMR_SCOPES: str = os.environ.get(
    "OPENEMR_SCOPES",
    "openid api:oemr api:fhir user/patient.read user/allergy.read",
)
