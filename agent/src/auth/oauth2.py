"""OpenEMR OAuth2 client for obtaining and refreshing API tokens."""

from __future__ import annotations

import logging
import time

import httpx

from src.config import (
    OPENEMR_BASE_URL,
    OPENEMR_CLIENT_ID,
    OPENEMR_CLIENT_SECRET,
    OPENEMR_USERNAME,
    OPENEMR_PASSWORD,
    OPENEMR_SCOPES,
)

logger = logging.getLogger(__name__)

_REGISTRATION_PATH = "/oauth2/default/registration"
_TOKEN_PATH = "/oauth2/default/token"


class OpenEMRAuth:
    """Handles OAuth2 registration, token acquisition, and refresh for OpenEMR.

    After registering a client you must **enable** it before requesting tokens.
    Enable via the OpenEMR admin UI (Administration > System > API Clients) or
    by calling :pymethod:`enable_client_via_docker`.

    Usage::

        auth = OpenEMRAuth()
        async with auth.get_client() as client:
            resp = await client.get("/apis/default/api/patient")
    """

    def __init__(
        self,
        base_url: str = OPENEMR_BASE_URL,
        client_id: str = OPENEMR_CLIENT_ID,
        client_secret: str = OPENEMR_CLIENT_SECRET,
        username: str = OPENEMR_USERNAME,
        password: str = OPENEMR_PASSWORD,
        scopes: str = OPENEMR_SCOPES,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.scopes = scopes

        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register_client(self) -> tuple[str, str]:
        """Register a new OAuth2 client with OpenEMR.

        Returns ``(client_id, client_secret)``.  Only needs to be called once;
        persist the values in your ``.env`` file.

        .. note::

           The newly registered client is **disabled** by default.  You must
           enable it before you can request tokens — either through the admin
           UI or by calling :pymethod:`enable_client_via_docker`.
        """
        payload = {
            "application_type": "private",
            "redirect_uris": [self.base_url],
            "post_logout_redirect_uris": [self.base_url],
            "client_name": "AgentForge Healthcare Agent",
            "token_endpoint_auth_method": "client_secret_post",
            "contacts": ["agent@example.com"],
            "scope": self.scopes,
        }

        async with httpx.AsyncClient(verify=False) as client:
            resp = await client.post(
                f"{self.base_url}{_REGISTRATION_PATH}",
                json=payload,
            )
            resp.raise_for_status()

        data = resp.json()
        self.client_id = data["client_id"]
        self.client_secret = data["client_secret"]
        logger.info("Registered OAuth2 client: %s", self.client_id)
        return self.client_id, self.client_secret

    @staticmethod
    def enable_client_via_docker(
        client_id: str,
        container: str = "development-easy-openemr-1",
    ) -> None:
        """Enable an OAuth2 client by running SQL inside the Docker container.

        This is a convenience for local development.  In production you would
        enable the client via the OpenEMR admin UI.
        """
        import subprocess

        sql = (
            f"UPDATE oauth_clients SET is_enabled = 1 "
            f"WHERE client_id = '{client_id}';"
        )
        cmd = [
            "docker", "exec", container,
            "mysql", "-u", "root", "--password=root", "openemr",
            "-e", sql,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to enable client in Docker: {result.stderr.strip()}"
            )
        logger.info("Enabled OAuth2 client %s in container %s", client_id, container)

    async def ensure_token(self) -> str:
        """Return a valid access token, refreshing or acquiring as needed."""
        if self._access_token and time.time() < self._expires_at:
            return self._access_token

        if self._refresh_token:
            try:
                return await self._refresh()
            except httpx.HTTPStatusError:
                logger.warning("Refresh failed, re-authenticating with password grant")

        return await self._password_grant()

    def get_client(self) -> _AuthenticatedClient:
        """Return a context-managed ``httpx.AsyncClient`` with auth headers.

        Usage::

            async with auth.get_client() as client:
                resp = await client.get("/apis/default/api/patient")
        """
        return _AuthenticatedClient(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _password_grant(self) -> str:
        """Acquire tokens using the resource-owner password grant."""
        if not self.client_id:
            raise RuntimeError(
                "client_id not set. "
                "Call register_client() first or set it in .env."
            )

        form = {
            "grant_type": "password",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scopes,
            "user_role": "users",
            "username": self.username,
            "password": self.password,
        }

        async with httpx.AsyncClient(verify=False) as client:
            resp = await client.post(
                f"{self.base_url}{_TOKEN_PATH}",
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                logger.error(
                    "Token request failed (%s): %s", resp.status_code, resp.text
                )
            resp.raise_for_status()

        self._store_tokens(resp.json())
        logger.info("Obtained access token via password grant")
        return self._access_token

    async def _refresh(self) -> str:
        """Refresh the access token using the stored refresh_token."""
        form = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self._refresh_token,
        }

        async with httpx.AsyncClient(verify=False) as client:
            resp = await client.post(
                f"{self.base_url}{_TOKEN_PATH}",
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                logger.error(
                    "Refresh request failed (%s): %s", resp.status_code, resp.text
                )
            resp.raise_for_status()

        self._store_tokens(resp.json())
        logger.info("Refreshed access token")
        return self._access_token

    def _store_tokens(self, data: dict) -> None:
        self._access_token = data["access_token"]
        self._refresh_token = data.get("refresh_token", self._refresh_token)
        expires_in = int(data.get("expires_in", 3600))
        # Subtract 30 s buffer so we refresh before actual expiry.
        self._expires_at = time.time() + expires_in - 30


class _AuthenticatedClient:
    """Async context manager that yields an ``httpx.AsyncClient`` with a
    valid Bearer token.  Re-injects the token on every request via an
    event hook so long-lived clients stay authenticated.
    """

    def __init__(self, auth: OpenEMRAuth) -> None:
        self._auth = auth
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> httpx.AsyncClient:
        token = await self._auth.ensure_token()
        self._client = httpx.AsyncClient(
            base_url=self._auth.base_url,
            verify=False,
            headers={"Authorization": f"Bearer {token}"},
            event_hooks={"request": [self._inject_token]},
        )
        return self._client

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()

    async def _inject_token(self, request: httpx.Request) -> None:
        """Event hook: refresh the token if needed before each request."""
        token = await self._auth.ensure_token()
        request.headers["Authorization"] = f"Bearer {token}"
