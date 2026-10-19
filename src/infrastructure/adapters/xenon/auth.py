"""Pluggable auth for the xenon WS connection (decision D2).

Default ``NoAuthProvider`` is the trusted-network / loopback path xenon already
permits (it skips ticket validation for localhost or when CLERK is unconfigured).
``ApiKeyAuthProvider`` is xenon's service-client path: the upgrade request carries
``X-API-Key`` and no ``Origin`` header. A ``TicketAuthProvider`` (service-JWT) can
drop in later with no client change.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class AuthProvider(Protocol):
    """Yield a connection ticket and extra upgrade headers for the xenon WS."""

    async def ticket(self) -> Optional[str]: ...

    def headers(self) -> Dict[str, str]: ...


class NoAuthProvider:
    """Trusted-network path: no ticket, no headers."""

    async def ticket(self) -> Optional[str]:
        return None

    def headers(self) -> Dict[str, str]:
        return {}


class ApiKeyAuthProvider:
    """Service-client path: send ``X-API-Key`` on the WS upgrade.

    SECURITY: the key is a secret. ``__repr__`` redacts it; never log ``headers()``.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def __repr__(self) -> str:
        return "ApiKeyAuthProvider(api_key=<redacted>)"

    async def ticket(self) -> Optional[str]:
        return None

    def headers(self) -> Dict[str, str]:
        return {"X-API-Key": self._api_key}
