"""Pluggable auth for the xenon WS connection (decision D2).

Default ``NoAuthProvider`` is the trusted-network / loopback path xenon already
permits (it skips ticket validation for localhost or when CLERK is unconfigured).
A ``TicketAuthProvider`` (service-JWT) can drop in later with no client change.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class AuthProvider(Protocol):
    """Yield a connection ticket, or ``None`` for the no-auth path."""

    async def ticket(self) -> Optional[str]:
        ...


class NoAuthProvider:
    """Trusted-network path: no ticket. Returns ``None``."""

    async def ticket(self) -> Optional[str]:
        return None
