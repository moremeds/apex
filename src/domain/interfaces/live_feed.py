"""Port for a live market-data feed (e.g. xenon's IB tick WS).

A feed-agnostic surface so the subscription layer can open/drop a live
subscription per ticker without knowing the transport (xenon, a fake, etc.).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LiveFeedPort(Protocol):
    """Open a connection, (un)subscribe per symbol, and close it."""

    async def connect(self) -> None:
        """Start the feed connection (non-blocking: launches the background loop)."""
        ...

    async def subscribe(self, symbol: str) -> None:
        """Begin receiving live ticks for ``symbol``."""
        ...

    async def unsubscribe(self, symbol: str) -> None:
        """Stop receiving live ticks for ``symbol``."""
        ...

    async def close(self) -> None:
        """Tear down the feed connection and stop the background loop."""
        ...
