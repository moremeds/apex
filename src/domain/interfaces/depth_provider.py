"""Depth (order book) provider protocol for real-time level 2 data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True, slots=True)
class DepthLevel:
    """Single price level in the order book."""

    price: float
    volume: float
    order_count: int = 0


@dataclass(frozen=True, slots=True)
class DepthSnapshot:
    """Point-in-time order book snapshot."""

    symbol: str
    bids: Tuple[DepthLevel, ...]
    asks: Tuple[DepthLevel, ...]
    timestamp: datetime
    source: str = ""


@runtime_checkable
class DepthProvider(Protocol):
    """Protocol for real-time order book depth streaming."""

    async def subscribe_depth(self, symbols: List[str]) -> None:
        """Subscribe to depth updates for symbols."""
        ...

    async def unsubscribe_depth(self, symbols: List[str]) -> None:
        """Unsubscribe from depth updates."""
        ...

    def set_depth_callback(
        self, callback: Optional[Callable[[DepthSnapshot], None]]
    ) -> None:
        """Set callback for incoming depth snapshots."""
        ...

    def get_latest_depth(self, symbol: str) -> Optional[DepthSnapshot]:
        """Get the most recent cached depth for a symbol."""
        ...
