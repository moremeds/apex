"""
Thread-safe in-memory position store with event subscription.

OPT-014: Uses RCU pattern for lock-free reads on the main data path.
"""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING, Iterable, List, Optional

from ...models.position import Position
from ...utils.logging_setup import get_logger
from .rcu_store import RCUDict

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus

logger = get_logger(__name__)


class PositionStore:
    """
    Thread-safe in-memory position store keyed by Position.key().

    OPT-014: Uses RCU pattern for lock-free reads.
    """

    def __init__(self) -> None:
        # OPT-014: Use RCUDict for lock-free reads
        self._positions: RCUDict[tuple, Position] = RCUDict()
        self._refresh_lock = RLock()  # Only for refresh flag coordination
        self._needs_refresh = False  # Flag for position update signal

    def upsert_positions(self, positions: Iterable[Position]) -> None:
        """
        Insert or update positions.

        OPT-014: RCUDict.update() handles atomicity internally.

        Args:
            positions: Iterable of Position objects.
        """
        updates = {p.key(): p for p in positions}
        self._positions.update(updates)

    def get_all(self) -> List[Position]:
        """
        Get all positions (thread-safe copy).

        OPT-014: Lock-free read via RCUDict.values() snapshot.
        """
        return self._positions.values()

    def get_by_key(self, key: tuple) -> Optional[Position]:
        """
        Get position by key.

        OPT-014: Lock-free read via RCUDict.get().
        """
        return self._positions.get(key)

    def get_by_underlying(self, underlying: str) -> List[Position]:
        """
        Get all positions for a specific underlying.

        OPT-014: Lock-free read via RCUDict.values() snapshot.
        """
        return [p for p in self._positions.values() if p.underlying == underlying]

    def clear(self) -> None:
        """
        Clear all positions.

        OPT-014: RCUDict.clear() handles atomicity internally.
        """
        self._positions.clear()

    def count(self) -> int:
        """
        Get position count.

        OPT-014: Lock-free read via RCUDict.__len__().
        """
        return len(self._positions)

    def subscribe_to_events(self, event_bus: "EventBus") -> None:
        """
        Subscribe to position-related events.

        Args:
            event_bus: Event bus to subscribe to.
        """
        from ...domain.interfaces.event_bus import EventType

        event_bus.subscribe(EventType.POSITIONS_BATCH, self._on_positions_batch)
        event_bus.subscribe(EventType.POSITION_UPDATED, self._on_position_updated)
        logger.debug("PositionStore subscribed to events")

    def _on_positions_batch(self, payload: dict) -> None:
        """
        Handle batch position update event.

        Args:
            payload: Event payload with 'positions' list.
        """
        positions = payload.get("positions", [])
        if positions:
            self.upsert_positions(positions)
            logger.debug(f"PositionStore updated from batch: {len(positions)} positions")

    def _on_position_updated(self, payload: dict) -> None:
        """
        Handle single position update event (e.g., from trade deal).

        Marks that positions need refresh on next poll cycle.

        Args:
            payload: Event payload with position update info.
        """
        with self._refresh_lock:
            self._needs_refresh = True
        logger.debug(f"Position update signal: {payload.get('symbol', 'unknown')}")

    def needs_refresh(self) -> bool:
        """Check if positions need refresh (trade deal received)."""
        with self._refresh_lock:
            return self._needs_refresh

    def clear_refresh_flag(self) -> None:
        """Clear the refresh flag after positions have been refreshed."""
        with self._refresh_lock:
            self._needs_refresh = False
