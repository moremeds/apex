"""Thread-safe in-memory position store with event subscription."""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING
from threading import RLock
import logging

from ...models.position import Position

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus

logger = logging.getLogger(__name__)


class PositionStore:
    """Thread-safe in-memory position store keyed by Position.key()."""

    def __init__(self) -> None:
        self._positions: Dict[tuple, Position] = {}
        self._lock = RLock()
        self._needs_refresh = False  # Flag for position update signal

    def upsert_positions(self, positions: Iterable[Position]) -> None:
        """
        Insert or update positions.

        Args:
            positions: Iterable of Position objects.
        """
        with self._lock:
            for p in positions:
                self._positions[p.key()] = p

    def get_all(self) -> List[Position]:
        """Get all positions (thread-safe copy)."""
        with self._lock:
            return list(self._positions.values())

    def get_by_key(self, key: tuple) -> Optional[Position]:
        """Get position by key."""
        with self._lock:
            return self._positions.get(key)

    def get_by_underlying(self, underlying: str) -> List[Position]:
        """Get all positions for a specific underlying."""
        with self._lock:
            return [p for p in self._positions.values() if p.underlying == underlying]

    def clear(self) -> None:
        """Clear all positions."""
        with self._lock:
            self._positions.clear()

    def count(self) -> int:
        """Get position count."""
        with self._lock:
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
        with self._lock:
            self._needs_refresh = True
        logger.debug(f"Position update signal: {payload.get('symbol', 'unknown')}")

    def needs_refresh(self) -> bool:
        """Check if positions need refresh (trade deal received)."""
        with self._lock:
            return self._needs_refresh

    def clear_refresh_flag(self) -> None:
        """Clear the refresh flag after positions have been refreshed."""
        with self._lock:
            self._needs_refresh = False
