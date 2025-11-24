"""Thread-safe in-memory position store."""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional
from threading import RLock
from ...models.position import Position


class PositionStore:
    """Thread-safe in-memory position store keyed by Position.key()."""

    def __init__(self) -> None:
        self._positions: Dict[tuple, Position] = {}
        self._lock = RLock()

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
