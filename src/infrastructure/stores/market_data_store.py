"""Thread-safe in-memory market data store."""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional
from threading import RLock
from ...models.market_data import MarketData


class MarketDataStore:
    """Thread-safe in-memory market data store keyed by symbol."""

    def __init__(self) -> None:
        self._market_data: Dict[str, MarketData] = {}
        self._lock = RLock()

    def upsert(self, market_data: Iterable[MarketData]) -> None:
        """
        Insert or update market data.

        Args:
            market_data: Iterable of MarketData objects.
        """
        with self._lock:
            for md in market_data:
                self._market_data[md.symbol] = md

    def get(self, symbol: str) -> Optional[MarketData]:
        """Get market data for a symbol."""
        with self._lock:
            return self._market_data.get(symbol)

    def get_all(self) -> Dict[str, MarketData]:
        """Get all market data (thread-safe copy)."""
        with self._lock:
            return dict(self._market_data)

    def get_symbols(self) -> List[str]:
        """Get all symbols."""
        with self._lock:
            return list(self._market_data.keys())

    def clear(self) -> None:
        """Clear all market data."""
        with self._lock:
            self._market_data.clear()

    def count(self) -> int:
        """Get market data count."""
        with self._lock:
            return len(self._market_data)
