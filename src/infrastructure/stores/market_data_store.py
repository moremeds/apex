"""Thread-safe in-memory market data store with separate price/Greeks caching."""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional
from threading import RLock
from datetime import datetime, timedelta
from ...models.market_data import MarketData


class MarketDataStore:
    """
    Thread-safe in-memory market data store keyed by symbol.

    Features:
    - Separate TTLs for price vs Greeks (prices need real-time, Greeks change slowly)
    - Automatic staleness detection
    - Thread-safe operations
    """

    def __init__(
        self,
        price_ttl_seconds: int = 5,
        greeks_ttl_seconds: int = 60,
    ) -> None:
        """
        Initialize market data store.

        Args:
            price_ttl_seconds: Time-to-live for price data in seconds.
                               Prices older than this trigger a refresh.
                               Default: 5 seconds (need near real-time prices).
            greeks_ttl_seconds: Time-to-live for Greeks cache in seconds.
                                Greeks older than this will be considered stale.
                                Default: 60 seconds (Greeks change slowly for most instruments).
        """
        self._market_data: Dict[str, MarketData] = {}
        self._lock = RLock()
        self._price_ttl_seconds = price_ttl_seconds
        self._greeks_ttl_seconds = greeks_ttl_seconds

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
        """
        Get market data for a symbol.

        Note: Returns cached data even if Greeks are stale.
        Use is_greeks_stale() to check freshness.
        """
        with self._lock:
            return self._market_data.get(symbol)

    def is_greeks_stale(self, symbol: str) -> bool:
        """
        Check if Greeks data is stale for a symbol.

        Args:
            symbol: Symbol to check.

        Returns:
            True if Greeks are stale or missing, False if fresh.
        """
        with self._lock:
            md = self._market_data.get(symbol)
            if not md or not md.timestamp:
                return True

            age = (datetime.now() - md.timestamp).total_seconds()
            return age > self._greeks_ttl_seconds

    def get_stale_symbols(self) -> List[str]:
        """
        Get list of symbols with stale Greeks.

        Returns:
            List of symbols that need Greeks refresh.
        """
        with self._lock:
            stale = []
            for symbol, md in self._market_data.items():
                if not md.timestamp:
                    stale.append(symbol)
                    continue

                age = (datetime.now() - md.timestamp).total_seconds()
                if age > self._greeks_ttl_seconds:
                    stale.append(symbol)

            return stale

    def get_all(self) -> Dict[str, MarketData]:
        """Get all market data (thread-safe copy)."""
        with self._lock:
            return dict(self._market_data)

    def get_symbols(self) -> List[str]:
        """Get all symbols."""
        with self._lock:
            return list(self._market_data.keys())

    def get_symbols_needing_refresh(self, requested_symbols: List[str]) -> List[str]:
        """
        Get symbols that need market data refresh (stale price or missing) - atomic operation.

        Uses price_ttl_seconds for staleness check (need real-time prices).
        Greeks staleness is checked separately via is_greeks_stale().

        Args:
            requested_symbols: List of symbols we want market data for.

        Returns:
            List of symbols that need refresh (price stale or not in cache).
        """
        with self._lock:
            stale_set = set()
            existing_set = set(self._market_data.keys())

            # Check for stale prices (using price TTL, not Greeks TTL)
            for symbol, md in self._market_data.items():
                if not md.timestamp:
                    stale_set.add(symbol)
                    continue
                age = (datetime.now() - md.timestamp).total_seconds()
                if age > self._price_ttl_seconds:
                    stale_set.add(symbol)

            # Return symbols that are stale OR not in cache
            requested_set = set(requested_symbols)
            missing_set = requested_set - existing_set
            return list(stale_set.union(missing_set) & requested_set)

    def clear(self) -> None:
        """Clear all market data."""
        with self._lock:
            self._market_data.clear()

    def count(self) -> int:
        """Get market data count."""
        with self._lock:
            return len(self._market_data)
