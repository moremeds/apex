"""Thread-safe in-memory market data store with separate price/Greeks caching."""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Union, TYPE_CHECKING
from threading import RLock
import time

from ...utils.logging_setup import get_logger
from ...models.market_data import MarketData
from ...utils.timezone import age_seconds

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus

logger = get_logger(__name__)

# Default eviction settings
DEFAULT_MAX_SYMBOLS = 10000
DEFAULT_EVICTION_AGE_SECONDS = 86400  # 24 hours


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
        max_symbols: int = DEFAULT_MAX_SYMBOLS,
        eviction_age_seconds: int = DEFAULT_EVICTION_AGE_SECONDS,
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
            max_symbols: Maximum number of symbols to cache before evicting oldest.
                        Default: 10,000 symbols.
            eviction_age_seconds: Evict entries older than this age (seconds).
                                  Default: 86400 (24 hours).
        """
        self._market_data: Dict[str, MarketData] = {}
        self._lock = RLock()
        self._price_ttl_seconds = price_ttl_seconds
        self._greeks_ttl_seconds = greeks_ttl_seconds
        self._max_symbols = max_symbols
        self._eviction_age_seconds = eviction_age_seconds
        self._last_eviction_time = time.time()

    def upsert(self, market_data: Iterable[MarketData]) -> None:
        """
        Insert or update market data.

        Args:
            market_data: Iterable of MarketData objects.
        """
        with self._lock:
            for md in market_data:
                self._market_data[md.symbol] = md

            # Periodic eviction check (every 60 seconds to avoid overhead)
            now = time.time()
            if now - self._last_eviction_time > 60:
                self._evict_stale_locked()
                self._last_eviction_time = now

    def _evict_stale_locked(self) -> None:
        """
        Evict stale entries (must be called with lock held).

        Evicts:
        1. Entries older than eviction_age_seconds
        2. Oldest entries if over max_symbols
        """
        # Evict by age first
        stale_symbols = []
        for symbol, md in self._market_data.items():
            if md.timestamp and age_seconds(md.timestamp) > self._eviction_age_seconds:
                stale_symbols.append(symbol)

        for symbol in stale_symbols:
            del self._market_data[symbol]

        if stale_symbols:
            logger.debug(f"Evicted {len(stale_symbols)} stale market data entries")

        # Evict by count if still over limit
        if len(self._market_data) > self._max_symbols:
            excess = len(self._market_data) - self._max_symbols
            # Sort by timestamp (oldest first)
            sorted_entries = sorted(
                self._market_data.items(),
                key=lambda x: x[1].timestamp.timestamp() if x[1].timestamp else 0
            )
            for symbol, _ in sorted_entries[:excess]:
                del self._market_data[symbol]
            logger.debug(f"Evicted {excess} oldest market data entries (over max {self._max_symbols})")

    def get(self, symbol: str) -> Optional[MarketData]:
        """
        Get market data for a symbol.

        Note: Returns cached data even if Greeks are stale.
        Use is_greeks_stale() to check freshness.
        """
        with self._lock:
            return self._market_data.get(symbol)

    def has_fresh_data(self, symbol: str) -> bool:
        """
        Check if we have fresh price data for a symbol.

        Used to determine if we need to subscribe to streaming for this symbol.

        Args:
            symbol: Symbol to check.

        Returns:
            True if we have fresh data (within price TTL), False otherwise.
        """
        with self._lock:
            md = self._market_data.get(symbol)
            if not md or not md.timestamp:
                return False
            return age_seconds(md.timestamp) <= self._price_ttl_seconds

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

            return age_seconds(md.timestamp) > self._greeks_ttl_seconds

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

                if age_seconds(md.timestamp) > self._greeks_ttl_seconds:
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
                if age_seconds(md.timestamp) > self._price_ttl_seconds:
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

    def subscribe_to_events(self, event_bus: "EventBus") -> None:
        """
        Subscribe to market data-related events.

        Args:
            event_bus: Event bus to subscribe to.
        """
        from ...domain.interfaces.event_bus import EventType

        event_bus.subscribe(EventType.MARKET_DATA_BATCH, self._on_market_data_batch)
        event_bus.subscribe(EventType.MARKET_DATA_TICK, self._on_market_data_tick)
        logger.debug("MarketDataStore subscribed to events")

    def _on_market_data_batch(self, payload: dict) -> None:
        """
        Handle batch market data update event.

        Args:
            payload: Event payload with 'market_data' list.
        """
        market_data_list = payload.get("market_data", [])
        if market_data_list:
            self.upsert(market_data_list)
            logger.debug(f"MarketDataStore updated from batch: {len(market_data_list)} symbols")

    def _on_market_data_tick(self, payload: Any) -> None:
        """
        Handle single market data tick event (from streaming).

        C3: Updated to handle MarketDataTickEvent (typed) instead of dict.

        Args:
            payload: MarketDataTickEvent or legacy dict with 'symbol' and 'data'.
        """
        from ...domain.events.domain_events import MarketDataTickEvent

        # Handle typed MarketDataTickEvent (new standard)
        if isinstance(payload, MarketDataTickEvent):
            # Update existing market data or create new entry
            symbol = payload.symbol
            existing = self.get(symbol)
            if existing:
                # Merge tick data into existing MarketData
                existing.bid = payload.bid if payload.bid is not None else existing.bid
                existing.ask = payload.ask if payload.ask is not None else existing.ask
                existing.last = payload.last if payload.last is not None else existing.last
                existing.delta = payload.delta if payload.delta is not None else existing.delta
                existing.gamma = payload.gamma if payload.gamma is not None else existing.gamma
                existing.vega = payload.vega if payload.vega is not None else existing.vega
                existing.theta = payload.theta if payload.theta is not None else existing.theta
                existing.iv = payload.iv if payload.iv is not None else existing.iv
                self.upsert([existing])
            else:
                # Create new MarketData from tick
                md = MarketData(
                    symbol=symbol,
                    bid=payload.bid,
                    ask=payload.ask,
                    last=payload.last,
                    delta=payload.delta,
                    gamma=payload.gamma,
                    vega=payload.vega,
                    theta=payload.theta,
                    iv=payload.iv,
                )
                self.upsert([md])
            return

        # Legacy dict handling (for backward compatibility during transition)
        if isinstance(payload, dict):
            data = payload.get("data")
            if data:
                self.upsert([data])
