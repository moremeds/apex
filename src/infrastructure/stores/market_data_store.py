"""
Thread-safe in-memory market data store with separate price/Greeks caching.

OPT-014: Uses RCU pattern for lock-free reads on the main data path.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Union, TYPE_CHECKING
from threading import RLock
import time

from ...utils.logging_setup import get_logger
from ...models.market_data import MarketData
from ...utils.timezone import age_seconds, now_utc
from .rcu_store import RCUDict

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
        # OPT-014: Use RCUDict for lock-free reads
        self._market_data: RCUDict[str, MarketData] = RCUDict()
        self._eviction_lock = RLock()  # Only for eviction coordination
        self._price_ttl_seconds = price_ttl_seconds
        self._greeks_ttl_seconds = greeks_ttl_seconds
        self._max_symbols = max_symbols
        self._eviction_age_seconds = eviction_age_seconds
        self._last_eviction_time = time.time()

        # HIGH-010: Tick buffer to avoid copy-on-write per tick
        # Buffer flushes when either threshold is reached
        self._tick_buffer: Dict[str, MarketData] = {}
        self._tick_buffer_lock = RLock()
        self._last_flush_time = time.time()
        self._tick_flush_interval_ms = 10  # Flush every 10ms
        self._tick_flush_count = 50  # Or every 50 ticks

    def upsert(self, market_data: Iterable[MarketData]) -> None:
        """
        Insert or update market data.

        OPT-014: RCUDict handles locking internally, so no external lock needed.

        Args:
            market_data: Iterable of MarketData objects.
        """
        # Build updates dict - RCUDict.update() handles atomicity internally
        updates = {md.symbol: md for md in market_data}
        self._market_data.update(updates)

        # Periodic eviction check (every 60 seconds to avoid overhead)
        # Use eviction lock for coordination only
        now = time.time()
        if now - self._last_eviction_time > 60:
            with self._eviction_lock:
                # Double-check after acquiring lock
                if now - self._last_eviction_time > 60:
                    self._evict_stale()
                    self._last_eviction_time = now

    def _buffer_tick(self, md: MarketData) -> None:
        """
        HIGH-010: Buffer tick updates to avoid per-tick copy-on-write.

        Ticks are accumulated in a buffer and flushed when either:
        - 50 ticks have accumulated
        - 10ms have elapsed since last flush

        Args:
            md: MarketData to buffer.
        """
        with self._tick_buffer_lock:
            self._tick_buffer[md.symbol] = md

            # Check if flush needed
            now = time.time()
            buffer_size = len(self._tick_buffer)
            elapsed_ms = (now - self._last_flush_time) * 1000

            if buffer_size >= self._tick_flush_count or elapsed_ms >= self._tick_flush_interval_ms:
                self._flush_tick_buffer_locked()

    def _maybe_flush_stale_buffer(self) -> None:
        """
        HIGH-010 FIX: Auto-flush buffer if stale (for sparse tick scenarios).

        Called on get() to ensure readers don't see stale data when tick flow is sparse.
        Uses try_lock to avoid blocking reads if flush is in progress.
        """
        # Quick check without lock - if buffer empty, nothing to do
        if not self._tick_buffer:
            return

        # Check if enough time has passed to warrant a flush
        elapsed_ms = (time.time() - self._last_flush_time) * 1000
        if elapsed_ms < self._tick_flush_interval_ms:
            return

        # Try to acquire lock without blocking
        if self._tick_buffer_lock.acquire(blocking=False):
            try:
                # Double-check after acquiring lock
                if self._tick_buffer and (time.time() - self._last_flush_time) * 1000 >= self._tick_flush_interval_ms:
                    self._flush_tick_buffer_locked()
            finally:
                self._tick_buffer_lock.release()

    def _flush_tick_buffer_locked(self) -> None:
        """Flush tick buffer to RCU store. Caller must hold _tick_buffer_lock."""
        if not self._tick_buffer:
            return
        # Single RCU update for all buffered ticks
        self._market_data.update(self._tick_buffer)
        self._tick_buffer.clear()
        self._last_flush_time = time.time()

    def flush_tick_buffer(self) -> None:
        """
        Force flush of tick buffer.

        Call this before operations that need latest data (e.g., snapshots).
        """
        with self._tick_buffer_lock:
            self._flush_tick_buffer_locked()

    def _evict_stale(self) -> None:
        """
        Evict stale entries.

        OPT-014: Uses RCUDict.delete() for atomic removal.
        Caller must hold _eviction_lock for coordination.

        Evicts:
        1. Entries older than eviction_age_seconds
        2. Oldest entries if over max_symbols
        """
        # Get snapshot for analysis (lock-free read)
        snapshot = self._market_data.items()

        # Evict by age first
        stale_symbols = []
        for symbol, md in snapshot:
            if md.timestamp and age_seconds(md.timestamp) > self._eviction_age_seconds:
                stale_symbols.append(symbol)

        # HIGH-008: Use batch_delete for O(n) instead of O(n²)
        if stale_symbols:
            deleted = self._market_data.batch_delete(stale_symbols)
            logger.debug(f"Evicted {deleted} stale market data entries")

        # Evict by count if still over limit
        current_size = len(self._market_data)
        if current_size > self._max_symbols:
            excess = current_size - self._max_symbols
            # Get fresh snapshot and sort by timestamp (oldest first)
            sorted_entries = sorted(
                self._market_data.items(),
                key=lambda x: x[1].timestamp.timestamp() if x[1].timestamp else 0
            )
            # HIGH-008: Use batch_delete for O(n) instead of O(n²)
            symbols_to_evict = [symbol for symbol, _ in sorted_entries[:excess]]
            deleted = self._market_data.batch_delete(symbols_to_evict)
            logger.debug(f"Evicted {deleted} oldest market data entries (over max {self._max_symbols})")

    def get(self, symbol: str) -> Optional[MarketData]:
        """
        Get market data for a symbol.

        OPT-014: Lock-free read via RCUDict.
        HIGH-010 FIX: Auto-flushes stale buffer for sparse tick scenarios.

        Note: Returns cached data even if Greeks are stale.
        Use is_greeks_stale() to check freshness.
        """
        # HIGH-010 FIX: Auto-flush stale buffer before read
        self._maybe_flush_stale_buffer()
        return self._market_data.get(symbol)

    def has_fresh_data(self, symbol: str) -> bool:
        """
        Check if we have fresh price data for a symbol.

        OPT-014: Lock-free read via RCUDict.

        Used to determine if we need to subscribe to streaming for this symbol.

        Args:
            symbol: Symbol to check.

        Returns:
            True if we have fresh data (within price TTL), False otherwise.
        """
        md = self._market_data.get(symbol)
        if not md or not md.timestamp:
            return False
        return age_seconds(md.timestamp) <= self._price_ttl_seconds

    def is_greeks_stale(self, symbol: str) -> bool:
        """
        Check if Greeks data is stale for a symbol.

        OPT-014: Lock-free read via RCUDict.

        Args:
            symbol: Symbol to check.

        Returns:
            True if Greeks are stale or missing, False if fresh.
        """
        md = self._market_data.get(symbol)
        if not md or not md.timestamp:
            return True
        return age_seconds(md.timestamp) > self._greeks_ttl_seconds

    def get_stale_symbols(self) -> List[str]:
        """
        Get list of symbols with stale Greeks.

        OPT-014: Lock-free read via RCUDict.items() snapshot.

        Returns:
            List of symbols that need Greeks refresh.
        """
        stale = []
        for symbol, md in self._market_data.items():
            if not md.timestamp:
                stale.append(symbol)
                continue
            if age_seconds(md.timestamp) > self._greeks_ttl_seconds:
                stale.append(symbol)
        return stale

    def get_all(self) -> Dict[str, MarketData]:
        """
        Get all market data (thread-safe copy).

        OPT-014: Lock-free read via RCUDict.get_all().
        Returns a copy for safety.
        """
        return dict(self._market_data.get_all())

    def get_symbols(self) -> List[str]:
        """
        Get all symbols.

        OPT-014: Lock-free read via RCUDict.keys() snapshot.
        """
        return self._market_data.keys()

    def get_symbols_needing_refresh(self, requested_symbols: List[str]) -> List[str]:
        """
        Get symbols that need market data refresh (stale price or missing).

        OPT-014: Lock-free read via RCUDict snapshot.
        Snapshot ensures consistency within this operation.

        Uses price_ttl_seconds for staleness check (need real-time prices).
        Greeks staleness is checked separately via is_greeks_stale().

        Args:
            requested_symbols: List of symbols we want market data for.

        Returns:
            List of symbols that need refresh (price stale or not in cache).
        """
        # Get atomic snapshot of current data
        snapshot = self._market_data.get_all()

        stale_set = set()
        existing_set = set(snapshot.keys())

        # Check for stale prices (using price TTL, not Greeks TTL)
        for symbol, md in snapshot.items():
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
        """
        Clear all market data.

        OPT-014: RCUDict.clear() handles atomicity internally.
        """
        self._market_data.clear()

    def count(self) -> int:
        """
        Get market data count.

        OPT-014: Lock-free read via RCUDict.__len__().
        """
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

        C3: Updated to handle MarketDataTickEvent (typed).
        M14: Removed legacy dict handling.
        OPT-016: Only recalculate mid when bid/ask actually change.

        Args:
            payload: MarketDataTickEvent (typed event).
        """
        from ...domain.events.domain_events import MarketDataTickEvent

        # Handle typed MarketDataTickEvent (new standard)
        if isinstance(payload, MarketDataTickEvent):
            symbol = payload.symbol
            existing = self.get(symbol)

            if existing:
                # OPT-016: Track if bid/ask changed to avoid redundant mid recalc
                bid_changed = payload.bid is not None and payload.bid != existing.bid
                ask_changed = payload.ask is not None and payload.ask != existing.ask

                # Merge tick data into existing MarketData (only non-None values)
                if payload.bid is not None:
                    existing.bid = payload.bid
                if payload.ask is not None:
                    existing.ask = payload.ask
                if payload.last is not None:
                    existing.last = payload.last
                if payload.delta is not None:
                    existing.delta = payload.delta
                if payload.gamma is not None:
                    existing.gamma = payload.gamma
                if payload.vega is not None:
                    existing.vega = payload.vega
                if payload.theta is not None:
                    existing.theta = payload.theta
                if payload.iv is not None:
                    existing.iv = payload.iv

                # OPT-016: Only recalculate mid if bid or ask actually changed
                if (bid_changed or ask_changed) and existing.bid is not None and existing.ask is not None:
                    existing.mid = (existing.bid + existing.ask) / 2

                existing.timestamp = now_utc()
                # HIGH-010: Buffer tick instead of per-tick upsert
                self._buffer_tick(existing)
            else:
                # Create new MarketData from tick
                mid = None
                if payload.bid is not None and payload.ask is not None:
                    mid = (payload.bid + payload.ask) / 2
                md = MarketData(
                    symbol=symbol,
                    bid=payload.bid,
                    ask=payload.ask,
                    mid=mid,
                    last=payload.last,
                    delta=payload.delta,
                    gamma=payload.gamma,
                    vega=payload.vega,
                    theta=payload.theta,
                    iv=payload.iv,
                    timestamp=now_utc(),
                )
                # HIGH-010: Buffer tick instead of per-tick upsert
                self._buffer_tick(md)
            return

        # M14: Removed legacy dict handling - all callers now use typed events
        logger.warning(f"Unexpected payload type in _on_market_data_tick: {type(payload)}")
