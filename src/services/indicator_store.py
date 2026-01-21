"""
Indicator Store - Cache for computed technical indicators.

A3: Centralized cache for TA-Lib computed values (ATR, RSI, SMA, etc.).
Prevents redundant computation when multiple consumers (scanner, alerts, UI)
need the same indicator values.

Features:
- Cache key: (symbol, timeframe, indicator, params_hash, as_of_date)
- Automatic invalidation when new bars arrive (via event bus)
- TTL-based expiry for stale data
- Metrics for cache hit/miss rates

Usage:
    store = IndicatorStore(event_bus)

    # Get or compute ATR
    atr = await store.get_or_compute(
        symbol="AAPL",
        indicator="ATR",
        params={"period": 14},
        timeframe="1d",
        compute_fn=lambda: ta_service.get_atr("AAPL", period=14)
    )
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import date
from threading import RLock
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional

from ..utils.logging_setup import get_logger
from ..utils.timezone import now_utc

if TYPE_CHECKING:
    from ..domain.interfaces.event_bus import EventBus

logger = get_logger(__name__)


@dataclass
class CachedIndicator:
    """Cached indicator value with metadata."""

    value: Any
    computed_at: float  # time.time()
    as_of_date: date  # Trading date this value is for
    hit_count: int = 0


@dataclass
class IndicatorStoreMetrics:
    """Metrics for indicator store performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    computations: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class IndicatorStore:
    """
    Cache for computed technical indicators.

    A3: Reduces CPU load by sharing computed indicators across consumers.
    Scanner, alerts, and UI all benefit from cached ATR/RSI values.
    """

    DEFAULT_TTL_SECONDS = 300  # 5 minutes default TTL
    DEFAULT_MAX_ENTRIES = 1000

    def __init__(
        self,
        event_bus: Optional["EventBus"] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        """
        Initialize indicator store.

        Args:
            event_bus: Optional event bus for bar arrival notifications.
            ttl_seconds: Time-to-live for cached values (default: 5 min).
            max_entries: Maximum cache entries before eviction (default: 1000).
        """
        self._cache: Dict[str, CachedIndicator] = {}
        self._lock = RLock()
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._metrics = IndicatorStoreMetrics()

        # Subscribe to bar events for invalidation
        if event_bus:
            self._subscribe_to_events(event_bus)

    def _subscribe_to_events(self, event_bus: "EventBus") -> None:
        """Subscribe to bar arrival events for cache invalidation."""
        from ..domain.interfaces.event_bus import EventType

        # Invalidate cache when new bars arrive (using BAR_CLOSE as the signal for new data)
        event_bus.subscribe(EventType.BAR_CLOSE, self._on_new_bars)
        logger.debug("IndicatorStore subscribed to BAR_CLOSE events for cache invalidation")

    def _on_new_bars(self, payload: dict) -> None:
        """
        Handle new bar data - invalidate affected cache entries.

        Args:
            payload: Event payload with symbol and timeframe info.
        """
        symbol = payload.get("symbol")
        timeframe = payload.get("timeframe")

        if not symbol:
            return

        # Invalidate all entries for this symbol/timeframe combination
        invalidated = self.invalidate(symbol=symbol, timeframe=timeframe)
        if invalidated > 0:
            logger.debug(
                "IndicatorStore invalidated %d entries for %s/%s", invalidated, symbol, timeframe
            )

    @staticmethod
    def _make_cache_key(
        symbol: str,
        indicator: str,
        params: Dict[str, Any],
        timeframe: str,
        as_of_date: date,
    ) -> str:
        """
        Generate cache key from indicator parameters.

        Key format: symbol:indicator:timeframe:date:params_hash
        """
        # Hash params for consistent key regardless of dict ordering
        params_str = str(sorted(params.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        return f"{symbol}:{indicator}:{timeframe}:{as_of_date.isoformat()}:{params_hash}"

    def get(
        self,
        symbol: str,
        indicator: str,
        params: Dict[str, Any],
        timeframe: str = "1d",
        as_of_date: Optional[date] = None,
    ) -> Optional[Any]:
        """
        Get cached indicator value if available and fresh.

        Args:
            symbol: Stock symbol.
            indicator: Indicator name (ATR, RSI, SMA, etc.).
            params: Indicator parameters (e.g., {"period": 14}).
            timeframe: Bar timeframe (default: "1d").
            as_of_date: Date for indicator (default: today).

        Returns:
            Cached value or None if not found/expired.
        """
        if as_of_date is None:
            as_of_date = now_utc().date()

        key = self._make_cache_key(symbol, indicator, params, timeframe, as_of_date)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._metrics.misses += 1
                return None

            # Check TTL
            age = time.time() - entry.computed_at
            if age > self._ttl_seconds:
                del self._cache[key]
                self._metrics.misses += 1
                self._metrics.evictions += 1
                return None

            # Cache hit
            entry.hit_count += 1
            self._metrics.hits += 1
            return entry.value

    def put(
        self,
        symbol: str,
        indicator: str,
        params: Dict[str, Any],
        value: Any,
        timeframe: str = "1d",
        as_of_date: Optional[date] = None,
    ) -> None:
        """
        Store indicator value in cache.

        Args:
            symbol: Stock symbol.
            indicator: Indicator name.
            params: Indicator parameters.
            value: Computed indicator value.
            timeframe: Bar timeframe.
            as_of_date: Date for indicator.
        """
        if as_of_date is None:
            as_of_date = now_utc().date()

        key = self._make_cache_key(symbol, indicator, params, timeframe, as_of_date)

        with self._lock:
            # Evict if over capacity
            if len(self._cache) >= self._max_entries:
                self._evict_oldest()

            self._cache[key] = CachedIndicator(
                value=value,
                computed_at=time.time(),
                as_of_date=as_of_date,
            )

    async def get_or_compute(
        self,
        symbol: str,
        indicator: str,
        params: Dict[str, Any],
        compute_fn: Callable[[], Awaitable[Any]],
        timeframe: str = "1d",
        as_of_date: Optional[date] = None,
    ) -> Any:
        """
        Get cached value or compute and cache if missing.

        This is the primary API for consumers - it handles cache lookup,
        computation on miss, and caching the result.

        Args:
            symbol: Stock symbol.
            indicator: Indicator name (ATR, RSI, etc.).
            params: Indicator parameters.
            compute_fn: Async function to compute the value on cache miss.
            timeframe: Bar timeframe.
            as_of_date: Date for indicator.

        Returns:
            Indicator value (cached or freshly computed).
        """
        # Try cache first
        cached = self.get(symbol, indicator, params, timeframe, as_of_date)
        if cached is not None:
            return cached

        # Compute on miss
        self._metrics.computations += 1
        value = await compute_fn()

        # Cache the result
        if value is not None:
            self.put(symbol, indicator, params, value, timeframe, as_of_date)

        return value

    def invalidate(
        self,
        symbol: Optional[str] = None,
        indicator: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries matching the criteria.

        Args:
            symbol: Invalidate entries for this symbol (None = all).
            indicator: Invalidate entries for this indicator (None = all).
            timeframe: Invalidate entries for this timeframe (None = all).

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            to_remove = []

            for key in self._cache:
                parts = key.split(":")
                if len(parts) < 3:
                    continue

                key_symbol, key_indicator, key_timeframe = parts[0], parts[1], parts[2]

                # Check if entry matches criteria
                if symbol and key_symbol != symbol:
                    continue
                if indicator and key_indicator != indicator:
                    continue
                if timeframe and key_timeframe != timeframe:
                    continue

                to_remove.append(key)

            for key in to_remove:
                del self._cache[key]
                self._metrics.evictions += 1

            return len(to_remove)

    def _evict_oldest(self) -> None:
        """Evict oldest entries to make room (must hold lock)."""
        if not self._cache:
            return

        # Sort by computed_at (oldest first)
        sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k].computed_at)

        # Evict 10% of cache
        evict_count = max(1, len(self._cache) // 10)
        for key in sorted_keys[:evict_count]:
            del self._cache[key]
            self._metrics.evictions += 1

        logger.debug("IndicatorStore evicted %d oldest entries", evict_count)

    def clear(self) -> None:
        """Clear all cached indicators."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("IndicatorStore cleared %d entries", count)

    def get_metrics(self) -> IndicatorStoreMetrics:
        """Get cache performance metrics."""
        return self._metrics

    def get_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._metrics.hits,
                "misses": self._metrics.misses,
                "hit_rate": f"{self._metrics.hit_rate:.1f}%",
                "evictions": self._metrics.evictions,
                "computations": self._metrics.computations,
            }
