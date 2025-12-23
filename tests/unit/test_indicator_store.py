"""
Tests for IndicatorStore (A3).

Verifies:
- Cache hit/miss behavior
- TTL expiration
- get_or_compute() pattern
- Invalidation by symbol/timeframe
- Eviction when over capacity
- Metrics tracking
"""

import pytest
import asyncio
import time
from datetime import date
from unittest.mock import AsyncMock, MagicMock

from src.services.indicator_store import IndicatorStore, CachedIndicator


class TestIndicatorStoreBasics:
    """Basic cache operations."""

    def test_cache_miss_returns_none(self):
        """Empty cache returns None."""
        store = IndicatorStore()
        result = store.get("AAPL", "ATR", {"period": 14})
        assert result is None

    def test_cache_put_and_get(self):
        """Can store and retrieve values."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)

        result = store.get("AAPL", "ATR", {"period": 14})
        assert result == 2.5

    def test_different_params_are_separate_keys(self):
        """Different params create separate cache entries."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)
        store.put("AAPL", "ATR", {"period": 20}, value=3.0)

        assert store.get("AAPL", "ATR", {"period": 14}) == 2.5
        assert store.get("AAPL", "ATR", {"period": 20}) == 3.0

    def test_different_symbols_are_separate_keys(self):
        """Different symbols create separate cache entries."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)
        store.put("MSFT", "ATR", {"period": 14}, value=1.8)

        assert store.get("AAPL", "ATR", {"period": 14}) == 2.5
        assert store.get("MSFT", "ATR", {"period": 14}) == 1.8


class TestIndicatorStoreTTL:
    """TTL expiration behavior."""

    def test_expired_entry_returns_none(self):
        """Expired entries are treated as cache miss."""
        store = IndicatorStore(ttl_seconds=1)
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)

        # Entry is fresh
        assert store.get("AAPL", "ATR", {"period": 14}) == 2.5

        # Wait for expiration
        time.sleep(1.1)

        # Entry is now expired
        assert store.get("AAPL", "ATR", {"period": 14}) is None

    def test_expired_entry_increments_eviction_count(self):
        """Expired entries increment eviction metric."""
        store = IndicatorStore(ttl_seconds=0)  # Immediate expiry
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)

        # Force expiry check
        time.sleep(0.01)
        store.get("AAPL", "ATR", {"period": 14})

        assert store.get_metrics().evictions >= 1


class TestIndicatorStoreGetOrCompute:
    """get_or_compute() pattern."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_computation(self):
        """Cached value avoids calling compute_fn."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)

        compute_fn = AsyncMock(return_value=9.9)

        result = await store.get_or_compute(
            "AAPL", "ATR", {"period": 14}, compute_fn
        )

        assert result == 2.5
        compute_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_compute(self):
        """Cache miss triggers computation."""
        store = IndicatorStore()
        compute_fn = AsyncMock(return_value=2.5)

        result = await store.get_or_compute(
            "AAPL", "ATR", {"period": 14}, compute_fn
        )

        assert result == 2.5
        compute_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_computed_value_is_cached(self):
        """Computed value is stored for future use."""
        store = IndicatorStore()
        compute_fn = AsyncMock(return_value=2.5)

        # First call computes
        await store.get_or_compute("AAPL", "ATR", {"period": 14}, compute_fn)

        # Second call should hit cache
        compute_fn.reset_mock()
        result = await store.get_or_compute(
            "AAPL", "ATR", {"period": 14}, compute_fn
        )

        assert result == 2.5
        compute_fn.assert_not_called()


class TestIndicatorStoreInvalidation:
    """Cache invalidation."""

    def test_invalidate_by_symbol(self):
        """Invalidate all entries for a symbol."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)
        store.put("AAPL", "RSI", {"period": 14}, value=65.0)
        store.put("MSFT", "ATR", {"period": 14}, value=1.8)

        count = store.invalidate(symbol="AAPL")

        assert count == 2
        assert store.get("AAPL", "ATR", {"period": 14}) is None
        assert store.get("AAPL", "RSI", {"period": 14}) is None
        assert store.get("MSFT", "ATR", {"period": 14}) == 1.8

    def test_invalidate_by_indicator(self):
        """Invalidate all entries for an indicator type."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)
        store.put("MSFT", "ATR", {"period": 14}, value=1.8)
        store.put("AAPL", "RSI", {"period": 14}, value=65.0)

        count = store.invalidate(indicator="ATR")

        assert count == 2
        assert store.get("AAPL", "RSI", {"period": 14}) == 65.0

    def test_clear_removes_all(self):
        """clear() removes all entries."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)
        store.put("MSFT", "RSI", {"period": 14}, value=65.0)

        store.clear()

        assert store.get("AAPL", "ATR", {"period": 14}) is None
        assert store.get("MSFT", "RSI", {"period": 14}) is None


class TestIndicatorStoreEviction:
    """Capacity-based eviction."""

    def test_evicts_oldest_when_over_capacity(self):
        """Oldest entries evicted when max_entries reached."""
        store = IndicatorStore(max_entries=5)

        # Fill cache
        for i in range(5):
            store.put(f"SYM{i}", "ATR", {"period": 14}, value=float(i))
            time.sleep(0.01)  # Ensure different timestamps

        # Add one more (should trigger eviction)
        store.put("NEW", "ATR", {"period": 14}, value=99.0)

        # Oldest should be evicted
        assert store.get("SYM0", "ATR", {"period": 14}) is None
        # Newest should remain
        assert store.get("NEW", "ATR", {"period": 14}) == 99.0


class TestIndicatorStoreMetrics:
    """Metrics tracking."""

    def test_tracks_hits_and_misses(self):
        """Metrics track cache hits and misses."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)

        # Miss
        store.get("MSFT", "ATR", {"period": 14})
        # Hit
        store.get("AAPL", "ATR", {"period": 14})
        # Hit
        store.get("AAPL", "ATR", {"period": 14})

        metrics = store.get_metrics()
        assert metrics.hits == 2
        assert metrics.misses == 1

    def test_hit_rate_calculation(self):
        """Hit rate is calculated correctly."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)

        # 1 miss, 3 hits = 75% hit rate
        store.get("MISS", "ATR", {"period": 14})
        store.get("AAPL", "ATR", {"period": 14})
        store.get("AAPL", "ATR", {"period": 14})
        store.get("AAPL", "ATR", {"period": 14})

        assert store.get_metrics().hit_rate == 75.0

    def test_get_stats_returns_dict(self):
        """get_stats() returns monitoring-friendly dict."""
        store = IndicatorStore()
        store.put("AAPL", "ATR", {"period": 14}, value=2.5)

        stats = store.get_stats()

        assert "entries" in stats
        assert "hit_rate" in stats
        assert stats["entries"] == 1
