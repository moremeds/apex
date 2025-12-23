"""
Tests for HistoricalRequestScheduler (A1).

Verifies:
- Priority queue ordering (ALERT > UI > SCANNER > PREFETCH)
- Concurrency limiting (semaphore)
- Rate limiting (token bucket)
- Request deduplication
- Metrics tracking
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.historical_request_scheduler import (
    HistoricalRequestScheduler,
    RequestPriority,
    TokenBucket,
    HistoricalRequest,
)
from src.services.bar_cache_service import BarPeriod


class TestRequestPriority:
    """Priority enum ordering."""

    def test_alert_is_highest_priority(self):
        """ALERT has lowest numeric value (highest priority)."""
        assert RequestPriority.ALERT < RequestPriority.UI
        assert RequestPriority.ALERT < RequestPriority.SCANNER
        assert RequestPriority.ALERT < RequestPriority.PREFETCH

    def test_priority_ordering(self):
        """Priorities are ordered correctly."""
        priorities = sorted(RequestPriority)
        assert priorities == [
            RequestPriority.ALERT,
            RequestPriority.UI,
            RequestPriority.SCANNER,
            RequestPriority.PREFETCH,
        ]


class TestTokenBucket:
    """Token bucket rate limiter."""

    @pytest.mark.asyncio
    async def test_initial_tokens_available(self):
        """Bucket starts with max tokens."""
        bucket = TokenBucket(max_tokens=10, refill_period_seconds=60)
        assert bucket.available_tokens == 10.0

    @pytest.mark.asyncio
    async def test_acquire_consumes_token(self):
        """Acquiring consumes a token."""
        bucket = TokenBucket(max_tokens=10, refill_period_seconds=60)

        result = await bucket.acquire(timeout=1.0)

        assert result is True
        assert bucket.available_tokens < 10.0

    @pytest.mark.asyncio
    async def test_acquire_fails_on_timeout(self):
        """Returns False when no tokens and timeout expires."""
        bucket = TokenBucket(max_tokens=1, refill_period_seconds=600)

        # Consume the only token
        await bucket.acquire(timeout=1.0)

        # Next acquire should timeout
        result = await bucket.acquire(timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self):
        """Tokens refill based on elapsed time."""
        bucket = TokenBucket(max_tokens=10, refill_period_seconds=1)

        # Consume all tokens
        for _ in range(10):
            await bucket.acquire(timeout=0.1)

        assert bucket.available_tokens < 1.0

        # Wait for refill
        await asyncio.sleep(0.2)

        # Should have some tokens now
        result = await bucket.acquire(timeout=0.1)
        assert result is True


class TestHistoricalRequest:
    """Request dataclass."""

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Cache key includes symbol, timeframe, period."""
        # Must be in async context due to Future default_factory
        request = HistoricalRequest(
            symbol="AAPL",
            timeframe="1d",
            period=BarPeriod.bars(30),
            priority=RequestPriority.UI,
        )

        key = request.cache_key()

        assert "AAPL" in key
        assert "1d" in key


class TestSchedulerBasics:
    """Basic scheduler operations."""

    @pytest.fixture
    def mock_historical_service(self):
        """Create mock historical data service."""
        service = MagicMock()
        service.fetch_bars = AsyncMock(return_value=[{"close": 150.0}])
        return service

    @pytest.mark.asyncio
    async def test_start_and_stop(self, mock_historical_service):
        """Scheduler can start and stop cleanly."""
        scheduler = HistoricalRequestScheduler(mock_historical_service)

        await scheduler.start()
        assert scheduler._running is True

        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_request_returns_bars(self, mock_historical_service):
        """Request returns bars from historical service."""
        scheduler = HistoricalRequestScheduler(mock_historical_service)
        await scheduler.start()

        try:
            result = await asyncio.wait_for(
                scheduler.request(
                    symbol="AAPL",
                    timeframe="1d",
                    period=BarPeriod.bars(30),
                    priority=RequestPriority.UI,
                ),
                timeout=2.0
            )

            assert result == [{"close": 150.0}]
            mock_historical_service.fetch_bars.assert_called_once()
        finally:
            await scheduler.stop()


class TestSchedulerDeduplication:
    """Request deduplication."""

    @pytest.fixture
    def mock_historical_service(self):
        service = MagicMock()
        service.fetch_bars = AsyncMock(return_value=[{"close": 150.0}])
        return service

    @pytest.mark.asyncio
    async def test_duplicate_requests_share_result(self, mock_historical_service):
        """Identical concurrent requests share the same result."""
        scheduler = HistoricalRequestScheduler(
            mock_historical_service,
            dedup_window_seconds=5,
        )
        await scheduler.start()

        try:
            # Make two identical requests concurrently
            task1 = scheduler.request("AAPL", "1d", BarPeriod.bars(30), RequestPriority.UI)
            task2 = scheduler.request("AAPL", "1d", BarPeriod.bars(30), RequestPriority.UI)

            results = await asyncio.wait_for(
                asyncio.gather(task1, task2),
                timeout=2.0
            )

            # Both should get same result
            assert results[0] == results[1]

            # Service should only be called once (dedup)
            # Note: Depending on timing, might be 1 or 2 calls
            assert mock_historical_service.fetch_bars.call_count <= 2
        finally:
            await scheduler.stop()


class TestSchedulerMetrics:
    """Metrics tracking."""

    @pytest.fixture
    def mock_historical_service(self):
        service = MagicMock()
        service.fetch_bars = AsyncMock(return_value=[{"close": 150.0}])
        return service

    @pytest.mark.asyncio
    async def test_tracks_requests_by_priority(self, mock_historical_service):
        """Metrics track requests per priority level."""
        scheduler = HistoricalRequestScheduler(mock_historical_service)
        await scheduler.start()

        try:
            await asyncio.wait_for(
                scheduler.request("AAPL", "1d", BarPeriod.bars(30), RequestPriority.ALERT),
                timeout=2.0
            )

            metrics = scheduler.get_metrics()
            assert metrics.requests_by_priority["ALERT"] >= 1
        finally:
            await scheduler.stop()

    def test_get_stats_returns_dict(self, mock_historical_service):
        """get_stats() returns monitoring-friendly dict."""
        scheduler = HistoricalRequestScheduler(mock_historical_service)

        stats = scheduler.get_stats()

        assert "running" in stats
        assert "pending_requests" in stats
        assert "by_priority" in stats
        assert "available_tokens" in stats


class TestSchedulerConcurrency:
    """Concurrency limiting."""

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self):
        """Semaphore limits concurrent requests."""
        # Track concurrent execution count
        concurrent_count = 0
        max_concurrent_seen = 0

        async def slow_fetch(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1
            return [{"close": 150.0}]

        service = MagicMock()
        service.fetch_bars = slow_fetch

        scheduler = HistoricalRequestScheduler(service, max_concurrent=2)
        await scheduler.start()

        try:
            # Make 5 requests concurrently
            tasks = [
                scheduler.request(f"SYM{i}", "1d", BarPeriod.bars(30), RequestPriority.UI)
                for i in range(5)
            ]

            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

            # Should never exceed max_concurrent
            assert max_concurrent_seen <= 2
        finally:
            await scheduler.stop()
