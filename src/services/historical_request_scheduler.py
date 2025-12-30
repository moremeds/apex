"""
Historical Request Scheduler - Priority-based rate-limited request queue.

A1: Centralized scheduler for IB historical data requests with:
- Priority queue with 4 levels (ALERT > UI > SCANNER > PREFETCH)
- Semaphore limiting concurrent requests to 6
- Token bucket enforcing 50 requests / 10 minutes (IB pacing limit)
- 15-second deduplication for identical requests
- Background queue processor task

Supersedes: C5, C9, M11, M12, M13

Usage:
    scheduler = HistoricalRequestScheduler(historical_service)
    await scheduler.start()

    # High-priority alert request (processed immediately)
    bars = await scheduler.request(
        symbol="AAPL",
        timeframe="1d",
        period=BarPeriod.bars(30),
        priority=RequestPriority.ALERT
    )

    # Low-priority prefetch (queued, processed when capacity available)
    await scheduler.request(
        symbol="MSFT",
        timeframe="1d",
        period=BarPeriod.bars(30),
        priority=RequestPriority.PREFETCH
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from collections import OrderedDict

from ..utils.logging_setup import get_logger
from .bar_cache_service import BarPeriod

if TYPE_CHECKING:
    from .historical_data_service import HistoricalDataService
    from ..domain.interfaces.event_bus import EventBus

logger = get_logger(__name__)


class RequestPriority(IntEnum):
    """
    Request priority levels (lower number = higher priority).

    ALERT: Risk alerts, stop-loss triggers (must complete <100ms)
    UI: User-facing dashboard updates
    SCANNER: Background scanning tasks
    PREFETCH: Proactive cache warming
    """
    ALERT = 0
    UI = 1
    SCANNER = 2
    PREFETCH = 3


@dataclass
class HistoricalRequest:
    """Queued historical data request."""
    symbol: str
    timeframe: str
    period: BarPeriod
    priority: RequestPriority
    created_at: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())

    def cache_key(self) -> str:
        """Generate cache key for deduplication."""
        return f"{self.symbol}:{self.timeframe}:{self.period}"


@dataclass
class SchedulerMetrics:
    """Metrics for scheduler performance."""
    requests_total: int = 0
    requests_by_priority: Dict[str, int] = field(default_factory=lambda: {
        "ALERT": 0, "UI": 0, "SCANNER": 0, "PREFETCH": 0
    })
    dedup_hits: int = 0
    rate_limit_waits: int = 0
    avg_wait_ms: float = 0.0
    max_wait_ms: float = 0.0
    _wait_times: List[float] = field(default_factory=list)

    def record_wait(self, wait_ms: float) -> None:
        """Record a wait time."""
        self._wait_times.append(wait_ms)
        # Keep last 100 for rolling average
        if len(self._wait_times) > 100:
            self._wait_times.pop(0)
        self.avg_wait_ms = sum(self._wait_times) / len(self._wait_times)
        self.max_wait_ms = max(self.max_wait_ms, wait_ms)


@dataclass
class SchedulerStatus:
    """
    OPT-015: Current scheduler status for display and monitoring.

    Provides visibility into rate limit status, queue depths, and performance.
    """
    tokens_available: float
    tokens_max: float
    queue_depth: Dict[str, int]  # priority name -> count
    total_queued: int
    in_flight: int
    max_concurrent: int
    requests_completed: int
    requests_failed: int
    dedup_hits: int
    avg_request_time_ms: float
    max_request_time_ms: float
    rate_limited: bool
    rate_limit_wait_sec: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class TokenBucket:
    """
    Token bucket rate limiter for IB pacing.

    IB allows ~50 historical requests per 10 minutes.
    Tokens are added at a steady rate and consumed per request.
    """

    def __init__(
        self,
        max_tokens: int = 50,
        refill_period_seconds: int = 600,  # 10 minutes
    ):
        """
        Initialize token bucket.

        Args:
            max_tokens: Maximum tokens (requests) in bucket.
            refill_period_seconds: Time to fully refill bucket.
        """
        self._max_tokens = max_tokens
        self._tokens = float(max_tokens)
        self._refill_rate = max_tokens / refill_period_seconds
        self._last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire a token (wait if necessary).

        Args:
            timeout: Maximum time to wait for token.

        Returns:
            True if token acquired, False if timeout.
        """
        start = time.time()

        while True:
            async with self._lock:
                self._refill()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            # Wait for next refill opportunity
            if time.time() - start > timeout:
                return False

            # Sleep for time to get 1 token
            wait_time = min(1.0 / self._refill_rate, timeout - (time.time() - start))
            await asyncio.sleep(max(0.1, wait_time))

    def _refill(self) -> None:
        """Refill tokens based on elapsed time (must hold lock)."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (approximate)."""
        return self._tokens


class HistoricalRequestScheduler:
    """
    Priority-based scheduler for IB historical data requests.

    A1: Ensures IB pacing limits are never exceeded while prioritizing
    critical requests (alerts) over background tasks (prefetch).

    OPT-015: Enhanced with status visibility and event publishing.
    """

    # IB limits
    MAX_CONCURRENT = 6  # Max concurrent historical requests
    DEDUP_WINDOW_SECONDS = 15  # Deduplication window
    STATUS_PUBLISH_INTERVAL = 5.0  # OPT-015: Publish status every 5 seconds

    def __init__(
        self,
        historical_service: "HistoricalDataService",
        max_concurrent: int = MAX_CONCURRENT,
        dedup_window_seconds: int = DEDUP_WINDOW_SECONDS,
        event_bus: Optional["EventBus"] = None,
    ):
        """
        Initialize scheduler.

        Args:
            historical_service: Service for actual data fetching.
            max_concurrent: Max concurrent requests (default: 6).
            dedup_window_seconds: Deduplication window (default: 15s).
            event_bus: Optional event bus for status publishing (OPT-015).
        """
        self._historical = historical_service
        self._max_concurrent = max_concurrent
        self._dedup_window = dedup_window_seconds
        self._event_bus = event_bus  # OPT-015

        # Priority queues (one per priority level)
        self._queues: Dict[RequestPriority, asyncio.Queue] = {
            p: asyncio.Queue() for p in RequestPriority
        }

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._token_bucket = TokenBucket()

        # Deduplication cache: key -> (result_future, timestamp)
        self._pending: OrderedDict[str, tuple] = OrderedDict()
        self._pending_lock = asyncio.Lock()

        # State
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None  # OPT-015
        self._metrics = SchedulerMetrics()

    async def start(self) -> None:
        """Start the background queue processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())

        # OPT-015: Start status publishing if event bus available
        if self._event_bus:
            self._status_task = asyncio.create_task(self._periodic_status_publish())

        logger.info("HistoricalRequestScheduler started")

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

        # OPT-015: Stop status publishing
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
            self._status_task = None

        logger.info("HistoricalRequestScheduler stopped")

    async def request(
        self,
        symbol: str,
        timeframe: str,
        period: BarPeriod,
        priority: RequestPriority = RequestPriority.UI,
    ) -> List[Any]:
        """
        Request historical bars with priority scheduling.

        High-priority requests are processed before lower priority.
        Identical requests within dedup window share the same result.

        Args:
            symbol: Stock symbol.
            timeframe: Bar timeframe (e.g., "1d", "1h").
            period: Bar period specification.
            priority: Request priority level.

        Returns:
            List of bar data.
        """
        self._metrics.requests_total += 1
        self._metrics.requests_by_priority[priority.name] += 1
        start_time = time.time()

        # Check deduplication cache
        cache_key = f"{symbol}:{timeframe}:{period}"

        async with self._pending_lock:
            if cache_key in self._pending:
                pending_future, timestamp = self._pending[cache_key]
                # Check if still within dedup window
                if time.time() - timestamp < self._dedup_window:
                    self._metrics.dedup_hits += 1
                    logger.debug(
                        "Dedup hit for %s (priority=%s)",
                        cache_key, priority.name
                    )
                    # Wait for existing request to complete
                    return await pending_future

        # Create new request
        request = HistoricalRequest(
            symbol=symbol,
            timeframe=timeframe,
            period=period,
            priority=priority,
        )

        # Register in pending cache
        async with self._pending_lock:
            self._pending[cache_key] = (request.future, time.time())
            # Cleanup old entries
            self._cleanup_pending()

        # Add to priority queue
        await self._queues[priority].put(request)

        # Wait for result
        try:
            result = await request.future
            wait_ms = (time.time() - start_time) * 1000
            self._metrics.record_wait(wait_ms)
            return result
        finally:
            # Remove from pending cache
            async with self._pending_lock:
                self._pending.pop(cache_key, None)

    async def _process_queue(self) -> None:
        """
        Background task that processes queued requests.

        Processes highest priority queue first, respecting rate limits.
        """
        while self._running:
            try:
                # Find next request (highest priority first)
                request = await self._get_next_request()

                if request is None:
                    await asyncio.sleep(0.01)  # Brief sleep when all queues empty
                    continue

                # Process with concurrency limit
                asyncio.create_task(self._process_request(request))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler error: %s", e)
                await asyncio.sleep(0.1)

    async def _get_next_request(self) -> Optional[HistoricalRequest]:
        """Get next request from highest priority non-empty queue."""
        for priority in RequestPriority:
            queue = self._queues[priority]
            if not queue.empty():
                try:
                    return queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
        return None

    async def _process_request(self, request: HistoricalRequest) -> None:
        """
        Process a single request with rate limiting.

        Acquires semaphore and token before making request.
        """
        try:
            # Acquire concurrency semaphore
            async with self._semaphore:
                # Acquire rate limit token
                if not await self._token_bucket.acquire(timeout=30.0):
                    self._metrics.rate_limit_waits += 1
                    logger.warning(
                        "Rate limit timeout for %s (priority=%s)",
                        request.symbol, request.priority.name
                    )
                    request.future.set_exception(
                        TimeoutError("Rate limit timeout")
                    )
                    return

                # Make the actual request
                try:
                    bars = await self._historical.fetch_bars(
                        request.symbol,
                        request.timeframe,
                        request.period,
                    )
                    request.future.set_result(bars)
                except Exception as e:
                    request.future.set_exception(e)

        except Exception as e:
            if not request.future.done():
                request.future.set_exception(e)

    def _cleanup_pending(self) -> None:
        """Remove expired entries from pending cache (must hold lock)."""
        now = time.time()
        expired = [
            key for key, (_, timestamp) in self._pending.items()
            if now - timestamp > self._dedup_window * 2
        ]
        for key in expired:
            self._pending.pop(key, None)

    def get_metrics(self) -> SchedulerMetrics:
        """Get scheduler performance metrics."""
        return self._metrics

    def get_stats(self) -> dict:
        """Get scheduler statistics for monitoring."""
        return {
            "running": self._running,
            "pending_requests": sum(q.qsize() for q in self._queues.values()),
            "by_priority": {p.name: self._queues[p].qsize() for p in RequestPriority},
            "concurrent_active": self._max_concurrent - self._semaphore._value,
            "available_tokens": self._token_bucket.available_tokens,
            "requests_total": self._metrics.requests_total,
            "requests_by_priority": self._metrics.requests_by_priority,
            "dedup_hits": self._metrics.dedup_hits,
            "rate_limit_waits": self._metrics.rate_limit_waits,
            "avg_wait_ms": f"{self._metrics.avg_wait_ms:.1f}",
            "max_wait_ms": f"{self._metrics.max_wait_ms:.1f}",
        }

    # -------------------------------------------------------------------------
    # OPT-015: Status Visibility
    # -------------------------------------------------------------------------

    def get_status(self) -> SchedulerStatus:
        """
        OPT-015: Get current scheduler status for display.

        Returns a SchedulerStatus object with all relevant metrics
        for dashboard display and monitoring.
        """
        queue_depth = {p.name: self._queues[p].qsize() for p in RequestPriority}
        total_queued = sum(queue_depth.values())
        tokens = self._token_bucket.available_tokens
        rate_limited = tokens < 1.0

        # Calculate wait time if rate limited
        rate_limit_wait_sec = None
        if rate_limited:
            # Time to get 1 token: (1 - tokens) / refill_rate
            rate_limit_wait_sec = (1.0 - tokens) / self._token_bucket._refill_rate

        return SchedulerStatus(
            tokens_available=tokens,
            tokens_max=self._token_bucket._max_tokens,
            queue_depth=queue_depth,
            total_queued=total_queued,
            in_flight=self._max_concurrent - self._semaphore._value,
            max_concurrent=self._max_concurrent,
            requests_completed=self._metrics.requests_total,
            requests_failed=self._metrics.rate_limit_waits,
            dedup_hits=self._metrics.dedup_hits,
            avg_request_time_ms=self._metrics.avg_wait_ms,
            max_request_time_ms=self._metrics.max_wait_ms,
            rate_limited=rate_limited,
            rate_limit_wait_sec=rate_limit_wait_sec,
        )

    async def _periodic_status_publish(self) -> None:
        """
        OPT-015: Periodically publish status to event bus.

        Runs as background task, publishes status every STATUS_PUBLISH_INTERVAL seconds.
        """
        from ..domain.interfaces.event_bus import EventType

        while self._running:
            try:
                await asyncio.sleep(self.STATUS_PUBLISH_INTERVAL)

                if not self._event_bus:
                    continue

                status = self.get_status()

                # Publish status event
                self._event_bus.publish(
                    EventType.HEALTH_CHECK,
                    {
                        "source": "historical_scheduler",
                        "type": "scheduler_status",
                        "tokens_available": round(status.tokens_available, 1),
                        "tokens_max": status.tokens_max,
                        "queue_depth": status.queue_depth,
                        "total_queued": status.total_queued,
                        "in_flight": status.in_flight,
                        "rate_limited": status.rate_limited,
                        "rate_limit_wait_sec": round(status.rate_limit_wait_sec, 1) if status.rate_limit_wait_sec else None,
                    }
                )

                # Log if rate limited
                if status.rate_limited:
                    logger.info(
                        f"Historical rate limit: tokens={status.tokens_available:.1f}/{status.tokens_max}, "
                        f"wait={status.rate_limit_wait_sec:.1f}s, queued={status.total_queued}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in status publish: {e}")

    def _publish_rate_limit_event(self, wait_sec: float) -> None:
        """
        OPT-015: Publish rate limit event when waiting for tokens.

        Called when a request must wait for rate limit tokens.
        """
        if not self._event_bus:
            return

        from ..domain.interfaces.event_bus import EventType

        self._event_bus.publish(
            EventType.HEALTH_CHECK,
            {
                "source": "historical_scheduler",
                "type": "rate_limit_wait",
                "tokens_available": self._token_bucket.available_tokens,
                "estimated_wait_sec": round(wait_sec, 1),
                "queue_depth": sum(q.qsize() for q in self._queues.values()),
            }
        )
