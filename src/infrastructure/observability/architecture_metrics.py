"""
Architecture Components Metrics - OpenTelemetry gauges for A1-A4.

Exposes Prometheus-compatible metrics for:
- A1: HistoricalRequestScheduler - queue depths, wait times, rate limits
- A2: MarketDataRouter - line usage, subscriptions, fanout
- A3: IndicatorStore - cache hit rates, computations
- A4: IbConnectionPool - connection status per role

Usage:
    metrics = ArchitectureMetrics(meter)
    metrics.record_scheduler_stats(scheduler.get_stats())
    metrics.record_router_stats(router.get_stats())
    metrics.record_indicator_stats(store.get_stats())
    metrics.record_pool_stats(pool.get_status())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from ...utils.logging_setup import get_logger

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter

logger = get_logger(__name__)


class ArchitectureMetrics:
    """
    OpenTelemetry metrics for Architecture Components (A1-A4).

    Provides gauges and counters for monitoring component health
    and performance via Prometheus scraping.
    """

    def __init__(self, meter: Optional["Meter"] = None):
        """
        Initialize architecture metrics.

        Args:
            meter: OpenTelemetry Meter instance. If None, metrics are no-op.
        """
        self._meter = meter
        self._initialized = False

        # Counter state for incremental metrics
        self._last_scheduler_total: int = 0
        self._last_scheduler_dedup: int = 0
        self._last_router_total: int = 0
        self._last_router_rejections: int = 0
        self._last_indicator_hits: int = 0
        self._last_indicator_misses: int = 0
        self._last_indicator_computations: int = 0

        if meter:
            self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize OpenTelemetry metrics instruments."""
        if not self._meter:
            return

        # A1: HistoricalRequestScheduler metrics
        self._scheduler_pending = self._meter.create_observable_gauge(
            name="apex_scheduler_pending_requests",
            description="Number of pending historical data requests",
            callbacks=[self._get_scheduler_pending],
        )

        self._scheduler_concurrent = self._meter.create_observable_gauge(
            name="apex_scheduler_concurrent_active",
            description="Number of concurrent active requests",
            callbacks=[self._get_scheduler_concurrent],
        )

        self._scheduler_tokens = self._meter.create_observable_gauge(
            name="apex_scheduler_available_tokens",
            description="Available rate limit tokens",
            callbacks=[self._get_scheduler_tokens],
        )

        self._scheduler_wait_ms = self._meter.create_histogram(
            name="apex_scheduler_wait_ms",
            description="Request wait time in milliseconds",
            unit="ms",
        )

        self._scheduler_requests = self._meter.create_counter(
            name="apex_scheduler_requests_total",
            description="Total historical data requests",
        )

        self._scheduler_dedup = self._meter.create_counter(
            name="apex_scheduler_dedup_hits_total",
            description="Deduplication cache hits",
        )

        # A2: MarketDataRouter metrics
        self._router_active_lines = self._meter.create_observable_gauge(
            name="apex_router_active_lines",
            description="Active streaming market data lines",
            callbacks=[self._get_router_lines],
        )

        self._router_available_lines = self._meter.create_observable_gauge(
            name="apex_router_available_lines",
            description="Available streaming lines",
            callbacks=[self._get_router_available],
        )

        self._router_subscriptions = self._meter.create_counter(
            name="apex_router_subscriptions_total",
            description="Total subscription requests",
        )

        self._router_rejections = self._meter.create_counter(
            name="apex_router_line_limit_rejections_total",
            description="Subscriptions rejected due to line limit",
        )

        # A3: IndicatorStore metrics
        self._indicator_entries = self._meter.create_observable_gauge(
            name="apex_indicator_cache_entries",
            description="Number of cached indicator values",
            callbacks=[self._get_indicator_entries],
        )

        self._indicator_hit_rate = self._meter.create_observable_gauge(
            name="apex_indicator_cache_hit_rate",
            description="Cache hit rate percentage",
            callbacks=[self._get_indicator_hit_rate],
        )

        self._indicator_hits = self._meter.create_counter(
            name="apex_indicator_cache_hits_total",
            description="Total cache hits",
        )

        self._indicator_misses = self._meter.create_counter(
            name="apex_indicator_cache_misses_total",
            description="Total cache misses",
        )

        self._indicator_computations = self._meter.create_counter(
            name="apex_indicator_computations_total",
            description="Total indicator computations",
        )

        # A4: IbConnectionPool metrics
        self._pool_connected = self._meter.create_observable_gauge(
            name="apex_pool_connected",
            description="Connection pool fully connected (1=yes, 0=no)",
            callbacks=[self._get_pool_connected],
        )

        self._pool_monitoring = self._meter.create_observable_gauge(
            name="apex_pool_monitoring_connected",
            description="Monitoring connection status (1=connected, 0=disconnected)",
            callbacks=[self._get_pool_monitoring],
        )

        self._pool_historical = self._meter.create_observable_gauge(
            name="apex_pool_historical_connected",
            description="Historical connection status (1=connected, 0=disconnected)",
            callbacks=[self._get_pool_historical],
        )

        self._pool_execution = self._meter.create_observable_gauge(
            name="apex_pool_execution_connected",
            description="Execution connection status (1=connected, 0=disconnected)",
            callbacks=[self._get_pool_execution],
        )

        self._initialized = True
        logger.info("Architecture metrics initialized")

    # Cached stats for observable gauges
    _scheduler_stats: Dict[str, Any] = {}
    _router_stats: Dict[str, Any] = {}
    _indicator_stats: Dict[str, Any] = {}
    _pool_stats: Dict[str, Any] = {}

    def record_scheduler_stats(self, stats: Dict[str, Any]) -> None:
        """
        Record scheduler statistics snapshot.

        Args:
            stats: Stats dict from scheduler.get_stats()
        """
        ArchitectureMetrics._scheduler_stats = stats

        if self._initialized and self._meter:
            # Record counters (incremental)
            total = stats.get("requests_total", 0)
            delta = total - self._last_scheduler_total
            if delta > 0:
                self._scheduler_requests.add(delta)
            self._last_scheduler_total = total

            dedup = stats.get("dedup_hits", 0)
            delta = dedup - self._last_scheduler_dedup
            if delta > 0:
                self._scheduler_dedup.add(delta)
            self._last_scheduler_dedup = dedup

    def record_router_stats(self, stats: Dict[str, Any]) -> None:
        """Record router statistics snapshot."""
        ArchitectureMetrics._router_stats = stats

        if self._initialized and self._meter:
            total = stats.get("subscriptions_total", 0)
            delta = total - self._last_router_total
            if delta > 0:
                self._router_subscriptions.add(delta)
            self._last_router_total = total

            rejections = stats.get("line_limit_rejections", 0)
            delta = rejections - self._last_router_rejections
            if delta > 0:
                self._router_rejections.add(delta)
            self._last_router_rejections = rejections

    def record_indicator_stats(self, stats: Dict[str, Any]) -> None:
        """Record indicator store statistics snapshot."""
        ArchitectureMetrics._indicator_stats = stats

        if self._initialized and self._meter:
            hits = stats.get("hits", 0)
            delta = hits - self._last_indicator_hits
            if delta > 0:
                self._indicator_hits.add(delta)
            self._last_indicator_hits = hits

            misses = stats.get("misses", 0)
            delta = misses - self._last_indicator_misses
            if delta > 0:
                self._indicator_misses.add(delta)
            self._last_indicator_misses = misses

            computations = stats.get("computations", 0)
            delta = computations - self._last_indicator_computations
            if delta > 0:
                self._indicator_computations.add(delta)
            self._last_indicator_computations = computations

    def record_pool_stats(self, stats: Dict[str, Any]) -> None:
        """Record connection pool statistics snapshot."""
        ArchitectureMetrics._pool_stats = stats

    # Observable gauge callbacks
    def _get_scheduler_pending(self, options: Any) -> Any:
        yield ArchitectureMetrics._scheduler_stats.get("pending_requests", 0)

    def _get_scheduler_concurrent(self, options: Any) -> Any:
        yield ArchitectureMetrics._scheduler_stats.get("concurrent_active", 0)

    def _get_scheduler_tokens(self, options: Any) -> Any:
        yield ArchitectureMetrics._scheduler_stats.get("available_tokens", 0)

    def _get_router_lines(self, options: Any) -> Any:
        yield ArchitectureMetrics._router_stats.get("active_lines", 0)

    def _get_router_available(self, options: Any) -> Any:
        yield ArchitectureMetrics._router_stats.get("available_lines", 0)

    def _get_indicator_entries(self, options: Any) -> Any:
        yield ArchitectureMetrics._indicator_stats.get("entries", 0)

    def _get_indicator_hit_rate(self, options: Any) -> Any:
        rate_str = ArchitectureMetrics._indicator_stats.get("hit_rate", "0.0%")
        try:
            yield float(rate_str.replace("%", ""))
        except ValueError:
            yield 0.0

    def _get_pool_connected(self, options: Any) -> Any:
        yield 1 if ArchitectureMetrics._pool_stats.get("connected", False) else 0

    def _get_pool_monitoring(self, options: Any) -> Any:
        mon = ArchitectureMetrics._pool_stats.get("monitoring", {})
        yield 1 if mon.get("connected", False) else 0

    def _get_pool_historical(self, options: Any) -> Any:
        hist = ArchitectureMetrics._pool_stats.get("historical", {})
        yield 1 if hist.get("connected", False) else 0

    def _get_pool_execution(self, options: Any) -> Any:
        exec_ = ArchitectureMetrics._pool_stats.get("execution", {})
        yield 1 if exec_.get("connected", False) else 0
