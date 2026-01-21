"""
System health metrics instrumentation.

Tracks operational health of the Apex system:
- Broker connectivity status
- Market data coverage and staleness
- Event bus queue depths
- System readiness state
- Adapter health per component

Note: OpenTelemetry is an optional dependency. When not installed,
HealthMetrics will accept a None meter and operate in no-op mode.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Set

from ...utils.logging_setup import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter

# Default threshold for considering a symbol "active" (received tick recently)
DEFAULT_ACTIVE_THRESHOLD_SECONDS = 30.0


class _NoopInstrument:
    """No-op instrument that accepts but ignores all metric calls."""

    def add(self, value: Any, attributes: Any = None) -> None:
        pass

    def set(self, value: Any, attributes: Any = None) -> None:
        pass

    def record(self, value: Any, attributes: Any = None) -> None:
        pass


class HealthMetrics:
    """
    Instruments for system health monitoring.

    Tracks broker connections, market data coverage, queue depths,
    and component health for operational monitoring and alerting.

    All metrics are prefixed with 'apex_' for namespace isolation.

    When meter is None (OpenTelemetry not installed), operates in no-op mode.
    """

    # Type annotations for instruments (Any allows both real and noop instruments)
    _broker_connected: Any
    _adapter_health: Any
    _market_data_coverage: Any
    _last_tick_timestamp: Any
    _last_any_tick_timestamp: Any
    _event_bus_queue_size: Any
    _system_ready: Any
    _startup_duration_seconds: Any
    _tick_to_store_ms: Any
    _event_process_duration_ms: Any
    _slow_lane_max_gap_ms: Any
    _connection_attempts: Any
    _connection_failures: Any
    _reconnection_count: Any
    _subscribed_symbols_count: Any
    _active_symbols_count: Any
    _tick_reception_coverage: Any

    def __init__(self, meter: "Meter | None"):
        """
        Initialize health metrics instruments.

        Args:
            meter: OpenTelemetry Meter for creating instruments, or None for no-op mode.
        """
        self._meter = meter
        self._noop = meter is None

        # Track last tick times for staleness calculation (always needed)
        self._last_tick_times: Dict[str, float] = {}
        # Track subscribed symbols (OBS-001) (always needed)
        self._subscribed_symbols: Set[str] = set()

        if self._noop:
            # No-op mode - use NoopInstrument that accepts but ignores calls
            noop = _NoopInstrument()
            self._broker_connected = noop
            self._adapter_health = noop
            self._market_data_coverage = noop
            self._last_tick_timestamp = noop
            self._last_any_tick_timestamp = noop
            self._event_bus_queue_size = noop
            self._system_ready = noop
            self._startup_duration_seconds = noop
            self._tick_to_store_ms = noop
            self._event_process_duration_ms = noop
            self._slow_lane_max_gap_ms = noop
            self._connection_attempts = noop
            self._connection_failures = noop
            self._reconnection_count = noop
            self._subscribed_symbols_count = noop
            self._active_symbols_count = noop
            self._tick_reception_coverage = noop
            return

        # mypy: meter is guaranteed non-None after noop check
        assert meter is not None

        # Broker connectivity gauge (1=connected, 0=disconnected)
        self._broker_connected = meter.create_gauge(
            name="apex_broker_connected",
            description="Broker connection status: 1=connected, 0=disconnected",
        )

        # Adapter health gauge (1=healthy, 0=unhealthy, -1=degraded)
        self._adapter_health = meter.create_gauge(
            name="apex_adapter_health",
            description="Adapter health status: 1=healthy, 0=unhealthy, -1=degraded",
        )

        # Market data coverage ratio (0.0-1.0)
        self._market_data_coverage = meter.create_gauge(
            name="apex_market_data_coverage",
            description="Market data coverage ratio (0.0-1.0)",
        )

        # Last tick timestamp per provider (unix seconds) - low cardinality
        self._last_tick_timestamp = meter.create_gauge(
            name="apex_last_tick_timestamp",
            description="Unix timestamp of last tick received by provider",
            unit="seconds",
        )

        # Global last tick timestamp (any symbol)
        self._last_any_tick_timestamp = meter.create_gauge(
            name="apex_last_any_tick_timestamp",
            description="Unix timestamp of last tick received from any symbol",
            unit="seconds",
        )

        # Event bus queue sizes
        self._event_bus_queue_size = meter.create_gauge(
            name="apex_event_bus_queue_size",
            description="Event bus queue depth",
        )

        # System readiness (1=ready, 0=not ready)
        self._system_ready = meter.create_gauge(
            name="apex_system_ready",
            description="System readiness: 1=ready, 0=not ready",
        )

        # Startup metrics
        self._startup_duration_seconds = meter.create_gauge(
            name="apex_startup_duration_seconds",
            description="Time from start to first valid snapshot",
            unit="seconds",
        )

        # Tick processing latency histogram
        self._tick_to_store_ms = meter.create_histogram(
            name="apex_tick_to_store_ms",
            description="Tick processing latency (receive to store)",
            unit="ms",
        )

        # Event processing latency histogram
        self._event_process_duration_ms = meter.create_histogram(
            name="apex_event_process_duration_ms",
            description="Event processing duration by type",
            unit="ms",
        )

        # Slow lane gap tracking (for priority event bus)
        self._slow_lane_max_gap_ms = meter.create_gauge(
            name="apex_eventbus_slow_max_gap_ms",
            description="Maximum time between slow lane dispatches",
            unit="ms",
        )

        # Connection attempt counters
        self._connection_attempts = meter.create_counter(
            name="apex_connection_attempts_total",
            description="Total connection attempts by broker",
        )
        self._connection_failures = meter.create_counter(
            name="apex_connection_failures_total",
            description="Total connection failures by broker",
        )

        # Reconnection tracking
        self._reconnection_count = meter.create_counter(
            name="apex_reconnection_total",
            description="Total reconnection events",
        )

        # Subscription coverage metrics (OBS-001)
        self._subscribed_symbols_count = meter.create_gauge(
            name="apex_subscribed_symbols",
            description="Number of symbols with active subscriptions",
        )
        self._active_symbols_count = meter.create_gauge(
            name="apex_active_symbols",
            description="Symbols with ticks received in the last 30 seconds",
        )
        self._tick_reception_coverage = meter.create_gauge(
            name="apex_tick_reception_coverage",
            description="Ratio of active symbols to subscribed symbols (0.0-1.0)",
        )

    def record_broker_status(self, broker: str, connected: bool) -> None:
        """
        Record broker connection status.

        Args:
            broker: Broker name (e.g., "ib", "futu").
            connected: True if connected, False otherwise.
        """
        logger.debug(f"Broker {broker} status: {'connected' if connected else 'disconnected'}")
        if self._noop:
            return
        self._broker_connected.set(1 if connected else 0, {"broker": broker})

    def record_adapter_health(self, adapter: str, status: str) -> None:
        """
        Record adapter health status.

        Args:
            adapter: Adapter name (e.g., "ib_live", "futu_trade").
            status: Health status ("healthy", "unhealthy", "degraded").
        """
        if self._noop:
            return
        status_map = {"healthy": 1, "degraded": -1, "unhealthy": 0}
        value = status_map.get(status.lower(), 0)
        self._adapter_health.set(value, {"adapter": adapter})

    def record_market_data_coverage(self, coverage: float) -> None:
        """
        Record market data coverage ratio.

        Args:
            coverage: Coverage ratio from 0.0 to 1.0.
        """
        if self._noop:
            return
        self._market_data_coverage.set(coverage)

    def record_tick_received(self, symbol: str, provider: str = "unknown") -> None:
        """
        Record that a tick was received for a symbol.

        Note: Metrics are tracked by provider (low cardinality) not symbol (high cardinality).
        Per-symbol timestamps are kept internally for staleness calculation.

        Args:
            symbol: Symbol that received a tick.
            provider: Data provider name (e.g., "ib", "futu").
        """
        now = time.time()
        # Keep internal per-symbol tracking for staleness checks (always needed)
        self._last_tick_times[symbol] = now
        if self._noop:
            return
        # Low-cardinality metric: track by provider, not by symbol (MAJ-007)
        self._last_tick_timestamp.set(now, {"provider": provider})
        self._last_any_tick_timestamp.set(now)

    def record_tick_latency(self, latency_ms: float) -> None:
        """
        Record tick processing latency.

        Args:
            latency_ms: Latency from tick receive to store update in ms.
        """
        if self._noop:
            return
        self._tick_to_store_ms.record(latency_ms)

    def record_queue_size(self, lane: str, size: int) -> None:
        """
        Record event bus queue size.

        Args:
            lane: Queue lane name (e.g., "fast", "slow", "critical").
            size: Current queue depth.
        """
        if self._noop:
            return
        self._event_bus_queue_size.set(size, {"lane": lane})

    def record_slow_lane_gap(self, gap_ms: float) -> None:
        """
        Record maximum gap between slow lane dispatches.

        Used to verify slow lane starvation fix.

        Args:
            gap_ms: Gap in milliseconds.
        """
        if self._noop:
            return
        self._slow_lane_max_gap_ms.set(gap_ms)

    def record_system_ready(self, ready: bool) -> None:
        """
        Record system readiness state.

        Args:
            ready: True if system is ready, False otherwise.
        """
        if self._noop:
            return
        self._system_ready.set(1 if ready else 0)

    def record_startup_duration(self, duration_seconds: float) -> None:
        """
        Record time from startup to first valid snapshot.

        Args:
            duration_seconds: Startup duration in seconds.
        """
        if self._noop:
            return
        self._startup_duration_seconds.set(duration_seconds)

    def record_event_processing(self, event_type: str, duration_ms: float) -> None:
        """
        Record event processing duration.

        Args:
            event_type: Type of event processed.
            duration_ms: Processing duration in milliseconds.
        """
        if self._noop:
            return
        self._event_process_duration_ms.record(duration_ms, {"event_type": event_type})

    def record_connection_attempt(self, broker: str, success: bool) -> None:
        """
        Record a connection attempt.

        Args:
            broker: Broker name.
            success: True if successful, False if failed.
        """
        if self._noop:
            return
        self._connection_attempts.add(1, {"broker": broker})
        if not success:
            self._connection_failures.add(1, {"broker": broker})

    def record_reconnection(self, broker: str) -> None:
        """
        Record a reconnection event.

        Args:
            broker: Broker that reconnected.
        """
        if self._noop:
            return
        self._reconnection_count.add(1, {"broker": broker})

    def get_stale_symbols(self, threshold_seconds: float = 30.0) -> list[str]:
        """
        Get list of symbols with stale market data.

        Args:
            threshold_seconds: Staleness threshold in seconds.

        Returns:
            List of symbol names with stale data.
        """
        now = time.time()
        stale = []
        for symbol, last_time in self._last_tick_times.items():
            if now - last_time > threshold_seconds:
                stale.append(symbol)
        return stale

    def calculate_coverage(self, total_positions: int, positions_with_data: int) -> float:
        """
        Calculate and record market data coverage.

        Args:
            total_positions: Total number of positions.
            positions_with_data: Positions with valid market data.

        Returns:
            Coverage ratio (0.0-1.0).
        """
        if total_positions == 0:
            coverage = 1.0
        else:
            coverage = positions_with_data / total_positions

        self.record_market_data_coverage(coverage)
        return coverage

    # =========================================================================
    # Subscription Coverage (OBS-001)
    # =========================================================================

    def record_subscriptions(self, symbols: list[str]) -> None:
        """
        Record symbols that we have subscribed to for market data.

        Call this when subscribing to market data to track expected symbols.

        Args:
            symbols: List of symbols being subscribed to.
        """
        self._subscribed_symbols.update(symbols)
        if self._noop:
            return
        self._subscribed_symbols_count.set(len(self._subscribed_symbols))

    def remove_subscriptions(self, symbols: list[str]) -> None:
        """
        Remove symbols from the subscription tracking.

        Call this when unsubscribing from market data.

        Args:
            symbols: List of symbols being unsubscribed.
        """
        for symbol in symbols:
            self._subscribed_symbols.discard(symbol)
        if self._noop:
            return
        self._subscribed_symbols_count.set(len(self._subscribed_symbols))

    def update_tick_reception_coverage(
        self, active_threshold_seconds: float = DEFAULT_ACTIVE_THRESHOLD_SECONDS
    ) -> float:
        """
        Calculate and record tick reception coverage.

        Coverage = (active symbols) / (subscribed symbols)
        where "active" means received a tick within threshold.

        Args:
            active_threshold_seconds: Threshold for considering a symbol active.

        Returns:
            Coverage ratio (0.0-1.0).
        """
        now = time.time()

        # Count symbols that received a tick recently
        active_count = sum(
            1
            for symbol in self._subscribed_symbols
            if symbol in self._last_tick_times
            and (now - self._last_tick_times[symbol]) < active_threshold_seconds
        )

        subscribed_count = len(self._subscribed_symbols)

        if subscribed_count == 0:
            coverage = 1.0  # No subscriptions = full coverage by definition
        else:
            coverage = active_count / subscribed_count

        if self._noop:
            return coverage

        # Record metrics
        self._active_symbols_count.set(active_count)
        self._tick_reception_coverage.set(coverage)
        return coverage

    def get_inactive_subscribed_symbols(
        self, threshold_seconds: float = DEFAULT_ACTIVE_THRESHOLD_SECONDS
    ) -> list[str]:
        """
        Get list of subscribed symbols that haven't received ticks recently.

        Useful for diagnosing market data issues.

        Args:
            threshold_seconds: Threshold for considering a symbol inactive.

        Returns:
            List of symbol names that are subscribed but inactive.
        """
        now = time.time()
        inactive = []
        for symbol in self._subscribed_symbols:
            last_tick = self._last_tick_times.get(symbol)
            if last_tick is None or (now - last_tick) > threshold_seconds:
                inactive.append(symbol)
        return inactive
