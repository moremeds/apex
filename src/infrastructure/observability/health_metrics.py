"""
System health metrics instrumentation.

Tracks operational health of the Apex system:
- Broker connectivity status
- Market data coverage and staleness
- Event bus queue depths
- System readiness state
- Adapter health per component
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from opentelemetry import metrics
from ...utils.logging_setup import get_logger

logger = get_logger(__name__)


class HealthMetrics:
    """
    Instruments for system health monitoring.

    Tracks broker connections, market data coverage, queue depths,
    and component health for operational monitoring and alerting.

    All metrics are prefixed with 'apex_' for namespace isolation.
    """

    def __init__(self, meter: metrics.Meter):
        """
        Initialize health metrics instruments.

        Args:
            meter: OpenTelemetry Meter for creating instruments.
        """
        self._meter = meter

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

        # Last tick timestamp per symbol (unix seconds)
        self._last_tick_timestamp = meter.create_gauge(
            name="apex_last_tick_timestamp",
            description="Unix timestamp of last tick received",
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

        # Track last tick times for staleness calculation
        self._last_tick_times: Dict[str, float] = {}

    def record_broker_status(self, broker: str, connected: bool) -> None:
        """
        Record broker connection status.

        Args:
            broker: Broker name (e.g., "ib", "futu").
            connected: True if connected, False otherwise.
        """
        self._broker_connected.set(1 if connected else 0, {"broker": broker})
        logger.debug(f"Broker {broker} status: {'connected' if connected else 'disconnected'}")

    def record_adapter_health(self, adapter: str, status: str) -> None:
        """
        Record adapter health status.

        Args:
            adapter: Adapter name (e.g., "ib_live", "futu_trade").
            status: Health status ("healthy", "unhealthy", "degraded").
        """
        status_map = {"healthy": 1, "degraded": -1, "unhealthy": 0}
        value = status_map.get(status.lower(), 0)
        self._adapter_health.set(value, {"adapter": adapter})

    def record_market_data_coverage(self, coverage: float) -> None:
        """
        Record market data coverage ratio.

        Args:
            coverage: Coverage ratio from 0.0 to 1.0.
        """
        self._market_data_coverage.set(coverage)

    def record_tick_received(self, symbol: str) -> None:
        """
        Record that a tick was received for a symbol.

        Updates both per-symbol and global last tick timestamps.

        Args:
            symbol: Symbol that received a tick.
        """
        now = time.time()
        self._last_tick_timestamp.set(now, {"symbol": symbol})
        self._last_any_tick_timestamp.set(now)
        self._last_tick_times[symbol] = now

    def record_tick_latency(self, latency_ms: float) -> None:
        """
        Record tick processing latency.

        Args:
            latency_ms: Latency from tick receive to store update in ms.
        """
        self._tick_to_store_ms.record(latency_ms)

    def record_queue_size(self, lane: str, size: int) -> None:
        """
        Record event bus queue size.

        Args:
            lane: Queue lane name (e.g., "fast", "slow", "critical").
            size: Current queue depth.
        """
        self._event_bus_queue_size.set(size, {"lane": lane})

    def record_slow_lane_gap(self, gap_ms: float) -> None:
        """
        Record maximum gap between slow lane dispatches.

        Used to verify slow lane starvation fix.

        Args:
            gap_ms: Gap in milliseconds.
        """
        self._slow_lane_max_gap_ms.set(gap_ms)

    def record_system_ready(self, ready: bool) -> None:
        """
        Record system readiness state.

        Args:
            ready: True if system is ready, False otherwise.
        """
        self._system_ready.set(1 if ready else 0)

    def record_startup_duration(self, duration_seconds: float) -> None:
        """
        Record time from startup to first valid snapshot.

        Args:
            duration_seconds: Startup duration in seconds.
        """
        self._startup_duration_seconds.set(duration_seconds)

    def record_event_processing(self, event_type: str, duration_ms: float) -> None:
        """
        Record event processing duration.

        Args:
            event_type: Type of event processed.
            duration_ms: Processing duration in milliseconds.
        """
        self._event_process_duration_ms.record(duration_ms, {"event_type": event_type})

    def record_connection_attempt(self, broker: str, success: bool) -> None:
        """
        Record a connection attempt.

        Args:
            broker: Broker name.
            success: True if successful, False if failed.
        """
        self._connection_attempts.add(1, {"broker": broker})
        if not success:
            self._connection_failures.add(1, {"broker": broker})

    def record_reconnection(self, broker: str) -> None:
        """
        Record a reconnection event.

        Args:
            broker: Broker that reconnected.
        """
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
