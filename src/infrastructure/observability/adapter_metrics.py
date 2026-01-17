"""
Adapter-specific metrics instrumentation.

Exposes per-adapter operational metrics for Prometheus:
- Connection status and reconnect counts
- Data throughput (quotes, bars, fills received)
- Error counts and types
- Request/response latencies
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from opentelemetry import metrics

from ...utils.logging_setup import get_logger

logger = get_logger(__name__)


class AdapterMetrics:
    """
    Instruments for adapter operational metrics.

    Tracks per-adapter health and performance. All metrics include
    adapter_name and broker labels for filtering in Prometheus.

    Metrics prefixed with 'apex_adapter_' for namespace isolation.
    """

    def __init__(self, meter: metrics.Meter):
        """
        Initialize adapter metrics instruments.

        Args:
            meter: OpenTelemetry Meter for creating instruments.
        """
        self._meter = meter

        # Connection status gauge (1=connected, 0=disconnected)
        self._connection_status = meter.create_gauge(
            name="apex_adapter_connected",
            description="Adapter connection status: 1=connected, 0=disconnected",
        )

        # Reconnect counter
        self._reconnect_total = meter.create_counter(
            name="apex_adapter_reconnect_total",
            description="Total reconnection attempts per adapter",
        )

        # Data throughput counters
        self._quotes_received = meter.create_counter(
            name="apex_adapter_quotes_received_total",
            description="Total quote ticks received per adapter",
        )
        self._bars_received = meter.create_counter(
            name="apex_adapter_bars_received_total",
            description="Total bars/candles received per adapter",
        )
        self._fills_received = meter.create_counter(
            name="apex_adapter_fills_received_total",
            description="Total order fills received per adapter",
        )
        self._orders_submitted = meter.create_counter(
            name="apex_adapter_orders_submitted_total",
            description="Total orders submitted per adapter",
        )

        # Error counters
        self._errors_total = meter.create_counter(
            name="apex_adapter_errors_total",
            description="Total errors per adapter by type",
        )

        # Latency histograms
        self._request_latency = meter.create_histogram(
            name="apex_adapter_request_latency_ms",
            description="Request latency in milliseconds",
            unit="ms",
        )
        self._callback_latency = meter.create_histogram(
            name="apex_adapter_callback_latency_ms",
            description="Callback processing latency in milliseconds",
            unit="ms",
        )

        # Queue depth gauges
        self._pending_requests = meter.create_gauge(
            name="apex_adapter_pending_requests",
            description="Number of pending requests per adapter",
        )

        # Subscription gauges
        self._active_subscriptions = meter.create_gauge(
            name="apex_adapter_active_subscriptions",
            description="Number of active symbol subscriptions per adapter",
        )

        # Last activity timestamp
        self._last_activity_timestamp = meter.create_gauge(
            name="apex_adapter_last_activity_timestamp",
            description="Unix timestamp of last activity per adapter",
            unit="seconds",
        )

    def record_connection_status(
        self,
        adapter_name: str,
        broker: str,
        connected: bool,
        adapter_type: str = "unknown",
    ) -> None:
        """
        Record adapter connection status.

        Args:
            adapter_name: Unique adapter identifier.
            broker: Broker name (e.g., "ib", "futu").
            connected: Whether adapter is connected.
            adapter_type: Type of adapter ("live", "historical", "execution").
        """
        labels = {
            "adapter": adapter_name,
            "broker": broker,
            "type": adapter_type,
        }
        self._connection_status.set(1 if connected else 0, labels)

    def record_reconnect(
        self,
        adapter_name: str,
        broker: str,
        adapter_type: str = "unknown",
    ) -> None:
        """
        Record a reconnection attempt.

        Args:
            adapter_name: Unique adapter identifier.
            broker: Broker name.
            adapter_type: Type of adapter.
        """
        labels = {
            "adapter": adapter_name,
            "broker": broker,
            "type": adapter_type,
        }
        self._reconnect_total.add(1, labels)

    def record_quote_received(
        self,
        adapter_name: str,
        broker: str,
        symbol: str = "",
    ) -> None:
        """
        Record a quote tick received.

        Args:
            adapter_name: Adapter that received the quote.
            broker: Broker name.
            symbol: Symbol for the quote (optional, for per-symbol metrics).
        """
        labels = {"adapter": adapter_name, "broker": broker}
        self._quotes_received.add(1, labels)
        self._update_last_activity(adapter_name, broker)

    def record_bar_received(
        self,
        adapter_name: str,
        broker: str,
        timeframe: str = "",
    ) -> None:
        """
        Record a bar/candle received.

        Args:
            adapter_name: Adapter that received the bar.
            broker: Broker name.
            timeframe: Bar timeframe (e.g., "1m", "5m").
        """
        labels = {"adapter": adapter_name, "broker": broker}
        if timeframe:
            labels["timeframe"] = timeframe
        self._bars_received.add(1, labels)
        self._update_last_activity(adapter_name, broker)

    def record_fill_received(
        self,
        adapter_name: str,
        broker: str,
        side: str = "",
    ) -> None:
        """
        Record an order fill received.

        Args:
            adapter_name: Adapter that received the fill.
            broker: Broker name.
            side: Order side ("BUY" or "SELL").
        """
        labels = {"adapter": adapter_name, "broker": broker}
        if side:
            labels["side"] = side
        self._fills_received.add(1, labels)
        self._update_last_activity(adapter_name, broker)

    def record_order_submitted(
        self,
        adapter_name: str,
        broker: str,
        order_type: str = "",
    ) -> None:
        """
        Record an order submission.

        Args:
            adapter_name: Adapter that submitted the order.
            broker: Broker name.
            order_type: Order type ("MARKET", "LIMIT", etc.).
        """
        labels = {"adapter": adapter_name, "broker": broker}
        if order_type:
            labels["order_type"] = order_type
        self._orders_submitted.add(1, labels)

    def record_error(
        self,
        adapter_name: str,
        broker: str,
        error_type: str = "unknown",
    ) -> None:
        """
        Record an adapter error.

        Args:
            adapter_name: Adapter that had the error.
            broker: Broker name.
            error_type: Type of error (e.g., "connection", "timeout", "parse").
        """
        labels = {
            "adapter": adapter_name,
            "broker": broker,
            "error_type": error_type,
        }
        self._errors_total.add(1, labels)

    def record_request_latency(
        self,
        adapter_name: str,
        broker: str,
        latency_ms: float,
        operation: str = "",
    ) -> None:
        """
        Record request latency.

        Args:
            adapter_name: Adapter making the request.
            broker: Broker name.
            latency_ms: Latency in milliseconds.
            operation: Operation type (e.g., "subscribe", "fetch_bars").
        """
        labels = {"adapter": adapter_name, "broker": broker}
        if operation:
            labels["operation"] = operation
        self._request_latency.record(latency_ms, labels)

    def record_callback_latency(
        self,
        adapter_name: str,
        broker: str,
        latency_ms: float,
        callback_type: str = "",
    ) -> None:
        """
        Record callback processing latency.

        Args:
            adapter_name: Adapter processing the callback.
            broker: Broker name.
            latency_ms: Processing time in milliseconds.
            callback_type: Type of callback (e.g., "tick", "fill", "order").
        """
        labels = {"adapter": adapter_name, "broker": broker}
        if callback_type:
            labels["callback_type"] = callback_type
        self._callback_latency.record(latency_ms, labels)

    def record_pending_requests(
        self,
        adapter_name: str,
        broker: str,
        count: int,
    ) -> None:
        """
        Record number of pending requests.

        Args:
            adapter_name: Adapter with pending requests.
            broker: Broker name.
            count: Number of pending requests.
        """
        labels = {"adapter": adapter_name, "broker": broker}
        self._pending_requests.set(count, labels)

    def record_active_subscriptions(
        self,
        adapter_name: str,
        broker: str,
        count: int,
    ) -> None:
        """
        Record number of active subscriptions.

        Args:
            adapter_name: Adapter with subscriptions.
            broker: Broker name.
            count: Number of active symbol subscriptions.
        """
        labels = {"adapter": adapter_name, "broker": broker}
        self._active_subscriptions.set(count, labels)

    def _update_last_activity(self, adapter_name: str, broker: str) -> None:
        """Update last activity timestamp for an adapter."""
        labels = {"adapter": adapter_name, "broker": broker}
        self._last_activity_timestamp.set(time.time(), labels)

    def record_adapter_status(self, status: Any) -> None:
        """
        Record all metrics from an adapter status object.

        Args:
            status: Object with name, broker, adapter_type, connected, last_updated attributes.
        """
        labels = {
            "adapter": status.name,
            "broker": status.broker,
            "type": status.adapter_type,
        }

        self._connection_status.set(1 if status.connected else 0, labels)

        if status.last_updated:
            self._last_activity_timestamp.set(
                status.last_updated.timestamp(),
                {"adapter": status.name, "broker": status.broker},
            )


class AdapterMetricsContext:
    """
    Context manager for timing adapter operations.

    Usage:
        with AdapterMetricsContext(metrics, "ib_live", "ib", "subscribe") as ctx:
            # ... do operation ...
        # Latency automatically recorded
    """

    def __init__(
        self,
        adapter_metrics: Optional[AdapterMetrics],
        adapter_name: str,
        broker: str,
        operation: str,
    ):
        """
        Initialize timing context.

        Args:
            adapter_metrics: AdapterMetrics instance (can be None for no-op).
            adapter_name: Name of adapter being timed.
            broker: Broker name.
            operation: Operation being timed.
        """
        self._metrics = adapter_metrics
        self._adapter_name = adapter_name
        self._broker = broker
        self._operation = operation
        self._start: float = 0

    def __enter__(self) -> "AdapterMetricsContext":
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and record duration."""
        if self._metrics:
            latency_ms = (time.perf_counter() - self._start) * 1000
            self._metrics.record_request_latency(
                self._adapter_name,
                self._broker,
                latency_ms,
                self._operation,
            )


@contextmanager
def time_adapter_operation(
    adapter_metrics: Optional[AdapterMetrics],
    adapter_name: str,
    broker: str,
    operation: str,
) -> Generator[None, None, None]:
    """
    Context manager for timing adapter operations.

    Args:
        adapter_metrics: AdapterMetrics instance (can be None for no-op).
        adapter_name: Name of adapter being timed.
        broker: Broker name.
        operation: Operation being timed.

    Yields:
        None

    Example:
        with time_adapter_operation(metrics, "ib_live", "ib", "fetch_positions"):
            positions = await adapter.fetch_positions()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if adapter_metrics:
            latency_ms = (time.perf_counter() - start) * 1000
            adapter_metrics.record_request_latency(adapter_name, broker, latency_ms, operation)
