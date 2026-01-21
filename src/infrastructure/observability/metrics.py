"""
OpenTelemetry metrics with Prometheus exporter.

This module provides a MetricsManager class for initializing and managing
OpenTelemetry metrics with Prometheus export capability. Metrics are exposed
on a configurable HTTP port for Prometheus scraping.

Usage:
    metrics_mgr = MetricsManager(port=8000)
    metrics_mgr.start()

    meter = metrics_mgr.get_meter("apex.risk")
    counter = meter.create_counter("requests")

Note: OpenTelemetry is an optional dependency. When not installed, this module
provides no-op implementations that allow the rest of the system to function.
"""

from __future__ import annotations

from typing import Any, Optional

from ...utils.logging_setup import get_logger

logger = get_logger(__name__)

# Lazy import of opentelemetry - it's an optional dependency
try:
    from opentelemetry import metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from prometheus_client import start_http_server

    HAS_OPENTELEMETRY = True
except ImportError:
    metrics = None  # type: ignore[assignment]
    PrometheusMetricReader = None  # type: ignore[assignment, misc]
    MeterProvider = None  # type: ignore[assignment, misc]
    start_http_server = None  # type: ignore[assignment]
    HAS_OPENTELEMETRY = False
    logger.debug("OpenTelemetry not installed - metrics will be disabled")


class _NoopMeter:
    """No-op meter that returns no-op instruments when opentelemetry is unavailable."""

    def create_counter(self, name: str, **kwargs: Any) -> "_NoopInstrument":
        return _NoopInstrument()

    def create_gauge(self, name: str, **kwargs: Any) -> "_NoopInstrument":
        return _NoopInstrument()

    def create_histogram(self, name: str, **kwargs: Any) -> "_NoopInstrument":
        return _NoopInstrument()

    def create_observable_gauge(self, name: str, **kwargs: Any) -> "_NoopInstrument":
        return _NoopInstrument()


class _NoopInstrument:
    """No-op instrument that accepts but ignores all metric calls."""

    def add(self, value: Any, attributes: Any = None) -> None:
        pass

    def set(self, value: Any, attributes: Any = None) -> None:
        pass

    def record(self, value: Any, attributes: Any = None) -> None:
        pass


class MetricsManager:
    """
    Manages OpenTelemetry metrics with Prometheus export.

    Provides centralized metrics initialization and access. Exposes
    a /metrics endpoint on the specified port for Prometheus to scrape.

    Thread-safe: Can be accessed from multiple threads after start().

    When OpenTelemetry is not installed, this class operates in no-op mode,
    returning no-op meters that accept but ignore all metric operations.
    """

    def __init__(self, port: int = 8000, service_name: str = "apex"):
        """
        Initialize metrics manager.

        Args:
            port: HTTP port for /metrics endpoint (default: 8000).
            service_name: Service name for metric labeling.
        """
        self._port = port
        self._service_name = service_name
        self._provider: Any = None
        self._started = False
        self._noop = not HAS_OPENTELEMETRY

    def start(self) -> None:
        """
        Initialize metrics and start /metrics HTTP endpoint.

        Idempotent: Safe to call multiple times.
        When OpenTelemetry is not installed, this is a no-op.
        """
        if self._started:
            logger.debug("MetricsManager already started")
            return

        if self._noop:
            logger.info("MetricsManager in no-op mode (OpenTelemetry not installed)")
            self._started = True
            return

        try:
            # Create Prometheus metric reader
            reader = PrometheusMetricReader()

            # Create meter provider with Prometheus exporter
            self._provider = MeterProvider(metric_readers=[reader])
            metrics.set_meter_provider(self._provider)

            # Start HTTP server for /metrics endpoint
            start_http_server(self._port)
            self._started = True

            logger.info(f"Metrics server started on port {self._port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

    def get_meter(self, name: str) -> Any:
        """
        Get a meter for creating metric instruments.

        Args:
            name: Meter name (e.g., "apex.risk", "apex.health").

        Returns:
            OpenTelemetry Meter instance, or NoopMeter if OTel not installed.
        """
        if self._noop:
            return _NoopMeter()
        return metrics.get_meter(name)

    def shutdown(self) -> None:
        """Gracefully shutdown the metrics provider."""
        if self._provider:
            self._provider.shutdown()
            logger.info("MetricsManager shutdown complete")

    @property
    def is_started(self) -> bool:
        """Check if metrics server has been started."""
        return self._started

    @property
    def port(self) -> int:
        """Get the metrics server port."""
        return self._port

    @property
    def is_noop(self) -> bool:
        """Check if metrics manager is in no-op mode."""
        return self._noop


# Global singleton for easy access
_metrics_manager: Optional[MetricsManager] = None


def get_metrics_manager(port: int = 8000) -> MetricsManager:
    """
    Get or create the global metrics manager singleton.

    Args:
        port: HTTP port for /metrics endpoint (only used on first call).

    Returns:
        Shared MetricsManager instance.
    """
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager(port=port)
    return _metrics_manager


def reset_metrics_manager() -> None:
    """
    Reset the global metrics manager (for testing only).

    Warning: This does not shutdown the HTTP server, which cannot be
    restarted on the same port within the same process.
    """
    global _metrics_manager
    if _metrics_manager:
        _metrics_manager.shutdown()
    _metrics_manager = None
