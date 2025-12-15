"""
Observability module for Apex risk management system.

Provides OpenTelemetry instrumentation with Prometheus export for:
- Risk metrics (Greeks, P&L, breaches)
- System health metrics (connections, coverage, queues)
- Adapter metrics (connections, throughput, latency)
- Performance metrics (latencies, durations)
"""

from .metrics import MetricsManager, get_metrics_manager
from .risk_metrics import RiskMetrics, RiskMetricsContext
from .health_metrics import HealthMetrics
from .adapter_metrics import AdapterMetrics, AdapterMetricsContext, time_adapter_operation

__all__ = [
    "MetricsManager",
    "get_metrics_manager",
    "RiskMetrics",
    "RiskMetricsContext",
    "HealthMetrics",
    "AdapterMetrics",
    "AdapterMetricsContext",
    "time_adapter_operation",
]
