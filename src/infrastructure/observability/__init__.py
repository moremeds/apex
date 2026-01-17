"""
Observability module for Apex risk management system.

Provides OpenTelemetry instrumentation with Prometheus export for:
- Risk metrics (Greeks, P&L, breaches)
- System health metrics (connections, coverage, queues)
- Adapter metrics (connections, throughput, latency)
- Signal pipeline metrics (bars, indicators, signals, confluence)
- Performance metrics (latencies, durations)
"""

from .adapter_metrics import AdapterMetrics, AdapterMetricsContext, time_adapter_operation
from .health_metrics import HealthMetrics
from .metrics import MetricsManager, get_metrics_manager
from .risk_metrics import RiskMetrics, RiskMetricsContext
from .signal_metrics import (
    SignalMetrics,
    time_alignment_calculation,
    time_confluence_calculation,
    time_indicator_computation,
    time_rule_evaluation,
)

__all__ = [
    "MetricsManager",
    "get_metrics_manager",
    "RiskMetrics",
    "RiskMetricsContext",
    "HealthMetrics",
    "AdapterMetrics",
    "AdapterMetricsContext",
    "time_adapter_operation",
    "SignalMetrics",
    "time_confluence_calculation",
    "time_alignment_calculation",
    "time_indicator_computation",
    "time_rule_evaluation",
]
