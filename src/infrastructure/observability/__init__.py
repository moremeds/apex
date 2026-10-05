"""
Observability module for Apex risk management system.

Provides OpenTelemetry instrumentation with Prometheus export for:
- Risk metrics (Greeks, P&L, breaches)
- System health metrics (connections, coverage, queues)
- Adapter metrics (connections, throughput, latency)
- Signal pipeline metrics (bars, indicators, signals, confluence)
- Performance metrics (latencies, durations)

Note: OpenTelemetry is an optional dependency. When not installed, all metric
classes operate in no-op mode, accepting calls but doing nothing. This allows
the rest of the system to function without observability support.
"""

from .signal_metrics import (
    SignalMetrics,
    time_alignment_calculation,
    time_confluence_calculation,
    time_indicator_computation,
    time_rule_evaluation,
)

__all__ = [
    "SignalMetrics",
    "time_confluence_calculation",
    "time_alignment_calculation",
    "time_indicator_computation",
    "time_rule_evaluation",
]
