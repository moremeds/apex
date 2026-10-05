"""
Observability for the signal pipeline.

`SignalMetrics` records OpenTelemetry metrics for bars, indicators, signals and confluence
when a meter is passed in; no exporter ships with apex.

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
