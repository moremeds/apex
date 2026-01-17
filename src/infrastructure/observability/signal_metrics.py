"""
Signal pipeline metrics instrumentation.

Exposes key signal pipeline metrics for Prometheus monitoring:
- Bar aggregation throughput and latency
- Indicator computation performance
- Rule evaluation and signal emission rates
- Confluence calculation metrics
- Error tracking by module

All metrics are prefixed with 'apex_signal_' for namespace isolation.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional

from opentelemetry import metrics

from ...utils.logging_setup import get_logger

logger = get_logger(__name__)


class SignalMetrics:
    """
    Metrics for the signal pipeline (bar aggregation → indicators → signals).

    Provides counters, gauges, and histograms for monitoring:
    - Throughput: bars/indicators/signals per second
    - Latency: processing time at each pipeline stage
    - Resource utilization: cache sizes, queue depths
    - Errors: failures by module and operation
    """

    def __init__(self, meter: metrics.Meter) -> None:
        """
        Initialize signal pipeline metrics.

        Args:
            meter: OpenTelemetry Meter for creating instruments.
        """
        self._meter = meter

        # -------------------------------------------------------------------------
        # Counters (monotonically increasing)
        # -------------------------------------------------------------------------

        self._bars_emitted = meter.create_counter(
            name="apex_signal_bars_emitted_total",
            description="Total bars emitted by aggregators",
        )

        self._indicators_computed = meter.create_counter(
            name="apex_signal_indicators_computed_total",
            description="Total indicator computations",
        )

        self._rules_evaluated = meter.create_counter(
            name="apex_signal_rules_evaluated_total",
            description="Total rule evaluations",
        )

        self._signals_emitted = meter.create_counter(
            name="apex_signal_signals_emitted_total",
            description="Total trading signals emitted",
        )

        self._signals_blocked = meter.create_counter(
            name="apex_signal_signals_blocked_total",
            description="Total signals blocked (cooldown, filter, etc.)",
        )

        self._confluence_calcs = meter.create_counter(
            name="apex_signal_confluence_calculations_total",
            description="Total confluence score calculations",
        )

        self._alignment_calcs = meter.create_counter(
            name="apex_signal_alignment_calculations_total",
            description="Total MTF alignment calculations",
        )

        self._errors = meter.create_counter(
            name="apex_signal_errors_total",
            description="Total errors in signal pipeline by module",
        )

        # -------------------------------------------------------------------------
        # Gauges (current values)
        # -------------------------------------------------------------------------

        self._indicator_state_cache_size = meter.create_gauge(
            name="apex_signal_indicator_state_cache_size",
            description="Number of entries in indicator state cache",
        )

        self._cooldown_entries = meter.create_gauge(
            name="apex_signal_cooldown_entries",
            description="Number of active signal cooldown entries",
        )

        self._bar_builders_active = meter.create_gauge(
            name="apex_signal_bar_builders_active",
            description="Number of active bar builders",
        )

        # -------------------------------------------------------------------------
        # Histograms (latency distributions)
        # -------------------------------------------------------------------------

        self._bar_aggregation_ms = meter.create_histogram(
            name="apex_signal_bar_aggregation_ms",
            description="Bar close processing latency in milliseconds",
            unit="ms",
        )

        self._indicator_compute_ms = meter.create_histogram(
            name="apex_signal_indicator_compute_ms",
            description="Per-indicator computation latency in milliseconds",
            unit="ms",
        )

        self._rule_evaluation_ms = meter.create_histogram(
            name="apex_signal_rule_evaluation_ms",
            description="Rule evaluation latency in milliseconds",
            unit="ms",
        )

        self._confluence_calc_ms = meter.create_histogram(
            name="apex_signal_confluence_calc_ms",
            description="Confluence calculation latency in milliseconds",
            unit="ms",
        )

        self._alignment_calc_ms = meter.create_histogram(
            name="apex_signal_alignment_calc_ms",
            description="MTF alignment calculation latency in milliseconds",
            unit="ms",
        )

    # -------------------------------------------------------------------------
    # Counter Recording Methods
    # -------------------------------------------------------------------------

    def record_bar_emitted(self, timeframe: str) -> None:
        """Record a bar emission event."""
        self._bars_emitted.add(1, {"timeframe": timeframe})

    def record_indicator_computed(self, indicator: str) -> None:
        """Record an indicator computation."""
        self._indicators_computed.add(1, {"indicator": indicator})

    def record_rule_evaluated(self, rule_id: str) -> None:
        """Record a rule evaluation."""
        self._rules_evaluated.add(1, {"rule_id": rule_id})

    def record_signal_emitted(self, rule_id: str, direction: str) -> None:
        """Record a trading signal emission."""
        self._signals_emitted.add(1, {"rule_id": rule_id, "direction": direction})

    def record_signal_blocked(self, rule_id: str, reason: str) -> None:
        """Record a signal being blocked (cooldown, filter, etc.)."""
        self._signals_blocked.add(1, {"rule_id": rule_id, "reason": reason})

    def record_confluence_calculated(self) -> None:
        """Record a confluence calculation."""
        self._confluence_calcs.add(1)

    def record_alignment_calculated(self) -> None:
        """Record an MTF alignment calculation."""
        self._alignment_calcs.add(1)

    def record_error(self, module: str, operation: str) -> None:
        """Record an error in the signal pipeline."""
        self._errors.add(1, {"module": module, "operation": operation})

    # -------------------------------------------------------------------------
    # Gauge Recording Methods
    # -------------------------------------------------------------------------

    def set_indicator_state_cache_size(self, size: int) -> None:
        """Set the current indicator state cache size."""
        self._indicator_state_cache_size.set(size)

    def set_cooldown_entries(self, count: int) -> None:
        """Set the current number of cooldown entries."""
        self._cooldown_entries.set(count)

    def set_bar_builders_active(self, count: int, timeframe: str = "") -> None:
        """Set the number of active bar builders."""
        attrs = {"timeframe": timeframe} if timeframe else {}
        self._bar_builders_active.set(count, attrs)

    # -------------------------------------------------------------------------
    # Histogram Recording Methods
    # -------------------------------------------------------------------------

    def record_bar_aggregation_latency(self, duration_ms: float, timeframe: str) -> None:
        """Record bar aggregation latency."""
        self._bar_aggregation_ms.record(duration_ms, {"timeframe": timeframe})

    def record_indicator_compute_latency(self, duration_ms: float, indicator: str) -> None:
        """Record indicator computation latency."""
        self._indicator_compute_ms.record(duration_ms, {"indicator": indicator})

    def record_rule_evaluation_latency(self, duration_ms: float) -> None:
        """Record rule evaluation latency."""
        self._rule_evaluation_ms.record(duration_ms)

    def record_confluence_latency(self, duration_ms: float) -> None:
        """Record confluence calculation latency."""
        self._confluence_calc_ms.record(duration_ms)

    def record_alignment_latency(self, duration_ms: float) -> None:
        """Record MTF alignment calculation latency."""
        self._alignment_calc_ms.record(duration_ms)


# -----------------------------------------------------------------------------
# Context Managers for Timing
# -----------------------------------------------------------------------------


@contextmanager
def time_confluence_calculation(
    metrics: Optional[SignalMetrics],
) -> Generator[None, None, None]:
    """
    Context manager for timing confluence calculations.

    Usage:
        with time_confluence_calculation(signal_metrics):
            score = analyzer.analyze(symbol, timeframe, states)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if metrics:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_confluence_latency(duration_ms)
            metrics.record_confluence_calculated()


@contextmanager
def time_alignment_calculation(
    metrics: Optional[SignalMetrics],
) -> Generator[None, None, None]:
    """
    Context manager for timing MTF alignment calculations.

    Usage:
        with time_alignment_calculation(signal_metrics):
            alignment = analyzer.analyze(symbol, timeframes, states_by_tf)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if metrics:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_alignment_latency(duration_ms)
            metrics.record_alignment_calculated()


@contextmanager
def time_indicator_computation(
    metrics: Optional[SignalMetrics],
    indicator: str,
) -> Generator[None, None, None]:
    """
    Context manager for timing indicator computations.

    Usage:
        with time_indicator_computation(signal_metrics, "rsi"):
            state = indicator.get_state(current, previous)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if metrics:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_indicator_compute_latency(duration_ms, indicator)
            metrics.record_indicator_computed(indicator)


@contextmanager
def time_rule_evaluation(
    metrics: Optional[SignalMetrics],
) -> Generator[None, None, None]:
    """
    Context manager for timing rule evaluations.

    Usage:
        with time_rule_evaluation(signal_metrics):
            triggered = rule.check_condition(event)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if metrics:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_rule_evaluation_latency(duration_ms)
