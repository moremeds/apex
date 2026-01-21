"""
Risk-specific metrics instrumentation.

Exposes key risk indicators for Prometheus alerting:
- Portfolio Greeks (delta, gamma, vega, theta)
- P&L metrics (unrealized, daily)
- Risk breach levels
- Concentration metrics
- Calculation performance

Note: OpenTelemetry is an optional dependency. When not installed,
RiskMetrics will accept a None meter and operate in no-op mode.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Optional

from ...utils.logging_setup import get_logger

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter

    from src.models.risk_snapshot import RiskSnapshot

logger = get_logger(__name__)


class _NoopInstrument:
    """No-op instrument that accepts but ignores all metric calls."""

    def add(self, value: Any, attributes: Any = None) -> None:
        pass

    def set(self, value: Any, attributes: Any = None) -> None:
        pass

    def record(self, value: Any, attributes: Any = None) -> None:
        pass


class RiskMetrics:
    """
    Instruments for risk monitoring metrics.

    Exposes portfolio-level risk indicators that can be scraped by Prometheus
    for alerting on breaches, P&L limits, and Greeks thresholds.

    All metrics are prefixed with 'apex_' for namespace isolation.

    When meter is None (OpenTelemetry not installed), operates in no-op mode.
    """

    # Type annotations for instruments (Any allows both real and noop instruments)
    _breach_level: Any
    _portfolio_delta: Any
    _portfolio_gamma: Any
    _portfolio_vega: Any
    _portfolio_theta: Any
    _unrealized_pnl: Any
    _daily_pnl: Any
    _gross_notional: Any
    _net_notional: Any
    _max_single_name_pct: Any
    _near_term_gamma_notional: Any
    _near_term_vega_notional: Any
    _margin_utilization: Any
    _buying_power: Any
    _net_liquidation: Any
    _position_count: Any
    _positions_missing_md: Any
    _positions_missing_greeks: Any
    _calc_duration: Any
    _snapshot_build_duration: Any
    _breach_counter: Any
    _last_snapshot_timestamp: Any

    def __init__(self, meter: Optional["Meter"]):
        """
        Initialize risk metrics instruments.

        Args:
            meter: OpenTelemetry Meter for creating instruments, or None for no-op mode.
        """
        self._meter = meter
        self._noop = meter is None

        if self._noop:
            # No-op mode - use NoopInstrument that accepts but ignores calls
            noop = _NoopInstrument()
            self._breach_level = noop
            self._portfolio_delta = noop
            self._portfolio_gamma = noop
            self._portfolio_vega = noop
            self._portfolio_theta = noop
            self._unrealized_pnl = noop
            self._daily_pnl = noop
            self._gross_notional = noop
            self._net_notional = noop
            self._max_single_name_pct = noop
            self._near_term_gamma_notional = noop
            self._near_term_vega_notional = noop
            self._margin_utilization = noop
            self._buying_power = noop
            self._net_liquidation = noop
            self._position_count = noop
            self._positions_missing_md = noop
            self._positions_missing_greeks = noop
            self._calc_duration = noop
            self._snapshot_build_duration = noop
            self._breach_counter = noop
            self._last_snapshot_timestamp = noop
            return

        # mypy: meter is guaranteed non-None after noop check
        assert meter is not None

        # Risk breach level gauge (0=OK, 1=soft, 2=hard)
        self._breach_level = meter.create_gauge(
            name="apex_risk_breach_level",
            description="Risk breach level: 0=OK, 1=soft, 2=hard",
            unit="level",
        )

        # Portfolio Greeks gauges
        self._portfolio_delta = meter.create_gauge(
            name="apex_portfolio_delta",
            description="Total portfolio delta (share equivalent exposure)",
        )
        self._portfolio_gamma = meter.create_gauge(
            name="apex_portfolio_gamma",
            description="Total portfolio gamma (delta change per $1 move)",
        )
        self._portfolio_vega = meter.create_gauge(
            name="apex_portfolio_vega",
            description="Total portfolio vega (P&L per 1% IV change)",
        )
        self._portfolio_theta = meter.create_gauge(
            name="apex_portfolio_theta",
            description="Total portfolio theta (daily time decay)",
        )

        # P&L gauges
        self._unrealized_pnl = meter.create_gauge(
            name="apex_unrealized_pnl",
            description="Unrealized P&L in USD",
            unit="USD",
        )
        self._daily_pnl = meter.create_gauge(
            name="apex_daily_pnl",
            description="Daily P&L in USD",
            unit="USD",
        )

        # Notional exposure gauges
        self._gross_notional = meter.create_gauge(
            name="apex_gross_notional",
            description="Total gross notional exposure",
            unit="USD",
        )
        self._net_notional = meter.create_gauge(
            name="apex_net_notional",
            description="Total net notional exposure",
            unit="USD",
        )

        # Concentration gauges
        self._max_single_name_pct = meter.create_gauge(
            name="apex_max_single_name_pct",
            description="Maximum single-name concentration percentage",
            unit="percent",
        )
        self._near_term_gamma_notional = meter.create_gauge(
            name="apex_near_term_gamma_notional",
            description="Near-term (0-7 DTE) gamma notional",
            unit="USD",
        )
        self._near_term_vega_notional = meter.create_gauge(
            name="apex_near_term_vega_notional",
            description="Near-term (0-30 DTE) vega notional",
            unit="USD",
        )

        # Account metrics gauges
        self._margin_utilization = meter.create_gauge(
            name="apex_margin_utilization",
            description="Margin utilization percentage",
            unit="percent",
        )
        self._buying_power = meter.create_gauge(
            name="apex_buying_power",
            description="Available buying power",
            unit="USD",
        )
        self._net_liquidation = meter.create_gauge(
            name="apex_net_liquidation",
            description="Net liquidation value",
            unit="USD",
        )

        # Position counts
        self._position_count = meter.create_gauge(
            name="apex_total_positions",
            description="Total number of positions",
        )
        self._positions_missing_md = meter.create_gauge(
            name="apex_positions_missing_md",
            description="Positions with missing market data",
        )
        self._positions_missing_greeks = meter.create_gauge(
            name="apex_positions_missing_greeks",
            description="Positions with missing Greeks",
        )

        # Performance histogram
        self._calc_duration = meter.create_histogram(
            name="apex_risk_calc_duration_ms",
            description="Risk calculation duration in milliseconds",
            unit="ms",
        )
        self._snapshot_build_duration = meter.create_histogram(
            name="apex_snapshot_build_duration_ms",
            description="Snapshot build duration in milliseconds",
            unit="ms",
        )

        # Breach counters by rule
        self._breach_counter = meter.create_counter(
            name="apex_risk_breach_total",
            description="Total risk breaches by rule and level",
        )

        # Snapshot timestamp (when data was last updated)
        self._last_snapshot_timestamp = meter.create_gauge(
            name="apex_last_snapshot_timestamp",
            description="Unix timestamp of last snapshot build",
            unit="seconds",
        )

    def record_snapshot(self, snapshot: "RiskSnapshot") -> None:
        """
        Record metrics from a RiskSnapshot.

        Args:
            snapshot: Risk snapshot containing aggregated metrics.
        """
        if self._noop:
            return

        # Portfolio Greeks
        self._portfolio_delta.set(snapshot.portfolio_delta or 0)
        self._portfolio_gamma.set(snapshot.portfolio_gamma or 0)
        self._portfolio_vega.set(snapshot.portfolio_vega or 0)
        self._portfolio_theta.set(snapshot.portfolio_theta or 0)

        # P&L
        self._unrealized_pnl.set(snapshot.total_unrealized_pnl or 0)
        self._daily_pnl.set(snapshot.total_daily_pnl or 0)

        # Notional
        self._gross_notional.set(snapshot.total_gross_notional or 0)
        self._net_notional.set(snapshot.total_net_notional or 0)

        # Concentration
        self._max_single_name_pct.set((snapshot.concentration_pct or 0) * 100)
        self._near_term_gamma_notional.set(snapshot.gamma_notional_near_term or 0)
        self._near_term_vega_notional.set(snapshot.vega_notional_near_term or 0)

        # Account
        self._margin_utilization.set((snapshot.margin_utilization or 0) * 100)
        self._buying_power.set(snapshot.buying_power or 0)
        self._net_liquidation.set(snapshot.total_net_liquidation or 0)

        # Position counts
        self._position_count.set(snapshot.total_positions or 0)
        self._positions_missing_md.set(snapshot.positions_with_missing_md or 0)
        self._positions_missing_greeks.set(snapshot.missing_greeks_count or 0)

        # Snapshot timestamp
        if snapshot.timestamp:
            self._last_snapshot_timestamp.set(snapshot.timestamp.timestamp())
        else:
            self._last_snapshot_timestamp.set(time.time())

        logger.debug(
            f"Recorded snapshot metrics: delta={snapshot.portfolio_delta:.2f}, "
            f"pnl={snapshot.total_unrealized_pnl:.2f}"
        )

    def record_breach(self, rule_name: str, level: int) -> None:
        """
        Record a risk breach event.

        Args:
            rule_name: Name of the breached rule (e.g., "delta_limit", "concentration").
            level: Breach level (0=OK, 1=soft breach, 2=hard breach).
        """
        if self._noop:
            return
        self._breach_level.set(level, {"rule": rule_name})

        # Also increment breach counter for historical tracking
        if level > 0:
            breach_type = "soft" if level == 1 else "hard"
            self._breach_counter.add(1, {"rule": rule_name, "level": breach_type})

    def record_calc_duration(self, duration_ms: float) -> None:
        """
        Record risk calculation duration.

        Args:
            duration_ms: Calculation duration in milliseconds.
        """
        if self._noop:
            return
        self._calc_duration.record(duration_ms)

    def record_snapshot_build_duration(self, duration_ms: float) -> None:
        """
        Record snapshot build duration.

        Args:
            duration_ms: Build duration in milliseconds.
        """
        if self._noop:
            return
        self._snapshot_build_duration.record(duration_ms)

    def clear_breaches(self) -> None:
        """Clear all breach level gauges (set to 0)."""
        if self._noop:
            return
        self._breach_level.set(0, {"rule": "all"})


class RiskMetricsContext:
    """
    Context manager for timing risk calculations.

    Usage:
        with RiskMetricsContext(risk_metrics) as ctx:
            # ... do calculation ...
        # Duration automatically recorded
    """

    def __init__(self, risk_metrics: Optional[RiskMetrics]):
        """
        Initialize timing context.

        Args:
            risk_metrics: RiskMetrics instance (can be None for no-op).
        """
        self._metrics = risk_metrics
        self._start: float = 0

    def __enter__(self) -> "RiskMetricsContext":
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and record duration."""
        if self._metrics:
            duration_ms = (time.perf_counter() - self._start) * 1000
            self._metrics.record_snapshot_build_duration(duration_ms)


@contextmanager
def time_risk_calculation(risk_metrics: Optional[RiskMetrics]) -> Generator[None, None, None]:
    """
    Context manager for timing risk calculations.

    Functional alternative to RiskMetricsContext class.

    Args:
        risk_metrics: RiskMetrics instance (can be None for no-op).

    Yields:
        None

    Example:
        with time_risk_calculation(metrics):
            snapshot = engine.build_snapshot(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if risk_metrics:
            duration_ms = (time.perf_counter() - start) * 1000
            risk_metrics.record_snapshot_build_duration(duration_ms)
