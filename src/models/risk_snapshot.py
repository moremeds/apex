"""Risk snapshot model for aggregated portfolio metrics."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class PositionRisk:
    """
    Per-position risk view calculated by the RiskEngine.

    Contains pricing, P&L, Greeks, and grouping metadata needed by the
    presentation layer without re-computation.
    """

    symbol: str
    underlying: str
    expiry: Optional[str]
    strike: Optional[float]
    right: Optional[str]
    quantity: float
    avg_price: float
    mark: Optional[float]
    market_value: float
    notional: float
    unrealized_pnl: float
    daily_pnl: float
    delta_contribution: float
    gamma_contribution: float
    vega_contribution: float
    theta_contribution: float
    delta_dollars: float
    expiry_bucket: str
    days_to_expiry: int | None
    gamma_notional_near_term: float
    vega_notional_near_term: float
    has_missing_md: bool
    has_missing_greeks: bool


@dataclass
class RiskSnapshot:
    """
    Aggregated portfolio risk snapshot.

    Output of RiskEngine.build_snapshot() - contains portfolio-level metrics,
    Greeks, concentration, and expiry bucket analysis.
    """

    timestamp: datetime = field(default_factory=datetime.now)

    # Portfolio P&L
    total_unrealized_pnl: float = 0.0
    total_daily_pnl: float = 0.0

    # Portfolio Greeks
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0

    # Notional exposure
    total_gross_notional: float = 0.0
    total_net_notional: float = 0.0

    # Concentration metrics
    max_underlying_notional: float = 0.0
    max_underlying_symbol: str = ""
    concentration_pct: float = 0.0

    # Greeks by underlying
    delta_by_underlying: Dict[str, float] = field(default_factory=dict)
    notional_by_underlying: Dict[str, float] = field(default_factory=dict)

    # Expiry bucket analysis
    expiry_buckets: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Greeks concentration (near-term)
    gamma_notional_near_term: float = 0.0  # 0-7 DTE
    vega_notional_near_term: float = 0.0   # 0-30 DTE

    # Account metrics
    margin_utilization: float = 0.0
    buying_power: float = 0.0

    # Risk limit breaches
    breached_limits: List[str] = field(default_factory=list)

    # Per-position metrics for presentation (single source of truth)
    positions: List[PositionRisk] = field(default_factory=list)

    # Data quality
    positions_with_missing_md: int = 0
    total_positions: int = 0
    missing_greeks_count: int = 0
