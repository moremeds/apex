"""Risk snapshot model for aggregated portfolio metrics."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, TYPE_CHECKING

from ..utils.timezone import now_utc

if TYPE_CHECKING:
    from .position_risk import PositionRisk


@dataclass
class RiskSnapshot:
    """
    Aggregated portfolio risk snapshot.

    Output of RiskEngine.build_snapshot() - contains portfolio-level metrics,
    Greeks, concentration, and expiry bucket analysis.
    """

    timestamp: datetime = field(default_factory=now_utc)

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

    # Account metrics (aggregated)
    margin_utilization: float = 0.0
    buying_power: float = 0.0

    # Per-broker account info for separate display
    ib_net_liquidation: float = 0.0
    ib_buying_power: float = 0.0
    futu_net_liquidation: float = 0.0
    futu_buying_power: float = 0.0
    total_net_liquidation: float = 0.0

    # NOTE: breached_limits field removed - replaced by RiskSignal model.
    # Risk signals are now published via RISK_SIGNAL events and stored separately.

    # Data quality
    positions_with_missing_md: int = 0
    total_positions: int = 0
    missing_greeks_count: int = 0

    # Preview mode indicator (positions shown before full risk calculation)
    is_preview: bool = False

    # Per-position risk breakdown (calculated by RiskEngine)
    # This is the single source of truth for all position-level calculations
    position_risks: List["PositionRisk"] = field(default_factory=list)
