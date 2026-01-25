"""
Hierarchical Regime Service Models.

Defines models for 3-level regime hierarchy and trading action resolution:
- HierarchicalRegime: Combined market/sector/stock regime state
- TradingAction: GO, GO_SMALL, NO_GO, HARD_NO
- ActionContext: Specific recommendations (DTE, delta, type)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.domain.signals.indicators.regime import MarketRegime


class TradingAction(Enum):
    """
    Trading action based on hierarchical regime synthesis.

    Severity ordering (highest to lowest restriction):
    - HARD_NO (4): No trading allowed, defensive only
    - NO_GO (3): No new positions
    - GO_SMALL (2): Reduced size, defined-risk only
    - GO (1): Full trading allowed
    """

    GO = 1
    GO_SMALL = 2
    NO_GO = 3
    HARD_NO = 4

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            1: "Go",
            2: "Go Small",
            3: "No Go",
            4: "Hard No",
        }
        return names.get(self.value, "Unknown")

    @property
    def allows_new_positions(self) -> bool:
        """Whether this action allows opening new positions."""
        return self.value <= 2  # GO or GO_SMALL

    @property
    def requires_defined_risk(self) -> bool:
        """Whether positions must be defined-risk (spreads)."""
        return self.value == 2  # GO_SMALL


class AccountType(Enum):
    """
    Account type for action resolution.

    Different accounts have different risk profiles and regime mappings.
    """

    SHORT_PUT = "A"  # Account A: Short put strategies
    SWING = "B"  # Account B: Swing/position trading


# Action mapping per account type
# These map from MarketRegime to TradingAction
ACTION_MAP_SHORT_PUT = {
    MarketRegime.R0_HEALTHY_UPTREND: TradingAction.GO,
    MarketRegime.R1_CHOPPY_EXTENDED: TradingAction.NO_GO,
    MarketRegime.R2_RISK_OFF: TradingAction.HARD_NO,
    MarketRegime.R3_REBOUND_WINDOW: TradingAction.GO_SMALL,  # Allowed with defined-risk
}

ACTION_MAP_SWING = {
    MarketRegime.R0_HEALTHY_UPTREND: TradingAction.GO,
    MarketRegime.R1_CHOPPY_EXTENDED: TradingAction.NO_GO,
    MarketRegime.R2_RISK_OFF: TradingAction.HARD_NO,
    MarketRegime.R3_REBOUND_WINDOW: TradingAction.NO_GO,  # Higher vol = riskier for swing
}

ACTION_MAPS = {
    AccountType.SHORT_PUT: ACTION_MAP_SHORT_PUT,
    AccountType.SWING: ACTION_MAP_SWING,
}


@dataclass
class ActionContext:
    """
    Context-specific action recommendations.

    Provides concrete guidance based on regime and account type.
    """

    # Position parameters
    dte_min: Optional[int] = None  # Minimum days to expiration
    dte_max: Optional[int] = None  # Maximum days to expiration
    delta_min: Optional[float] = None  # Minimum delta
    delta_max: Optional[float] = None  # Maximum delta
    position_type: str = "any"  # "put", "put_spread", "call_spread", "any"
    size_factor: float = 1.0  # Position size multiplier (0.0 to 1.0)

    # Risk parameters
    max_loss_per_trade: Optional[float] = None
    require_stop: bool = False
    require_profit_target: bool = False

    # Descriptive
    rationale: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "dte_min": self.dte_min,
            "dte_max": self.dte_max,
            "delta_min": self.delta_min,
            "delta_max": self.delta_max,
            "position_type": self.position_type,
            "size_factor": self.size_factor,
            "max_loss_per_trade": self.max_loss_per_trade,
            "require_stop": self.require_stop,
            "require_profit_target": self.require_profit_target,
            "rationale": self.rationale,
            "warnings": self.warnings,
        }


# Decision table: (market_regime, sector_regime, stock_regime) -> ActionContext
# For Account A (Short Put)
#
# The lookup function tries progressively less specific keys:
# 1. (market, sector, stock) - most specific
# 2. (market, sector, None)
# 3. (market, None, stock)
# 4. (market, None, None) - fallback baseline per market regime
#
# CRITICAL: Each market regime MUST have a (market, None, None) baseline fallback
DECISION_TABLE_SHORT_PUT: Dict[Tuple[str, Optional[str], Optional[str]], ActionContext] = {
    # ========== R0 Market (Healthy Uptrend) ==========
    # Full specificity: market + sector + stock
    ("R0", "R0", "R0"): ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.15,
        delta_max=0.25,
        position_type="put_or_spread",
        size_factor=1.0,
        rationale="Healthy conditions across all levels - full size allowed",
    ),
    ("R0", "R0", "R1"): ActionContext(
        dte_min=45,
        dte_max=60,
        delta_min=0.10,
        delta_max=0.15,
        position_type="put_spread",
        size_factor=0.5,
        rationale="Stock choppy but sector/market healthy - reduced size, spread only",
    ),
    ("R0", "R0", "R2"): ActionContext(
        size_factor=0.0,
        rationale="Stock in risk-off despite healthy market/sector - avoid",
        warnings=["Stock-specific weakness - wait for recovery"],
    ),
    ("R0", "R0", "R3"): ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.10,
        delta_max=0.15,
        position_type="put_spread",
        size_factor=0.4,
        rationale="Stock rebounding with healthy market/sector support",
    ),
    # Partial specificity: market + sector
    ("R0", "R1", None): ActionContext(
        size_factor=0.0,
        rationale="Sector choppy - wait for better conditions",
    ),
    ("R0", "R2", None): ActionContext(
        size_factor=0.0,
        rationale="Sector in risk-off despite healthy market - avoid sector",
        warnings=["Sector-specific weakness"],
    ),
    ("R0", "R3", None): ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.10,
        delta_max=0.15,
        position_type="put_spread",
        size_factor=0.3,
        rationale="Sector rebounding with healthy market - small positions",
    ),
    # Partial specificity: market + stock
    ("R0", None, "R0"): ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.15,
        delta_max=0.20,
        position_type="put_or_spread",
        size_factor=0.8,
        rationale="Healthy market and stock - near full size",
    ),
    ("R0", None, "R1"): ActionContext(
        dte_min=45,
        dte_max=60,
        delta_min=0.10,
        delta_max=0.15,
        position_type="put_spread",
        size_factor=0.4,
        rationale="Stock choppy despite healthy market - conservative",
    ),
    ("R0", None, "R2"): ActionContext(
        size_factor=0.0,
        rationale="Stock in risk-off - avoid despite healthy market",
    ),
    ("R0", None, "R3"): ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.10,
        delta_max=0.15,
        position_type="put_spread",
        size_factor=0.3,
        rationale="Stock rebounding with market support",
    ),
    # FALLBACK: market only (required baseline)
    ("R0", None, None): ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.12,
        delta_max=0.20,
        position_type="put_or_spread",
        size_factor=0.7,
        rationale="Healthy market - moderate size when sector/stock unknown",
    ),
    # ========== R1 Market (Choppy/Extended) ==========
    # FALLBACK: market only (required baseline)
    ("R1", None, None): ActionContext(
        size_factor=0.0,
        rationale="Market choppy - reduce frequency, no new positions",
    ),
    # ========== R2 Market (Risk-Off) ==========
    # FALLBACK: market only (required baseline)
    ("R2", None, None): ActionContext(
        size_factor=0.0,
        rationale="Risk-off - no new positions, defensive only",
        warnings=["Consider closing existing short puts", "Monitor for protective actions"],
    ),
    # ========== R3 Market (Rebound Window) ==========
    # Full specificity: market + stock
    ("R3", None, "R3"): ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.15,
        delta_max=0.20,
        position_type="put_spread",
        size_factor=0.3,
        rationale="Rebound window confirmed - small defined-risk positions only",
        require_stop=True,
    ),
    ("R3", None, "R0"): ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.12,
        delta_max=0.18,
        position_type="put_spread",
        size_factor=0.25,
        rationale="Stock healthy in rebound window - very cautious",
        require_stop=True,
    ),
    ("R3", None, "R1"): ActionContext(
        size_factor=0.0,
        rationale="Stock choppy during market rebound - wait",
    ),
    ("R3", None, "R2"): ActionContext(
        size_factor=0.0,
        rationale="Stock still in risk-off during market rebound - avoid",
        warnings=["Stock lagging market recovery"],
    ),
    # FALLBACK: market only (required baseline)
    ("R3", None, None): ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.12,
        delta_max=0.18,
        position_type="put_spread",
        size_factor=0.2,
        rationale="Rebound window - very small defined-risk positions",
        require_stop=True,
        warnings=["Market still volatile - extra caution required"],
    ),
}

# Default contexts by action
DEFAULT_CONTEXTS = {
    TradingAction.GO: ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.15,
        delta_max=0.25,
        position_type="put_or_spread",
        size_factor=1.0,
        rationale="Full trading allowed",
    ),
    TradingAction.GO_SMALL: ActionContext(
        dte_min=30,
        dte_max=45,
        delta_min=0.10,
        delta_max=0.20,
        position_type="put_spread",
        size_factor=0.3,
        rationale="Reduced size, defined-risk only",
        require_stop=True,
    ),
    TradingAction.NO_GO: ActionContext(
        size_factor=0.0,
        rationale="No new positions - wait for better conditions",
    ),
    TradingAction.HARD_NO: ActionContext(
        size_factor=0.0,
        rationale="Trading suspended - defensive mode only",
        warnings=["Consider reducing exposure", "Monitor for protective actions"],
    ),
}


@dataclass
class HierarchicalRegime:
    """
    Combined regime state across all hierarchy levels.

    Synthesizes market, sector, and stock regimes into an actionable
    trading decision with context-specific recommendations.
    """

    symbol: str
    timestamp: datetime

    # Level 1: Market
    market_regime: MarketRegime
    market_confidence: int
    market_symbol: str  # "QQQ" or "SPY" (or resolved from both)

    # Level 2: Sector
    sector_regime: Optional[MarketRegime] = None
    sector_confidence: Optional[int] = None
    sector_symbol: Optional[str] = None  # "SMH", "XLV", etc.

    # Level 3: Stock
    stock_regime: Optional[MarketRegime] = None
    stock_confidence: Optional[int] = None

    # Synthesized action
    action: TradingAction = TradingAction.NO_GO
    action_context: ActionContext = field(default_factory=ActionContext)

    # Account context
    account_type: AccountType = AccountType.SHORT_PUT

    # Multi-timeframe
    weekly_veto_active: bool = False
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "market": {
                "regime": self.market_regime.value,
                "regime_name": self.market_regime.display_name,
                "confidence": self.market_confidence,
                "symbol": self.market_symbol,
            },
            "sector": (
                {
                    "regime": self.sector_regime.value if self.sector_regime else None,
                    "regime_name": self.sector_regime.display_name if self.sector_regime else None,
                    "confidence": self.sector_confidence,
                    "symbol": self.sector_symbol,
                }
                if self.sector_regime
                else None
            ),
            "stock": (
                {
                    "regime": self.stock_regime.value if self.stock_regime else None,
                    "regime_name": self.stock_regime.display_name if self.stock_regime else None,
                    "confidence": self.stock_confidence,
                }
                if self.stock_regime
                else None
            ),
            "action": {
                "action": self.action.display_name,
                "allows_new_positions": self.action.allows_new_positions,
                "requires_defined_risk": self.action.requires_defined_risk,
                "context": self.action_context.to_dict(),
            },
            "account_type": self.account_type.value,
            "weekly_veto_active": self.weekly_veto_active,
            "alerts": self.alerts,
        }


# =============================================================================
# Derived Mappings from Universe YAML (Single Source of Truth)
# =============================================================================
#
# These mappings are derived from config/universe.yaml
# via the universe_loader module. This eliminates manual synchronization.
#
# To add a new stock or sector:
# 1. Edit config/universe.yaml
# 2. Run tests to verify consistency: pytest tests/unit/test_universe_consistency.py
#
# The loader handles:
# - STOCK_TO_SECTOR: Maps stock symbol -> sector ETF
# - SECTOR_NAMES: Maps sector ETF -> human-readable name
# - MARKET_BENCHMARKS: Market-level ETFs
# - SECTOR_ETFS: All sector ETF symbols
# =============================================================================

from .universe_loader import get_universe

# Lazy-load universe to avoid circular imports and allow testing
_universe = get_universe()

# Derived mappings (single source of truth from YAML)
STOCK_TO_SECTOR = _universe.stock_to_sector
SECTOR_NAMES = _universe.sector_names
MARKET_BENCHMARKS = _universe.market_benchmarks
SECTOR_ETFS = set(_universe.sector_etfs)
