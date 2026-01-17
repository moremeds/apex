"""
Regime Service - Hierarchical Regime Detection.

Provides 3-level hierarchical regime detection and action resolution:
- Level 1: Market Regime (QQQ/SPY) - Gate/Veto
- Level 2: Sector Regime (SMH, XLV, XLF, XLE) - Weight/Selection
- Level 3: Single-Name Regime (NVDA, TSLA, AAPL) - Entry/Sizing

Parameter optimization support via params_store:
    from src.domain.services.regime import get_regime_params
    params = get_regime_params("NVDA")
"""

from .action_resolver import (
    get_action_summary,
    get_defensive_actions,
    get_position_sizing,
    resolve_action,
    should_reduce_exposure,
)
from .models import (
    ACTION_MAPS,
    DECISION_TABLE_SHORT_PUT,
    DEFAULT_CONTEXTS,
    MARKET_BENCHMARKS,
    SECTOR_ETFS,
    SECTOR_NAMES,
    STOCK_TO_SECTOR,
    AccountType,
    ActionContext,
    HierarchicalRegime,
    TradingAction,
)
from .params_store import (
    DEFAULT_PARAMS,
    REGIME_PARAMS,
    get_regime_params,
    update_regime_params,
    validate_params,
)
from .regime_hierarchy import (
    apply_weekly_veto,
    get_4h_alerts,
    get_dynamic_sector,
    get_hierarchy_level,
    get_sector_for_symbol,
    is_market_benchmark,
    is_sector_etf,
    resolve_market_action,
    synthesize_regimes,
)

__all__ = [
    # Models
    "TradingAction",
    "AccountType",
    "ActionContext",
    "HierarchicalRegime",
    # Mappings
    "STOCK_TO_SECTOR",
    "SECTOR_NAMES",
    "MARKET_BENCHMARKS",
    "SECTOR_ETFS",
    "ACTION_MAPS",
    "DECISION_TABLE_SHORT_PUT",
    "DEFAULT_CONTEXTS",
    # Parameter store
    "DEFAULT_PARAMS",
    "REGIME_PARAMS",
    "get_regime_params",
    "update_regime_params",
    "validate_params",
    # Hierarchy functions
    "resolve_market_action",
    "get_sector_for_symbol",
    "get_dynamic_sector",
    "synthesize_regimes",
    "apply_weekly_veto",
    "is_market_benchmark",
    "is_sector_etf",
    "get_hierarchy_level",
    "get_4h_alerts",
    # Action resolver functions
    "resolve_action",
    "get_position_sizing",
    "get_action_summary",
    "should_reduce_exposure",
    "get_defensive_actions",
]
