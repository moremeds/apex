"""
M2 Validation Framework for Regime Detection.

Implements nested walk-forward cross-validation with anti-overfitting measures:
- Strict nested CV: Optuna only sees inner folds, outer test is evaluation-only
- Frozen labeler: Ground truth thresholds are immutable + versioned
- Symbol-level statistics: n ~ 140 symbols gives real statistical power
- Block bootstrap: CIs respect time correlation
- Unified bar units: No day/bar confusion for multi-timeframe
"""

from .labeler_contract import (
    RegimeLabel,
    RegimeLabeler,
    RegimeLabelerConfig,
)
from .nested_cv import (
    NestedCVConfig,
    NestedCVResult,
    NestedWalkForwardCV,
    OuterFold,
    OuterFoldResult,
    TimeWindow,
)
from .statistics import (
    StatisticalResult,
    SymbolMetrics,
    block_bootstrap_ci,
    compute_cohens_d,
    compute_symbol_level_stats,
)
from .time_units import ValidationTimeConfig, validate_time_config
from .earliness import (
    EarlinessResult,
    SignalEvent,
    TrendEpisode,
    compute_earliness,
    compute_multi_tf_earliness,
    detect_trend_episodes,
    find_first_signal_date,
)
from .confirmation import (
    ConfirmationResult,
    StrategyMetrics,
    apply_and_rule,
    apply_majority_vote,
    compare_strategies,
    compute_strategy_metrics,
)
from .schemas import (
    BarValidation,
    GateResult,
    HorizonConfig,
    LabelerThreshold,
    SplitConfig,
    ValidationOutput,
    create_fast_validation_output,
    create_full_validation_output,
)
from .validation_service import (
    ValidationService,
    ValidationServiceConfig,
    SymbolValidationResult,
)

__all__ = [
    # Time units
    "ValidationTimeConfig",
    "validate_time_config",
    # Labeler
    "RegimeLabel",
    "RegimeLabeler",
    "RegimeLabelerConfig",
    # Nested CV
    "NestedCVConfig",
    "NestedCVResult",
    "NestedWalkForwardCV",
    "OuterFold",
    "OuterFoldResult",
    "TimeWindow",
    # Statistics
    "StatisticalResult",
    "SymbolMetrics",
    "block_bootstrap_ci",
    "compute_cohens_d",
    "compute_symbol_level_stats",
    # Earliness
    "EarlinessResult",
    "SignalEvent",
    "TrendEpisode",
    "compute_earliness",
    "compute_multi_tf_earliness",
    "detect_trend_episodes",
    "find_first_signal_date",
    # Confirmation
    "ConfirmationResult",
    "StrategyMetrics",
    "apply_and_rule",
    "apply_majority_vote",
    "compare_strategies",
    "compute_strategy_metrics",
    # Schemas
    "BarValidation",
    "GateResult",
    "HorizonConfig",
    "LabelerThreshold",
    "SplitConfig",
    "ValidationOutput",
    "create_fast_validation_output",
    "create_full_validation_output",
]
