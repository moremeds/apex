"""
Regime Detection Indicators.

Provides 3-level hierarchical regime detection:
- Level 1: Market Regime (QQQ/SPY) - Gate/Veto
- Level 2: Sector Regime (SMH, XLV, XLF, XLE) - Weight/Selection
- Level 3: Single-Name Regime (NVDA, TSLA, AAPL) - Entry/Sizing

Schema version: regime_output@1.0
"""

from .models import (  # Enums; Core dataclasses; Explainability dataclasses (PR1)
    ENTRY_HYSTERESIS,
    EXIT_HYSTERESIS,
    MARKET_BENCHMARKS,
    BarSnapshot,
    ChopState,
    ComponentStates,
    ComponentValues,
    DataQuality,
    DataWindow,
    DerivedMetrics,
    ExtState,
    FallbackReason,
    InputsUsed,
    IVState,
    MarketRegime,
    RegimeOutput,
    RegimeState,
    RegimeTransitionState,
    TrendState,
    VolState,
)
from .regime_detector import RegimeDetectorIndicator
from .rule_trace import (
    RuleTrace,
    ThresholdInfo,
    format_rule_result,
    generate_counterfactual,
)

# Phase 5: Composite scoring system
from .composite_scorer import CompositeRegimeScorer, CompositeWeights, ScoreBands
from .factor_normalizer import FactorNormalizer, NormalizedFactors, compute_normalized_factors
from .regime_validation import FailureCriteria, RegimeValidator, ValidationResult
from .score_hysteresis import HysteresisBands, ScoreHysteresisStateMachine
from .weight_learner import LearningResult, TargetLabelGenerator, WeightLearner

__all__ = [
    # Indicator
    "RegimeDetectorIndicator",
    # Enums
    "MarketRegime",
    "TrendState",
    "VolState",
    "IVState",
    "ChopState",
    "ExtState",
    "FallbackReason",
    # Core dataclasses
    "RegimeState",
    "ComponentStates",
    "ComponentValues",
    "RegimeOutput",
    # Explainability dataclasses (PR1)
    "DataWindow",
    "BarSnapshot",
    "InputsUsed",
    "DerivedMetrics",
    "DataQuality",
    "RegimeTransitionState",
    # Rule tracing
    "RuleTrace",
    "ThresholdInfo",
    "generate_counterfactual",
    "format_rule_result",
    # Constants
    "ENTRY_HYSTERESIS",
    "EXIT_HYSTERESIS",
    "MARKET_BENCHMARKS",
    # Phase 5: Composite scoring system
    "CompositeRegimeScorer",
    "CompositeWeights",
    "ScoreBands",
    "FactorNormalizer",
    "NormalizedFactors",
    "compute_normalized_factors",
    "WeightLearner",
    "TargetLabelGenerator",
    "LearningResult",
    "ScoreHysteresisStateMachine",
    "HysteresisBands",
    "RegimeValidator",
    "FailureCriteria",
    "ValidationResult",
]
