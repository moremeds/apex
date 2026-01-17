"""
Regime Detection Indicators.

Provides 3-level hierarchical regime detection:
- Level 1: Market Regime (QQQ/SPY) - Gate/Veto
- Level 2: Sector Regime (SMH, XLV, XLF, XLE) - Weight/Selection
- Level 3: Single-Name Regime (NVDA, TSLA, AAPL) - Entry/Sizing

Schema version: regime_output@1.0
"""

from .models import (
    ENTRY_HYSTERESIS,
    EXIT_HYSTERESIS,
    MARKET_BENCHMARKS,
    # Enums
    ChopState,
    ExtState,
    FallbackReason,
    IVState,
    MarketRegime,
    TrendState,
    VolState,
    # Core dataclasses
    ComponentStates,
    ComponentValues,
    RegimeOutput,
    RegimeState,
    # Explainability dataclasses (PR1)
    BarSnapshot,
    DataQuality,
    DataWindow,
    DerivedMetrics,
    InputsUsed,
    RegimeTransitionState,
)
from .regime_detector import RegimeDetectorIndicator
from .rule_trace import (
    RuleTrace,
    ThresholdInfo,
    format_rule_result,
    generate_counterfactual,
)

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
]
