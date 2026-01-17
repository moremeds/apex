"""
Regime Detection Indicators.

Provides 3-level hierarchical regime detection:
- Level 1: Market Regime (QQQ/SPY) - Gate/Veto
- Level 2: Sector Regime (SMH, XLV, XLF, XLE) - Weight/Selection
- Level 3: Single-Name Regime (NVDA, TSLA, AAPL) - Entry/Sizing
"""

from .models import (
    ENTRY_HYSTERESIS,
    EXIT_HYSTERESIS,
    ChopState,
    ComponentStates,
    ComponentValues,
    ExtState,
    IVState,
    MarketRegime,
    RegimeOutput,
    RegimeState,
    TrendState,
    VolState,
)
from .regime_detector import RegimeDetectorIndicator

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
    # Dataclasses
    "RegimeState",
    "ComponentStates",
    "ComponentValues",
    "RegimeOutput",
    # Constants
    "ENTRY_HYSTERESIS",
    "EXIT_HYSTERESIS",
]
