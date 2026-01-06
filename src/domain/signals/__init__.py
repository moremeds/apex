"""
Trading Signal Engine - Real-time technical indicator calculation and signal generation.

This module provides:
- TradingSignal: Domain model for generated trading signals
- SignalRule: Configurable rule definitions for signal generation
- Indicator protocol: Unified interface for all technical indicators
- IndicatorRegistry: Auto-discovery and management of indicators
- UniverseProvider: Configurable ticker universe management
"""

from .models import (
    TradingSignal,
    SignalRule,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    ConditionType,
    Divergence,
    DivergenceType,
)

__all__ = [
    "TradingSignal",
    "SignalRule",
    "SignalCategory",
    "SignalDirection",
    "SignalPriority",
    "ConditionType",
    "Divergence",
    "DivergenceType",
]
