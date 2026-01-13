"""
Trading Signal Engine - Real-time technical indicator calculation and signal generation.

This module provides:
- TradingSignal: Domain model for generated trading signals
- SignalRule: Configurable rule definitions for signal generation
- Indicator protocol: Unified interface for all technical indicators
- IndicatorRegistry: Auto-discovery and management of indicators
- IndicatorEngine: Calculates indicators on BAR_CLOSE events
- RuleEngine: Evaluates rules on INDICATOR_UPDATE events
- Divergence detection: Price/indicator divergence analysis

Usage:
    from src.domain.signals import (
        IndicatorEngine,
        RuleEngine,
        RuleRegistry,
    )
    from src.domain.signals.rules import ALL_RULES

    # Setup engines
    registry = RuleRegistry()
    registry.add_rules(ALL_RULES)

    indicator_engine = IndicatorEngine(event_bus)
    rule_engine = RuleEngine(event_bus, registry)

    # Start processing
    indicator_engine.start()
    rule_engine.start()
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
    ConfluenceScore,
)

from .indicator_engine import IndicatorEngine
from .rule_engine import RuleEngine, RuleRegistry
from .confluence_calculator import ConfluenceCalculator

from .data import BarBuilder, BarAggregator

__all__ = [
    # Models
    "TradingSignal",
    "SignalRule",
    "SignalCategory",
    "SignalDirection",
    "SignalPriority",
    "ConditionType",
    "Divergence",
    "DivergenceType",
    "ConfluenceScore",
    # Engines
    "IndicatorEngine",
    "RuleEngine",
    "RuleRegistry",
    "ConfluenceCalculator",
    # Data pipeline
    "BarBuilder",
    "BarAggregator",
]
