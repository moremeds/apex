"""
Analysis layer for systematic backtesting.

This module provides:
- Aggregation: Median/MAD statistics across runs
- Validation: PBO, DSR, Monte Carlo
- Constraints: p10_sharpe, median_max_dd thresholds
- Metrics: Comprehensive metrics calculator
- Reporting: HTML tearsheets
"""

from .aggregator import AggregationConfig, Aggregator
from .metrics_calculator import MetricsCalculator, Trade
from .statistics import DSRCalculator, MonteCarloSimulator, PBOCalculator
from .validator import Constraint, ConstraintValidator

__all__ = [
    # Aggregation
    "Aggregator",
    "AggregationConfig",
    # Validation
    "PBOCalculator",
    "DSRCalculator",
    "MonteCarloSimulator",
    # Constraints
    "ConstraintValidator",
    "Constraint",
    # Metrics
    "MetricsCalculator",
    "Trade",
]
