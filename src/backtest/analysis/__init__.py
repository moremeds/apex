"""
Analysis layer for systematic backtesting.

This module provides:
- Aggregation: Median/MAD statistics across runs
- Validation: PBO, DSR, Monte Carlo
- Constraints: p10_sharpe, median_max_dd thresholds
- Metrics: Comprehensive metrics calculator
- Reporting: HTML tearsheets
"""

from .aggregator import Aggregator, AggregationConfig
from .statistics import PBOCalculator, DSRCalculator, MonteCarloSimulator
from .validator import ConstraintValidator, Constraint
from .metrics_calculator import MetricsCalculator, Trade

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
