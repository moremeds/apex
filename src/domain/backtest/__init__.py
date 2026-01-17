"""
Backtest domain models and schemas.

This module provides:
- BacktestSpec: Configuration for a backtest run
- BacktestResult: Results and metrics from a backtest
- Performance, Risk, Trade metrics dataclasses
"""

from .backtest_result import (
    BacktestResult,
    CostMetrics,
    ExposureMetrics,
    PerformanceMetrics,
    RiskMetrics,
    TradeMetrics,
    TradeRecord,
)
from .backtest_spec import BacktestSpec, DataSpecConfig, ExecutionSpecConfig, StrategySpecConfig

__all__ = [
    # Spec
    "BacktestSpec",
    "StrategySpecConfig",
    "DataSpecConfig",
    "ExecutionSpecConfig",
    # Result
    "BacktestResult",
    "PerformanceMetrics",
    "RiskMetrics",
    "TradeMetrics",
    "CostMetrics",
    "ExposureMetrics",
    "TradeRecord",
]
