"""
Backtest domain models and schemas.

This module provides:
- BacktestSpec: Configuration for a backtest run
- BacktestResult: Results and metrics from a backtest
- Performance, Risk, Trade metrics dataclasses
"""

from .backtest_spec import BacktestSpec, StrategySpecConfig, DataSpecConfig, ExecutionSpecConfig
from .backtest_result import (
    BacktestResult,
    PerformanceMetrics,
    RiskMetrics,
    TradeMetrics,
    CostMetrics,
    ExposureMetrics,
    TradeRecord,
)

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
