"""
Execution layer for systematic backtesting.

This module provides:
- SystematicRunner: Main experiment orchestrator
- ParallelRunner: Multi-process execution
- Engines: VectorBT and Apex backtest adapters
- Parity: Engine comparison testing for drift detection
"""

from .systematic import SystematicRunner, RunnerConfig
from .parallel import (
    ParallelRunner,
    ParallelConfig,
    ExecutionProgress,
    ProgressMonitor,
    is_transient_error,
)
from .engines import (
    BacktestEngine,
    BaseEngine,
    EngineConfig,
    EngineType,
    VectorBTEngine,
    VectorBTConfig,
)
from .parity import (
    StrategyParityHarness,
    ParityConfig,
    ParityResult,
    DriftType,
    DriftDetail,
)

__all__ = [
    # Runner
    "SystematicRunner",
    "RunnerConfig",
    # Parallel execution
    "ParallelRunner",
    "ParallelConfig",
    "ExecutionProgress",
    "ProgressMonitor",
    "is_transient_error",
    # Engines
    "BacktestEngine",
    "BaseEngine",
    "EngineConfig",
    "EngineType",
    "VectorBTEngine",
    "VectorBTConfig",
    # Parity testing
    "StrategyParityHarness",
    "ParityConfig",
    "ParityResult",
    "DriftType",
    "DriftDetail",
]
