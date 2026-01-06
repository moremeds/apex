"""
Execution layer for systematic backtesting.

This module provides:
- SystematicRunner: Main experiment orchestrator
- ParallelRunner: Multi-process execution
- Engines: VectorBT and Apex backtest adapters
- Parity: Engine comparison testing for drift detection
- SimulatedExecution: Order matching and fill simulation
- TradeTracker: Entry/exit matching for completed trades
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
    ApexEngine,
    ApexEngineConfig,
    EventDrivenEngine,
    BacktestConfig,
)
from .backtrader_adapter import (
    ApexStrategyWrapper,
    BacktraderScheduler,
    run_backtest_with_backtrader,
    BACKTRADER_AVAILABLE,
)
from .parity import (
    StrategyParityHarness,
    ParityConfig,
    ParityResult,
    DriftType,
    DriftDetail,
    # Signal parity testing
    SignalParityResult,
    compare_signal_parity,
    compare_directional_signal_parity,
    SignalCapture,
    DirectionalSignalCapture,
)
from .simulated import SimulatedExecution, FillModel, SimulatedOrder, SimulatedPosition
from .order_matching import OrderMatcher
from .ledger import PositionLedger
from .trade_tracker import TradeTracker, MatchingMethod, OpenPosition

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
    "ApexEngine",
    "ApexEngineConfig",
    "EventDrivenEngine",
    "BacktestConfig",
    # Backtrader integration
    "ApexStrategyWrapper",
    "BacktraderScheduler",
    "run_backtest_with_backtrader",
    "BACKTRADER_AVAILABLE",
    # Parity testing
    "StrategyParityHarness",
    "ParityConfig",
    "ParityResult",
    "DriftType",
    "DriftDetail",
    # Signal parity
    "SignalParityResult",
    "compare_signal_parity",
    "compare_directional_signal_parity",
    "SignalCapture",
    "DirectionalSignalCapture",
    # Simulated execution
    "SimulatedExecution",
    "FillModel",
    "SimulatedOrder",
    "SimulatedPosition",
    "OrderMatcher",
    "PositionLedger",
    # Trade tracking
    "TradeTracker",
    "MatchingMethod",
    "OpenPosition",
]
