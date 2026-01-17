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

from .backtrader_adapter import (
    BACKTRADER_AVAILABLE,
    ApexStrategyWrapper,
    BacktraderScheduler,
    run_backtest_with_backtrader,
)
from .engines import (
    ApexEngine,
    ApexEngineConfig,
    BacktestConfig,
    BacktestEngine,
    BaseEngine,
    EngineConfig,
    EngineType,
    EventDrivenEngine,
    VectorBTConfig,
    VectorBTEngine,
)
from .ledger import PositionLedger
from .order_matching import OrderMatcher
from .parallel import (
    ExecutionProgress,
    ParallelConfig,
    ParallelRunner,
    ProgressMonitor,
    is_transient_error,
)
from .parity import (  # Signal parity testing
    DirectionalSignalCapture,
    DriftDetail,
    DriftType,
    ParityConfig,
    ParityResult,
    SignalCapture,
    SignalParityResult,
    StrategyParityHarness,
    compare_directional_signal_parity,
    compare_signal_parity,
)
from .simulated import FillModel, SimulatedExecution, SimulatedOrder, SimulatedPosition
from .systematic import RunnerConfig, SystematicRunner
from .trade_tracker import MatchingMethod, OpenPosition, TradeTracker

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
