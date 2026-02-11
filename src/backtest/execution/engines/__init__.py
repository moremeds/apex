"""
Backtest execution engines.

Provides:
- BacktestEngine: Protocol for engine interface (RunSpec-based)
- VectorBTEngine: Fast vectorized backtesting for screening
- ApexEngine: Event-driven backtesting with full feature support (RunSpec wrapper)
- EventDrivenEngine: Bar-by-bar event-driven engine (BacktestConfig-based)

Two-Stage Pipeline:
1. Screening (VectorBT): Fast evaluation of 10,000+ parameter combinations
2. Validation (Apex/EventDriven): Full-featured validation of top candidates

Example:
    from src.backtest.execution.engines import VectorBTEngine, ApexEngine

    # Fast screening
    engine = VectorBTEngine(VectorBTConfig(strategy_type="rsi_mean_reversion"))
    result = engine.run(run_spec)

    # Full validation (via RunSpec)
    apex = ApexEngine(ApexEngineConfig(reality_pack_name="ib"))
    result = apex.run(run_spec)

    # Direct event-driven (via BacktestConfig)
    from src.backtest.execution.engines import EventDrivenEngine, BacktestConfig
    engine = EventDrivenEngine(config)
    result = await engine.run()
"""

from .apex_engine import ApexEngine, ApexEngineConfig

# Worker functions for multiprocessing
from .apex_worker import (
    create_apex_backtest_fn,
    init_apex_worker,
    is_apex_required,
    run_apex_backtest,
)

# Event-driven engine (bar-by-bar, async) - renamed to avoid Protocol conflict
from .backtest_engine import BacktestConfig
from .backtest_engine import BacktestEngine as EventDrivenEngine
from .interface import (
    BacktestEngine,
    BaseEngine,
    EngineConfig,
    EngineType,
)
from .vectorbt_engine import VectorBTConfig, VectorBTEngine
from .vectorbt_worker import (
    create_vectorbt_backtest_fn,
    init_vectorbt_worker,
    run_vectorbt_backtest,
)

__all__ = [
    # Interface (Protocol)
    "BacktestEngine",
    "BaseEngine",
    "EngineConfig",
    "EngineType",
    # Systematic Engines (RunSpec-based)
    "VectorBTEngine",
    "VectorBTConfig",
    "ApexEngine",
    "ApexEngineConfig",
    # Event-Driven Engine (BacktestConfig-based)
    "EventDrivenEngine",
    "BacktestConfig",
    # Worker functions
    "init_vectorbt_worker",
    "run_vectorbt_backtest",
    "create_vectorbt_backtest_fn",
    "init_apex_worker",
    "run_apex_backtest",
    "create_apex_backtest_fn",
    "is_apex_required",
]
