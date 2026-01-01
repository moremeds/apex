"""
Backtest execution engines.

Provides:
- BacktestEngine: Protocol for engine interface
- VectorBTEngine: Fast vectorized backtesting for screening
- ApexEngine: Event-driven backtesting with full feature support (future)

Two-Stage Pipeline:
1. Screening (VectorBT): Fast evaluation of 10,000+ parameter combinations
2. Validation (Apex): Full-featured validation of top candidates

Example:
    from src.backtest.execution.engines import VectorBTEngine, VectorBTConfig

    engine = VectorBTEngine(VectorBTConfig(strategy_type="ma_cross"))
    result = engine.run(run_spec)

    # Batch with vectorization
    results = engine.run_batch(specs)
"""

from .interface import (
    BacktestEngine,
    BaseEngine,
    EngineConfig,
    EngineType,
)
from .vectorbt_engine import VectorBTEngine, VectorBTConfig

__all__ = [
    # Interface
    "BacktestEngine",
    "BaseEngine",
    "EngineConfig",
    "EngineType",
    # Engines
    "VectorBTEngine",
    "VectorBTConfig",
]
