#!/usr/bin/env python3
"""
Unified Backtest Runner

Main entry point for all backtesting operations:
- Single backtests (bar-by-bar with ApexEngine)
- Systematic experiments (vectorized with VectorBTEngine)

Usage:
    # Single backtest (ApexEngine - full execution simulation)
    python -m src.backtest.runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30

    # Systematic experiment (VectorBTEngine - fast parameter optimization)
    python -m src.backtest.runner --spec config/backtest/playbook/ta_metrics.yaml

    # Force specific engine
    python -m src.backtest.runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30 --engine vectorbt

    # List strategies
    python -m src.backtest.runner --list-strategies

This module serves as a facade, re-exporting from the refactored modules:
- config/loaders.py: Configuration loading
- execution/single_backtest.py: SingleBacktestRunner
- execution/systematic_experiment.py: run_systematic_experiment, prefetch_data
- execution/backtrader_runner.py: BacktraderRunner
- execution/engines/vectorbt_worker.py: VectorBT worker functions
- execution/engines/apex_worker.py: Apex worker functions
"""

from __future__ import annotations

# Re-export CLI main function
from .cli import main

# Re-export configuration loaders
from .config import (
    DEFAULT_CONFIG_PATH,
    load_historical_data_config,
    load_ib_config,
)

# Re-export runners
from .execution import (
    BacktraderRunner,
    SingleBacktestRunner,
    prefetch_data,
    run_systematic_experiment,
)

# Re-export worker functions for backward compatibility
from .execution.engines import (
    create_apex_backtest_fn,
    create_vectorbt_backtest_fn,
    is_apex_required,
)

__all__ = [
    # Configuration
    "DEFAULT_CONFIG_PATH",
    "load_ib_config",
    "load_historical_data_config",
    # Runners
    "SingleBacktestRunner",
    "BacktraderRunner",
    "run_systematic_experiment",
    "prefetch_data",
    # Worker functions
    "create_vectorbt_backtest_fn",
    "create_apex_backtest_fn",
    "is_apex_required",
    # CLI
    "main",
]

if __name__ == "__main__":
    main()
