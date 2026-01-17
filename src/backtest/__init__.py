"""
Systematic Backtesting Framework.

This module provides a comprehensive framework for systematic strategy experimentation:

- **Experiment Management**: Define experiments with parameter grids, universes, and temporal splits
- **Walk-Forward Validation**: Proper train/test splits with purge/embargo gaps
- **Multi-Level Aggregation**: Trial -> Symbol -> Run hierarchy with robust statistics
- **Statistical Validation**: PBO, DSR, Monte Carlo for overfitting detection
- **Parallel Execution**: Scale across CPU cores with progress tracking
- **Storage**: DuckDB/Parquet persistence for fast queries

Module Structure:
- core/: Specs, results, and hashing utilities
- data/: Splitters, storage, and trading calendar
- execution/: Runner, engines, and parity testing
- analysis/: Aggregation, validation, and constraints
- optimization/: Grid and Bayesian optimizers
- _internal/: Logging and utilities

Example:
    from src.backtest import ExperimentSpec, SystematicRunner

    spec = ExperimentSpec.from_yaml("experiments/momentum.yaml")
    runner = SystematicRunner(db_path="results/backtest.db")
    experiment_id = runner.run(spec)
"""

# Internal: Logging
from ._internal import (
    ContextLogger,
    LogContext,
    StructuredFormatter,
    get_logger,
    setup_logging,
)

# Analysis: Aggregation, Validation, Constraints
from .analysis import (
    AggregationConfig,
    Aggregator,
    Constraint,
    ConstraintValidator,
    DSRCalculator,
    MonteCarloSimulator,
    PBOCalculator,
)

# Core: Specs, Results, Hashing
from .core import (  # Specs; Results; Hashing
    ExperimentResult,
    ExperimentSpec,
    OptimizationConfig,
    ParameterDef,
    ProfileConfig,
    ReproducibilityConfig,
    RunMetrics,
    RunResult,
    RunSpec,
    RunStatus,
    TemporalConfig,
    TimeWindow,
    TrialAggregates,
    TrialResult,
    TrialSpec,
    UniverseConfig,
    canonical_json,
    generate_experiment_id,
    generate_run_id,
    generate_trial_id,
    get_git_sha,
    quantize_float,
)

# Data: Splitters, Storage, Calendar
from .data import (
    CPCVConfig,
    CPCVSplitter,
    DatabaseManager,
    ExperimentRepository,
    RunRepository,
    SplitConfig,
    TradingCalendar,
    TrialRepository,
    WalkForwardSplitter,
    WeekdayCalendar,
    get_calendar,
)

# Execution: Runners
from .execution import (
    ExecutionProgress,
    ParallelConfig,
    ParallelRunner,
    ProgressMonitor,
    RunnerConfig,
    SystematicRunner,
)

# Optimization
from .optimization import (
    BayesianOptimizer,
    GridOptimizer,
)

__all__ = [
    # Core - Specs
    "ExperimentSpec",
    "TrialSpec",
    "RunSpec",
    "TimeWindow",
    "ParameterDef",
    "UniverseConfig",
    "TemporalConfig",
    "OptimizationConfig",
    "ProfileConfig",
    "ReproducibilityConfig",
    # Core - Results
    "RunResult",
    "RunMetrics",
    "RunStatus",
    "TrialResult",
    "TrialAggregates",
    "ExperimentResult",
    # Core - Hashing
    "canonical_json",
    "generate_experiment_id",
    "generate_run_id",
    "generate_trial_id",
    "get_git_sha",
    "quantize_float",
    # Data - Splitters
    "WalkForwardSplitter",
    "CPCVSplitter",
    "SplitConfig",
    "CPCVConfig",
    # Data - Storage
    "DatabaseManager",
    "ExperimentRepository",
    "TrialRepository",
    "RunRepository",
    # Data - Calendar
    "TradingCalendar",
    "WeekdayCalendar",
    "get_calendar",
    # Execution
    "SystematicRunner",
    "RunnerConfig",
    "ParallelRunner",
    "ParallelConfig",
    "ExecutionProgress",
    "ProgressMonitor",
    # Analysis
    "Aggregator",
    "AggregationConfig",
    "PBOCalculator",
    "DSRCalculator",
    "MonteCarloSimulator",
    "ConstraintValidator",
    "Constraint",
    # Optimization
    "GridOptimizer",
    "BayesianOptimizer",
    # Logging
    "LogContext",
    "ContextLogger",
    "StructuredFormatter",
    "setup_logging",
    "get_logger",
]
