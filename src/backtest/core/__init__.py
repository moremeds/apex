"""
Core data structures for systematic backtesting.

This module provides:
- Specifications (ExperimentSpec, TrialSpec, RunSpec)
- Results (RunResult, TrialResult, ExperimentResult)
- Hashing utilities for reproducibility
"""

from .experiment import (
    ExperimentSpec,
    OptimizationConfig,
    ParameterDef,
    ProfileConfig,
    ReproducibilityConfig,
    TemporalConfig,
    UniverseConfig,
)
from .experiment_result import ExperimentResult
from .hashing import (
    canonical_json,
    generate_experiment_id,
    generate_run_id,
    generate_trial_id,
    get_git_sha,
    quantize_float,
)
from .run import RunSpec, TimeWindow
from .run_result import RunMetrics, RunResult, RunStatus
from .trial import TrialSpec
from .trial_result import TrialAggregates, TrialResult

__all__ = [
    # Specs
    "ExperimentSpec",
    "ParameterDef",
    "UniverseConfig",
    "TemporalConfig",
    "OptimizationConfig",
    "ProfileConfig",
    "ReproducibilityConfig",
    "TrialSpec",
    "RunSpec",
    "TimeWindow",
    # Results
    "RunResult",
    "RunMetrics",
    "RunStatus",
    "TrialResult",
    "TrialAggregates",
    "ExperimentResult",
    # Hashing
    "canonical_json",
    "generate_experiment_id",
    "generate_run_id",
    "generate_trial_id",
    "get_git_sha",
    "quantize_float",
]
