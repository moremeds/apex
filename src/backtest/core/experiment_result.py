"""
Experiment result - final summary of entire experiment.

Contains top trials, statistical validation results, and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .trial_result import TrialResult


@dataclass
class ExperimentResult:
    """
    Complete result for an experiment.

    Summarizes all trials and provides:
    - Top performing trials
    - Statistical validation results
    - Experiment metadata
    """

    # Identification
    experiment_id: str
    name: str
    strategy: str

    # Configuration summary
    total_parameter_combinations: int = 0
    symbols_tested: List[str] = field(default_factory=list)
    temporal_folds: int = 0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0

    # Results summary
    total_trials: int = 0
    successful_trials: int = 0
    total_runs: int = 0
    successful_runs: int = 0

    # Top trials (sorted by trial_score descending)
    top_trials: List[TrialResult] = field(default_factory=list)

    # Statistical validation
    pbo: Optional[float] = None  # Probability of Backtest Overfit
    dsr: Optional[float] = None  # Deflated Sharpe Ratio
    dsr_p_value: Optional[float] = None
    monte_carlo_p_value: Optional[float] = None

    # Best trial details
    best_trial_id: Optional[str] = None
    best_params: Optional[Dict[str, Any]] = None
    best_sharpe: Optional[float] = None
    best_return: Optional[float] = None

    # Reproducibility
    data_version: str = ""
    random_seed: int = 42

    def to_dict(self, include_trials: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for storage/display."""
        result = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "strategy": self.strategy,
            "total_parameter_combinations": self.total_parameter_combinations,
            "symbols_tested": self.symbols_tested,
            "temporal_folds": self.temporal_folds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": self.total_duration_seconds,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "pbo": self.pbo,
            "dsr": self.dsr,
            "dsr_p_value": self.dsr_p_value,
            "monte_carlo_p_value": self.monte_carlo_p_value,
            "best_trial_id": self.best_trial_id,
            "best_params": self.best_params,
            "best_sharpe": self.best_sharpe,
            "best_return": self.best_return,
            "data_version": self.data_version,
            "random_seed": self.random_seed,
        }

        if include_trials:
            result["top_trials"] = [t.to_dict() for t in self.top_trials]

        return result

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.name}")
        print(f"ID: {self.experiment_id}")
        print(f"Strategy: {self.strategy}")
        print(f"{'='*60}")

        print(f"\nConfiguration:")
        print(f"  Parameter combinations: {self.total_parameter_combinations}")
        print(f"  Symbols: {len(self.symbols_tested)}")
        print(f"  Temporal folds: {self.temporal_folds}")

        print(f"\nExecution:")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Successful: {self.successful_trials}")
        print(f"  Total runs: {self.total_runs}")
        print(f"  Duration: {self.total_duration_seconds:.1f}s")

        if self.best_trial_id:
            print(f"\nBest Trial:")
            print(f"  ID: {self.best_trial_id}")
            print(f"  Params: {self.best_params}")
            print(f"  Sharpe: {self.best_sharpe:.3f}")
            print(f"  Return: {self.best_return:.2%}")

        if self.pbo is not None:
            print(f"\nStatistical Validation:")
            print(f"  PBO: {self.pbo:.3f}")
            if self.pbo > 0.5:
                print("    (Warning: High probability of overfitting)")
            if self.dsr is not None:
                print(f"  DSR: {self.dsr:.3f} (p-value: {self.dsr_p_value:.4f})")

        print(f"\n{'='*60}\n")
