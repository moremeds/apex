"""
Trial result - aggregated across symbols and windows.

Uses robust statistics (median, MAD) for aggregation to reduce
sensitivity to outliers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .run_result import RunResult


@dataclass
class TrialAggregates:
    """
    Aggregated statistics across all runs in a trial.

    Uses median and MAD (median absolute deviation) for robustness.
    Also computes percentiles for constraint checking.
    """

    # Central tendency
    median_sharpe: float = 0.0
    median_return: float = 0.0
    median_max_dd: float = 0.0
    median_win_rate: float = 0.0
    median_profit_factor: float = 0.0

    # Dispersion (MAD = median absolute deviation)
    mad_sharpe: float = 0.0
    mad_return: float = 0.0
    mad_max_dd: float = 0.0

    # Percentiles for constraint checking
    p10_sharpe: float = 0.0  # 10th percentile (worst case)
    p90_sharpe: float = 0.0  # 90th percentile (best case)
    p10_max_dd: float = 0.0  # 10th percentile max drawdown
    p90_max_dd: float = 0.0

    # Stability metrics
    stability_score: float = 0.0  # Cross-window consistency
    degradation_ratio: float = 0.0  # IS to OOS performance drop

    # Count statistics
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0

    # Separate IS/OOS aggregates
    is_median_sharpe: float = 0.0  # In-sample median
    oos_median_sharpe: float = 0.0  # Out-of-sample median

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "median_sharpe": self.median_sharpe,
            "median_return": self.median_return,
            "median_max_dd": self.median_max_dd,
            "median_win_rate": self.median_win_rate,
            "median_profit_factor": self.median_profit_factor,
            "mad_sharpe": self.mad_sharpe,
            "mad_return": self.mad_return,
            "mad_max_dd": self.mad_max_dd,
            "p10_sharpe": self.p10_sharpe,
            "p90_sharpe": self.p90_sharpe,
            "p10_max_dd": self.p10_max_dd,
            "p90_max_dd": self.p90_max_dd,
            "stability_score": self.stability_score,
            "degradation_ratio": self.degradation_ratio,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "is_median_sharpe": self.is_median_sharpe,
            "oos_median_sharpe": self.oos_median_sharpe,
        }


@dataclass
class TrialResult:
    """
    Complete result for a trial (parameter combination).

    Aggregates results from all runs (symbols × windows) to produce
    robust statistics for comparing against other trials.
    """

    # Identification
    trial_id: str
    experiment_id: str
    params: Dict[str, Any]

    # Aggregates
    aggregates: TrialAggregates = field(default_factory=TrialAggregates)

    # Composite score for ranking
    trial_score: float = 0.0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0

    # Run details (optional, may be omitted for memory)
    runs: List[RunResult] = field(default_factory=list)

    # Optimization metadata
    trial_index: Optional[int] = None
    suggested_by: Optional[str] = None  # "grid", "TPE", "random"

    # Constraint satisfaction
    constraints_met: bool = True
    constraint_violations: List[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if trial has any successful runs."""
        return self.aggregates.successful_runs > 0

    @property
    def success_rate(self) -> float:
        """Fraction of runs that succeeded."""
        if self.aggregates.total_runs == 0:
            return 0.0
        return self.aggregates.successful_runs / self.aggregates.total_runs

    def to_dict(self, include_runs: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "params": self.params,
            "trial_score": self.trial_score,
            "constraints_met": self.constraints_met,
            "constraint_violations": self.constraint_violations,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": self.total_duration_seconds,
            "trial_index": self.trial_index,
            "suggested_by": self.suggested_by,
            **self.aggregates.to_dict(),
        }

        if include_runs:
            result["runs"] = [r.to_dict() for r in self.runs]

        return result

    @classmethod
    def from_runs(
        cls,
        trial_id: str,
        experiment_id: str,
        params: Dict[str, Any],
        runs: List[RunResult],
    ) -> "TrialResult":
        """Create trial result from list of run results."""
        # Will be computed by Aggregator
        return cls(
            trial_id=trial_id,
            experiment_id=experiment_id,
            params=params,
            runs=runs,
        )
