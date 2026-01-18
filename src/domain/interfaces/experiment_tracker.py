"""
Experiment Tracker Port for ML training runs.

This port defines the contract for recording and querying experiment results.
Used to:
1. Record training run metrics and parameters
2. Compare new models against baselines
3. Track model performance over time

Implementations:
- InMemoryExperimentTracker - testing/development
- FileExperimentTracker - local JSON files
- MLflowExperimentTracker - MLflow integration (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable


@dataclass
class BaselineMetrics:
    """Baseline metrics for a symbol used for comparison."""

    symbol: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    recorded_at: datetime
    model_version: str


@dataclass
class ComparisonResult:
    """Result of comparing a new model against baseline."""

    symbol: str
    roc_auc_improvement: float  # New - baseline (positive = better)
    pr_auc_improvement: float
    brier_improvement: float  # Negative = better (lower is better)
    decision: Literal["promote", "reject", "no_baseline"]
    reason: str
    meets_threshold: bool  # True if improvement meets promotion threshold

    @property
    def is_better(self) -> bool:
        """Check if new model is better than baseline."""
        return self.decision == "promote"


@dataclass
class TrainingRunRecord:
    """
    Record of a single training run.

    Captures all information needed to reproduce and audit the run.
    """

    run_id: str
    symbol: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float

    # Model config
    model_type: str
    cv_splits: int
    label_horizon: int
    embargo: int

    # Dataset info
    dataset_start: datetime
    dataset_end: datetime
    dataset_hash: str
    n_samples: int
    n_positive_top: int
    n_positive_bottom: int

    # Performance metrics
    roc_auc_top: float
    roc_auc_bottom: float
    pr_auc_top: float
    pr_auc_bottom: float
    brier_top: float
    brier_bottom: float

    # Cross-validation details
    cv_roc_auc_std: float
    cv_pr_auc_std: float

    # Comparison with baseline
    baseline_comparison: Optional[ComparisonResult] = None

    # Outcome
    was_promoted: bool = False
    model_path: Optional[str] = None

    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Additional context
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "model_type": self.model_type,
            "cv_splits": self.cv_splits,
            "label_horizon": self.label_horizon,
            "embargo": self.embargo,
            "dataset_start": self.dataset_start.isoformat(),
            "dataset_end": self.dataset_end.isoformat(),
            "dataset_hash": self.dataset_hash,
            "n_samples": self.n_samples,
            "n_positive_top": self.n_positive_top,
            "n_positive_bottom": self.n_positive_bottom,
            "roc_auc_top": self.roc_auc_top,
            "roc_auc_bottom": self.roc_auc_bottom,
            "pr_auc_top": self.pr_auc_top,
            "pr_auc_bottom": self.pr_auc_bottom,
            "brier_top": self.brier_top,
            "brier_bottom": self.brier_bottom,
            "cv_roc_auc_std": self.cv_roc_auc_std,
            "cv_pr_auc_std": self.cv_pr_auc_std,
            "baseline_comparison": (
                {
                    "symbol": self.baseline_comparison.symbol,
                    "roc_auc_improvement": self.baseline_comparison.roc_auc_improvement,
                    "pr_auc_improvement": self.baseline_comparison.pr_auc_improvement,
                    "brier_improvement": self.baseline_comparison.brier_improvement,
                    "decision": self.baseline_comparison.decision,
                    "reason": self.baseline_comparison.reason,
                    "meets_threshold": self.baseline_comparison.meets_threshold,
                }
                if self.baseline_comparison
                else None
            ),
            "was_promoted": self.was_promoted,
            "model_path": self.model_path,
            "feature_importance": self.feature_importance,
            "notes": self.notes,
        }


@runtime_checkable
class ExperimentTrackerPort(Protocol):
    """
    Port for experiment tracking and baseline comparison.

    Domain services use this interface to record training runs and
    compare models against baselines without knowing storage details.

    Usage:
        # In training service
        async def train(self, symbol: str) -> TrainingRunRecord:
            # ... train model ...
            baseline = await self._tracker.get_baseline(symbol)
            comparison = await self._tracker.compare_to_baseline(result)
            await self._tracker.record_run(record)
            return record
    """

    async def record_run(self, record: TrainingRunRecord) -> None:
        """
        Record a training run.

        Args:
            record: Training run record with all metrics and context.
        """
        ...

    async def get_baseline(self, symbol: str) -> Optional[BaselineMetrics]:
        """
        Get baseline metrics for a symbol.

        The baseline is the current active model's performance.

        Args:
            symbol: Trading symbol.

        Returns:
            BaselineMetrics or None if no baseline exists.
        """
        ...

    async def set_baseline(self, symbol: str, metrics: BaselineMetrics) -> None:
        """
        Set baseline metrics for a symbol.

        Called when a model is promoted to active.

        Args:
            symbol: Trading symbol.
            metrics: New baseline metrics.
        """
        ...

    async def compare_to_baseline(
        self,
        symbol: str,
        new_roc_auc: float,
        new_pr_auc: float,
        new_brier: float,
        improvement_threshold: float = 0.01,
    ) -> ComparisonResult:
        """
        Compare new model metrics against baseline.

        Args:
            symbol: Trading symbol.
            new_roc_auc: New model's ROC-AUC.
            new_pr_auc: New model's PR-AUC.
            new_brier: New model's Brier score.
            improvement_threshold: Minimum ROC-AUC improvement for promotion.

        Returns:
            ComparisonResult with decision and reasoning.
        """
        ...

    async def get_recent_runs(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[TrainingRunRecord]:
        """
        Get recent training runs.

        Args:
            symbol: Filter by symbol (None = all symbols).
            limit: Maximum number of runs to return.

        Returns:
            List of TrainingRunRecord, newest first.
        """
        ...

    async def get_run(self, run_id: str) -> Optional[TrainingRunRecord]:
        """
        Get a specific training run by ID.

        Args:
            run_id: Unique run identifier.

        Returns:
            TrainingRunRecord or None if not found.
        """
        ...


class ExperimentTrackerPortABC(ABC):
    """
    Abstract base class version of ExperimentTrackerPort.

    Use this for inheritance-based implementations.
    """

    @abstractmethod
    async def record_run(self, record: TrainingRunRecord) -> None:
        """Record a training run."""

    @abstractmethod
    async def get_baseline(self, symbol: str) -> Optional[BaselineMetrics]:
        """Get baseline metrics."""

    @abstractmethod
    async def set_baseline(self, symbol: str, metrics: BaselineMetrics) -> None:
        """Set baseline metrics."""

    @abstractmethod
    async def compare_to_baseline(
        self,
        symbol: str,
        new_roc_auc: float,
        new_pr_auc: float,
        new_brier: float,
        improvement_threshold: float = 0.01,
    ) -> ComparisonResult:
        """Compare to baseline."""

    @abstractmethod
    async def get_recent_runs(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[TrainingRunRecord]:
        """Get recent runs."""

    @abstractmethod
    async def get_run(self, run_id: str) -> Optional[TrainingRunRecord]:
        """Get run by ID."""
