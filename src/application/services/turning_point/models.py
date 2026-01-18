"""
Typed Models for Turning Point Training Service.

All training results use typed dataclasses instead of Dict[str, Any].
This improves:
- Type safety (mypy catches errors)
- IDE autocomplete
- Self-documenting code
- Serialization consistency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TrainingConfig:
    """
    Configuration for a model training run.

    Passed to TurningPointTrainingService.train() to control training behavior.
    """

    symbols: List[str]
    days: int = 750  # Days of historical data

    # Model configuration
    model_type: Literal["logistic", "lightgbm"] = "logistic"
    cv_splits: int = 5
    label_horizon: int = 10  # Bars to look ahead for labeling
    embargo: int = 2  # Embargo bars between train/test

    # Training behavior
    force_update: bool = False  # Update even if not better than baseline
    eval_only: bool = False  # Evaluate only, no model promotion

    # Feature extraction
    atr_period: int = 14
    zigzag_threshold: float = 2.0
    risk_threshold: float = 1.5

    # Parallelization
    max_workers: int = 2  # Parallel symbol training

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbols": self.symbols,
            "days": self.days,
            "model_type": self.model_type,
            "cv_splits": self.cv_splits,
            "label_horizon": self.label_horizon,
            "embargo": self.embargo,
            "force_update": self.force_update,
            "eval_only": self.eval_only,
            "atr_period": self.atr_period,
            "zigzag_threshold": self.zigzag_threshold,
            "risk_threshold": self.risk_threshold,
            "max_workers": self.max_workers,
        }


@dataclass
class SymbolTrainingResult:
    """
    Training result for a single symbol.

    Contains all metrics and feature importance from training.
    """

    symbol: str
    trained_at: datetime
    dataset_hash: str  # SHA256 of training data for reproducibility

    # Dataset info
    n_samples: int
    n_positive_top: int
    n_positive_bottom: int
    dataset_start: datetime
    dataset_end: datetime

    # TOP_RISK model metrics
    roc_auc_top: float
    roc_auc_top_std: float
    pr_auc_top: float
    pr_auc_top_std: float
    brier_top: float

    # BOTTOM_RISK model metrics
    roc_auc_bottom: float
    roc_auc_bottom_std: float
    pr_auc_bottom: float
    pr_auc_bottom_std: float
    brier_bottom: float

    # Feature importance (top 10 features)
    feature_importance_top: Dict[str, float] = field(default_factory=dict)
    feature_importance_bottom: Dict[str, float] = field(default_factory=dict)

    # Calibration metrics
    ece_top: Optional[float] = None  # Expected calibration error
    ece_bottom: Optional[float] = None

    # Training time
    training_seconds: float = 0.0

    @property
    def roc_auc_combined(self) -> float:
        """Average ROC-AUC across top and bottom models."""
        return (self.roc_auc_top + self.roc_auc_bottom) / 2

    @property
    def pr_auc_combined(self) -> float:
        """Average PR-AUC across top and bottom models."""
        return (self.pr_auc_top + self.pr_auc_bottom) / 2

    @property
    def brier_combined(self) -> float:
        """Average Brier score across top and bottom models."""
        return (self.brier_top + self.brier_bottom) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "trained_at": self.trained_at.isoformat(),
            "dataset_hash": self.dataset_hash,
            "n_samples": self.n_samples,
            "n_positive_top": self.n_positive_top,
            "n_positive_bottom": self.n_positive_bottom,
            "dataset_start": self.dataset_start.isoformat(),
            "dataset_end": self.dataset_end.isoformat(),
            "roc_auc_top": self.roc_auc_top,
            "roc_auc_top_std": self.roc_auc_top_std,
            "pr_auc_top": self.pr_auc_top,
            "pr_auc_top_std": self.pr_auc_top_std,
            "brier_top": self.brier_top,
            "roc_auc_bottom": self.roc_auc_bottom,
            "roc_auc_bottom_std": self.roc_auc_bottom_std,
            "pr_auc_bottom": self.pr_auc_bottom,
            "pr_auc_bottom_std": self.pr_auc_bottom_std,
            "brier_bottom": self.brier_bottom,
            "feature_importance_top": self.feature_importance_top,
            "feature_importance_bottom": self.feature_importance_bottom,
            "ece_top": self.ece_top,
            "ece_bottom": self.ece_bottom,
            "training_seconds": self.training_seconds,
        }


@dataclass
class ModelComparisonResult:
    """
    Result of comparing candidate model against baseline.

    Used to determine whether to promote the candidate.
    """

    symbol: str
    candidate_roc_auc: float
    baseline_roc_auc: Optional[float]

    # Improvement metrics
    improvement_pct: float  # Positive = better
    decision: Literal["promote", "reject", "no_baseline"]
    reason: str

    # Detailed metrics
    candidate_pr_auc: float
    baseline_pr_auc: Optional[float] = None
    candidate_brier: float = 0.0
    baseline_brier: Optional[float] = None

    @property
    def should_promote(self) -> bool:
        """Whether the candidate should be promoted."""
        return self.decision == "promote"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "candidate_roc_auc": self.candidate_roc_auc,
            "baseline_roc_auc": self.baseline_roc_auc,
            "improvement_pct": self.improvement_pct,
            "decision": self.decision,
            "reason": self.reason,
            "candidate_pr_auc": self.candidate_pr_auc,
            "baseline_pr_auc": self.baseline_pr_auc,
            "candidate_brier": self.candidate_brier,
            "baseline_brier": self.baseline_brier,
        }


@dataclass
class TrainingRunResult:
    """
    Complete result of a training run across multiple symbols.

    Returned by TurningPointTrainingService.train().
    """

    run_id: str
    started_at: datetime
    completed_at: datetime
    config: TrainingConfig

    # Per-symbol results
    results: Dict[str, SymbolTrainingResult] = field(default_factory=dict)
    comparisons: Dict[str, ModelComparisonResult] = field(default_factory=dict)

    # Summary lists
    promoted: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)  # symbol -> error message

    @property
    def duration_seconds(self) -> float:
        """Total run duration in seconds."""
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def success_count(self) -> int:
        """Number of successfully trained symbols."""
        return len(self.results)

    @property
    def total_count(self) -> int:
        """Total number of symbols attempted."""
        return len(self.config.symbols)

    @property
    def success_rate(self) -> float:
        """Fraction of symbols successfully trained."""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Training Run: {self.run_id}",
            f"Duration: {self.duration_seconds:.1f}s",
            f"Symbols: {self.success_count}/{self.total_count} successful",
            f"Promoted: {len(self.promoted)} ({', '.join(self.promoted) or 'none'})",
            f"Rejected: {len(self.rejected)} ({', '.join(self.rejected) or 'none'})",
        ]
        if self.failed:
            lines.append(f"Failed: {len(self.failed)} ({', '.join(self.failed)})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "config": self.config.to_dict(),
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "comparisons": {k: v.to_dict() for k, v in self.comparisons.items()},
            "promoted": self.promoted,
            "rejected": self.rejected,
            "failed": self.failed,
            "errors": self.errors,
            "summary": {
                "success_count": self.success_count,
                "total_count": self.total_count,
                "success_rate": self.success_rate,
            },
        }
