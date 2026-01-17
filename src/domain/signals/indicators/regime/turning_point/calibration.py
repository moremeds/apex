"""
Probability Calibration Evidence (Phase 4).

Provides calibration evidence for turning point predictions:
1. Reliability diagram / calibration curve with raw bucket statistics
2. Brier score as supplementary metric

Problem: Brier score alone can mislead - lower Brier doesn't guarantee
better calibration (see sklearn docs).

Solution: Two-part calibration evidence with full bucket transparency.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationEvidence:
    """
    Evidence of probability calibration quality.

    Contains raw bucket statistics for reliability diagram plus
    summary metrics. This allows users to judge calibration quality
    without relying solely on aggregate scores.
    """

    # Reliability diagram buckets (typically 10 bins)
    bucket_edges: List[float] = field(default_factory=list)  # [0.0, 0.1, ..., 1.0]
    bucket_counts: List[int] = field(default_factory=list)  # Samples per bucket
    bucket_mean_predicted: List[float] = field(default_factory=list)  # Mean predicted prob
    bucket_mean_actual: List[float] = field(default_factory=list)  # Mean actual label (0/1)

    # Summary metrics
    brier_score: float = 1.0  # Lower is better, 0 = perfect
    calibration_error: float = 1.0  # Mean |predicted - actual| across buckets
    expected_calibration_error: float = 1.0  # Weighted by bucket count

    # Auxiliary metrics
    n_samples: int = 0
    event_rate: float = 0.0  # Base rate of positive class

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bucket_edges": self.bucket_edges,
            "bucket_counts": self.bucket_counts,
            "bucket_mean_predicted": self.bucket_mean_predicted,
            "bucket_mean_actual": self.bucket_mean_actual,
            "brier_score": self.brier_score,
            "calibration_error": self.calibration_error,
            "expected_calibration_error": self.expected_calibration_error,
            "n_samples": self.n_samples,
            "event_rate": self.event_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationEvidence":
        """Deserialize from dictionary."""
        return cls(
            bucket_edges=data.get("bucket_edges", []),
            bucket_counts=data.get("bucket_counts", []),
            bucket_mean_predicted=data.get("bucket_mean_predicted", []),
            bucket_mean_actual=data.get("bucket_mean_actual", []),
            brier_score=data.get("brier_score", 1.0),
            calibration_error=data.get("calibration_error", 1.0),
            expected_calibration_error=data.get("expected_calibration_error", 1.0),
            n_samples=data.get("n_samples", 0),
            event_rate=data.get("event_rate", 0.0),
        )

    def to_reliability_diagram_data(self) -> Dict[str, Any]:
        """
        Export data for plotting reliability diagram.

        Returns:
            Dictionary with x (mean predicted), y (mean actual),
            sizes (bucket counts), and reference line data.
        """
        return {
            "x_predicted": self.bucket_mean_predicted,
            "y_actual": self.bucket_mean_actual,
            "sizes": self.bucket_counts,
            "reference_line": {"x": [0, 1], "y": [0, 1]},  # Perfect calibration
            "metrics": {
                "brier_score": self.brier_score,
                "calibration_error": self.calibration_error,
            },
        }

    def is_well_calibrated(self, threshold: float = 0.1) -> bool:
        """
        Check if model is reasonably well-calibrated.

        Args:
            threshold: Maximum acceptable calibration error

        Returns:
            True if calibration error is below threshold
        """
        return self.expected_calibration_error < threshold


def compute_calibration_evidence(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationEvidence:
    """
    Compute calibration evidence from predictions.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities [0, 1]
        n_bins: Number of bins for reliability diagram
        strategy: Binning strategy - "uniform" (equal width) or "quantile" (equal count)

    Returns:
        CalibrationEvidence with bucket statistics and metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    # Clip probabilities to valid range
    y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)

    n_samples = len(y_true)
    if n_samples == 0:
        return CalibrationEvidence()

    # Event rate
    event_rate = np.mean(y_true)

    # Brier score: mean squared error of probability predictions
    brier_score = np.mean((y_pred_proba - y_true) ** 2)

    # Create bins
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        # Equal count bins based on prediction distribution
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(y_pred_proba, percentiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)  # Remove duplicates
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    bucket_edges = bin_edges.tolist()
    bucket_counts = []
    bucket_mean_predicted = []
    bucket_mean_actual = []

    # Compute bucket statistics
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]

        # Include right edge for last bucket
        if i == len(bin_edges) - 2:
            mask = (y_pred_proba >= low) & (y_pred_proba <= high)
        else:
            mask = (y_pred_proba >= low) & (y_pred_proba < high)

        count = np.sum(mask)
        bucket_counts.append(int(count))

        if count > 0:
            mean_pred = np.mean(y_pred_proba[mask])
            mean_actual = np.mean(y_true[mask])
        else:
            mean_pred = (low + high) / 2  # Bin center
            mean_actual = 0.0

        bucket_mean_predicted.append(float(mean_pred))
        bucket_mean_actual.append(float(mean_actual))

    # Calibration error: mean absolute difference per bucket
    cal_errors = [
        abs(p - a) for p, a in zip(bucket_mean_predicted, bucket_mean_actual)
    ]
    calibration_error = np.mean(cal_errors) if cal_errors else 1.0

    # Expected calibration error: weighted by bucket count
    if sum(bucket_counts) > 0:
        ece = sum(
            c * abs(p - a)
            for c, p, a in zip(bucket_counts, bucket_mean_predicted, bucket_mean_actual)
        ) / sum(bucket_counts)
    else:
        ece = 1.0

    return CalibrationEvidence(
        bucket_edges=bucket_edges,
        bucket_counts=bucket_counts,
        bucket_mean_predicted=bucket_mean_predicted,
        bucket_mean_actual=bucket_mean_actual,
        brier_score=float(brier_score),
        calibration_error=float(calibration_error),
        expected_calibration_error=float(ece),
        n_samples=n_samples,
        event_rate=float(event_rate),
    )


def compute_roc_and_pr_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute ROC-AUC and PR-AUC.

    PR-AUC is more informative for sparse events (imbalanced classes).
    Baseline for random classifier = event_rate.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        Tuple of (roc_auc, pr_auc)
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        # Only one class present
        return 0.5, float(np.mean(y_true))

    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_pred_proba)
    except ValueError:
        pr_auc = float(np.mean(y_true))

    return float(roc_auc), float(pr_auc)
