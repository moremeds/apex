"""
Multi-Timeframe Confirmation Analysis.

Compares strategies to measure whether multi-TF confirmation reduces
false positives without sacrificing precision.

Strategies:
- S1: Single TF only (e.g., 1d only)
- S2: Multi-TF confirmation (e.g., 1d AND 4h both signal)
- S3: Majority vote (e.g., 2 of 3 TFs signal)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .statistics import block_bootstrap_ci


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy."""

    strategy_name: str
    precision: float
    recall: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int

    @property
    def f1_score(self) -> float:
        """Compute F1 score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "precision": self.precision,
            "recall": self.recall,
            "false_positive_rate": self.false_positive_rate,
            "f1_score": self.f1_score,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "total_samples": self.total_samples,
        }


@dataclass
class ConfirmationResult:
    """Result comparing two strategies."""

    s1: StrategyMetrics
    s2: StrategyMetrics

    # CIs for precision and FP rate
    s1_ci_precision: Tuple[float, float]
    s1_ci_fp_rate: Tuple[float, float]
    s2_ci_precision: Tuple[float, float]
    s2_ci_fp_rate: Tuple[float, float]

    # Deltas
    delta_precision: float  # S2 - S1
    delta_fp_rate: float  # S2 - S1 (negative is better)

    # Overall assessment
    confirmation_value: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "strategy_comparison": {
                f"S1_{self.s1.strategy_name}": {
                    "precision": self.s1.precision,
                    "false_positive_rate": self.s1.false_positive_rate,
                    "ci_95_precision": list(self.s1_ci_precision),
                    "ci_95_fp_rate": list(self.s1_ci_fp_rate),
                },
                f"S2_{self.s2.strategy_name}": {
                    "precision": self.s2.precision,
                    "false_positive_rate": self.s2.false_positive_rate,
                    "ci_95_precision": list(self.s2_ci_precision),
                    "ci_95_fp_rate": list(self.s2_ci_fp_rate),
                },
                "delta_precision": self.delta_precision,
                "delta_fp_rate": self.delta_fp_rate,
                "confirmation_value": self.confirmation_value,
            }
        }

    def passes_gates(
        self,
        min_fp_reduction: float = 0.05,
        max_precision_drop: float = 0.02,
    ) -> Tuple[bool, List[str]]:
        """
        Check if confirmation result passes gates.

        Args:
            min_fp_reduction: Minimum FP rate reduction (S2 vs S1)
            max_precision_drop: Maximum precision drop allowed

        Returns:
            Tuple of (passes, list_of_failures)
        """
        failures = []

        # FP reduction: S2 - S1 should be negative (lower is better)
        if -self.delta_fp_rate < min_fp_reduction:
            failures.append(f"FP rate reduction ({-self.delta_fp_rate:.3f}) < {min_fp_reduction}")

        # Precision drop: S2 - S1, negative means drop
        if -self.delta_precision > max_precision_drop:
            failures.append(f"Precision drop ({-self.delta_precision:.3f}) > {max_precision_drop}")

        return len(failures) == 0, failures


def compute_strategy_metrics(
    predictions: List[bool],
    actuals: List[bool],
    strategy_name: str,
) -> StrategyMetrics:
    """
    Compute metrics for a single strategy.

    Args:
        predictions: Predicted positives (True = signal)
        actuals: Actual positives (True = actually trending)
        strategy_name: Name of the strategy

    Returns:
        StrategyMetrics with precision, recall, FP rate
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")

    n = len(predictions)
    if n == 0:
        return StrategyMetrics(
            strategy_name=strategy_name,
            precision=0.0,
            recall=0.0,
            false_positive_rate=0.0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            total_samples=0,
        )

    # Confusion matrix
    tp = sum(1 for p, a in zip(predictions, actuals) if p and a)
    fp = sum(1 for p, a in zip(predictions, actuals) if p and not a)
    tn = sum(1 for p, a in zip(predictions, actuals) if not p and not a)
    fn = sum(1 for p, a in zip(predictions, actuals) if not p and a)

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return StrategyMetrics(
        strategy_name=strategy_name,
        precision=precision,
        recall=recall,
        false_positive_rate=fp_rate,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        total_samples=n,
    )


def compare_strategies(
    s1_predictions: List[bool],
    s2_predictions: List[bool],
    actuals: List[bool],
    s1_name: str = "1d_only",
    s2_name: str = "1d_and_4h",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> ConfirmationResult:
    """
    Compare two strategies.

    Args:
        s1_predictions: Strategy 1 predictions
        s2_predictions: Strategy 2 predictions
        actuals: Actual labels
        s1_name: Name for strategy 1
        s2_name: Name for strategy 2
        n_bootstrap: Bootstrap samples for CIs
        seed: Random seed

    Returns:
        ConfirmationResult comparing the strategies
    """
    s1_metrics = compute_strategy_metrics(s1_predictions, actuals, s1_name)
    s2_metrics = compute_strategy_metrics(s2_predictions, actuals, s2_name)

    # Bootstrap CIs for precision and FP rate
    s1_precision_samples = _bootstrap_metric(
        s1_predictions, actuals, "precision", n_bootstrap, seed
    )
    s1_fp_samples = _bootstrap_metric(s1_predictions, actuals, "fp_rate", n_bootstrap, seed)
    s2_precision_samples = _bootstrap_metric(
        s2_predictions, actuals, "precision", n_bootstrap, seed + 1
    )
    s2_fp_samples = _bootstrap_metric(s2_predictions, actuals, "fp_rate", n_bootstrap, seed + 1)

    s1_ci_precision = _percentile_ci(s1_precision_samples)
    s1_ci_fp_rate = _percentile_ci(s1_fp_samples)
    s2_ci_precision = _percentile_ci(s2_precision_samples)
    s2_ci_fp_rate = _percentile_ci(s2_fp_samples)

    # Deltas
    delta_precision = s2_metrics.precision - s1_metrics.precision
    delta_fp_rate = s2_metrics.false_positive_rate - s1_metrics.false_positive_rate

    # Assess confirmation value
    # Positive: S2 has lower FP rate without much precision loss
    if delta_fp_rate < -0.03 and delta_precision > -0.05:
        confirmation_value = "POSITIVE"
    elif delta_fp_rate > 0.03 or delta_precision < -0.05:
        confirmation_value = "NEGATIVE"
    else:
        confirmation_value = "NEUTRAL"

    return ConfirmationResult(
        s1=s1_metrics,
        s2=s2_metrics,
        s1_ci_precision=s1_ci_precision,
        s1_ci_fp_rate=s1_ci_fp_rate,
        s2_ci_precision=s2_ci_precision,
        s2_ci_fp_rate=s2_ci_fp_rate,
        delta_precision=delta_precision,
        delta_fp_rate=delta_fp_rate,
        confirmation_value=confirmation_value,
    )


def _bootstrap_metric(
    predictions: List[bool],
    actuals: List[bool],
    metric: str,
    n_samples: int,
    seed: int,
) -> List[float]:
    """Bootstrap samples for a metric."""
    rng = np.random.default_rng(seed)
    n = len(predictions)
    samples = []

    preds_arr = np.array(predictions)
    acts_arr = np.array(actuals)

    for _ in range(n_samples):
        idx = rng.choice(n, size=n, replace=True)
        boot_preds = preds_arr[idx].tolist()
        boot_acts = acts_arr[idx].tolist()

        m = compute_strategy_metrics(boot_preds, boot_acts, "boot")
        if metric == "precision":
            samples.append(m.precision)
        elif metric == "fp_rate":
            samples.append(m.false_positive_rate)
        elif metric == "recall":
            samples.append(m.recall)

    return samples


def _percentile_ci(samples: List[float], level: float = 0.95) -> Tuple[float, float]:
    """Compute percentile CI from samples."""
    if not samples:
        return (0.0, 0.0)
    alpha = 1 - level
    lower = float(np.percentile(samples, alpha / 2 * 100))
    upper = float(np.percentile(samples, (1 - alpha / 2) * 100))
    return (lower, upper)


def apply_and_rule(
    signals_tf1: List[bool],
    signals_tf2: List[bool],
) -> List[bool]:
    """
    Apply AND rule: signal only when both TFs agree.

    Args:
        signals_tf1: Signals from TF1
        signals_tf2: Signals from TF2

    Returns:
        Combined signals (AND logic)
    """
    if len(signals_tf1) != len(signals_tf2):
        raise ValueError("Signal lists must have same length")
    return [a and b for a, b in zip(signals_tf1, signals_tf2)]


def apply_majority_vote(
    signals_by_tf: List[List[bool]],
    min_agree: Optional[int] = None,
) -> List[bool]:
    """
    Apply majority vote rule.

    Args:
        signals_by_tf: List of signal lists for each TF
        min_agree: Minimum TFs that must agree (default: majority)

    Returns:
        Combined signals (majority vote)
    """
    if not signals_by_tf:
        return []

    n_samples = len(signals_by_tf[0])
    n_tfs = len(signals_by_tf)

    if any(len(s) != n_samples for s in signals_by_tf):
        raise ValueError("All signal lists must have same length")

    if min_agree is None:
        min_agree = (n_tfs // 2) + 1  # Simple majority

    result = []
    for i in range(n_samples):
        votes = sum(1 for signals in signals_by_tf if signals[i])
        result.append(votes >= min_agree)

    return result
