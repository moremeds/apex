"""
Objective Functions for Regime Parameter Optimization.

Defines three objective functions for walk-forward parameter optimization:
1. regime_stability: Minimize transition rate (fewer regime switches)
2. turning_point_quality: Maximize PR-AUC for turning point detection
3. trading_proxy: Maximize forward return separation by regime bucket

Each objective is independent and can be weighted differently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ObjectiveResult:
    """Result from evaluating a single objective function."""

    name: str
    value: float
    direction: str  # "minimize" or "maximize"
    weight: float
    details: Dict[str, Any]

    @property
    def weighted_value(self) -> float:
        """Value adjusted by weight, normalized for direction."""
        # For maximization objectives, higher is better
        # For minimization objectives, lower is better
        # We normalize by negating minimization objectives
        if self.direction == "minimize":
            return -self.value * self.weight
        return self.value * self.weight

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "direction": self.direction,
            "weight": self.weight,
            "weighted_value": self.weighted_value,
            "details": self.details,
        }


class ObjectiveFunction(ABC):
    """Base class for objective functions."""

    def __init__(self, name: str, direction: str, weight: float = 1.0):
        self.name = name
        self.direction = direction  # "minimize" or "maximize"
        self.weight = weight

    @abstractmethod
    def evaluate(
        self,
        ohlcv: pd.DataFrame,
        regime_series: pd.Series,
        params: Dict[str, Any],
    ) -> ObjectiveResult:
        """
        Evaluate the objective function.

        Args:
            ohlcv: OHLCV DataFrame with price data
            regime_series: Series of regime labels (R0, R1, R2, R3) indexed by date
            params: Current parameter values being evaluated

        Returns:
            ObjectiveResult with value and details
        """


class RegimeStabilityObjective(ObjectiveFunction):
    """
    Minimize regime transition rate.

    Lower transition rate means more stable regime classification.
    Target: Fewer switches = more actionable signals.
    """

    def __init__(self, weight: float = 0.4):
        super().__init__(
            name="regime_stability",
            direction="minimize",
            weight=weight,
        )

    def evaluate(
        self,
        ohlcv: pd.DataFrame,
        regime_series: pd.Series,
        params: Dict[str, Any],
    ) -> ObjectiveResult:
        """Calculate transition rate per 100 bars."""
        if len(regime_series) < 2:
            return ObjectiveResult(
                name=self.name,
                value=0.0,
                direction=self.direction,
                weight=self.weight,
                details={"error": "insufficient_data", "n_bars": len(regime_series)},
            )

        # Count regime transitions
        transitions = (regime_series != regime_series.shift(1)).sum() - 1
        n_bars = len(regime_series)

        # Transition rate per 100 bars
        transition_rate = (transitions / n_bars) * 100

        # Regime distribution
        regime_counts = regime_series.value_counts().to_dict()

        return ObjectiveResult(
            name=self.name,
            value=transition_rate,
            direction=self.direction,
            weight=self.weight,
            details={
                "transitions": int(transitions),
                "n_bars": n_bars,
                "transition_rate_per_100": round(transition_rate, 2),
                "regime_distribution": {str(k): int(v) for k, v in regime_counts.items()},
            },
        )


class TurningPointQualityObjective(ObjectiveFunction):
    """
    Maximize turning point detection quality.

    Uses PR-AUC as the primary metric (more informative for rare events).
    Falls back to ROC-AUC if PR-AUC unavailable.
    """

    def __init__(self, weight: float = 0.3):
        super().__init__(
            name="turning_point_quality",
            direction="maximize",
            weight=weight,
        )

    def evaluate(
        self,
        ohlcv: pd.DataFrame,
        regime_series: pd.Series,
        params: Dict[str, Any],
    ) -> ObjectiveResult:
        """
        Evaluate turning point detection quality.

        This requires the turning point model to be trained and predictions available.
        If not available, returns a neutral score.
        """
        # Try to load experiment results for turning point quality
        import json
        from pathlib import Path

        symbol = params.get("symbol", "SPY")
        exp_path = Path(f"experiments/turning_point/{symbol.lower()}_latest.json")

        if not exp_path.exists():
            return ObjectiveResult(
                name=self.name,
                value=0.5,  # Neutral score
                direction=self.direction,
                weight=self.weight,
                details={"error": "no_experiment_results", "symbol": symbol},
            )

        try:
            with open(exp_path) as f:
                exp_data = json.load(f)

            # Use median PR-AUC for top detection (primary metric)
            top_pr_auc = exp_data.get("median_top_pr_auc", 0.5)
            bottom_pr_auc = exp_data.get("median_bottom_pr_auc", 0.5)

            # Combined score (weighted average, top detection slightly more important)
            combined_score = 0.6 * top_pr_auc + 0.4 * bottom_pr_auc

            return ObjectiveResult(
                name=self.name,
                value=combined_score,
                direction=self.direction,
                weight=self.weight,
                details={
                    "top_pr_auc": round(top_pr_auc, 4),
                    "bottom_pr_auc": round(bottom_pr_auc, 4),
                    "combined_score": round(combined_score, 4),
                    "experiment_id": exp_data.get("experiment_id", "unknown"),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to load turning point experiment: {e}")
            return ObjectiveResult(
                name=self.name,
                value=0.5,
                direction=self.direction,
                weight=self.weight,
                details={"error": str(e)},
            )


class TradingProxyObjective(ObjectiveFunction):
    """
    Maximize forward return separation by regime bucket.

    Measures how well regime classification predicts forward returns.
    Higher separation = better predictive power for trading decisions.
    """

    def __init__(self, weight: float = 0.3, forward_bars: int = 5):
        super().__init__(
            name="trading_proxy",
            direction="maximize",
            weight=weight,
        )
        self.forward_bars = forward_bars

    def evaluate(
        self,
        ohlcv: pd.DataFrame,
        regime_series: pd.Series,
        params: Dict[str, Any],
    ) -> ObjectiveResult:
        """
        Calculate forward return separation by regime.

        Measures the spread between best and worst regime buckets
        in terms of forward returns.
        """
        if len(ohlcv) < self.forward_bars + 10:
            return ObjectiveResult(
                name=self.name,
                value=0.0,
                direction=self.direction,
                weight=self.weight,
                details={"error": "insufficient_data"},
            )

        # Calculate forward returns
        returns = ohlcv["close"].pct_change(self.forward_bars).shift(-self.forward_bars)

        # Align with regime series
        aligned = pd.DataFrame({"regime": regime_series, "fwd_return": returns}).dropna()

        if len(aligned) < 20:
            return ObjectiveResult(
                name=self.name,
                value=0.0,
                direction=self.direction,
                weight=self.weight,
                details={"error": "insufficient_aligned_data", "n_aligned": len(aligned)},
            )

        # Calculate mean forward return per regime
        regime_returns = aligned.groupby("regime")["fwd_return"].agg(["mean", "std", "count"])

        # Separation score: range of mean returns across regimes
        # Higher range = better separation
        mean_returns = regime_returns["mean"]
        separation = mean_returns.max() - mean_returns.min()

        # Normalize by overall return volatility to make it comparable across assets
        overall_std = aligned["fwd_return"].std()
        if overall_std > 0:
            normalized_separation = separation / overall_std
        else:
            normalized_separation = 0.0

        return ObjectiveResult(
            name=self.name,
            value=normalized_separation,
            direction=self.direction,
            weight=self.weight,
            details={
                "separation": round(separation, 6),
                "normalized_separation": round(normalized_separation, 4),
                "regime_mean_returns": {
                    str(k): round(v, 6) for k, v in mean_returns.to_dict().items()
                },
                "regime_counts": {
                    str(k): int(v) for k, v in regime_returns["count"].to_dict().items()
                },
                "forward_bars": self.forward_bars,
            },
        )


@dataclass
class CombinedObjectiveResult:
    """Combined result from all objective functions."""

    objective_results: List[ObjectiveResult]
    total_score: float
    param_set: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "objective_results": [r.to_dict() for r in self.objective_results],
            "total_score": self.total_score,
            "param_set": self.param_set,
        }


class ObjectiveEvaluator:
    """
    Evaluates multiple objective functions and combines results.
    """

    def __init__(
        self,
        objectives: Optional[List[ObjectiveFunction]] = None,
    ):
        """
        Initialize evaluator with objective functions.

        Args:
            objectives: List of objective functions. If None, uses default set.
        """
        self.objectives = objectives or [
            RegimeStabilityObjective(weight=0.4),
            TurningPointQualityObjective(weight=0.3),
            TradingProxyObjective(weight=0.3),
        ]

    def evaluate(
        self,
        ohlcv: pd.DataFrame,
        regime_series: pd.Series,
        params: Dict[str, Any],
    ) -> CombinedObjectiveResult:
        """
        Evaluate all objectives and combine into single score.

        Args:
            ohlcv: OHLCV DataFrame
            regime_series: Series of regime labels
            params: Parameter values being evaluated

        Returns:
            CombinedObjectiveResult with all objective values and total score
        """
        results = []
        for obj in self.objectives:
            result = obj.evaluate(ohlcv, regime_series, params)
            results.append(result)

        # Combine weighted values
        total_score = sum(r.weighted_value for r in results)

        return CombinedObjectiveResult(
            objective_results=results,
            total_score=total_score,
            param_set=params,
        )

    def get_objective_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all objectives and their weights."""
        return {
            obj.name: {
                "direction": obj.direction,
                "weight": obj.weight,
            }
            for obj in self.objectives
        }
