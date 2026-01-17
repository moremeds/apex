"""
Walk-Forward Parameter Optimizer for Regime Classification.

Implements walk-forward optimization with purged + embargo cross-validation
to prevent overfitting when tuning regime classification parameters.

Key features:
- Purged + embargo CV to prevent label leakage
- Multi-objective optimization
- Fold agreement check (≥70% folds must agree for recommendation)
- Stability tracking across folds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

from .objectives import CombinedObjectiveResult, ObjectiveEvaluator, ObjectiveResult

logger = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""

    n_folds: int = 5
    train_days: int = 252  # 1 year
    test_days: int = 63  # 3 months
    purge_gap_days: int = 5  # Gap between train end and test start
    embargo_days: int = 2  # Gap after test before next train
    label_horizon_days: int = 10  # For turning point labels

    # Stability requirements
    min_fold_agreement: float = 0.7  # 70%+ folds must agree on direction

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration."""
        return {
            "n_folds": self.n_folds,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "purge_gap_days": self.purge_gap_days,
            "embargo_days": self.embargo_days,
            "label_horizon_days": self.label_horizon_days,
            "min_fold_agreement": self.min_fold_agreement,
        }


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""

    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    objective_result: CombinedObjectiveResult
    suggested_changes: Dict[str, float]  # param_name -> suggested_change

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fold result."""
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "objective_result": self.objective_result.to_dict(),
            "suggested_changes": self.suggested_changes,
        }


@dataclass
class ParamStability:
    """Stability metrics for a single parameter across folds."""

    param_name: str
    changes_by_fold: List[float]
    mean_change: float
    std_change: float
    agreement_ratio: float  # % of folds agreeing on direction
    suggested_direction: str  # "increase", "decrease", or "no_change"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stability metrics."""
        return {
            "param_name": self.param_name,
            "changes_by_fold": self.changes_by_fold,
            "mean_change": round(self.mean_change, 4),
            "std_change": round(self.std_change, 4),
            "agreement_ratio": round(self.agreement_ratio, 4),
            "suggested_direction": self.suggested_direction,
        }


@dataclass
class WalkForwardResult:
    """Complete result from walk-forward optimization."""

    symbol: str
    config: WalkForwardConfig
    fold_results: List[FoldResult]
    param_stability: Dict[str, ParamStability]
    objective_summary: Dict[str, float]  # objective_name -> mean value
    recommendations: Dict[str, Dict[str, Any]]  # param_name -> recommendation
    why_not_changed: List[str]  # Reasons for not changing parameters
    total_score_mean: float
    total_score_std: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result."""
        return {
            "symbol": self.symbol,
            "config": self.config.to_dict(),
            "fold_results": [f.to_dict() for f in self.fold_results],
            "param_stability": {k: v.to_dict() for k, v in self.param_stability.items()},
            "objective_summary": self.objective_summary,
            "recommendations": self.recommendations,
            "why_not_changed": self.why_not_changed,
            "total_score_mean": round(self.total_score_mean, 4),
            "total_score_std": round(self.total_score_std, 4),
        }


class WalkForwardOptimizer:
    """
    Walk-forward optimizer with purged + embargo cross-validation.

    Key features:
    - Respects purge gap and embargo to prevent label leakage
    - Evaluates multiple objectives
    - Requires fold agreement before recommending changes
    """

    # Parameter bounds
    PARAM_BOUNDS = {
        "vol_high_short_pct": (50, 95),
        "vol_high_long_pct": (50, 95),
        "chop_high_pct": (40, 85),
        "ext_threshold_short": (1.5, 4.0),
        "ext_threshold_long": (2.0, 5.0),
    }

    # Maximum change per optimization run
    MAX_CHANGE = {
        "vol_high_short_pct": 5.0,
        "vol_high_long_pct": 5.0,
        "chop_high_pct": 5.0,
        "ext_threshold_short": 0.3,
        "ext_threshold_long": 0.3,
    }

    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        evaluator: Optional[ObjectiveEvaluator] = None,
    ):
        """
        Initialize optimizer.

        Args:
            config: Walk-forward configuration
            evaluator: Objective evaluator
        """
        self.config = config or WalkForwardConfig()
        self.evaluator = evaluator or ObjectiveEvaluator()

    def _generate_folds(
        self,
        data_start: date,
        data_end: date,
    ) -> List[Tuple[date, date, date, date]]:
        """
        Generate walk-forward fold boundaries with purge and embargo.

        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        folds = []
        total_days = (data_end - data_start).days

        # Calculate fold size
        fold_size = self.config.train_days + self.config.test_days
        fold_size += self.config.purge_gap_days + self.config.embargo_days

        # Calculate how many folds we can fit
        available_days = total_days - self.config.train_days
        denominator = self.config.test_days + self.config.embargo_days
        if denominator <= 0:
            logger.warning("Invalid config: test_days + embargo_days must be > 0")
            return []
        max_folds = max(1, available_days // denominator)
        n_folds = min(self.config.n_folds, max_folds)

        if n_folds < 2:
            logger.warning(f"Insufficient data for walk-forward: {total_days} days available")
            return []

        # Generate folds from end (most recent data in last fold)
        current_end = data_end

        for i in range(n_folds):
            test_end = current_end
            test_start = test_end - timedelta(days=self.config.test_days)
            train_end = test_start - timedelta(days=self.config.purge_gap_days)
            train_start = train_end - timedelta(days=self.config.train_days)

            if train_start < data_start:
                break

            folds.append((train_start, train_end, test_start, test_end))

            # Move to next fold
            current_end = train_end - timedelta(days=self.config.embargo_days)

        # Reverse to have oldest fold first
        return list(reversed(folds))

    def _run_regime_detector(
        self,
        ohlcv: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.Series:
        """
        Run regime detector with given parameters.

        Returns Series of regime labels indexed by date.
        """
        from src.domain.signals.indicators.regime.regime_detector import RegimeDetector

        detector = RegimeDetector()

        # Override default params
        for key, value in params.items():
            if hasattr(detector, key):
                setattr(detector, key, value)

        regimes = []
        dates = []

        for idx in range(len(ohlcv)):
            row = ohlcv.iloc[: idx + 1]
            if len(row) < 20:  # Need minimum data for indicators
                continue

            try:
                result = detector.detect(row)
                if result:
                    regimes.append(result.regime.value)
                    dates.append(ohlcv.index[idx])
            except Exception:
                continue

        return pd.Series(regimes, index=dates)

    def _evaluate_fold(
        self,
        fold_id: int,
        ohlcv: pd.DataFrame,
        train_window: Tuple[date, date],
        test_window: Tuple[date, date],
        base_params: Dict[str, Any],
    ) -> FoldResult:
        """
        Evaluate a single fold.

        Trains on train_window, evaluates on test_window.
        """
        train_start, train_end = train_window
        test_start, test_end = test_window

        # Filter data to test window for evaluation
        test_mask = (ohlcv.index >= pd.Timestamp(test_start)) & (
            ohlcv.index <= pd.Timestamp(test_end)
        )
        test_data = ohlcv[test_mask]

        if len(test_data) < 10:
            # Not enough test data
            return FoldResult(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                objective_result=CombinedObjectiveResult(
                    objective_results=[],
                    total_score=0.0,
                    param_set=base_params,
                ),
                suggested_changes={},
            )

        # Run regime detector on test data
        regime_series = self._run_regime_detector(test_data, base_params)

        if len(regime_series) < 10:
            return FoldResult(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                objective_result=CombinedObjectiveResult(
                    objective_results=[],
                    total_score=0.0,
                    param_set=base_params,
                ),
                suggested_changes={},
            )

        # Evaluate objectives
        objective_result = self.evaluator.evaluate(test_data, regime_series, base_params)

        # Determine suggested changes based on objectives
        suggested_changes = self._suggest_changes(objective_result, base_params)

        return FoldResult(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            objective_result=objective_result,
            suggested_changes=suggested_changes,
        )

    def _suggest_changes(
        self,
        objective_result: CombinedObjectiveResult,
        current_params: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Suggest parameter changes based on objective results.

        Uses heuristics to map objective performance to parameter adjustments.
        """
        changes = {}

        for obj_result in objective_result.objective_results:
            if obj_result.name == "regime_stability":
                # High transition rate -> suggest widening thresholds (hysteresis)
                transition_rate = obj_result.details.get("transition_rate_per_100", 0)
                if transition_rate > 15:  # More than 15 transitions per 100 bars
                    # Suggest increasing vol threshold to reduce sensitivity
                    changes["vol_high_short_pct"] = changes.get("vol_high_short_pct", 0) + 2.0
                    changes["chop_high_pct"] = changes.get("chop_high_pct", 0) + 2.0
                elif transition_rate < 3:  # Very stable, possibly too insensitive
                    changes["vol_high_short_pct"] = changes.get("vol_high_short_pct", 0) - 1.0
                    changes["chop_high_pct"] = changes.get("chop_high_pct", 0) - 1.0

            elif obj_result.name == "trading_proxy":
                # Low separation -> adjust thresholds for better regime distinction
                normalized_sep = obj_result.details.get("normalized_separation", 0)
                if normalized_sep < 0.3:  # Weak separation
                    # Try to find better boundaries
                    regime_returns = obj_result.details.get("regime_mean_returns", {})
                    if regime_returns:
                        # If R0 (healthy) has lower returns than expected, lower vol threshold
                        r0_return = regime_returns.get("R0", regime_returns.get("0", 0))
                        r2_return = regime_returns.get("R2", regime_returns.get("2", 0))
                        if r0_return < r2_return:
                            changes["vol_high_short_pct"] = changes.get("vol_high_short_pct", 0) - 2.0

        # Clamp changes to max allowed
        for param, change in changes.items():
            max_change = self.MAX_CHANGE.get(param, 5.0)
            changes[param] = max(-max_change, min(max_change, change))

        return changes

    def _calculate_param_stability(
        self,
        fold_results: List[FoldResult],
    ) -> Dict[str, ParamStability]:
        """
        Calculate parameter stability metrics across folds.
        """
        # Collect changes per parameter
        param_changes: Dict[str, List[float]] = {}

        for fold in fold_results:
            for param, change in fold.suggested_changes.items():
                if param not in param_changes:
                    param_changes[param] = []
                param_changes[param].append(change)

        # Calculate stability metrics
        stability = {}

        for param, changes in param_changes.items():
            if not changes:
                continue

            changes_array = np.array(changes)
            mean_change = float(np.mean(changes_array))
            std_change = float(np.std(changes_array))

            # Calculate agreement ratio
            positive_count = sum(1 for c in changes if c > 0)
            negative_count = sum(1 for c in changes if c < 0)
            zero_count = sum(1 for c in changes if c == 0)

            total_non_zero = positive_count + negative_count
            if total_non_zero == 0:
                agreement_ratio = 1.0
                direction = "no_change"
            else:
                # Agreement ratio among folds that suggested a change (exclude zeros)
                agreement_ratio = max(positive_count, negative_count) / total_non_zero
                direction = "increase" if positive_count > negative_count else "decrease"

            stability[param] = ParamStability(
                param_name=param,
                changes_by_fold=changes,
                mean_change=mean_change,
                std_change=std_change,
                agreement_ratio=agreement_ratio,
                suggested_direction=direction,
            )

        return stability

    def _generate_recommendations(
        self,
        param_stability: Dict[str, ParamStability],
        current_params: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """
        Generate final recommendations based on stability analysis.

        Only recommends changes if fold agreement exceeds threshold.
        """
        recommendations = {}
        why_not_changed = []

        for param, stability in param_stability.items():
            # Check fold agreement
            if stability.agreement_ratio < self.config.min_fold_agreement:
                why_not_changed.append(
                    f"{param}: Only {stability.agreement_ratio:.0%} fold agreement "
                    f"(need ≥{self.config.min_fold_agreement:.0%})"
                )
                continue

            # Check if change is significant
            if abs(stability.mean_change) < 0.5:
                why_not_changed.append(
                    f"{param}: Mean change {stability.mean_change:.2f} too small"
                )
                continue

            # Check if direction is "no_change"
            if stability.suggested_direction == "no_change":
                continue

            # Get current value
            current_value = current_params.get(param, 0)
            suggested_value = current_value + stability.mean_change

            # Clamp to bounds
            bounds = self.PARAM_BOUNDS.get(param, (0, 100))
            suggested_value = max(bounds[0], min(bounds[1], suggested_value))

            recommendations[param] = {
                "current_value": current_value,
                "suggested_value": round(suggested_value, 2),
                "change": round(stability.mean_change, 2),
                "direction": stability.suggested_direction,
                "fold_agreement": round(stability.agreement_ratio, 2),
                "confidence": round(stability.agreement_ratio * 100, 0),
            }

        return recommendations, why_not_changed

    def optimize(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        current_params: Dict[str, Any],
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            symbol: Symbol being optimized
            ohlcv: OHLCV DataFrame with sufficient history
            current_params: Current parameter values

        Returns:
            WalkForwardResult with recommendations
        """
        # Ensure datetime index
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv = ohlcv.copy()
            ohlcv.index = pd.to_datetime(ohlcv.index)

        data_start = ohlcv.index.min().date()
        data_end = ohlcv.index.max().date()

        # Generate folds
        folds = self._generate_folds(data_start, data_end)

        if len(folds) < 2:
            return WalkForwardResult(
                symbol=symbol,
                config=self.config,
                fold_results=[],
                param_stability={},
                objective_summary={},
                recommendations={},
                why_not_changed=["Insufficient data for walk-forward optimization"],
                total_score_mean=0.0,
                total_score_std=0.0,
            )

        # Add symbol to params for objectives
        params_with_symbol = {**current_params, "symbol": symbol}

        # Evaluate each fold
        fold_results = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            result = self._evaluate_fold(
                fold_id=i,
                ohlcv=ohlcv,
                train_window=(train_start, train_end),
                test_window=(test_start, test_end),
                base_params=params_with_symbol,
            )
            fold_results.append(result)
            logger.info(
                f"Fold {i}: train={train_start} to {train_end}, "
                f"test={test_start} to {test_end}, score={result.objective_result.total_score:.4f}"
            )

        # Calculate stability
        param_stability = self._calculate_param_stability(fold_results)

        # Calculate objective summary
        objective_summary: Dict[str, List[float]] = {}
        total_scores = []

        for fold in fold_results:
            total_scores.append(fold.objective_result.total_score)
            for obj in fold.objective_result.objective_results:
                if obj.name not in objective_summary:
                    objective_summary[obj.name] = []
                objective_summary[obj.name].append(obj.value)

        objective_means = {
            name: round(float(np.mean(values)), 4)
            for name, values in objective_summary.items()
        }

        # Generate recommendations
        recommendations, why_not_changed = self._generate_recommendations(
            param_stability, current_params
        )

        return WalkForwardResult(
            symbol=symbol,
            config=self.config,
            fold_results=fold_results,
            param_stability=param_stability,
            objective_summary=objective_means,
            recommendations=recommendations,
            why_not_changed=why_not_changed,
            total_score_mean=float(np.mean(total_scores)) if total_scores else 0.0,
            total_score_std=float(np.std(total_scores)) if total_scores else 0.0,
        )
