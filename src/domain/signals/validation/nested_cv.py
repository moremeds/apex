"""
Nested Walk-Forward Cross-Validation Framework.

Implements strict separation between optimization and evaluation:
- Outer CV: Evaluation ONLY (no parameter access)
- Inner CV: Optimization (Optuna sees these results)

Key invariant: Optuna ONLY sees inner CV results. Outer test is evaluation-only.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from .statistics import SymbolMetrics
from .time_units import ValidationTimeConfig


@dataclass(frozen=True)
class TimeWindow:
    """Time window for train/test splits."""

    start_date: date
    end_date: date

    @property
    def days(self) -> int:
        """Number of calendar days in window."""
        return (self.end_date - self.start_date).days


@dataclass
class OuterFold:
    """Outer fold definition."""

    fold_id: int
    train_window: TimeWindow
    test_window: TimeWindow


@dataclass
class OuterFoldResult:
    """Result from evaluating one outer fold."""

    fold_id: int
    best_params: Dict[str, Any]
    symbol_metrics: List[SymbolMetrics]
    inner_cv_score: float = 0.0  # Best score from inner CV


@dataclass
class NestedCVResult:
    """Complete nested CV result."""

    outer_results: List[OuterFoldResult]
    aggregated_metrics: List[SymbolMetrics]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "n_outer_folds": len(self.outer_results),
            "outer_folds": [
                {
                    "fold_id": r.fold_id,
                    "best_params": r.best_params,
                    "inner_cv_score": r.inner_cv_score,
                    "n_symbol_metrics": len(r.symbol_metrics),
                }
                for r in self.outer_results
            ],
            "n_aggregated_symbols": len(self.aggregated_metrics),
        }


class RegimeDetectorProtocol(Protocol):
    """Protocol for regime detector."""

    def classify(self, symbol: str, data: Any) -> str:
        """Classify regime for a symbol."""
        ...


@dataclass
class NestedCVConfig:
    """Configuration for nested cross-validation."""

    # Outer CV (evaluation)
    outer_folds: int = 5
    outer_train_pct: float = 0.7

    # Inner CV (optimization) - runs inside each outer_train
    inner_folds: int = 3
    inner_max_trials: int = 20  # Optuna trials per inner CV

    # Time config (bars-based)
    time_config: ValidationTimeConfig = field(
        default_factory=lambda: ValidationTimeConfig.from_days("1d", 20, 5, 3)
    )

    # Frozen labeler config version
    labeler_version: str = "v1.0"

    def validate(self) -> None:
        """Validate config parameters."""
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least 2")
        if self.inner_folds < 2:
            raise ValueError("inner_folds must be at least 2")
        if not 0.5 <= self.outer_train_pct <= 0.9:
            raise ValueError("outer_train_pct must be between 0.5 and 0.9")


class NestedWalkForwardCV:
    """
    Nested walk-forward cross-validation.

    Outer loop: Evaluation only (no parameter access)
    Inner loop: Optimization (Optuna sees these results)

    Key design principle: Optuna ONLY sees inner folds.
    Outer test is pure evaluation with NO parameter tuning.
    """

    def __init__(self, config: NestedCVConfig):
        """
        Initialize nested CV.

        Args:
            config: Nested CV configuration
        """
        self.config = config
        config.validate()

    def generate_outer_splits(
        self,
        start_date: date,
        end_date: date,
    ) -> Iterator[OuterFold]:
        """
        Generate outer CV splits (walk-forward expanding window).

        Args:
            start_date: Start of data range
            end_date: End of data range

        Yields:
            OuterFold objects defining train/test windows
        """
        total_days = (end_date - start_date).days
        test_days = int(total_days * (1 - self.config.outer_train_pct) / self.config.outer_folds)
        min_train_days = int(total_days * 0.3)  # At least 30% for first fold

        for fold_id in range(self.config.outer_folds):
            # Expanding window: train grows, test moves forward
            test_end_offset = total_days - fold_id * test_days
            test_start_offset = test_end_offset - test_days
            train_end_offset = test_start_offset - self.config.time_config.purge_days

            if train_end_offset < min_train_days:
                continue  # Skip if not enough training data

            from datetime import timedelta

            train_window = TimeWindow(
                start_date=start_date,
                end_date=start_date + timedelta(days=train_end_offset),
            )
            test_window = TimeWindow(
                start_date=start_date + timedelta(days=test_start_offset),
                end_date=start_date + timedelta(days=test_end_offset),
            )

            yield OuterFold(
                fold_id=fold_id,
                train_window=train_window,
                test_window=test_window,
            )

    def generate_inner_splits(
        self,
        train_window: TimeWindow,
    ) -> Iterator[Tuple[TimeWindow, TimeWindow]]:
        """
        Generate inner CV splits within outer training window.

        Args:
            train_window: Outer training window to split

        Yields:
            Tuples of (inner_train_window, inner_test_window)
        """
        total_days = train_window.days
        test_days = total_days // (self.config.inner_folds + 2)  # More conservative split
        min_train_days = max(30, test_days)  # At least 30 days or one test period

        from datetime import timedelta

        for fold_id in range(self.config.inner_folds):
            # Walk-forward within training window
            test_end_offset = total_days - fold_id * test_days
            test_start_offset = test_end_offset - test_days
            train_end_offset = test_start_offset - self.config.time_config.purge_days

            if train_end_offset < min_train_days:
                continue

            inner_train = TimeWindow(
                start_date=train_window.start_date,
                end_date=train_window.start_date + timedelta(days=train_end_offset),
            )
            inner_test = TimeWindow(
                start_date=train_window.start_date + timedelta(days=test_start_offset),
                end_date=train_window.start_date + timedelta(days=test_end_offset),
            )

            yield (inner_train, inner_test)

    def run_inner_optimization(
        self,
        symbols: List[str],
        train_window: TimeWindow,
        objective_fn: Callable[[Dict[str, Any], TimeWindow, TimeWindow], float],
        param_space: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run Optuna optimization on INNER folds only.

        Args:
            symbols: Symbols to use for optimization
            train_window: Outer training window (inner CV runs within this)
            objective_fn: Function(params, train_window, test_window) -> score
            param_space: Parameter space for Optuna

        Returns:
            Tuple of (best_params, best_score)
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError as e:
            logger.warning("Optuna not available, using default params: %s", e)
            return ({}, 0.0)

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def inner_objective(trial: optuna.Trial) -> float:
            # Suggest parameters from space
            params = {}
            for name, spec in param_space.items():
                if spec["type"] == "float":
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"], log=spec.get("log", False)
                    )
                elif spec["type"] == "int":
                    params[name] = trial.suggest_int(name, spec["low"], spec["high"])
                elif spec["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, spec["choices"])

            # Evaluate on inner folds
            inner_scores = []
            for inner_train, inner_test in self.generate_inner_splits(train_window):
                score = objective_fn(params, inner_train, inner_test)
                inner_scores.append(score)

            return float(np.mean(inner_scores)) if inner_scores else 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
        )
        study.optimize(
            inner_objective,
            n_trials=self.config.inner_max_trials,
            show_progress_bar=False,
        )

        return study.best_params, study.best_value

    def evaluate_outer_test(
        self,
        symbols: List[str],
        test_window: TimeWindow,
        params: Dict[str, Any],
        evaluate_symbol_fn: Callable[[str, TimeWindow, Dict[str, Any]], SymbolMetrics],
    ) -> List[SymbolMetrics]:
        """
        Evaluate on outer test - NO PARAMETER ACCESS.

        This is pure evaluation with fixed parameters from inner CV.

        Args:
            symbols: Symbols to evaluate
            test_window: Test window (outer)
            params: Fixed parameters from inner optimization
            evaluate_symbol_fn: Function(symbol, window, params) -> SymbolMetrics

        Returns:
            Per-symbol metrics for statistical aggregation
        """
        symbol_metrics = []

        for symbol in symbols:
            metrics = evaluate_symbol_fn(symbol, test_window, params)
            symbol_metrics.append(metrics)

        return symbol_metrics

    def run(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        objective_fn: Callable[[Dict[str, Any], TimeWindow, TimeWindow], float],
        evaluate_symbol_fn: Callable[[str, TimeWindow, Dict[str, Any]], SymbolMetrics],
        param_space: Dict[str, Any],
    ) -> NestedCVResult:
        """
        Run nested CV with strict separation.

        Args:
            symbols: Training universe (NOT holdout)
            start_date: Start of data range
            end_date: End of data range
            objective_fn: Inner CV objective function
            evaluate_symbol_fn: Outer test evaluation function
            param_space: Optuna parameter space

        Returns:
            NestedCVResult with outer fold results and aggregated metrics
        """
        outer_results: List[OuterFoldResult] = []

        for outer_fold in self.generate_outer_splits(start_date, end_date):
            # INNER CV: Optimize on outer_train only
            best_params, inner_score = self.run_inner_optimization(
                symbols=symbols,
                train_window=outer_fold.train_window,
                objective_fn=objective_fn,
                param_space=param_space,
            )

            # OUTER TEST: Evaluate with best_params (NO TUNING)
            fold_metrics = self.evaluate_outer_test(
                symbols=symbols,
                test_window=outer_fold.test_window,
                params=best_params,
                evaluate_symbol_fn=evaluate_symbol_fn,
            )

            outer_results.append(
                OuterFoldResult(
                    fold_id=outer_fold.fold_id,
                    best_params=best_params,
                    symbol_metrics=fold_metrics,
                    inner_cv_score=inner_score,
                )
            )

        # Aggregate across folds
        from .statistics import aggregate_symbol_metrics_across_folds

        all_fold_metrics = [r.symbol_metrics for r in outer_results]
        aggregated = aggregate_symbol_metrics_across_folds(all_fold_metrics)

        return NestedCVResult(
            outer_results=outer_results,
            aggregated_metrics=aggregated,
        )


def create_default_param_space() -> Dict[str, Any]:
    """
    Create default parameter space for regime detector optimization.

    Returns:
        Dictionary defining parameter ranges for Optuna
    """
    return {
        "ma50_period": {"type": "int", "low": 40, "high": 60},
        "ma200_period": {"type": "int", "low": 180, "high": 220},
        "atr_period": {"type": "int", "low": 14, "high": 25},
        "vol_high_short_pct": {"type": "int", "low": 70, "high": 90},
        "chop_high_pct": {"type": "int", "low": 60, "high": 80},
    }
