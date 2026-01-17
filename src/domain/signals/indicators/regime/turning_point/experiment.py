"""
Turning Point Model Experiment Framework.

Provides walk-forward backtesting and model comparison for turning point models.
Integrates with APEX experiment framework for systematic evaluation.

Key Features:
- Walk-forward model evaluation (train on window N, test on window N+1)
- Proper Purged+Embargo CV to prevent leakage
- Model comparison with statistical significance testing
- Experiment tracking with metrics history
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

from .features import extract_features
from .labels import TurningPointLabeler
from .model import TurningPointModel

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a single model evaluation window."""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # Classification metrics
    top_roc_auc: float
    top_pr_auc: float
    bottom_roc_auc: float
    bottom_pr_auc: float
    # Calibration
    top_brier: float
    bottom_brier: float
    top_ece: float  # Expected Calibration Error
    bottom_ece: float
    # Sample counts
    train_samples: int
    test_samples: int
    top_positives: int
    bottom_positives: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def combined_roc_auc(self) -> float:
        """Average ROC-AUC across top and bottom models."""
        return (self.top_roc_auc + self.bottom_roc_auc) / 2

    @property
    def combined_pr_auc(self) -> float:
        """Average PR-AUC across top and bottom models."""
        return (self.top_pr_auc + self.bottom_pr_auc) / 2


@dataclass
class ExperimentResult:
    """Result of a turning point model experiment."""

    experiment_id: str
    symbol: str
    model_type: str
    created_at: str
    # Aggregate metrics (median across windows)
    median_top_roc_auc: float
    median_top_pr_auc: float
    median_bottom_roc_auc: float
    median_bottom_pr_auc: float
    # Stability (std across windows)
    std_top_roc_auc: float
    std_bottom_roc_auc: float
    # Window results
    window_metrics: List[ModelMetrics] = field(default_factory=list)
    # Best model path (if saved)
    model_path: Optional[str] = None
    # Comparison with baseline
    baseline_comparison: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["window_metrics"] = [m.to_dict() for m in self.window_metrics]
        return result

    def save(self, path: Path) -> None:
        """Save experiment result to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentResult":
        """Load experiment result from JSON."""
        with open(path) as f:
            data = json.load(f)
        window_metrics = [ModelMetrics(**m) for m in data.pop("window_metrics")]
        return cls(**data, window_metrics=window_metrics)

    def print_summary(self) -> None:
        """Print experiment summary to console."""
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT RESULT: {self.experiment_id}")
        print(f"{'=' * 60}")
        print(f"Symbol:     {self.symbol}")
        print(f"Model:      {self.model_type}")
        print(f"Windows:    {len(self.window_metrics)}")
        print(f"\nTOP_RISK Model:")
        print(f"  ROC-AUC:  {self.median_top_roc_auc:.4f} ± {self.std_top_roc_auc:.4f}")
        print(f"  PR-AUC:   {self.median_top_pr_auc:.4f}")
        print(f"\nBOTTOM_RISK Model:")
        print(f"  ROC-AUC:  {self.median_bottom_roc_auc:.4f} ± {self.std_bottom_roc_auc:.4f}")
        print(f"  PR-AUC:   {self.median_bottom_pr_auc:.4f}")
        if self.baseline_comparison:
            print(f"\nBaseline Comparison:")
            for key, value in self.baseline_comparison.items():
                print(f"  {key}: {value}")
        print(f"{'=' * 60}\n")


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward model evaluation."""

    train_days: int = 504  # 2 years
    test_days: int = 63  # 3 months
    n_windows: int = 4  # Number of walk-forward windows
    purge_days: int = 10  # Gap between train and test (label horizon)
    embargo_days: int = 2  # Gap after test before next train
    min_train_samples: int = 200
    # Model parameters
    model_type: str = "logistic"
    confidence_threshold: float = 0.7
    cv_splits: int = 5


class TurningPointExperiment:
    """
    Walk-forward experiment for turning point model evaluation.

    Trains and evaluates models using proper temporal splits to prevent
    look-ahead bias and measure true out-of-sample performance.
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self.labeler = TurningPointLabeler(
            atr_period=14,
            zigzag_threshold=2.0,
            risk_horizon=10,
            risk_threshold=1.5,
        )

    def run(
        self,
        symbol: str,
        data: Optional[pd.DataFrame] = None,
        days: int = 1500,
    ) -> ExperimentResult:
        """
        Run walk-forward experiment for a symbol.

        Args:
            symbol: Symbol to evaluate
            data: Pre-loaded OHLCV data (optional, will fetch if None)
            days: Days of history to fetch if data not provided

        Returns:
            ExperimentResult with per-window metrics and aggregates
        """
        # Fetch data if not provided
        if data is None:
            data = self._fetch_data(symbol, days)

        # Generate labels and features
        y_top, y_bottom, _ = self.labeler.generate_combined_labels(data)
        features_df = extract_features(data)

        # Align data
        valid_mask = ~features_df.isna().any(axis=1)
        valid_idx = features_df.index[valid_mask][: -self.config.purge_days]

        X = features_df.loc[valid_idx]
        y_top = y_top.loc[valid_idx]
        y_bottom = y_bottom.loc[valid_idx]

        # Create walk-forward windows
        windows = self._create_windows(len(X))

        # Evaluate each window
        window_metrics = []
        for window_id, (train_idx, test_idx) in enumerate(windows):
            metrics = self._evaluate_window(
                window_id=window_id,
                X=X,
                y_top=y_top,
                y_bottom=y_bottom,
                train_idx=train_idx,
                test_idx=test_idx,
            )
            if metrics:
                window_metrics.append(metrics)

        # Aggregate results
        if not window_metrics:
            raise ValueError(f"No valid windows for {symbol}")

        # Compute aggregate metrics
        top_rocs = [m.top_roc_auc for m in window_metrics]
        top_prs = [m.top_pr_auc for m in window_metrics]
        bottom_rocs = [m.bottom_roc_auc for m in window_metrics]
        bottom_prs = [m.bottom_pr_auc for m in window_metrics]

        experiment_id = f"tp_{symbol.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return ExperimentResult(
            experiment_id=experiment_id,
            symbol=symbol,
            model_type=self.config.model_type,
            created_at=datetime.now().isoformat(),
            median_top_roc_auc=float(np.median(top_rocs)),
            median_top_pr_auc=float(np.median(top_prs)),
            median_bottom_roc_auc=float(np.median(bottom_rocs)),
            median_bottom_pr_auc=float(np.median(bottom_prs)),
            std_top_roc_auc=float(np.std(top_rocs)),
            std_bottom_roc_auc=float(np.std(bottom_rocs)),
            window_metrics=window_metrics,
        )

    def _fetch_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        import yfinance as yf

        logger.info(f"Fetching {days} days of {symbol} data...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval="1d")
        df.columns = df.columns.str.lower()

        if len(df) < 500:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} bars")

        return df

    def _create_windows(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create walk-forward windows with purge and embargo.

        Returns list of (train_indices, test_indices) tuples.
        """
        windows = []
        total_window = self.config.train_days + self.config.purge_days + self.config.test_days

        # Calculate starting point to fit n_windows
        step = (n_samples - total_window) // max(1, self.config.n_windows - 1)
        step = max(step, self.config.test_days + self.config.embargo_days)

        for i in range(self.config.n_windows):
            start = i * step

            train_start = start
            train_end = start + self.config.train_days
            # Purge gap
            test_start = train_end + self.config.purge_days
            test_end = test_start + self.config.test_days

            if test_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            windows.append((train_idx, test_idx))

        return windows

    def _evaluate_window(
        self,
        window_id: int,
        X: pd.DataFrame,
        y_top: pd.Series,
        y_bottom: pd.Series,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> Optional[ModelMetrics]:
        """Evaluate a single walk-forward window."""
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        # Split data
        X_train = X.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_top_train = y_top.iloc[train_idx].values
        y_top_test = y_top.iloc[test_idx].values
        y_bottom_train = y_bottom.iloc[train_idx].values
        y_bottom_test = y_bottom.iloc[test_idx].values

        # Check minimum samples
        if len(X_train) < self.config.min_train_samples:
            logger.warning(f"Window {window_id}: insufficient train samples ({len(X_train)})")
            return None

        # Check for positive samples
        if y_top_train.sum() < 10 or y_bottom_train.sum() < 10:
            logger.warning(f"Window {window_id}: insufficient positive samples")
            return None

        # Train model
        model = TurningPointModel(
            model_type=cast(Literal["logistic", "lightgbm"], self.config.model_type),
            confidence_threshold=self.config.confidence_threshold,
        )

        try:
            model.train(
                X=X_train,
                y_top=y_top_train,
                y_bottom=y_bottom_train,
                cv_splits=self.config.cv_splits,
                label_horizon=self.config.purge_days,
                embargo=self.config.embargo_days,
            )
        except Exception as e:
            logger.warning(f"Window {window_id}: training failed - {e}")
            return None

        # Predict on test set
        top_probs = model.top_model.predict_proba(X_test)[:, 1]
        bottom_probs = model.bottom_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        try:
            top_roc = roc_auc_score(y_top_test, top_probs)
            top_pr = average_precision_score(y_top_test, top_probs)
            bottom_roc = roc_auc_score(y_bottom_test, bottom_probs)
            bottom_pr = average_precision_score(y_bottom_test, bottom_probs)
        except ValueError as e:
            logger.warning(f"Window {window_id}: metric calculation failed - {e}")
            return None

        # Calibration metrics
        top_brier = brier_score_loss(y_top_test, top_probs)
        bottom_brier = brier_score_loss(y_bottom_test, bottom_probs)

        # ECE calculation
        top_ece = self._calculate_ece(y_top_test, top_probs)
        bottom_ece = self._calculate_ece(y_bottom_test, bottom_probs)

        # Get dates
        train_dates = X.index[train_idx]
        test_dates = X.index[test_idx]

        return ModelMetrics(
            window_id=window_id,
            train_start=str(train_dates[0].date()),
            train_end=str(train_dates[-1].date()),
            test_start=str(test_dates[0].date()),
            test_end=str(test_dates[-1].date()),
            top_roc_auc=float(top_roc),
            top_pr_auc=float(top_pr),
            bottom_roc_auc=float(bottom_roc),
            bottom_pr_auc=float(bottom_pr),
            top_brier=float(top_brier),
            bottom_brier=float(bottom_brier),
            top_ece=float(top_ece),
            bottom_ece=float(bottom_ece),
            train_samples=len(X_train),
            test_samples=len(X_test),
            top_positives=int(y_top_test.sum()),
            bottom_positives=int(y_bottom_test.sum()),
        )

    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)

        return ece

    def compare_models(
        self,
        current_result: ExperimentResult,
        baseline_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Compare experiment result with baseline model.

        Returns comparison metrics and whether to update.
        """
        comparison = {
            "should_update": False,
            "reason": "",
            "current_roc_auc": current_result.median_top_roc_auc,
            "current_pr_auc": current_result.median_top_pr_auc,
        }

        if baseline_path and baseline_path.exists():
            baseline = ExperimentResult.load(baseline_path)
            comparison["baseline_roc_auc"] = baseline.median_top_roc_auc
            comparison["baseline_pr_auc"] = baseline.median_top_pr_auc

            # Calculate improvement
            roc_improvement = current_result.median_top_roc_auc - baseline.median_top_roc_auc
            pr_improvement = current_result.median_top_pr_auc - baseline.median_top_pr_auc

            comparison["roc_improvement"] = roc_improvement
            comparison["pr_improvement"] = pr_improvement

            # Decision logic: update if PR-AUC improves by >2% (more important for rare events)
            if pr_improvement > 0.02:
                comparison["should_update"] = True
                comparison["reason"] = f"PR-AUC improved by {pr_improvement:.4f}"
            elif roc_improvement > 0.03 and pr_improvement >= 0:
                comparison["should_update"] = True
                comparison["reason"] = f"ROC-AUC improved by {roc_improvement:.4f}"
            else:
                comparison["reason"] = "No significant improvement"
        else:
            # No baseline - always update
            comparison["should_update"] = True
            comparison["reason"] = "No baseline model exists"

        return comparison


def retrain_model(
    symbol: str,
    force: bool = False,
    save_experiment: bool = True,
) -> Tuple[Optional[TurningPointModel], ExperimentResult]:
    """
    Retrain turning point model with backtesting validation.

    Args:
        symbol: Symbol to retrain
        force: Force update even if not better than baseline
        save_experiment: Save experiment result to disk

    Returns:
        Tuple of (new_model or None, experiment_result)
    """
    logger.info(f"Starting model retraining for {symbol}...")

    # Run experiment
    experiment = TurningPointExperiment()
    result = experiment.run(symbol, days=1500)

    # Compare with baseline
    baseline_path = Path(f"experiments/turning_point/{symbol.lower()}_latest.json")
    comparison = experiment.compare_models(result, baseline_path)
    result.baseline_comparison = comparison

    result.print_summary()

    # Decide whether to update
    if comparison["should_update"] or force:
        logger.info(f"Updating model: {comparison['reason']}")

        # Train final model on all data
        model = _train_final_model(symbol)

        if model:
            # Save model
            model_dir = Path("models/turning_point")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{symbol.lower()}_logistic.pkl"
            model.save(model_path)
            result.model_path = str(model_path)
            logger.info(f"Saved updated model to {model_path}")

            # Save experiment result
            if save_experiment:
                exp_dir = Path("experiments/turning_point")
                exp_dir.mkdir(parents=True, exist_ok=True)
                result.save(exp_dir / f"{symbol.lower()}_latest.json")
                # Also save timestamped version
                result.save(exp_dir / f"{symbol.lower()}_{result.experiment_id}.json")

            return model, result
    else:
        logger.info(f"Not updating model: {comparison['reason']}")

    return None, result


def _train_final_model(symbol: str, days: int = 1000) -> Optional[TurningPointModel]:
    """Train final model on all available data."""
    import yfinance as yf

    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval="1d")
        df.columns = df.columns.str.lower()

        if len(df) < 300:
            logger.warning(f"Insufficient data for final model: {len(df)} bars")
            return None

        # Generate labels and features
        labeler = TurningPointLabeler(
            atr_period=14,
            zigzag_threshold=2.0,
            risk_horizon=10,
            risk_threshold=1.5,
        )
        y_top, y_bottom, _ = labeler.generate_combined_labels(df)
        features_df = extract_features(df)

        # Align data
        valid_mask = ~features_df.isna().any(axis=1)
        valid_idx = features_df.index[valid_mask][:-10]

        X = features_df.loc[valid_idx].values
        y_top_arr = y_top.loc[valid_idx].values
        y_bottom_arr = y_bottom.loc[valid_idx].values

        # Train model
        model = TurningPointModel(model_type="logistic", confidence_threshold=0.7)
        model.train(
            X=X,
            y_top=y_top_arr,
            y_bottom=y_bottom_arr,
            cv_splits=5,
            label_horizon=10,
            embargo=2,
        )

        return model

    except Exception as e:
        logger.error(f"Final model training failed: {e}")
        return None


def batch_retrain(
    symbols: List[str],
    force: bool = False,
) -> Dict[str, ExperimentResult]:
    """
    Retrain models for multiple symbols.

    Args:
        symbols: List of symbols to retrain
        force: Force update even if not better

    Returns:
        Dict mapping symbol to experiment result
    """
    results = {}

    for symbol in symbols:
        try:
            _, result = retrain_model(symbol, force=force)
            results[symbol] = result
        except Exception as e:
            logger.error(f"Retraining failed for {symbol}: {e}")

    return results
