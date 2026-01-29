"""
Weight Learner - Learn regime factor weights from data.

Phase 5: Replaces guessed weights (40/30/20/10) with data-driven weights
learned via logistic regression or constrained optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .composite_scorer import CompositeWeights


@dataclass
class LearningResult:
    """Result from weight learning."""

    weights: CompositeWeights
    train_accuracy: float
    feature_importance: Dict[str, float]
    method: str


class TargetLabelGenerator:
    """
    Generate supervised learning targets for regime classification.

    Supports multiple objectives:
    - Risk: Predict forward drawdown > threshold
    - Return: Predict forward return > threshold
    - Volatility: Predict forward volatility spike
    """

    def __init__(self, forward_bars: int = 10) -> None:
        self.forward_bars = forward_bars

    def create_risk_labels(
        self,
        df: pd.DataFrame,
        drawdown_threshold_pct: float = 5.0,
    ) -> pd.Series:
        """
        Create risk labels: 1 = bad forward period (drawdown > threshold).

        Args:
            df: DataFrame with OHLCV
            drawdown_threshold_pct: Drawdown threshold in %

        Returns:
            Binary series: 1 = risk-off period, 0 = safe
        """
        close = df["close"]
        low = df["low"]

        # Forward min low (for drawdown calculation)
        forward_min = low.shift(-1).rolling(self.forward_bars).min()

        # Drawdown from current close to forward low
        forward_drawdown = (close - forward_min) / close

        labels = (forward_drawdown > drawdown_threshold_pct / 100).astype(int)
        return labels.rename("risk_label")

    def create_return_labels(
        self,
        df: pd.DataFrame,
        return_threshold_pct: float = 2.0,
    ) -> pd.Series:
        """
        Create return labels: 1 = good forward return.

        Args:
            df: DataFrame with close prices
            return_threshold_pct: Return threshold in %

        Returns:
            Binary series: 1 = bullish period, 0 = not
        """
        close = df["close"]
        forward_close = close.shift(-self.forward_bars)
        forward_return = (forward_close - close) / close

        labels = (forward_return > return_threshold_pct / 100).astype(int)
        return labels.rename("return_label")

    def create_volatility_labels(
        self,
        df: pd.DataFrame,
        vol_multiplier: float = 1.5,
    ) -> pd.Series:
        """
        Create volatility spike labels: 1 = vol spike incoming.

        Args:
            df: DataFrame with OHLCV
            vol_multiplier: How much higher than recent vol to trigger

        Returns:
            Binary series: 1 = volatility spike, 0 = normal
        """
        import talib

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        atr = pd.Series(talib.ATR(high, low, close, timeperiod=14), index=df.index)
        atr_ma = atr.rolling(20).mean()

        # Forward ATR
        forward_atr = atr.shift(-self.forward_bars)

        labels = (forward_atr > atr_ma * vol_multiplier).astype(int)
        return labels.rename("vol_label")


class WeightLearner:
    """
    Learn regime factor weights from data using supervised learning.

    Methods:
    - logistic: Logistic regression coefficients → normalized weights
    - optimize: Constrained optimization maximizing class separation
    """

    def __init__(self, method: str = "logistic") -> None:
        """
        Args:
            method: "logistic" or "optimize"
        """
        if method not in ("logistic", "optimize"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self._model = None

    def fit(
        self,
        factors: pd.DataFrame,
        labels: pd.Series,
        factor_names: Optional[List[str]] = None,
    ) -> LearningResult:
        """
        Fit weights on training data.

        Args:
            factors: DataFrame with columns: trend, momentum, volatility, breadth
            labels: Binary target series
            factor_names: Optional list of factor names to use

        Returns:
            LearningResult with learned weights
        """
        # Align and drop NaN
        aligned = factors.join(labels).dropna()
        if len(aligned) < 100:
            raise ValueError(f"Insufficient data: {len(aligned)} rows (need 100+)")

        cols = factor_names or ["trend", "momentum", "volatility", "breadth"]
        available_cols = [c for c in cols if c in aligned.columns]

        X = aligned[available_cols]
        y = aligned[labels.name]

        if self.method == "logistic":
            return self._fit_logistic(X, y)
        else:
            return self._fit_optimize(X, y)

    def _fit_logistic(self, X: pd.DataFrame, y: pd.Series) -> LearningResult:
        """Fit using logistic regression."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit logistic regression
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        self._model = model

        # Convert coefficients to weights
        # For risk labels (1 = bad), negative coef = good factor
        # Invert sign so positive weight = contributes to healthy regime
        raw_coef = -model.coef_[0]  # Invert for "healthy" interpretation

        # Handle volatility specially - high vol should reduce score
        vol_idx = list(X.columns).index("volatility") if "volatility" in X.columns else -1
        if vol_idx >= 0:
            raw_coef[vol_idx] = -raw_coef[vol_idx]  # Re-invert for vol

        # Normalize to sum to 1
        abs_coef = np.abs(raw_coef)
        norm_weights = abs_coef / abs_coef.sum()

        # Build weights dict
        weights_dict = dict(zip(X.columns, norm_weights))

        # Fill missing with default (all 5 Phase 5 factors)
        default_weights = {
            "trend": 0.13,
            "trend_short": 0.10,
            "momentum": 0.35,
            "volatility": 0.22,
            "breadth": 0.20,
        }
        for k in default_weights:
            if k not in weights_dict:
                weights_dict[k] = default_weights[k]

        # Renormalize to ensure sum = 1.0
        total = sum(weights_dict.values())
        weights_dict = {k: v / total for k, v in weights_dict.items()}

        weights = CompositeWeights(**weights_dict)
        accuracy = model.score(X_scaled, y)

        importance = dict(zip(X.columns, np.abs(model.coef_[0])))

        return LearningResult(
            weights=weights,
            train_accuracy=accuracy,
            feature_importance=importance,
            method="logistic",
        )

    def _fit_optimize(self, X: pd.DataFrame, y: pd.Series) -> LearningResult:
        """Fit using constrained optimization."""
        from scipy.optimize import minimize

        n_features = X.shape[1]
        X_arr = X.values

        def objective(w: np.ndarray) -> float:
            """Maximize separation between classes."""
            # For risk labels: y=1 means bad, y=0 means good
            # We want score to be LOW when y=1, HIGH when y=0
            score = X_arr @ w

            # Handle volatility inversion
            vol_idx = list(X.columns).index("volatility") if "volatility" in X.columns else -1
            if vol_idx >= 0:
                score = score - 2 * w[vol_idx] * X_arr[:, vol_idx]  # Invert vol contribution

            mean_bad = score[y == 1].mean() if (y == 1).any() else 0
            mean_good = score[y == 0].mean() if (y == 0).any() else 0

            # Minimize negative separation (maximize good - bad)
            return -(mean_good - mean_bad)

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0.05, 0.50)] * n_features  # Each weight between 5% and 50%

        result = minimize(
            objective,
            x0=np.ones(n_features) / n_features,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights_dict = dict(zip(X.columns, result.x))

        # Fill missing with all 5 Phase 5 factors
        default_weights = {
            "trend": 0.13,
            "trend_short": 0.10,
            "momentum": 0.35,
            "volatility": 0.22,
            "breadth": 0.20,
        }
        for k in default_weights:
            if k not in weights_dict:
                weights_dict[k] = default_weights[k]

        total = sum(weights_dict.values())
        weights_dict = {k: v / total for k, v in weights_dict.items()}

        weights = CompositeWeights(**weights_dict)

        # Compute accuracy (classify by threshold)
        final_score = X_arr @ result.x
        vol_idx = list(X.columns).index("volatility") if "volatility" in X.columns else -1
        if vol_idx >= 0:
            final_score = final_score - 2 * result.x[vol_idx] * X_arr[:, vol_idx]

        threshold = np.median(final_score)
        preds = (final_score < threshold).astype(int)  # Low score = bad
        accuracy = (preds == y).mean()

        return LearningResult(
            weights=weights,
            train_accuracy=accuracy,
            feature_importance=dict(zip(X.columns, result.x)),
            method="optimize",
        )


def learn_weights_from_data(
    df: pd.DataFrame,
    objective: str = "risk",
    method: str = "logistic",
    benchmark_df: Optional[pd.DataFrame] = None,
) -> LearningResult:
    """
    Convenience function to learn weights from OHLCV data.

    Args:
        df: DataFrame with OHLCV
        objective: "risk", "return", or "volatility"
        method: "logistic" or "optimize"
        benchmark_df: Optional benchmark for breadth calculation

    Returns:
        LearningResult with learned weights
    """
    from .factor_normalizer import compute_normalized_factors

    # Compute factors
    factors = compute_normalized_factors(df, benchmark_df)
    factors_df = factors.to_dataframe()

    # Generate labels
    label_gen = TargetLabelGenerator(forward_bars=10)

    if objective == "risk":
        labels = label_gen.create_risk_labels(df, drawdown_threshold_pct=5.0)
    elif objective == "return":
        labels = label_gen.create_return_labels(df, return_threshold_pct=2.0)
    elif objective == "volatility":
        labels = label_gen.create_volatility_labels(df, vol_multiplier=1.5)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    # Learn weights
    learner = WeightLearner(method=method)
    return learner.fit(factors_df, labels)
