"""
Composite Regime Scorer - Single 0-100 score from calibrated factors.

Phase 5: Replaces implicit decision tree weights with explicit composite scoring.
Score interpretation (with calibrated bands):
  - 70-100: R0 (Healthy Uptrend)
  - 30-70:  R1 (Choppy/Extended)
  - 0-30:   R2 (Risk-Off)

Calibration:
  - Weights learned from 35 symbols, 2y history via logistic regression
  - Achieves ~50-60% R1 vs previous ~75% "everything is R1" problem
  - Momentum is most predictive factor (importance=0.62)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .factor_normalizer import NormalizedFactors, compute_normalized_factors


@dataclass
class CompositeWeights:
    """
    Weights for composite score calculation.

    Constraints: All weights must sum to 1.0.

    Factor groups:
    - EMA Trend: trend (EMA20/50) + trend_short (EMA10/20)
    - Dual MACD: macd_trend (55/89) + macd_momentum (13/21)
    - RSI Momentum: momentum (RSI 14)
    - Volatility: volatility (ATR 14)
    - Breadth: breadth (relative performance)
    """

    trend: float = 0.10  # Long-term EMA trend (EMA20/50)
    trend_short: float = 0.08  # Short-term EMA trend (EMA10/20)
    macd_trend: float = 0.12  # Long MACD (55/89) - trend confirmation
    macd_momentum: float = 0.10  # Short MACD (13/21) - momentum timing
    momentum: float = 0.28  # RSI momentum
    volatility: float = 0.17  # Inverted in score (high vol = lower score)
    breadth: float = 0.15

    def __post_init__(self) -> None:
        total = (
            self.trend
            + self.trend_short
            + self.macd_trend
            + self.macd_momentum
            + self.momentum
            + self.volatility
            + self.breadth
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def to_dict(self) -> Dict[str, float]:
        return {
            "trend": self.trend,
            "trend_short": self.trend_short,
            "macd_trend": self.macd_trend,
            "macd_momentum": self.macd_momentum,
            "momentum": self.momentum,
            "volatility": self.volatility,
            "breadth": self.breadth,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "CompositeWeights":
        return cls(
            trend=d.get("trend", 0.10),
            trend_short=d.get("trend_short", 0.08),
            macd_trend=d.get("macd_trend", 0.12),
            macd_momentum=d.get("macd_momentum", 0.10),
            momentum=d.get("momentum", 0.28),
            volatility=d.get("volatility", 0.17),
            breadth=d.get("breadth", 0.15),
        )


@dataclass
class ScoreBands:
    """
    Score bands for regime classification.

    Uses banded thresholds to reduce flip-flopping:
    - Enter R0: score must exceed healthy_enter
    - Exit R0: score must drop below healthy_exit

    Calibrated to achieve ~50-60% R1 (vs original 75%+ "everything is R1").
    """

    # R0 (Healthy) bands - lowered from 75/65 to widen R0 zone
    healthy_enter: float = 70.0
    healthy_exit: float = 60.0

    # R2 (Risk-Off) bands - raised from 25/35 to widen R2 zone
    risk_off_enter: float = 30.0
    risk_off_exit: float = 40.0

    # Everything in between is R1 (Choppy)


@dataclass
class CompositeScoreOutput:
    """Output from composite score calculation."""

    score: float  # 0-100 composite score
    regime: str  # R0, R1, R2
    confidence: float  # 0-1 confidence level
    factors: Dict[str, float]  # Individual factor scores
    weights: Dict[str, float]  # Weights used


class CompositeRegimeScorer:
    """
    Compute composite regime score from calibrated factors.

    The composite score formula:
        score = w_trend * trend + w_trend_short * trend_short +
                w_macd_trend * macd_trend + w_macd_momentum * macd_momentum +
                w_momentum * momentum + w_volatility * (1 - volatility) +
                w_breadth * breadth

    Note: Volatility is inverted because high volatility = risk.
    """

    def __init__(
        self,
        weights: Optional[CompositeWeights] = None,
        bands: Optional[ScoreBands] = None,
    ) -> None:
        self.weights = weights or CompositeWeights()
        self.bands = bands or ScoreBands()

    def compute_score(self, factors: NormalizedFactors, idx: int = -1) -> float:
        """
        Compute composite score for a single bar.

        Args:
            factors: NormalizedFactors container
            idx: Index to compute score for (-1 = latest)

        Returns:
            Composite score in [0, 100] range
        """
        # Get factor values at index
        trend = factors.trend.iloc[idx] if not np.isnan(factors.trend.iloc[idx]) else 0.5
        trend_short = (
            factors.trend_short.iloc[idx]
            if not np.isnan(factors.trend_short.iloc[idx])
            else 0.5
        )
        macd_trend = (
            factors.macd_trend.iloc[idx]
            if not np.isnan(factors.macd_trend.iloc[idx])
            else 0.5
        )
        macd_momentum = (
            factors.macd_momentum.iloc[idx]
            if not np.isnan(factors.macd_momentum.iloc[idx])
            else 0.5
        )
        momentum = factors.momentum.iloc[idx] if not np.isnan(factors.momentum.iloc[idx]) else 0.5
        volatility = (
            factors.volatility.iloc[idx] if not np.isnan(factors.volatility.iloc[idx]) else 0.5
        )

        breadth = 0.5  # Neutral default if no benchmark
        if factors.breadth is not None and not np.isnan(factors.breadth.iloc[idx]):
            breadth = factors.breadth.iloc[idx]

        # Composite formula (volatility inverted)
        score = (
            self.weights.trend * trend
            + self.weights.trend_short * trend_short
            + self.weights.macd_trend * macd_trend
            + self.weights.macd_momentum * macd_momentum
            + self.weights.momentum * momentum
            + self.weights.volatility * (1.0 - volatility)  # High vol = lower score
            + self.weights.breadth * breadth
        )

        return float(score * 100)  # Scale to 0-100

    def compute_score_series(self, factors: NormalizedFactors) -> pd.Series:
        """Compute composite score for entire series."""
        trend = factors.trend.fillna(0.5)
        trend_short = factors.trend_short.fillna(0.5)
        macd_trend = factors.macd_trend.fillna(0.5)
        macd_momentum = factors.macd_momentum.fillna(0.5)
        momentum = factors.momentum.fillna(0.5)
        volatility = factors.volatility.fillna(0.5)

        breadth = pd.Series(0.5, index=factors.trend.index)
        if factors.breadth is not None:
            breadth = factors.breadth.fillna(0.5)

        score = (
            self.weights.trend * trend
            + self.weights.trend_short * trend_short
            + self.weights.macd_trend * macd_trend
            + self.weights.macd_momentum * macd_momentum
            + self.weights.momentum * momentum
            + self.weights.volatility * (1.0 - volatility)
            + self.weights.breadth * breadth
        )

        return score * 100  # Scale to 0-100

    def classify(self, score: float, current_regime: Optional[str] = None) -> str:
        """
        Classify regime from score with hysteresis.

        Args:
            score: Composite score (0-100)
            current_regime: Current regime for hysteresis (None = no hysteresis)

        Returns:
            Regime string: "R0", "R1", or "R2"
        """
        if current_regime is None:
            # No hysteresis - simple threshold
            if score >= self.bands.healthy_enter:
                return "R0"
            elif score <= self.bands.risk_off_enter:
                return "R2"
            else:
                return "R1"

        # With hysteresis - use entry/exit bands
        if current_regime == "R0":
            if score < self.bands.healthy_exit:
                if score <= self.bands.risk_off_enter:
                    return "R2"
                return "R1"
            return "R0"

        elif current_regime == "R2":
            if score > self.bands.risk_off_exit:
                if score >= self.bands.healthy_enter:
                    return "R0"
                return "R1"
            return "R2"

        else:  # R1
            if score >= self.bands.healthy_enter:
                return "R0"
            elif score <= self.bands.risk_off_enter:
                return "R2"
            return "R1"

    def score_and_classify(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Full scoring pipeline: compute factors, scores, and regimes.

        Args:
            df: OHLCV DataFrame
            benchmark_df: Optional benchmark for breadth calculation

        Returns:
            DataFrame with columns: score, regime, trend, trend_short, momentum, volatility, breadth
        """
        factors = compute_normalized_factors(df, benchmark_df)
        scores = self.compute_score_series(factors)

        # Classify with hysteresis
        regimes: List[Optional[str]] = []
        current: Optional[str] = None
        for score in scores:
            if np.isnan(score):
                regimes.append(None)
            else:
                regime = self.classify(score, current)
                regimes.append(regime)
                current = regime

        return pd.DataFrame(
            {
                "score": scores,
                "regime": regimes,
                "trend": factors.trend,
                "trend_short": factors.trend_short,
                "macd_trend": factors.macd_trend,
                "macd_momentum": factors.macd_momentum,
                "momentum": factors.momentum,
                "volatility": factors.volatility,
                "breadth": factors.breadth if factors.breadth is not None else np.nan,
            },
            index=df.index,
        )

    def get_output(
        self,
        factors: NormalizedFactors,
        idx: int = -1,
        current_regime: Optional[str] = None,
    ) -> CompositeScoreOutput:
        """Get full output for a single bar."""
        score = self.compute_score(factors, idx)
        regime = self.classify(score, current_regime)

        # Confidence based on distance from band boundaries
        if regime == "R0":
            confidence = min(1.0, (score - self.bands.healthy_enter) / 15)
        elif regime == "R2":
            confidence = min(1.0, (self.bands.risk_off_enter - score) / 15)
        else:  # R1
            mid = (self.bands.healthy_exit + self.bands.risk_off_exit) / 2
            distance_from_edge = min(
                abs(score - self.bands.healthy_exit), abs(score - self.bands.risk_off_exit)
            )
            confidence = distance_from_edge / (mid - self.bands.risk_off_exit)

        confidence = max(0.0, min(1.0, confidence))

        factor_values = {
            "trend": float(factors.trend.iloc[idx]) if not np.isnan(factors.trend.iloc[idx]) else 0,
            "trend_short": (
                float(factors.trend_short.iloc[idx])
                if not np.isnan(factors.trend_short.iloc[idx])
                else 0
            ),
            "macd_trend": (
                float(factors.macd_trend.iloc[idx])
                if not np.isnan(factors.macd_trend.iloc[idx])
                else 0
            ),
            "macd_momentum": (
                float(factors.macd_momentum.iloc[idx])
                if not np.isnan(factors.macd_momentum.iloc[idx])
                else 0
            ),
            "momentum": (
                float(factors.momentum.iloc[idx]) if not np.isnan(factors.momentum.iloc[idx]) else 0
            ),
            "volatility": (
                float(factors.volatility.iloc[idx])
                if not np.isnan(factors.volatility.iloc[idx])
                else 0
            ),
            "breadth": (
                float(factors.breadth.iloc[idx])
                if factors.breadth is not None and not np.isnan(factors.breadth.iloc[idx])
                else 0
            ),
        }

        return CompositeScoreOutput(
            score=score,
            regime=regime,
            confidence=confidence,
            factors=factor_values,
            weights=self.weights.to_dict(),
        )
