"""
Trend Health Analyzer.

Calculates a weighted composite trend health score (0-100) by combining:
- HH/HL swing patterns (30%)
- MA alignment (25%)
- ADX trend strength (20%)
- MA slope acceleration (15%)
- RSI health - not at extremes (10%)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

from .hh_hl_detector import HHHLDetector, HHHLResult
from .ma_alignment import MAAlignmentResult, MAAlignmentScorer


@dataclass
class TrendHealthResult:
    """Result from trend health analysis."""

    score: float  # 0-100 composite score
    direction: Literal["bullish", "bearish", "neutral"]
    components: Dict[str, float]  # Individual component scores
    confidence: float  # 0-1 based on data quality
    hh_hl_result: Optional[HHHLResult] = None
    ma_alignment_result: Optional[MAAlignmentResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "score": round(self.score, 2),
            "direction": self.direction,
            "components": {k: round(v, 2) for k, v in self.components.items()},
            "confidence": round(self.confidence, 3),
            "hh_hl": {
                "consecutive_hh": self.hh_hl_result.consecutive_hh if self.hh_hl_result else 0,
                "consecutive_hl": self.hh_hl_result.consecutive_hl if self.hh_hl_result else 0,
                "consecutive_lh": self.hh_hl_result.consecutive_lh if self.hh_hl_result else 0,
                "consecutive_ll": self.hh_hl_result.consecutive_ll if self.hh_hl_result else 0,
                "trend": self.hh_hl_result.trend_direction if self.hh_hl_result else "neutral",
            },
            "ma_alignment": {
                "order_correct": (
                    self.ma_alignment_result.order_correct if self.ma_alignment_result else False
                ),
                "spread_pct": (
                    self.ma_alignment_result.spread_pct if self.ma_alignment_result else 0.0
                ),
                "time_since_cross": (
                    self.ma_alignment_result.time_since_cross if self.ma_alignment_result else 0
                ),
                "direction": (
                    self.ma_alignment_result.direction if self.ma_alignment_result else "neutral"
                ),
            },
        }


class TrendHealthAnalyzer:
    """
    Calculates composite trend health score.

    Example:
        analyzer = TrendHealthAnalyzer()
        result = analyzer.calculate(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            adx=25.0,
            rsi=55.0,
        )
        print(f"Trend Health: {result.score:.1f}/100 ({result.direction})")
    """

    # Component weights (must sum to 1.0)
    WEIGHTS = {
        "hh_hl": 0.30,
        "ma_alignment": 0.25,
        "adx": 0.20,
        "slope": 0.15,
        "rsi_health": 0.10,
    }

    def __init__(
        self,
        hh_hl_detector: Optional[HHHLDetector] = None,
        ma_scorer: Optional[MAAlignmentScorer] = None,
    ) -> None:
        """
        Initialize trend health analyzer.

        Args:
            hh_hl_detector: HH/HL detector instance (creates default if None)
            ma_scorer: MA alignment scorer instance (creates default if None)
        """
        self.hh_hl = hh_hl_detector or HHHLDetector()
        self.ma_scorer = ma_scorer or MAAlignmentScorer()

    def calculate(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        adx: float = 25.0,
        rsi: float = 50.0,
    ) -> TrendHealthResult:
        """
        Calculate composite trend health score.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            adx: Current ADX value (0-100)
            rsi: Current RSI value (0-100)

        Returns:
            TrendHealthResult with score and components
        """
        # Calculate component scores
        hh_hl_result = self.hh_hl.calculate_score(high, low)
        ma_result = self.ma_scorer.calculate(close)
        adx_score = self._calc_adx_score(adx)
        slope_score = self._calc_slope_score(close)
        rsi_score = self._calc_rsi_health(rsi)

        components = {
            "hh_hl": hh_hl_result.score,
            "ma_alignment": ma_result.score,
            "adx": adx_score,
            "slope": slope_score,
            "rsi_health": rsi_score,
        }

        # Calculate weighted total
        total = sum(self.WEIGHTS[k] * v for k, v in components.items())

        # Determine direction based on HH/HL and MA alignment
        direction = self._determine_direction(hh_hl_result, ma_result)

        # Calculate confidence based on data quality
        confidence = self._calc_confidence(high, low, close)

        return TrendHealthResult(
            score=total,
            direction=direction,
            components=components,
            confidence=confidence,
            hh_hl_result=hh_hl_result,
            ma_alignment_result=ma_result,
        )

    def _calc_adx_score(self, adx: float) -> float:
        """
        Calculate ADX component score.

        ADX scoring:
        - 0-15: Weak trend → 0-30
        - 15-25: Moderate → 30-60
        - 25-40: Strong → 60-80
        - 40+: Very strong → 80-100
        """
        if adx < 15:
            return adx * 2  # 0-30
        elif adx < 25:
            return 30 + (adx - 15) * 3  # 30-60
        elif adx < 40:
            return 60 + (adx - 25) * 1.33  # 60-80
        else:
            return min(100, 80 + (adx - 40) * 0.5)  # 80-100

    def _calc_slope_score(self, close: pd.Series, lookback: int = 20) -> float:
        """
        Calculate MA slope score based on price momentum.

        Uses linear regression slope of recent prices normalized to 0-100.
        """
        if len(close) < lookback:
            return 50.0

        close_arr = close.values[-lookback:].astype(np.float64)

        # Calculate slope using linear regression
        x = np.arange(len(close_arr))
        slope = np.polyfit(x, close_arr, 1)[0]

        # Normalize slope as percentage of price
        avg_price = np.mean(close_arr)
        slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0

        # Convert to 0-100 score
        # Positive slope = higher score, negative = lower
        # Max score at ~2% daily slope (40% over 20 days)
        score = 50 + slope_pct * 25  # Center at 50
        return max(0, min(100, score))

    def _calc_rsi_health(self, rsi: float) -> float:
        """
        Calculate RSI health score.

        RSI scoring (healthy trends avoid extremes):
        - 40-60: Ideal → 100
        - 30-40 or 60-70: Acceptable → 70
        - 20-30 or 70-80: Concerning → 40
        - <20 or >80: Extreme → 10
        """
        if 40 <= rsi <= 60:
            return 100.0
        elif 30 <= rsi <= 70:
            # 30-40 or 60-70
            if rsi < 40:
                return 100 - (40 - rsi) * 3  # 70-100
            else:
                return 100 - (rsi - 60) * 3  # 70-100
        elif 20 <= rsi <= 80:
            # 20-30 or 70-80
            if rsi < 30:
                return 70 - (30 - rsi) * 3  # 40-70
            else:
                return 70 - (rsi - 70) * 3  # 40-70
        else:
            # <20 or >80
            return 10.0

    def _determine_direction(
        self,
        hh_hl_result: HHHLResult,
        ma_result: MAAlignmentResult,
    ) -> Literal["bullish", "bearish", "neutral"]:
        """
        Determine overall trend direction from components.

        Priority:
        1. Both agree → use that direction
        2. One neutral, one directional → use directional
        3. Conflicting → neutral
        """
        hh_hl_dir = hh_hl_result.trend_direction
        ma_dir = ma_result.direction

        # Map hh_hl "up"/"down" to "bullish"/"bearish"
        hh_hl_mapped = (
            "bullish" if hh_hl_dir == "up" else "bearish" if hh_hl_dir == "down" else "neutral"
        )

        # Both agree
        if hh_hl_mapped == ma_dir:
            return hh_hl_mapped  # type: ignore

        # One neutral
        if hh_hl_mapped == "neutral":
            return ma_dir
        if ma_dir == "neutral":
            return hh_hl_mapped  # type: ignore

        # Conflicting - return neutral
        return "neutral"

    def _calc_confidence(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> float:
        """
        Calculate confidence based on data quality.

        Factors:
        - Data completeness (no NaN)
        - Sufficient history (200+ bars)
        - Price consistency (no extreme gaps)
        """
        confidence = 1.0

        # Check for NaN values
        total_len = len(close)
        nan_count = close.isna().sum() + high.isna().sum() + low.isna().sum()
        if nan_count > 0:
            confidence *= max(0.5, 1 - nan_count / (total_len * 3))

        # Check data length (penalize if < 200 bars)
        if total_len < 200:
            confidence *= total_len / 200

        # Check for extreme gaps (> 10% daily moves)
        if total_len > 1:
            close_arr = close.dropna().values.astype(np.float64)
            if len(close_arr) > 1:
                returns = np.diff(close_arr) / close_arr[:-1]
                extreme_count = np.sum(np.abs(returns) > 0.10)
                if extreme_count > 0:
                    confidence *= max(0.7, 1 - extreme_count * 0.05)

        return min(1.0, max(0.0, confidence))
