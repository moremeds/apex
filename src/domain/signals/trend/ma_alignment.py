"""
Moving Average Alignment Scorer.

Evaluates trend quality based on:
- MA order alignment (SMA20 > SMA50 > SMA200 for uptrend)
- Spread between MAs (wider = stronger trend)
- Time since last golden/death cross

Scoring:
- All MAs in correct order: +50 base
- Spread between MAs: +20 max (wider = stronger)
- Time since cross: +30 max (longer = more established)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class MAAlignmentResult:
    """Result from MA alignment analysis."""

    order_correct: bool  # True if MAs in correct order for trend
    spread_pct: float  # % spread between fast/slow MA
    time_since_cross: int  # Bars since last golden/death cross
    score: float  # 0-100 alignment score
    direction: Literal["bullish", "bearish", "neutral"]
    ma_values: Dict[int, float]  # Current MA values by period


class MAAlignmentScorer:
    """
    Scores trend quality based on MA alignment.

    Example:
        scorer = MAAlignmentScorer(periods=[20, 50, 200])
        result = scorer.calculate(df['close'])
        print(f"MA Alignment: {result.direction}, Score: {result.score}")
    """

    def __init__(
        self,
        periods: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize MA alignment scorer.

        Args:
            periods: MA periods to analyze (default: [20, 50, 200])
        """
        self.periods = periods or [20, 50, 200]

    def calculate(
        self,
        close: pd.Series,
    ) -> MAAlignmentResult:
        """
        Calculate MA alignment score.

        Args:
            close: Series of close prices

        Returns:
            MAAlignmentResult with alignment score
        """
        # Need enough data for longest MA
        min_periods = max(self.periods)
        if len(close) < min_periods:
            return MAAlignmentResult(
                order_correct=False,
                spread_pct=0.0,
                time_since_cross=0,
                score=50.0,
                direction="neutral",
                ma_values={},
            )

        close_arr = close.values.astype(np.float64)

        # Calculate all MAs
        ma_values: Dict[int, float] = {}
        ma_series: Dict[int, np.ndarray] = {}
        for period in self.periods:
            ma = self._sma(close_arr, period)
            ma_values[period] = float(ma[-1])
            ma_series[period] = ma

        # Check if all MAs are in correct order
        # Bullish: short > medium > long (e.g., SMA20 > SMA50 > SMA200)
        # Bearish: short < medium < long
        sorted_periods = sorted(self.periods)
        current_values = [ma_values[p] for p in sorted_periods]

        is_bullish_order = all(
            current_values[i] > current_values[i + 1] for i in range(len(current_values) - 1)
        )
        is_bearish_order = all(
            current_values[i] < current_values[i + 1] for i in range(len(current_values) - 1)
        )

        # Determine direction
        if is_bullish_order:
            direction: Literal["bullish", "bearish", "neutral"] = "bullish"
            order_correct = True
        elif is_bearish_order:
            direction = "bearish"
            order_correct = True
        else:
            direction = "neutral"
            order_correct = False

        # Calculate spread between fast and slow MA
        fast_ma = ma_values[sorted_periods[0]]
        slow_ma = ma_values[sorted_periods[-1]]
        spread_pct = abs(fast_ma - slow_ma) / slow_ma * 100 if slow_ma > 0 else 0.0

        # Find time since last cross (between fast and medium MA)
        if len(sorted_periods) >= 2:
            fast_series = ma_series[sorted_periods[0]]
            medium_series = ma_series[sorted_periods[1]]
            time_since_cross = self._find_bars_since_cross(fast_series, medium_series)
        else:
            time_since_cross = 0

        # Calculate score
        score = self._calculate_alignment_score(
            order_correct=order_correct,
            direction=direction,
            spread_pct=spread_pct,
            time_since_cross=time_since_cross,
        )

        return MAAlignmentResult(
            order_correct=order_correct,
            spread_pct=round(spread_pct, 4),
            time_since_cross=time_since_cross,
            score=score,
            direction=direction,
            ma_values=ma_values,
        )

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        if len(data) < period:
            return np.full(len(data), np.nan)

        # Use cumsum for efficient SMA calculation
        cumsum = np.cumsum(np.insert(data, 0, 0))
        sma = (cumsum[period:] - cumsum[:-period]) / period

        # Pad front with NaN
        result = np.full(len(data), np.nan)
        result[period - 1 :] = sma
        return result

    def _find_bars_since_cross(
        self,
        fast: np.ndarray,
        slow: np.ndarray,
    ) -> int:
        """Find bars since last golden/death cross."""
        if len(fast) != len(slow):
            return 0

        # Find where fast-slow changes sign
        diff = fast - slow
        cross_points = np.where(np.diff(np.signbit(diff)))[0]

        if len(cross_points) == 0:
            return len(fast)  # No cross in data

        last_cross_idx = cross_points[-1]
        return int(len(fast) - last_cross_idx - 1)

    def _calculate_alignment_score(
        self,
        order_correct: bool,
        direction: str,
        spread_pct: float,
        time_since_cross: int,
    ) -> float:
        """
        Calculate alignment score from 0-100.

        Components:
        - Order correct: +50 base (bullish order for uptrend analysis)
        - Spread score: +20 max (wider spread = stronger trend)
        - Time since cross: +30 max (longer = more established)
        """
        score = 0.0

        # Base score for correct order
        if order_correct and direction == "bullish":
            score += 50.0
        elif order_correct and direction == "bearish":
            # For bearish trends, give lower score
            score += 25.0
        else:
            # Neutral/mixed - base 25
            score += 25.0

        # Spread score: max +20 for spread >= 5%
        spread_score = min(20.0, spread_pct * 4)
        score += spread_score

        # Time since cross: max +30 for 60+ bars since cross
        time_score = min(30.0, time_since_cross * 0.5)
        score += time_score

        return min(100.0, score)
