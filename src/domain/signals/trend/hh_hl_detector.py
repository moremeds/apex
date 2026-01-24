"""
Higher-High/Lower-Low (HH/HL) Swing Pattern Detector.

Identifies swing highs and lows using zigzag logic, then counts
consecutive HH/HL patterns for uptrends or LH/LL patterns for downtrends.

Scoring:
- 4+ consecutive HH+HL → 100 (strong uptrend)
- 3 consecutive HH+HL → 80
- 2 consecutive HH+HL → 60
- 1 HH or HL → 40
- Mixed/neutral → 50
- Downtrend patterns → 0-30
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    """A detected swing high or low."""

    index: int
    price: float
    type: Literal["high", "low"]


@dataclass
class HHHLResult:
    """Result from HH/HL pattern detection."""

    consecutive_hh: int  # Count of consecutive higher highs
    consecutive_hl: int  # Count of consecutive higher lows
    consecutive_lh: int  # Count of consecutive lower highs
    consecutive_ll: int  # Count of consecutive lower lows
    trend_direction: Literal["up", "down", "neutral"]
    score: float  # 0-100 trend quality score
    swing_points: List[SwingPoint]  # Detected swing points


class HHHLDetector:
    """
    Detects Higher-High/Higher-Low patterns using zigzag swing detection.

    Example:
        detector = HHHLDetector(lookback=20, swing_threshold=0.02)
        result = detector.calculate_score(df['high'], df['low'])
        print(f"Trend: {result.trend_direction}, Score: {result.score}")
    """

    def __init__(
        self,
        lookback: int = 20,
        swing_threshold: float = 0.02,
        min_bars_between_swings: int = 3,
    ) -> None:
        """
        Initialize the HH/HL detector.

        Args:
            lookback: Number of bars to analyze for swings
            swing_threshold: Minimum % move to qualify as swing (2% default)
            min_bars_between_swings: Minimum bars between swing points
        """
        self.lookback = lookback
        self.swing_threshold = swing_threshold
        self.min_bars_between_swings = min_bars_between_swings

    def detect_swings(
        self,
        high: pd.Series,
        low: pd.Series,
    ) -> List[SwingPoint]:
        """
        Identify swing highs and lows using zigzag logic.

        A swing high is a local maximum with lower prices on both sides.
        A swing low is a local minimum with higher prices on both sides.

        Args:
            high: Series of high prices
            low: Series of low prices

        Returns:
            List of SwingPoint objects in chronological order
        """
        if len(high) < self.lookback:
            return []

        # Use last N bars for analysis
        high_arr = high.values[-self.lookback :].astype(np.float64)
        low_arr = low.values[-self.lookback :].astype(np.float64)
        offset = len(high) - self.lookback

        swings: List[SwingPoint] = []
        last_swing_idx = -self.min_bars_between_swings - 1
        last_swing_type: Literal["high", "low"] | None = None

        # Detect swing highs and lows using local extrema
        for i in range(2, len(high_arr) - 2):
            # Check for swing high (local maximum)
            is_swing_high = (
                high_arr[i] > high_arr[i - 1]
                and high_arr[i] > high_arr[i - 2]
                and high_arr[i] > high_arr[i + 1]
                and high_arr[i] > high_arr[i + 2]
            )

            # Check for swing low (local minimum)
            is_swing_low = (
                low_arr[i] < low_arr[i - 1]
                and low_arr[i] < low_arr[i - 2]
                and low_arr[i] < low_arr[i + 1]
                and low_arr[i] < low_arr[i + 2]
            )

            # Ensure minimum bars between swings
            if i - last_swing_idx < self.min_bars_between_swings:
                continue

            # Alternate between highs and lows for proper zigzag
            if is_swing_high and last_swing_type != "high":
                # Verify swing magnitude meets threshold
                if swings:
                    last_low = next(
                        (s for s in reversed(swings) if s.type == "low"),
                        None,
                    )
                    if last_low:
                        magnitude = (high_arr[i] - last_low.price) / last_low.price
                        if magnitude < self.swing_threshold:
                            continue

                swings.append(
                    SwingPoint(
                        index=offset + i,
                        price=float(high_arr[i]),
                        type="high",
                    )
                )
                last_swing_idx = i
                last_swing_type = "high"

            elif is_swing_low and last_swing_type != "low":
                # Verify swing magnitude meets threshold
                if swings:
                    last_high = next(
                        (s for s in reversed(swings) if s.type == "high"),
                        None,
                    )
                    if last_high:
                        magnitude = (last_high.price - low_arr[i]) / last_high.price
                        if magnitude < self.swing_threshold:
                            continue

                swings.append(
                    SwingPoint(
                        index=offset + i,
                        price=float(low_arr[i]),
                        type="low",
                    )
                )
                last_swing_idx = i
                last_swing_type = "low"

        return swings

    def calculate_score(
        self,
        high: pd.Series,
        low: pd.Series,
    ) -> HHHLResult:
        """
        Calculate trend score based on HH/HL or LH/LL patterns.

        Args:
            high: Series of high prices
            low: Series of low prices

        Returns:
            HHHLResult with pattern counts and score
        """
        swings = self.detect_swings(high, low)

        if len(swings) < 3:
            return HHHLResult(
                consecutive_hh=0,
                consecutive_hl=0,
                consecutive_lh=0,
                consecutive_ll=0,
                trend_direction="neutral",
                score=50.0,
                swing_points=swings,
            )

        # Separate swing highs and lows
        swing_highs = [s for s in swings if s.type == "high"]
        swing_lows = [s for s in swings if s.type == "low"]

        # Count consecutive patterns
        consecutive_hh = self._count_consecutive_higher(swing_highs)
        consecutive_hl = self._count_consecutive_higher(swing_lows)
        consecutive_lh = self._count_consecutive_lower(swing_highs)
        consecutive_ll = self._count_consecutive_lower(swing_lows)

        # Determine trend direction and score
        uptrend_score = consecutive_hh + consecutive_hl
        downtrend_score = consecutive_lh + consecutive_ll

        if uptrend_score >= 4:
            trend_direction: Literal["up", "down", "neutral"] = "up"
            score = 100.0
        elif uptrend_score == 3:
            trend_direction = "up"
            score = 80.0
        elif uptrend_score == 2:
            trend_direction = "up"
            score = 60.0
        elif downtrend_score >= 4:
            trend_direction = "down"
            score = 0.0
        elif downtrend_score == 3:
            trend_direction = "down"
            score = 15.0
        elif downtrend_score == 2:
            trend_direction = "down"
            score = 30.0
        elif uptrend_score == 1:
            trend_direction = "neutral"
            score = 40.0
        else:
            trend_direction = "neutral"
            score = 50.0

        return HHHLResult(
            consecutive_hh=consecutive_hh,
            consecutive_hl=consecutive_hl,
            consecutive_lh=consecutive_lh,
            consecutive_ll=consecutive_ll,
            trend_direction=trend_direction,
            score=score,
            swing_points=swings,
        )

    def _count_consecutive_higher(self, swings: List[SwingPoint]) -> int:
        """Count consecutive higher swings from most recent."""
        if len(swings) < 2:
            return 0

        count = 0
        for i in range(len(swings) - 1, 0, -1):
            if swings[i].price > swings[i - 1].price:
                count += 1
            else:
                break

        return count

    def _count_consecutive_lower(self, swings: List[SwingPoint]) -> int:
        """Count consecutive lower swings from most recent."""
        if len(swings) < 2:
            return 0

        count = 0
        for i in range(len(swings) - 1, 0, -1):
            if swings[i].price < swings[i - 1].price:
                count += 1
            else:
                break

        return count
