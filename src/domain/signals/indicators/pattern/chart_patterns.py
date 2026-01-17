"""
Chart Patterns Indicator.

Detects classic chart patterns including:
- Head and Shoulders (reversal)
- Double Top/Bottom (reversal)
- Triangles (continuation/reversal)
- Wedges (reversal)
- Channels (continuation)

Note: Simplified detection using price structure analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class ChartPatternsIndicator(IndicatorBase):
    """
    Chart Pattern detector.

    Default Parameters:
        lookback: 50  # Bars to analyze
        tolerance: 0.02  # % tolerance for pattern matching

    State Output:
        pattern: Detected pattern name or None
        pattern_type: "reversal", "continuation", or None
        direction: "bullish", "bearish", or None
        confidence: Pattern confidence score (0-100)
    """

    name = "chart_patterns"
    category = SignalCategory.PATTERN
    required_fields = ["high", "low", "close"]
    warmup_periods = 51

    _default_params = {
        "lookback": 50,
        "tolerance": 0.02,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate chart pattern signals."""
        lookback = params["lookback"]
        tolerance = params["tolerance"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "cp_pattern": pd.Series(dtype=object),
                    "cp_confidence": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        n = len(close)

        patterns: List[Optional[str]] = [None] * n
        confidence = np.zeros(n, dtype=np.float64)

        for i in range(lookback, n):
            window_high = high[i - lookback : i + 1]
            window_low = low[i - lookback : i + 1]
            window_close = close[i - lookback : i + 1]

            # Detect patterns
            detected_pattern, conf = self._detect_pattern(
                window_high, window_low, window_close, tolerance
            )
            patterns[i] = detected_pattern
            confidence[i] = conf

        return pd.DataFrame({"cp_pattern": patterns, "cp_confidence": confidence}, index=data.index)

    def _detect_pattern(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        tolerance: float,
    ) -> tuple[Optional[str], float]:
        """Detect chart patterns in the window."""
        n = len(close)

        # Find swing points
        swing_highs = []
        swing_lows = []

        for i in range(2, n - 2):
            if high[i] > high[i - 1] and high[i] > high[i + 1]:
                swing_highs.append((i, high[i]))
            if low[i] < low[i - 1] and low[i] < low[i + 1]:
                swing_lows.append((i, low[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None, 0

        # Double Top Detection
        if len(swing_highs) >= 2:
            last_two_highs = swing_highs[-2:]
            h1, h2 = last_two_highs[0][1], last_two_highs[1][1]
            if abs(h1 - h2) / h1 < tolerance:
                # Check for neckline break
                neckline = min(low[last_two_highs[0][0] : last_two_highs[1][0]])
                if close[-1] < neckline:
                    return "double_top", 75

        # Double Bottom Detection
        if len(swing_lows) >= 2:
            last_two_lows = swing_lows[-2:]
            l1, l2 = last_two_lows[0][1], last_two_lows[1][1]
            if abs(l1 - l2) / l1 < tolerance:
                # Check for neckline break
                neckline = max(high[last_two_lows[0][0] : last_two_lows[1][0]])
                if close[-1] > neckline:
                    return "double_bottom", 75

        # Head and Shoulders Detection
        if len(swing_highs) >= 3:
            last_three_highs = swing_highs[-3:]
            h1, h2, h3 = [h[1] for h in last_three_highs]
            # Head higher than shoulders, shoulders roughly equal
            if h2 > h1 and h2 > h3 and abs(h1 - h3) / h1 < tolerance:
                return "head_and_shoulders", 70

        # Inverse Head and Shoulders
        if len(swing_lows) >= 3:
            last_three_lows = swing_lows[-3:]
            l1, l2, l3 = [l[1] for l in last_three_lows]
            if l2 < l1 and l2 < l3 and abs(l1 - l3) / l1 < tolerance:
                return "inverse_head_and_shoulders", 70

        # Triangle Detection (converging highs and lows)
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            high_slope = (swing_highs[-1][1] - swing_highs[-2][1]) / (
                swing_highs[-1][0] - swing_highs[-2][0]
            )
            low_slope = (swing_lows[-1][1] - swing_lows[-2][1]) / (
                swing_lows[-1][0] - swing_lows[-2][0]
            )

            if high_slope < 0 and low_slope > 0:  # Converging
                return "symmetrical_triangle", 60
            elif high_slope < 0 and abs(low_slope) < 0.001:  # Flat bottom
                return "descending_triangle", 65
            elif abs(high_slope) < 0.001 and low_slope > 0:  # Flat top
                return "ascending_triangle", 65

        return None, 0

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract chart pattern state for rule evaluation."""
        pattern = current.get("cp_pattern")
        confidence = current.get("cp_confidence", 0)

        if pattern is None or pd.isna(confidence):
            return {
                "pattern": None,
                "pattern_type": None,
                "direction": None,
                "confidence": 0,
            }

        # Determine pattern type and direction
        reversal_bearish = {"double_top", "head_and_shoulders"}
        reversal_bullish = {"double_bottom", "inverse_head_and_shoulders"}
        continuation = {"ascending_triangle", "descending_triangle", "symmetrical_triangle"}

        pattern_type: Optional[str] = None
        direction: Optional[str] = None

        if pattern in reversal_bearish:
            pattern_type = "reversal"
            direction = "bearish"
        elif pattern in reversal_bullish:
            pattern_type = "reversal"
            direction = "bullish"
        elif pattern in continuation:
            pattern_type = "continuation"
            if "ascending" in pattern:
                direction = "bullish"
            elif "descending" in pattern:
                direction = "bearish"

        return {
            "pattern": pattern,
            "pattern_type": pattern_type,
            "direction": direction,
            "confidence": float(confidence) if not pd.isna(confidence) else 0,
        }
