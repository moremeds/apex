"""
Trendline Indicator.

Automatically detects trendlines by connecting swing highs
(resistance trendline) and swing lows (support trendline).

Signals:
- Upward trendline: Bullish support line
- Downward trendline: Bearish resistance line
- Trendline break: Potential reversal
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class TrendlineIndicator(IndicatorBase):
    """
    Automatic Trendline detector.

    Default Parameters:
        lookback: 50  # Bars to analyze
        min_touches: 2  # Minimum points to form trendline

    State Output:
        support_slope: Slope of support trendline (positive = uptrend)
        resistance_slope: Slope of resistance trendline
        support_level: Current support trendline price
        resistance_level: Current resistance trendline price
        trend: "uptrend", "downtrend", or "sideways"
    """

    name = "trendline"
    category = SignalCategory.PATTERN
    required_fields = ["high", "low"]
    warmup_periods = 51

    _default_params = {
        "lookback": 50,
        "min_touches": 2,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate trendline values."""
        lookback = params["lookback"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "tl_support_slope": pd.Series(dtype=float),
                    "tl_support_level": pd.Series(dtype=float),
                    "tl_resistance_slope": pd.Series(dtype=float),
                    "tl_resistance_level": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(high)

        support_slope = np.full(n, np.nan, dtype=np.float64)
        support_level = np.full(n, np.nan, dtype=np.float64)
        resistance_slope = np.full(n, np.nan, dtype=np.float64)
        resistance_level = np.full(n, np.nan, dtype=np.float64)

        for i in range(lookback, n):
            # Get window
            window_high = high[i - lookback : i + 1]
            window_low = low[i - lookback : i + 1]

            # Simple linear regression on swing points
            # Support line: linear fit of recent lows
            x = np.arange(lookback + 1)

            # Fit support line (on lows)
            try:
                support_coeffs = np.polyfit(x, window_low, 1)
                support_slope[i] = support_coeffs[0]  # Slope
                support_level[i] = support_coeffs[0] * lookback + support_coeffs[1]
            except (np.linalg.LinAlgError, ValueError):
                pass

            # Fit resistance line (on highs)
            try:
                resistance_coeffs = np.polyfit(x, window_high, 1)
                resistance_slope[i] = resistance_coeffs[0]
                resistance_level[i] = resistance_coeffs[0] * lookback + resistance_coeffs[1]
            except (np.linalg.LinAlgError, ValueError):
                pass

        return pd.DataFrame(
            {
                "tl_support_slope": support_slope,
                "tl_support_level": support_level,
                "tl_resistance_slope": resistance_slope,
                "tl_resistance_level": resistance_level,
            },
            index=data.index,
        )

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract trendline state for rule evaluation."""
        support_slope = current.get("tl_support_slope", 0)
        support_level = current.get("tl_support_level", 0)
        resistance_slope = current.get("tl_resistance_slope", 0)
        resistance_level = current.get("tl_resistance_level", 0)

        if pd.isna(support_slope):
            support_slope = 0
        if pd.isna(resistance_slope):
            resistance_slope = 0

        # Determine trend from slopes
        avg_slope = (support_slope + resistance_slope) / 2
        if avg_slope > 0.001:  # Small positive threshold
            trend = "uptrend"
        elif avg_slope < -0.001:
            trend = "downtrend"
        else:
            trend = "sideways"

        return {
            "support_slope": float(support_slope),
            "resistance_slope": float(resistance_slope),
            "support_level": float(support_level) if not pd.isna(support_level) else 0,
            "resistance_level": float(resistance_level) if not pd.isna(resistance_level) else 0,
            "trend": trend,
        }
