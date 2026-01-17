"""
Aroon Indicator.

Identifies trend changes and strength by measuring time since
highest high and lowest low.

Signals:
- Aroon Up > 70: Strong uptrend
- Aroon Down > 70: Strong downtrend
- Aroon crossover: Trend change
- Oscillator > 0: Bullish, < 0: Bearish
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from ...models import SignalCategory
from ..base import IndicatorBase


class AroonIndicator(IndicatorBase):
    """
    Aroon indicator.

    Default Parameters:
        period: 25
        strong_threshold: 70

    State Output:
        aroon_up: Aroon Up value (0-100)
        aroon_down: Aroon Down value (0-100)
        oscillator: Aroon Up - Aroon Down (-100 to 100)
        trend: "bullish", "bearish", or "consolidating"
        cross: "bullish", "bearish", or None
    """

    name = "aroon"
    category = SignalCategory.TREND
    required_fields = ["high", "low"]
    warmup_periods = 26

    _default_params = {
        "period": 25,
        "strong_threshold": 70,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Aroon values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"aroon_up": pd.Series(dtype=float), "aroon_down": pd.Series(dtype=float)},
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)

        if HAS_TALIB:
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=period)
        else:
            aroon_up, aroon_down = self._calculate_manual(high, low, period)

        return pd.DataFrame({"aroon_up": aroon_up, "aroon_down": aroon_down}, index=data.index)

    def _calculate_manual(
        self, high: np.ndarray, low: np.ndarray, period: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Aroon without TA-Lib."""
        n = len(high)
        aroon_up = np.full(n, np.nan, dtype=np.float64)
        aroon_down = np.full(n, np.nan, dtype=np.float64)

        # TA-Lib AROON uses 'period' bars: from i-period+1 to i (inclusive)
        for i in range(period, n):
            # Window of exactly 'period' bars (matching TA-Lib behavior)
            window_high = high[i - period + 1 : i + 1]
            window_low = low[i - period + 1 : i + 1]

            # argmax returns index from start of window (0 = oldest, period-1 = current)
            # days_since = position from end of window
            days_since_high = (period - 1) - np.argmax(window_high)
            days_since_low = (period - 1) - np.argmin(window_low)

            aroon_up[i] = ((period - days_since_high) / period) * 100
            aroon_down[i] = ((period - days_since_low) / period) * 100

        return aroon_up, aroon_down

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Aroon state for rule evaluation."""
        aroon_up = current.get("aroon_up", 50)
        aroon_down = current.get("aroon_down", 50)
        strong = params["strong_threshold"]

        if pd.isna(aroon_up) or pd.isna(aroon_down):
            return {
                "aroon_up": 50,
                "aroon_down": 50,
                "oscillator": 0,
                "trend": "consolidating",
                "cross": None,
            }

        oscillator = aroon_up - aroon_down

        if aroon_up > strong and aroon_up > aroon_down:
            trend = "bullish"
        elif aroon_down > strong and aroon_down > aroon_up:
            trend = "bearish"
        else:
            trend = "consolidating"

        cross = None
        if previous is not None:
            prev_up = previous.get("aroon_up", 50)
            prev_down = previous.get("aroon_down", 50)
            if not pd.isna(prev_up) and not pd.isna(prev_down):
                if prev_up <= prev_down and aroon_up > aroon_down:
                    cross = "bullish"
                elif prev_up >= prev_down and aroon_up < aroon_down:
                    cross = "bearish"

        return {
            "aroon_up": float(aroon_up),
            "aroon_down": float(aroon_down),
            "oscillator": float(oscillator),
            "trend": trend,
            "cross": cross,
        }
