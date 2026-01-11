"""
SuperTrend Indicator.

ATR-based trend indicator that provides clear trend direction
with automatic trailing stop levels.

Signals:
- SuperTrend below price: Bullish trend
- SuperTrend above price: Bearish trend
- Flip: Trend reversal signal
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


class SuperTrendIndicator(IndicatorBase):
    """
    SuperTrend indicator.

    Default Parameters:
        period: 10
        multiplier: 3.0

    State Output:
        supertrend: SuperTrend value (support/resistance level)
        trend: "bullish" or "bearish"
        flip: True if trend just reversed, False otherwise
        distance: Percentage distance from price to supertrend
    """

    name = "supertrend"
    category = SignalCategory.TREND
    required_fields = ["high", "low", "close"]
    warmup_periods = 11

    _default_params = {
        "period": 10,
        "multiplier": 3.0,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate SuperTrend values."""
        period = params["period"]
        multiplier = params["multiplier"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "supertrend": pd.Series(dtype=float),
                    "supertrend_direction": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        n = len(close)

        # Calculate ATR
        if HAS_TALIB:
            atr = talib.ATR(high, low, close, timeperiod=period)
        else:
            atr = self._calculate_atr(high, low, close, period)

        # Calculate basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        # Initialize arrays
        supertrend = np.full(n, np.nan, dtype=np.float64)
        direction = np.full(n, 1.0, dtype=np.float64)

        # Start from period index where ATR is valid
        start_idx = period
        if start_idx >= n:
            return pd.DataFrame(
                {"supertrend": supertrend, "supertrend_direction": direction},
                index=data.index,
            )

        supertrend[start_idx] = lower_band[start_idx]
        direction[start_idx] = 1

        for i in range(start_idx + 1, n):
            if np.isnan(atr[i]):
                continue

            # Final upper and lower bands
            if lower_band[i] > supertrend[i - 1] or close[i - 1] < supertrend[i - 1]:
                final_lower = lower_band[i]
            else:
                final_lower = max(lower_band[i], supertrend[i - 1])

            if upper_band[i] < supertrend[i - 1] or close[i - 1] > supertrend[i - 1]:
                final_upper = upper_band[i]
            else:
                final_upper = min(upper_band[i], supertrend[i - 1])

            # Determine SuperTrend value based on price action
            if direction[i - 1] == 1:
                if close[i] < final_lower:
                    supertrend[i] = final_upper
                    direction[i] = -1
                else:
                    supertrend[i] = final_lower
                    direction[i] = 1
            else:
                if close[i] > final_upper:
                    supertrend[i] = final_lower
                    direction[i] = 1
                else:
                    supertrend[i] = final_upper
                    direction[i] = -1

        return pd.DataFrame(
            {"supertrend": supertrend, "supertrend_direction": direction},
            index=data.index,
        )

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate ATR without TA-Lib."""
        n = len(close)
        atr = np.full(n, np.nan)

        if n < period:
            return atr

        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr[period - 1] = np.mean(tr[:period])

        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract SuperTrend state for rule evaluation."""
        supertrend = current.get("supertrend", 0)
        direction = current.get("supertrend_direction", 0)

        if pd.isna(supertrend) or pd.isna(direction):
            return {
                "supertrend": 0,
                "trend": "neutral",
                "flip": False,
                "distance": 0,
            }

        trend = "bullish" if direction > 0 else "bearish"

        flip = False
        if previous is not None:
            prev_direction = previous.get("supertrend_direction", 0)
            if not pd.isna(prev_direction) and prev_direction != direction:
                flip = True

        return {
            "supertrend": float(supertrend),
            "trend": trend,
            "flip": flip,
            "distance": 0,  # Would need close price to calculate
        }
