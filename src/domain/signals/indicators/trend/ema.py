"""
EMA Crossover Indicator.

Exponential Moving Average crossover system using fast and slow EMAs.
Classic trend-following indicator.

Signals:
- Golden Cross: Fast EMA crosses above Slow EMA (bullish)
- Death Cross: Fast EMA crosses below Slow EMA (bearish)
- Trend direction: Fast EMA position relative to Slow EMA
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


class EMAIndicator(IndicatorBase):
    """
    EMA Crossover indicator.

    Default Parameters:
        fast_period: 12
        slow_period: 26

    State Output:
        fast_ema: Fast EMA value
        slow_ema: Slow EMA value
        trend: "bullish" if fast > slow, "bearish" otherwise
        cross: "golden" (bullish cross), "death" (bearish cross), or None
        spread: Percentage spread between EMAs
    """

    name = "ema"
    category = SignalCategory.TREND
    required_fields = ["close"]
    warmup_periods = 27

    _default_params = {
        "fast_period": 12,
        "slow_period": 26,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate EMA Crossover values."""
        fast_p = params["fast_period"]
        slow_p = params["slow_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"ema_fast": pd.Series(dtype=float), "ema_slow": pd.Series(dtype=float)},
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            ema_fast = talib.EMA(close, timeperiod=fast_p)
            ema_slow = talib.EMA(close, timeperiod=slow_p)
        else:
            ema_fast = self._calculate_ema(close, fast_p)
            ema_slow = self._calculate_ema(close, slow_p)

        return pd.DataFrame(
            {"ema_fast": ema_fast, "ema_slow": ema_slow}, index=data.index
        )

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA without TA-Lib."""
        n = len(data)
        ema = np.full(n, np.nan, dtype=np.float64)

        if n < period:
            return ema

        alpha = 2.0 / (period + 1)
        ema[period - 1] = np.mean(data[:period])

        for i in range(period, n):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract EMA Crossover state for rule evaluation."""
        ema_fast = current.get("ema_fast", 0)
        ema_slow = current.get("ema_slow", 0)

        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return {
                "fast_ema": 0,
                "slow_ema": 0,
                "trend": "neutral",
                "cross": None,
                "spread": 0,
            }

        trend = "bullish" if ema_fast > ema_slow else "bearish"
        spread = ((ema_fast - ema_slow) / ema_slow * 100) if ema_slow != 0 else 0

        cross = None
        if previous is not None:
            prev_fast = previous.get("ema_fast", 0)
            prev_slow = previous.get("ema_slow", 0)
            if not pd.isna(prev_fast) and not pd.isna(prev_slow):
                if prev_fast <= prev_slow and ema_fast > ema_slow:
                    cross = "golden"
                elif prev_fast >= prev_slow and ema_fast < ema_slow:
                    cross = "death"

        return {
            "fast_ema": float(ema_fast),
            "slow_ema": float(ema_slow),
            "trend": trend,
            "cross": cross,
            "spread": float(spread),
        }
