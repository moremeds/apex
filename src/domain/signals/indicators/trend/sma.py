"""
SMA Crossover Indicator.

Simple Moving Average crossover system using fast and slow SMAs.
Classic trend-following indicator with more lag than EMA.

Signals:
- Golden Cross: Fast SMA crosses above Slow SMA (bullish)
- Death Cross: Fast SMA crosses below Slow SMA (bearish)
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


class SMAIndicator(IndicatorBase):
    """
    SMA Crossover indicator.

    Default Parameters:
        fast_period: 50
        slow_period: 200

    State Output:
        sma_fast: Fast SMA value
        sma_slow: Slow SMA value
        trend: "bullish" if fast > slow, "bearish" otherwise
        cross: "golden" (bullish cross), "death" (bearish cross), or None
        spread: Percentage spread between SMAs
    """

    name = "sma"
    category = SignalCategory.TREND
    required_fields = ["close"]
    warmup_periods = 201

    _default_params = {
        "fast_period": 50,
        "slow_period": 200,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate SMA Crossover values."""
        fast_p = params["fast_period"]
        slow_p = params["slow_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"sma_fast": pd.Series(dtype=float), "sma_slow": pd.Series(dtype=float)},
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            sma_fast = talib.SMA(close, timeperiod=fast_p)
            sma_slow = talib.SMA(close, timeperiod=slow_p)
        else:
            sma_fast = self._calculate_sma(close, fast_p)
            sma_slow = self._calculate_sma(close, slow_p)

        return pd.DataFrame({"sma_fast": sma_fast, "sma_slow": sma_slow}, index=data.index)

    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA without TA-Lib."""
        n = len(data)
        sma = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            sma[i] = np.mean(data[i - period + 1 : i + 1])

        return sma

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract SMA Crossover state for rule evaluation."""
        sma_fast = current.get("sma_fast", 0)
        sma_slow = current.get("sma_slow", 0)

        if pd.isna(sma_fast) or pd.isna(sma_slow):
            return {
                "sma_fast": 0,
                "sma_slow": 0,
                "trend": "neutral",
                "cross": None,
                "spread": 0,
            }

        trend = "bullish" if sma_fast > sma_slow else "bearish"
        spread = ((sma_fast - sma_slow) / sma_slow * 100) if sma_slow != 0 else 0

        cross = None
        if previous is not None:
            prev_fast = previous.get("sma_fast", 0)
            prev_slow = previous.get("sma_slow", 0)
            if not pd.isna(prev_fast) and not pd.isna(prev_slow):
                if prev_fast <= prev_slow and sma_fast > sma_slow:
                    cross = "golden"
                elif prev_fast >= prev_slow and sma_fast < sma_slow:
                    cross = "death"

        return {
            "sma_fast": float(sma_fast),
            "sma_slow": float(sma_slow),
            "trend": trend,
            "cross": cross,
            "spread": float(spread),
        }
