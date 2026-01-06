"""
ATR (Average True Range) Indicator.

Measures market volatility by calculating the average of true ranges
over a specified period.

Signals:
- High ATR: High volatility
- Low ATR: Low volatility, potential breakout
- ATR expansion: Increasing volatility
- ATR contraction: Decreasing volatility
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


class ATRIndicator(IndicatorBase):
    """
    Average True Range indicator.

    Default Parameters:
        period: 14

    State Output:
        atr: ATR value in price units
        atr_percent: ATR as percentage of close
        volatility: "high", "normal", or "low" based on historical comparison
    """

    name = "atr"
    category = SignalCategory.VOLATILITY
    required_fields = ["high", "low", "close"]
    warmup_periods = 15

    _default_params = {
        "period": 14,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate ATR values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"atr": pd.Series(dtype=float)}, index=data.index)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            atr = talib.ATR(high, low, close, timeperiod=period)
        else:
            atr = self._calculate_manual(high, low, close, period)

        return pd.DataFrame({"atr": atr}, index=data.index)

    def _calculate_manual(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate ATR without TA-Lib."""
        n = len(close)
        atr = np.full(n, np.nan, dtype=np.float64)

        if n < period:
            return atr

        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Initial ATR = SMA of TR
        atr[period - 1] = np.mean(tr[:period])

        # Subsequent ATR = smoothed
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract ATR state for rule evaluation."""
        atr = current.get("atr", 0)

        if pd.isna(atr):
            return {"atr": 0, "atr_percent": 0, "volatility": "normal"}

        # Compare with previous to determine volatility level
        volatility = "normal"
        if previous is not None:
            prev_atr = previous.get("atr", 0)
            if not pd.isna(prev_atr) and prev_atr != 0:
                change = (atr - prev_atr) / prev_atr
                if change > 0.1:
                    volatility = "high"
                elif change < -0.1:
                    volatility = "low"

        return {
            "atr": float(atr),
            "atr_percent": 0,  # Would need close to calculate
            "volatility": volatility,
        }
