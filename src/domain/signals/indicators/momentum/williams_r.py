"""
Williams %R Indicator.

Momentum indicator measuring overbought/oversold levels.
Similar to Stochastic but inverted scale (-100 to 0).

Signals:
- Overbought: %R > -20
- Oversold: %R < -80
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


class WilliamsRIndicator(IndicatorBase):
    """
    Williams %R indicator.

    Default Parameters:
        period: 14
        overbought: -20
        oversold: -80

    State Output:
        value: Williams %R value (-100 to 0)
        zone: "overbought", "oversold", or "neutral"
    """

    name = "williams_r"
    category = SignalCategory.MOMENTUM
    required_fields = ["high", "low", "close"]
    warmup_periods = 14

    _default_params = {
        "period": 14,
        "overbought": -20,
        "oversold": -80,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Williams %R values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"willr": pd.Series(dtype=float)}, index=data.index)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            willr = talib.WILLR(high, low, close, timeperiod=period)
        else:
            willr = self._calculate_manual(high, low, close, period)

        return pd.DataFrame({"willr": willr}, index=data.index)

    def _calculate_manual(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate Williams %R without TA-Lib."""
        n = len(close)
        willr = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            highest = np.max(high[i - period + 1 : i + 1])
            lowest = np.min(low[i - period + 1 : i + 1])
            if highest != lowest:
                willr[i] = -100 * (highest - close[i]) / (highest - lowest)
            else:
                willr[i] = -50

        return willr

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Williams %R state for rule evaluation."""
        willr = current.get("willr", -50)
        overbought = params["overbought"]
        oversold = params["oversold"]

        if pd.isna(willr):
            return {"value": -50, "zone": "neutral"}

        if willr >= overbought:
            zone = "overbought"
        elif willr <= oversold:
            zone = "oversold"
        else:
            zone = "neutral"

        return {"value": float(willr), "zone": zone}
