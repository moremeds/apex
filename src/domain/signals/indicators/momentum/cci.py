"""
CCI (Commodity Channel Index) Indicator.

Measures the current price level relative to an average price level.
Originally designed for commodities, now used for all securities.

Signals:
- Overbought: CCI > 100
- Oversold: CCI < -100
- Trend: Zero line crossovers
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


class CCIIndicator(IndicatorBase):
    """
    Commodity Channel Index indicator.

    Default Parameters:
        period: 20
        overbought: 100
        oversold: -100

    State Output:
        value: CCI value
        zone: "overbought", "oversold", or "neutral"
    """

    name = "cci"
    category = SignalCategory.MOMENTUM
    required_fields = ["high", "low", "close"]
    warmup_periods = 20

    _default_params = {
        "period": 20,
        "overbought": 100,
        "oversold": -100,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate CCI values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"cci": pd.Series(dtype=float)}, index=data.index)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            cci = talib.CCI(high, low, close, timeperiod=period)
        else:
            cci = self._calculate_manual(high, low, close, period)

        return pd.DataFrame({"cci": cci}, index=data.index)

    def _calculate_manual(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate CCI without TA-Lib."""
        n = len(close)
        cci = np.full(n, np.nan, dtype=np.float64)

        # Typical price
        tp = (high + low + close) / 3

        for i in range(period - 1, n):
            tp_window = tp[i - period + 1 : i + 1]
            sma = np.mean(tp_window)
            mean_dev = np.mean(np.abs(tp_window - sma))
            if mean_dev != 0:
                cci[i] = (tp[i] - sma) / (0.015 * mean_dev)
            else:
                cci[i] = 0

        return cci

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract CCI state for rule evaluation."""
        cci = current.get("cci", 0)
        overbought = params["overbought"]
        oversold = params["oversold"]

        if pd.isna(cci):
            return {"value": 0, "zone": "neutral"}

        if cci >= overbought:
            zone = "overbought"
        elif cci <= oversold:
            zone = "oversold"
        else:
            zone = "neutral"

        return {"value": float(cci), "zone": zone}
