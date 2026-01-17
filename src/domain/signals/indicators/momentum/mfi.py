"""
MFI (Money Flow Index) Indicator.

Volume-weighted momentum indicator similar to RSI.
Measures buying and selling pressure using price and volume.

Signals:
- Overbought: MFI > 80
- Oversold: MFI < 20
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


class MFIIndicator(IndicatorBase):
    """
    Money Flow Index indicator.

    Default Parameters:
        period: 14
        overbought: 80
        oversold: 20

    State Output:
        value: MFI value (0-100)
        zone: "overbought", "oversold", or "neutral"
    """

    name = "mfi"
    category = SignalCategory.MOMENTUM
    required_fields = ["high", "low", "close", "volume"]
    warmup_periods = 15

    _default_params = {
        "period": 14,
        "overbought": 80,
        "oversold": 20,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate MFI values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"mfi": pd.Series(dtype=float)}, index=data.index)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)

        if HAS_TALIB:
            mfi = talib.MFI(high, low, close, volume, timeperiod=period)
        else:
            mfi = self._calculate_manual(high, low, close, volume, period)

        return pd.DataFrame({"mfi": mfi}, index=data.index)

    def _calculate_manual(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate MFI without TA-Lib."""
        n = len(close)
        mfi = np.full(n, np.nan, dtype=np.float64)

        # Typical price
        tp = (high + low + close) / 3

        # Raw money flow
        mf = tp * volume

        # Positive and negative money flow
        pos_mf = np.zeros(n)
        neg_mf = np.zeros(n)

        for i in range(1, n):
            if tp[i] > tp[i - 1]:
                pos_mf[i] = mf[i]
            elif tp[i] < tp[i - 1]:
                neg_mf[i] = mf[i]

        # Calculate MFI
        for i in range(period, n):
            pos_sum = np.sum(pos_mf[i - period + 1 : i + 1])
            neg_sum = np.sum(neg_mf[i - period + 1 : i + 1])

            if neg_sum == 0:
                mfi[i] = 100
            else:
                mf_ratio = pos_sum / neg_sum
                mfi[i] = 100 - (100 / (1 + mf_ratio))

        return mfi

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract MFI state for rule evaluation."""
        mfi = current.get("mfi", 50)
        overbought = params["overbought"]
        oversold = params["oversold"]

        if pd.isna(mfi):
            return {"value": 50, "zone": "neutral"}

        if mfi >= overbought:
            zone = "overbought"
        elif mfi <= oversold:
            zone = "oversold"
        else:
            zone = "neutral"

        return {"value": float(mfi), "zone": zone}
