"""
OBV (On-Balance Volume) Indicator.

Cumulative volume indicator that adds volume on up days and
subtracts volume on down days. Confirms price trends.

Signals:
- Rising OBV with rising price: Strong uptrend
- Falling OBV with falling price: Strong downtrend
- Divergence: Potential trend reversal
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


class OBVIndicator(IndicatorBase):
    """
    On-Balance Volume indicator.

    Default Parameters:
        signal_period: 20  # For OBV signal line (SMA)

    State Output:
        obv: OBV value
        obv_signal: OBV signal line (SMA)
        trend: "accumulation" (rising) or "distribution" (falling)
    """

    name = "obv"
    category = SignalCategory.VOLUME
    required_fields = ["close", "volume"]
    warmup_periods = 2

    _default_params = {
        "signal_period": 20,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate OBV values."""
        signal_period = params["signal_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"obv": pd.Series(dtype=float), "obv_signal": pd.Series(dtype=float)},
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)

        if HAS_TALIB:
            obv = talib.OBV(close, volume)
        else:
            obv = self._calculate_manual(close, volume)

        # Signal line (SMA of OBV)
        obv_signal = self._calculate_sma(obv, signal_period)

        return pd.DataFrame({"obv": obv, "obv_signal": obv_signal}, index=data.index)

    def _calculate_manual(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate OBV without TA-Lib."""
        n = len(close)
        obv = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        return obv

    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA."""
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
        """Extract OBV state for rule evaluation."""
        obv = current.get("obv", 0)
        obv_signal = current.get("obv_signal", 0)

        if pd.isna(obv):
            return {"obv": 0, "obv_signal": 0, "trend": "neutral"}

        trend = "neutral"
        if previous is not None:
            prev_obv = previous.get("obv", 0)
            if not pd.isna(prev_obv):
                if obv > prev_obv:
                    trend = "accumulation"
                elif obv < prev_obv:
                    trend = "distribution"

        return {
            "obv": float(obv),
            "obv_signal": float(obv_signal) if not pd.isna(obv_signal) else 0,
            "trend": trend,
        }
