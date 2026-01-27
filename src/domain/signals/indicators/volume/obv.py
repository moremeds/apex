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

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate OBV values."""
        signal_period = params["signal_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "obv": pd.Series(dtype=float),
                    "obv_signal": pd.Series(dtype=float),
                    "obv_close": pd.Series(dtype=float),
                },
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

        return pd.DataFrame(
            {"obv": obv, "obv_signal": obv_signal, "obv_close": close}, index=data.index
        )

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract OBV state for rule evaluation."""
        obv = current.get("obv", 0)
        obv_signal = current.get("obv_signal", 0)
        close = current.get("obv_close", np.nan)

        if pd.isna(obv):
            return {"obv": 0, "obv_signal": 0, "trend": "neutral", "divergence": "none"}

        trend = "neutral"
        divergence = "none"

        if previous is not None:
            prev_obv = previous.get("obv", 0)
            prev_obv_signal = previous.get("obv_signal", 0)
            prev_close = previous.get("obv_close", np.nan)

            if not pd.isna(prev_obv):
                if obv > prev_obv:
                    trend = "accumulation"
                elif obv < prev_obv:
                    trend = "distribution"

            # Divergence detection using OBV vs signal line:
            # - Bullish divergence: OBV crosses above signal line while price is falling
            #   (accumulation despite price decline - smart money buying)
            # - Bearish divergence: OBV crosses below signal line while price is rising
            #   (distribution despite price rise - smart money selling)
            if (
                not pd.isna(obv_signal)
                and not pd.isna(prev_obv_signal)
                and not pd.isna(prev_close)
                and not pd.isna(close)
            ):
                # OBV crossing above signal line
                obv_cross_above = obv > obv_signal and prev_obv <= prev_obv_signal
                # OBV crossing below signal line
                obv_cross_below = obv < obv_signal and prev_obv >= prev_obv_signal

                price_down = close < prev_close
                price_up = close > prev_close

                if obv_cross_above and price_down:
                    divergence = "bullish"  # OBV rising, price falling
                elif obv_cross_below and price_up:
                    divergence = "bearish"  # OBV falling, price rising

        return {
            "obv": float(obv),
            "obv_signal": float(obv_signal) if not pd.isna(obv_signal) else 0,
            "trend": trend,
            "divergence": divergence,
        }
