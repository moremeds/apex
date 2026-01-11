"""
Keltner Channels Indicator.

EMA-based channels using ATR for band width.
Similar to Bollinger Bands but uses ATR instead of standard deviation.

Signals:
- Price at upper band: Strong uptrend
- Price at lower band: Strong downtrend
- Price inside channels: Consolidation
- Breakout above/below: Trend continuation
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


class KeltnerChannelsIndicator(IndicatorBase):
    """
    Keltner Channels indicator.

    Default Parameters:
        ema_period: 20
        atr_period: 10
        multiplier: 2.0

    State Output:
        upper: Upper channel value
        middle: Middle line (EMA) value
        lower: Lower channel value
        position: "above", "below", or "inside"
    """

    name = "keltner"
    category = SignalCategory.VOLATILITY
    required_fields = ["high", "low", "close"]
    warmup_periods = 21

    _default_params = {
        "ema_period": 20,
        "atr_period": 10,
        "multiplier": 2.0,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Keltner Channels values."""
        ema_period = params["ema_period"]
        atr_period = params["atr_period"]
        multiplier = params["multiplier"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "kc_upper": pd.Series(dtype=float),
                    "kc_middle": pd.Series(dtype=float),
                    "kc_lower": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        # EMA of close
        if HAS_TALIB:
            ema = talib.EMA(close, timeperiod=ema_period)
            atr = talib.ATR(high, low, close, timeperiod=atr_period)
        else:
            ema = self._calculate_ema(close, ema_period)
            atr = self._calculate_atr(high, low, close, atr_period)

        kc_upper = ema + multiplier * atr
        kc_middle = ema
        kc_lower = ema - multiplier * atr

        return pd.DataFrame(
            {"kc_upper": kc_upper, "kc_middle": kc_middle, "kc_lower": kc_lower, "kc_close": close},
            index=data.index,
        )

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        n = len(data)
        ema = np.full(n, np.nan, dtype=np.float64)

        if n < period:
            return ema

        alpha = 2.0 / (period + 1)
        ema[period - 1] = np.mean(data[:period])

        for i in range(period, n):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate ATR."""
        n = len(close)
        atr = np.full(n, np.nan, dtype=np.float64)

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
        """Extract Keltner Channels state for rule evaluation."""
        upper = current.get("kc_upper", 0)
        middle = current.get("kc_middle", 0)
        lower = current.get("kc_lower", 0)
        close = current.get("kc_close", np.nan)

        if pd.isna(upper) or pd.isna(middle) or pd.isna(lower):
            return {
                "upper": 0,
                "middle": 0,
                "lower": 0,
                "position": "inside",
            }

        # Position relative to channels using close price
        position = "inside"
        if not pd.isna(close):
            if close > upper:
                position = "above"
            elif close < lower:
                position = "below"

        return {
            "upper": float(upper),
            "middle": float(middle),
            "lower": float(lower),
            "position": position,
        }
