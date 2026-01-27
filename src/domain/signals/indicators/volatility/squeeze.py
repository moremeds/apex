"""
Squeeze Indicator.

Identifies when Bollinger Bands are inside Keltner Channels,
indicating low volatility and potential for explosive moves.

Signals:
- Squeeze On: BB inside KC, low volatility
- Squeeze Off: BB outside KC, volatility expanding
- Momentum: Direction of expected breakout
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


class SqueezeIndicator(IndicatorBase):
    """
    Squeeze Momentum indicator (TTM Squeeze variant).

    Default Parameters:
        bb_period: 20
        bb_std: 2.0
        kc_period: 20
        kc_atr_period: 10
        kc_multiplier: 1.5
        mom_period: 12

    State Output:
        squeeze_on: True if BB is inside KC
        squeeze_off: True if BB is outside KC
        momentum: Momentum value (positive = bullish, negative = bearish)
        direction: "bullish", "bearish", or "neutral"
    """

    name = "squeeze"
    category = SignalCategory.VOLATILITY
    required_fields = ["high", "low", "close"]
    warmup_periods = 21

    _default_params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "kc_period": 20,
        "kc_atr_period": 10,
        "kc_multiplier": 1.5,
        "mom_period": 12,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Squeeze values."""
        bb_period = params["bb_period"]
        bb_std = params["bb_std"]
        kc_period = params["kc_period"]
        kc_atr_period = params["kc_atr_period"]
        kc_multiplier = params["kc_multiplier"]
        mom_period = params["mom_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "squeeze": pd.Series(dtype=float),
                    "squeeze_mom": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        n = len(close)

        # Bollinger Bands
        if HAS_TALIB:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std
            )
            kc_middle = talib.EMA(close, timeperiod=kc_period)
            atr = talib.ATR(high, low, close, timeperiod=kc_atr_period)
        else:
            bb_middle, bb_upper, bb_lower = self._calculate_bbands(close, bb_period, bb_std)
            kc_middle = self._calculate_ema(close, kc_period)
            atr = self._calculate_atr(high, low, close, kc_atr_period)

        kc_upper = kc_middle + kc_multiplier * atr
        kc_lower = kc_middle - kc_multiplier * atr

        # Squeeze = 1 if BB inside KC, -1 otherwise
        squeeze = np.full(n, np.nan, dtype=np.float64)
        for i in range(max(bb_period, kc_period, kc_atr_period), n):
            if not any(np.isnan(v) for v in [bb_upper[i], bb_lower[i], kc_upper[i], kc_lower[i]]):
                if bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]:
                    squeeze[i] = 1  # Squeeze on
                else:
                    squeeze[i] = -1  # Squeeze off

        # Momentum (using linear regression of price deviation)
        squeeze_mom = np.full(n, np.nan, dtype=np.float64)
        hl2 = (high + low) / 2
        for i in range(mom_period - 1, n):
            if not np.isnan(bb_middle[i]) and not np.isnan(kc_middle[i]):
                deviation = hl2[i] - (bb_middle[i] + kc_middle[i]) / 2
                squeeze_mom[i] = deviation

        return pd.DataFrame({"squeeze": squeeze, "squeeze_mom": squeeze_mom}, index=data.index)

    def _calculate_bbands(
        self, close: np.ndarray, period: int, std_dev: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        n = len(close)
        middle = np.full(n, np.nan, dtype=np.float64)
        upper = np.full(n, np.nan, dtype=np.float64)
        lower = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            window = close[i - period + 1 : i + 1]
            sma = np.mean(window)
            std = np.std(window, ddof=0)
            middle[i] = sma
            upper[i] = sma + std_dev * std
            lower[i] = sma - std_dev * std

        return middle, upper, lower

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
        """Extract Squeeze state for rule evaluation."""
        squeeze = current.get("squeeze", 0)
        squeeze_mom = current.get("squeeze_mom", 0)

        if pd.isna(squeeze):
            return {
                "squeeze_on": False,
                "squeeze_off": False,
                "momentum": 0,
                "direction": "neutral",
            }

        # Convert to Python bool for JSON serialization
        squeeze_on = bool(squeeze == 1)
        squeeze_off = bool(squeeze == -1)

        if pd.isna(squeeze_mom):
            direction = "neutral"
        elif squeeze_mom > 0:
            direction = "bullish"
        elif squeeze_mom < 0:
            direction = "bearish"
        else:
            direction = "neutral"

        # signal field: when squeeze releases (squeeze_off), signal direction
        # "bullish" if momentum > 0, "bearish" if momentum < 0
        signal = "neutral"
        if squeeze_off:
            signal = direction  # Use momentum direction when squeeze fires

        return {
            "squeeze_on": squeeze_on,
            "squeeze_off": squeeze_off,
            "momentum": float(squeeze_mom) if not pd.isna(squeeze_mom) else 0,
            "direction": direction,
            "signal": signal,
        }
