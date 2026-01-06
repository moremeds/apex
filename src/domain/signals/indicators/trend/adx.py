"""
ADX (Average Directional Index) Indicator.

Measures trend strength regardless of direction.
Combines +DI and -DI to determine overall trend strength.

Signals:
- ADX > 25: Strong trend
- ADX > 40: Very strong trend
- +DI > -DI: Bullish trend direction
- -DI > +DI: Bearish trend direction
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


class ADXIndicator(IndicatorBase):
    """
    Average Directional Index indicator.

    Default Parameters:
        period: 14
        strong_threshold: 25
        very_strong_threshold: 40

    State Output:
        adx: ADX value (0-100)
        plus_di: +DI value
        minus_di: -DI value
        trend_strength: "weak", "strong", or "very_strong"
        trend_direction: "bullish" if +DI > -DI, "bearish" otherwise
    """

    name = "adx"
    category = SignalCategory.TREND
    required_fields = ["high", "low", "close"]
    warmup_periods = 28

    _default_params = {
        "period": 14,
        "strong_threshold": 25,
        "very_strong_threshold": 40,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate ADX values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "adx": pd.Series(dtype=float),
                    "plus_di": pd.Series(dtype=float),
                    "minus_di": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            adx = talib.ADX(high, low, close, timeperiod=period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
        else:
            adx, plus_di, minus_di = self._calculate_manual(high, low, close, period)

        return pd.DataFrame(
            {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}, index=data.index
        )

    def _calculate_manual(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ADX without TA-Lib."""
        n = len(close)
        adx = np.full(n, np.nan, dtype=np.float64)
        plus_di = np.full(n, np.nan, dtype=np.float64)
        minus_di = np.full(n, np.nan, dtype=np.float64)

        if n < period + 1:
            return adx, plus_di, minus_di

        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smoothed TR, +DM, -DM
        atr = np.zeros(n)
        smoothed_plus_dm = np.zeros(n)
        smoothed_minus_dm = np.zeros(n)

        atr[period] = np.sum(tr[1 : period + 1])
        smoothed_plus_dm[period] = np.sum(plus_dm[1 : period + 1])
        smoothed_minus_dm[period] = np.sum(minus_dm[1 : period + 1])

        for i in range(period + 1, n):
            atr[i] = atr[i - 1] - (atr[i - 1] / period) + tr[i]
            smoothed_plus_dm[i] = (
                smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / period) + plus_dm[i]
            )
            smoothed_minus_dm[i] = (
                smoothed_minus_dm[i - 1]
                - (smoothed_minus_dm[i - 1] / period)
                + minus_dm[i]
            )

        # +DI and -DI
        for i in range(period, n):
            if atr[i] != 0:
                plus_di[i] = 100 * smoothed_plus_dm[i] / atr[i]
                minus_di[i] = 100 * smoothed_minus_dm[i] / atr[i]

        # DX and ADX
        dx = np.zeros(n)
        for i in range(period, n):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

        adx[2 * period - 1] = np.mean(dx[period : 2 * period])
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx, plus_di, minus_di

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract ADX state for rule evaluation."""
        adx = current.get("adx", 0)
        plus_di = current.get("plus_di", 0)
        minus_di = current.get("minus_di", 0)
        strong = params["strong_threshold"]
        very_strong = params["very_strong_threshold"]

        if pd.isna(adx):
            return {
                "adx": 0,
                "plus_di": 0,
                "minus_di": 0,
                "trend_strength": "weak",
                "trend_direction": "neutral",
            }

        if adx >= very_strong:
            trend_strength = "very_strong"
        elif adx >= strong:
            trend_strength = "strong"
        else:
            trend_strength = "weak"

        trend_direction = "bullish" if plus_di > minus_di else "bearish"

        return {
            "adx": float(adx),
            "plus_di": float(plus_di) if not pd.isna(plus_di) else 0,
            "minus_di": float(minus_di) if not pd.isna(minus_di) else 0,
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
        }
