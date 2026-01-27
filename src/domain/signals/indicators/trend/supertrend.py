"""
SuperTrend Indicator (ZLEMA-based, TradeCat spec).

ATR-based trend indicator using Zero-Lag EMA for smoother response.
Provides clear trend direction with automatic trailing stop levels.

TradeCat Parameters:
    period: 70 (ATR and ZLEMA period)
    multiplier: 1.2 (band width multiplier)
    lag: 34 (ZLEMA lag offset)

Signals:
- SuperTrend below price: Bullish trend
- SuperTrend above price: Bearish trend
- Flip: Trend reversal signal
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


class SuperTrendIndicator(IndicatorBase):
    """
    ZLEMA-based SuperTrend indicator (TradeCat implementation).

    Uses Zero-Lag EMA instead of simple HL2 midpoint for smoother,
    more responsive trend detection.

    Default Parameters (TradeCat):
        period: 70 (ATR and ZLEMA lookback)
        multiplier: 1.2 (band width multiplier)
        lag: 34 (ZLEMA lag offset for zero-lag calculation)

    State Output:
        supertrend: SuperTrend value (support/resistance level)
        direction: "bullish" or "bearish"
        flip: True if trend just reversed, False otherwise
        distance: Percentage distance from price to supertrend
    """

    name = "supertrend"
    category = SignalCategory.TREND
    required_fields = ["high", "low", "close"]
    warmup_periods = 105  # period (70) + lag (34) + buffer

    _default_params = {
        "period": 70,  # TradeCat: ATR and ZLEMA period
        "multiplier": 1.2,  # TradeCat: tighter bands than standard 3.0
        "lag": 34,  # TradeCat: ZLEMA lag offset
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate ZLEMA-based SuperTrend values."""
        period = params["period"]
        multiplier = params["multiplier"]
        lag = params["lag"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "supertrend": pd.Series(dtype=float),
                    "supertrend_direction": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        n = len(close)

        # Calculate ATR
        if HAS_TALIB:
            atr = talib.ATR(high, low, close, timeperiod=period)
        else:
            atr = self._calculate_atr(high, low, close, period)

        # Calculate ZLEMA (Zero-Lag EMA)
        # source = close + (close - close.shift(lag))
        # zlema = source.ewm(span=period).mean()
        zlema = self._calculate_zlema(close, period, lag)

        # Calculate bands using ZLEMA as center
        upper_band = zlema + multiplier * atr
        lower_band = zlema - multiplier * atr

        # Initialize arrays
        supertrend = np.full(n, np.nan, dtype=np.float64)
        direction = np.full(n, 1.0, dtype=np.float64)

        # Start from where both ATR and ZLEMA are valid
        start_idx = max(period, lag + period)
        if start_idx >= n:
            return pd.DataFrame(
                {"supertrend": supertrend, "supertrend_direction": direction},
                index=data.index,
            )

        supertrend[start_idx] = lower_band[start_idx]
        direction[start_idx] = 1

        for i in range(start_idx + 1, n):
            if np.isnan(atr[i]) or np.isnan(zlema[i]):
                continue

            # Final upper and lower bands (with trailing logic)
            if lower_band[i] > supertrend[i - 1] or close[i - 1] < supertrend[i - 1]:
                final_lower = lower_band[i]
            else:
                final_lower = max(lower_band[i], supertrend[i - 1])

            if upper_band[i] < supertrend[i - 1] or close[i - 1] > supertrend[i - 1]:
                final_upper = upper_band[i]
            else:
                final_upper = min(upper_band[i], supertrend[i - 1])

            # Determine SuperTrend value based on price action
            if direction[i - 1] == 1:
                if close[i] < final_lower:
                    supertrend[i] = final_upper
                    direction[i] = -1
                else:
                    supertrend[i] = final_lower
                    direction[i] = 1
            else:
                if close[i] > final_upper:
                    supertrend[i] = final_lower
                    direction[i] = 1
                else:
                    supertrend[i] = final_upper
                    direction[i] = -1

        return pd.DataFrame(
            {
                "supertrend": supertrend,
                "supertrend_direction": direction,
                "supertrend_close": close,
            },
            index=data.index,
        )

    def _calculate_zlema(self, close: np.ndarray, period: int, lag: int) -> np.ndarray:
        """Calculate Zero-Lag EMA.

        ZLEMA reduces lag by using: source = close + (close - close[lag_periods_ago])
        Then applying EMA to this adjusted source.
        """
        n = len(close)
        zlema = np.full(n, np.nan, dtype=np.float64)

        if n < lag + period:
            return zlema

        # Create lagged source: close + (close - close.shift(lag))
        source = np.full(n, np.nan, dtype=np.float64)
        for i in range(lag, n):
            source[i] = close[i] + (close[i] - close[i - lag])

        # Apply EMA to source
        alpha = 2.0 / (period + 1)
        start_idx = lag + period - 1

        # Initialize EMA with SMA of first valid period
        if start_idx < n:
            zlema[start_idx] = np.nanmean(source[lag : start_idx + 1])

            for i in range(start_idx + 1, n):
                if not np.isnan(source[i]):
                    zlema[i] = alpha * source[i] + (1 - alpha) * zlema[i - 1]

        return zlema

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate ATR without TA-Lib."""
        n = len(close)
        atr = np.full(n, np.nan)

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
        """Extract SuperTrend state for rule evaluation."""
        supertrend = current.get("supertrend", 0)
        dir_val = current.get("supertrend_direction", 0)
        close = current.get("supertrend_close", np.nan)

        if pd.isna(supertrend) or pd.isna(dir_val):
            return {
                "supertrend": 0,
                "direction": "neutral",
                "flip": False,
                "distance": 0,
            }

        direction = "bullish" if dir_val > 0 else "bearish"

        flip = False
        if previous is not None:
            prev_direction = previous.get("supertrend_direction", 0)
            if not pd.isna(prev_direction) and prev_direction != dir_val:
                flip = True

        # Calculate percentage distance from close to supertrend
        distance = 0.0
        if not pd.isna(close) and supertrend != 0:
            distance = abs(close - supertrend) / supertrend * 100

        return {
            "supertrend": float(supertrend),
            "direction": direction,
            "flip": flip,
            "distance": float(distance),
        }
