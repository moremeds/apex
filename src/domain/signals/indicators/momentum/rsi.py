"""
RSI (Relative Strength Index) Indicator.

Measures the speed and magnitude of recent price changes to evaluate
overbought or oversold conditions.

Signals:
- Overbought (>70): Potential sell signal
- Oversold (<30): Potential buy signal
- Zone transitions: Buy on oversold exit, sell on overbought exit
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from ...models import SignalCategory
from ..base import IndicatorBase


class RSIIndicator(IndicatorBase):
    """
    Relative Strength Index indicator.

    Calculates RSI and provides zone classification for rule evaluation.

    Default Parameters:
        period: 14
        overbought: 70
        oversold: 30

    State Output:
        value: Current RSI value (0-100)
        zone: "overbought", "oversold", or "neutral"
    """

    name = "rsi"
    category = SignalCategory.MOMENTUM
    required_fields = ["close"]
    warmup_periods = 15

    _default_params = {
        "period": 14,
        "overbought": 70,
        "oversold": 30,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate RSI values.

        Args:
            data: DataFrame with 'close' column
            params: Parameters including 'period'

        Returns:
            DataFrame with 'rsi' column
        """
        period = params["period"]

        # Handle empty/short data - return all NaN
        if len(data) == 0:
            return pd.DataFrame({"rsi": pd.Series(dtype=float)}, index=data.index)

        if len(data) <= period:
            return pd.DataFrame(
                {"rsi": np.full(len(data), np.nan)},
                index=data.index,
            )

        # Convert to float64 to handle integer dtypes safely
        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            rsi_values = talib.RSI(close, timeperiod=period)
        else:
            rsi_values = self._calculate_rsi_manual(close, period)

        return pd.DataFrame({"rsi": rsi_values}, index=data.index)

    def _calculate_rsi_manual(self, close: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate RSI without TA-Lib (fallback).

        Uses Wilder's smoothing method.

        Args:
            close: Close prices as float64 array (already converted in _calculate)
            period: RSI period

        Returns:
            RSI values as float64 array with NaN for warmup period
        """
        n = len(close)

        # Initialize output as float64 with NaN
        rsi = np.full(n, np.nan, dtype=np.float64)

        # Calculate price changes (delta[0] = 0 since no previous price)
        delta = np.zeros(n, dtype=np.float64)
        delta[1:] = close[1:] - close[:-1]

        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # Wilder's smoothing (exponential moving average)
        alpha = 1.0 / period
        avg_gains = np.zeros(n, dtype=np.float64)
        avg_losses = np.zeros(n, dtype=np.float64)

        # Initialize with SMA for first period
        avg_gains[period] = np.mean(gains[1:period + 1])
        avg_losses[period] = np.mean(losses[1:period + 1])

        # Calculate EMA for remaining periods
        for i in range(period + 1, n):
            avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i - 1]
            avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i - 1]

        # Calculate RS and RSI for valid periods only
        for i in range(period, n):
            if avg_losses[i] != 0:
                rs = avg_gains[i] / avg_losses[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                # No losses means RSI = 100
                rsi[i] = 100.0 if avg_gains[i] > 0 else 50.0

        return rsi

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract RSI state for rule evaluation.

        Args:
            current: Current indicator values
            previous: Previous indicator values
            params: Merged parameters including overbought/oversold thresholds

        Returns:
            State dict with value and zone
        """
        rsi = current.get("rsi", 50)
        overbought = params["overbought"]
        oversold = params["oversold"]

        if pd.isna(rsi):
            zone = "neutral"
            rsi = 50
        elif rsi >= overbought:
            zone = "overbought"
        elif rsi <= oversold:
            zone = "oversold"
        else:
            zone = "neutral"

        return {
            "value": float(rsi),
            "zone": zone,
        }
