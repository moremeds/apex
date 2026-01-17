"""
TRIX (Triple Exponential Average) Indicator.

Triple-smoothed EMA oscillator that filters out market noise.
Shows rate of change of a triple-smoothed EMA.

Signals:
- TRIX > 0: Bullish
- TRIX < 0: Bearish
- Signal line crossover: Trading signal
- Zero line crossover: Trend change
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


class TRIXIndicator(IndicatorBase):
    """
    TRIX indicator.

    Default Parameters:
        period: 15
        signal_period: 9

    State Output:
        trix: TRIX value
        signal: Signal line value
        direction: "bullish" if > 0, "bearish" if < 0
        cross: "bullish", "bearish", or None
    """

    name = "trix"
    category = SignalCategory.TREND
    required_fields = ["close"]
    warmup_periods = 46  # ~3 * period

    _default_params = {
        "period": 15,
        "signal_period": 9,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate TRIX values."""
        period = params["period"]
        signal_p = params["signal_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"trix": pd.Series(dtype=float), "trix_signal": pd.Series(dtype=float)},
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            trix = talib.TRIX(close, timeperiod=period)
            trix_signal = talib.EMA(trix, timeperiod=signal_p)
        else:
            trix = self._calculate_trix(close, period)
            trix_signal = self._calculate_ema(trix, signal_p)

        return pd.DataFrame({"trix": trix, "trix_signal": trix_signal}, index=data.index)

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        n = len(data)
        ema = np.full(n, np.nan, dtype=np.float64)

        # Find first valid value
        first_valid = 0
        while first_valid < n and np.isnan(data[first_valid]):
            first_valid += 1

        if first_valid >= n or first_valid + period > n:
            return ema

        alpha = 2.0 / (period + 1)
        ema[first_valid + period - 1] = np.nanmean(data[first_valid : first_valid + period])

        for i in range(first_valid + period, n):
            if not np.isnan(data[i]):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
            else:
                ema[i] = ema[i - 1]

        return ema

    def _calculate_trix(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate TRIX without TA-Lib."""
        # Triple EMA
        ema1 = self._calculate_ema(close, period)
        ema2 = self._calculate_ema(ema1, period)
        ema3 = self._calculate_ema(ema2, period)

        # Rate of change of triple EMA
        n = len(close)
        trix = np.full(n, np.nan, dtype=np.float64)

        for i in range(1, n):
            if not np.isnan(ema3[i]) and not np.isnan(ema3[i - 1]) and ema3[i - 1] != 0:
                trix[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100

        return trix

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract TRIX state for rule evaluation."""
        trix = current.get("trix", 0)
        trix_signal = current.get("trix_signal", 0)

        if pd.isna(trix):
            return {"trix": 0, "signal": 0, "direction": "neutral", "cross": None}

        direction = "bullish" if trix > 0 else "bearish" if trix < 0 else "neutral"

        cross = None
        if previous is not None and not pd.isna(trix_signal):
            prev_trix = previous.get("trix", 0)
            prev_signal = previous.get("trix_signal", 0)
            if not pd.isna(prev_trix) and not pd.isna(prev_signal):
                if prev_trix <= prev_signal and trix > trix_signal:
                    cross = "bullish"
                elif prev_trix >= prev_signal and trix < trix_signal:
                    cross = "bearish"

        return {
            "trix": float(trix),
            "signal": float(trix_signal) if not pd.isna(trix_signal) else 0,
            "direction": direction,
            "cross": cross,
        }
