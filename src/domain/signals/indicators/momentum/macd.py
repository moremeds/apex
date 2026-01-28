"""
MACD (Moving Average Convergence Divergence) Indicator.

Measures the relationship between two exponential moving averages.
Generates signals on line crossovers and histogram direction changes.

Signals:
- Bullish: MACD crosses above signal line
- Bearish: MACD crosses below signal line
- Histogram momentum: Rising/falling histogram
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


class MACDIndicator(IndicatorBase):
    """
    MACD indicator with line, signal, and histogram.

    Default Parameters:
        fast_period: 12
        slow_period: 26
        signal_period: 9
        histogram_multiplier: 2 (TradeCat uses 2x for amplified histogram)

    State Output:
        macd: MACD line value (DIF)
        signal: Signal line value (DEA)
        histogram: multiplier * (MACD - Signal), default 2x per TradeCat spec
        direction: "bullish" or "bearish" based on MACD vs Signal
    """

    name = "macd"
    category = SignalCategory.MOMENTUM
    required_fields = ["close"]
    warmup_periods = 35  # slow_period + signal_period

    _default_params = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "histogram_multiplier": 2,  # TradeCat uses 2x histogram
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate MACD line, signal line, and histogram."""
        fast = params["fast_period"]
        slow = params["slow_period"]
        signal = params["signal_period"]
        multiplier = params["histogram_multiplier"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "macd": pd.Series(dtype=float),
                    "signal": pd.Series(dtype=float),
                    "histogram": pd.Series(dtype=float),
                },
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            macd_line, signal_line, _ = talib.MACD(
                close, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            # Apply histogram multiplier (TradeCat uses 2x)
            histogram = multiplier * (macd_line - signal_line)
        else:
            macd_line, signal_line, histogram = self._calculate_manual(
                close, fast, slow, signal, multiplier
            )

        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram},
            index=data.index,
        )

    def _calculate_manual(
        self, close: np.ndarray, fast: int, slow: int, signal: int, multiplier: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD without TA-Lib."""
        n = len(close)

        def ema(data: np.ndarray, period: int) -> np.ndarray:
            result = np.full(n, np.nan, dtype=np.float64)
            if n < period:
                return result
            alpha = 2.0 / (period + 1)
            result[period - 1] = np.mean(data[:period])
            for i in range(period, n):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result

        fast_ema = ema(close, fast)
        slow_ema = ema(close, slow)
        macd_line = fast_ema - slow_ema

        # Signal line is EMA of MACD
        signal_line = np.full(n, np.nan, dtype=np.float64)
        valid_start = slow - 1 + signal - 1
        if n > valid_start:
            alpha = 2.0 / (signal + 1)
            signal_line[valid_start] = np.nanmean(macd_line[slow - 1 : valid_start + 1])
            for i in range(valid_start + 1, n):
                if not np.isnan(macd_line[i]):
                    signal_line[i] = alpha * macd_line[i] + (1 - alpha) * signal_line[i - 1]

        # Apply histogram multiplier (TradeCat uses 2x)
        histogram = multiplier * (macd_line - signal_line)
        return macd_line, signal_line, histogram

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract MACD state for rule evaluation."""
        macd = current.get("macd", 0)
        signal = current.get("signal", 0)
        histogram = current.get("histogram", 0)

        if pd.isna(macd) or pd.isna(signal):
            return {
                "macd": 0,
                "signal": 0,
                "histogram": 0,
                "direction": "neutral",
            }

        direction = "bullish" if macd > signal else "bearish"

        return {
            "macd": float(macd),
            "signal": float(signal),
            "histogram": float(histogram),
            "direction": direction,
        }
