"""
AD (Accumulation/Distribution) Line Indicator.

Measures the cumulative flow of money into and out of a security.
Considers where the close falls within the high-low range.

Signals:
- Rising AD: Accumulation, bullish
- Falling AD: Distribution, bearish
- Divergence with price: Potential reversal
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


class ADIndicator(IndicatorBase):
    """
    Accumulation/Distribution Line indicator.

    Default Parameters:
        signal_period: 20

    State Output:
        ad: A/D Line value
        ad_signal: Signal line (SMA)
        trend: "accumulation", "distribution", or "neutral"
    """

    name = "ad"
    category = SignalCategory.VOLUME
    required_fields = ["high", "low", "close", "volume"]
    warmup_periods = 1

    _default_params = {
        "signal_period": 20,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate A/D Line values."""
        signal_period = params["signal_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"ad": pd.Series(dtype=float), "ad_signal": pd.Series(dtype=float)},
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)

        if HAS_TALIB:
            ad = talib.AD(high, low, close, volume)
        else:
            ad = self._calculate_manual(high, low, close, volume)

        # Signal line
        ad_signal = self._calculate_sma(ad, signal_period)

        return pd.DataFrame({"ad": ad, "ad_signal": ad_signal}, index=data.index)

    def _calculate_manual(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Calculate A/D Line without TA-Lib."""
        n = len(close)
        ad = np.zeros(n, dtype=np.float64)

        for i in range(n):
            hl_range = high[i] - low[i]
            if hl_range != 0:
                # Money Flow Multiplier
                mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
                # Money Flow Volume
                mfv = mfm * volume[i]
            else:
                mfv = 0

            ad[i] = ad[i - 1] + mfv if i > 0 else mfv

        return ad

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
        """Extract A/D state for rule evaluation."""
        ad = current.get("ad", 0)
        ad_signal = current.get("ad_signal", 0)

        if pd.isna(ad):
            return {"ad": 0, "ad_signal": 0, "trend": "neutral"}

        trend = "neutral"
        if previous is not None:
            prev_ad = previous.get("ad", 0)
            if not pd.isna(prev_ad):
                if ad > prev_ad:
                    trend = "accumulation"
                elif ad < prev_ad:
                    trend = "distribution"

        return {
            "ad": float(ad),
            "ad_signal": float(ad_signal) if not pd.isna(ad_signal) else 0,
            "trend": trend,
        }
