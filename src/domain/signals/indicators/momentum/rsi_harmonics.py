"""
RSI Harmonics Indicator.

Multi-period RSI analysis that examines RSI across short, medium, and long
periods to identify confluence and divergence between timeframes.

Signals:
- All periods overbought/oversold: Strong signal
- Divergence between periods: Potential reversal
- Confluence score: Agreement level across periods
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


class RSIHarmonicsIndicator(IndicatorBase):
    """
    RSI Harmonics indicator - multi-period RSI analysis.

    Default Parameters:
        short_period: 7
        medium_period: 14
        long_period: 21
        overbought: 70
        oversold: 30

    State Output:
        short_rsi: Short period RSI value
        medium_rsi: Medium period RSI value
        long_rsi: Long period RSI value
        avg_rsi: Average of all three RSI values
        confluence: "bullish", "bearish", or "mixed"
        zone: "overbought", "oversold", or "neutral" (based on avg)
        alignment_score: -100 to 100 (negative=bearish, positive=bullish)
    """

    name = "rsi_harmonics"
    category = SignalCategory.MOMENTUM
    required_fields = ["close"]
    warmup_periods = 22

    _default_params = {
        "short_period": 7,
        "medium_period": 14,
        "long_period": 21,
        "overbought": 70,
        "oversold": 30,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate RSI Harmonics values."""
        short_p = params["short_period"]
        medium_p = params["medium_period"]
        long_p = params["long_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "rsi_short": pd.Series(dtype=float),
                    "rsi_medium": pd.Series(dtype=float),
                    "rsi_long": pd.Series(dtype=float),
                },
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            rsi_short = talib.RSI(close, timeperiod=short_p)
            rsi_medium = talib.RSI(close, timeperiod=medium_p)
            rsi_long = talib.RSI(close, timeperiod=long_p)
        else:
            rsi_short = self._calculate_rsi(close, short_p)
            rsi_medium = self._calculate_rsi(close, medium_p)
            rsi_long = self._calculate_rsi(close, long_p)

        return pd.DataFrame(
            {"rsi_short": rsi_short, "rsi_medium": rsi_medium, "rsi_long": rsi_long},
            index=data.index,
        )

    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI without TA-Lib."""
        n = len(close)
        rsi = np.full(n, np.nan, dtype=np.float64)

        if n < period + 1:
            return rsi

        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            rsi[period] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))

        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract RSI Harmonics state for rule evaluation."""
        rsi_short = current.get("rsi_short", 50)
        rsi_medium = current.get("rsi_medium", 50)
        rsi_long = current.get("rsi_long", 50)
        overbought = params["overbought"]
        oversold = params["oversold"]

        if pd.isna(rsi_short) or pd.isna(rsi_medium) or pd.isna(rsi_long):
            return {
                "short_rsi": 50,
                "medium_rsi": 50,
                "long_rsi": 50,
                "avg_rsi": 50,
                "confluence": "mixed",
                "zone": "neutral",
                "alignment_score": 0,
            }

        avg_rsi = (rsi_short + rsi_medium + rsi_long) / 3

        # Determine zone based on average
        if avg_rsi >= overbought:
            zone = "overbought"
        elif avg_rsi <= oversold:
            zone = "oversold"
        else:
            zone = "neutral"

        # Calculate confluence
        bullish_count = sum(1 for r in [rsi_short, rsi_medium, rsi_long] if r > 50)
        bearish_count = 3 - bullish_count

        if bullish_count == 3:
            confluence = "bullish"
        elif bearish_count == 3:
            confluence = "bearish"
        else:
            confluence = "mixed"

        # Alignment score: -100 to 100
        # All oversold = -100, all overbought = +100, neutral = 0
        alignment_score = ((avg_rsi - 50) / 50) * 100

        return {
            "short_rsi": float(rsi_short),
            "medium_rsi": float(rsi_medium),
            "long_rsi": float(rsi_long),
            "avg_rsi": float(avg_rsi),
            "confluence": confluence,
            "zone": zone,
            "alignment_score": float(alignment_score),
        }
