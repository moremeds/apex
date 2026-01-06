"""
Bollinger Bands Indicator.

Standard deviation bands around a moving average.
Measures volatility and identifies overbought/oversold conditions.

Signals:
- Price at upper band: Overbought
- Price at lower band: Oversold
- Band squeeze: Low volatility, potential breakout
- Band expansion: High volatility, trend in progress
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


class BollingerBandsIndicator(IndicatorBase):
    """
    Bollinger Bands indicator.

    Default Parameters:
        period: 20
        std_dev: 2.0

    State Output:
        upper: Upper band value
        middle: Middle band (SMA) value
        lower: Lower band value
        bandwidth: (Upper - Lower) / Middle * 100
        percent_b: (Close - Lower) / (Upper - Lower) * 100
        zone: "overbought", "oversold", or "neutral"
    """

    name = "bollinger"
    category = SignalCategory.VOLATILITY
    required_fields = ["close"]
    warmup_periods = 21

    _default_params = {
        "period": 20,
        "std_dev": 2.0,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Bollinger Bands values."""
        period = params["period"]
        std_dev = params["std_dev"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "bb_upper": pd.Series(dtype=float),
                    "bb_middle": pd.Series(dtype=float),
                    "bb_lower": pd.Series(dtype=float),
                },
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
        else:
            bb_middle, bb_upper, bb_lower = self._calculate_manual(close, period, std_dev)

        return pd.DataFrame(
            {"bb_upper": bb_upper, "bb_middle": bb_middle, "bb_lower": bb_lower, "bb_close": close},
            index=data.index,
        )

    def _calculate_manual(
        self, close: np.ndarray, period: int, std_dev: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands without TA-Lib."""
        n = len(close)
        bb_middle = np.full(n, np.nan, dtype=np.float64)
        bb_upper = np.full(n, np.nan, dtype=np.float64)
        bb_lower = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            window = close[i - period + 1 : i + 1]
            sma = np.mean(window)
            std = np.std(window, ddof=0)

            bb_middle[i] = sma
            bb_upper[i] = sma + std_dev * std
            bb_lower[i] = sma - std_dev * std

        return bb_middle, bb_upper, bb_lower

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Bollinger Bands state for rule evaluation."""
        upper = current.get("bb_upper", 0)
        middle = current.get("bb_middle", 0)
        lower = current.get("bb_lower", 0)
        close = current.get("bb_close", np.nan)

        if pd.isna(upper) or pd.isna(middle) or pd.isna(lower):
            return {
                "upper": 0,
                "middle": 0,
                "lower": 0,
                "bandwidth": 0,
                "percent_b": 50,
                "zone": "neutral",
            }

        bandwidth = ((upper - lower) / middle * 100) if middle != 0 else 0
        band_width = upper - lower

        # Calculate %B using close price (not middle band)
        if pd.isna(close):
            percent_b = 50
            zone = "neutral"
        else:
            percent_b = ((close - lower) / band_width * 100) if band_width != 0 else 50
            if percent_b >= 100:
                zone = "overbought"
            elif percent_b <= 0:
                zone = "oversold"
            else:
                zone = "neutral"

        return {
            "upper": float(upper),
            "middle": float(middle),
            "lower": float(lower),
            "bandwidth": float(bandwidth),
            "percent_b": float(percent_b),
            "zone": zone,
        }
