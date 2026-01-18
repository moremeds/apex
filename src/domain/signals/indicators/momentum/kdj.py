"""
KDJ (Stochastic Oscillator) Indicator.

Measures momentum by comparing closing price to price range.
K and D are the standard stochastic lines, J is a more sensitive derivative.

Signals:
- Overbought: K > 80
- Oversold: K < 20
- Bullish cross: K crosses above D
- Bearish cross: K crosses below D
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


class KDJIndicator(IndicatorBase):
    """
    KDJ/Stochastic oscillator with K, D, and J lines.

    Default Parameters:
        fastk_period: 14
        slowk_period: 3
        slowd_period: 3
        overbought: 80
        oversold: 20

    State Output:
        k: %K value (0-100)
        d: %D value (0-100)
        j: J value (3*K - 2*D)
        zone: "overbought", "oversold", or "neutral"
    """

    name = "kdj"
    category = SignalCategory.MOMENTUM
    required_fields = ["high", "low", "close"]
    warmup_periods = 20

    _default_params = {
        "fastk_period": 14,
        "slowk_period": 3,
        "slowd_period": 3,
        "overbought": 80,
        "oversold": 20,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate K, D, and J values."""
        fastk = params["fastk_period"]
        slowk = params["slowk_period"]
        slowd = params["slowd_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "k": pd.Series(dtype=float),
                    "d": pd.Series(dtype=float),
                    "j": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            k, d = talib.STOCH(
                high,
                low,
                close,
                fastk_period=fastk,
                slowk_period=slowk,
                slowk_matype=talib.MA_Type.SMA,
                slowd_period=slowd,
                slowd_matype=talib.MA_Type.SMA,
            )
        else:
            k, d = self._calculate_manual(high, low, close, fastk, slowk, slowd)

        # J = 3*K - 2*D (more sensitive)
        j = 3 * k - 2 * d

        return pd.DataFrame({"k": k, "d": d, "j": j}, index=data.index)

    def _calculate_manual(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        fastk: int,
        slowk: int,
        slowd: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic without TA-Lib."""
        n = len(close)
        raw_k = np.full(n, np.nan, dtype=np.float64)

        # Calculate raw %K
        for i in range(fastk - 1, n):
            highest = np.max(high[i - fastk + 1 : i + 1])
            lowest = np.min(low[i - fastk + 1 : i + 1])
            if highest != lowest:
                raw_k[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                raw_k[i] = 50  # Middle value when range is zero

        # Slow %K is SMA of raw %K
        k = np.full(n, np.nan, dtype=np.float64)
        for i in range(fastk + slowk - 2, n):
            k[i] = np.nanmean(raw_k[i - slowk + 1 : i + 1])

        # %D is SMA of slow %K
        d = np.full(n, np.nan, dtype=np.float64)
        for i in range(fastk + slowk + slowd - 3, n):
            d[i] = np.nanmean(k[i - slowd + 1 : i + 1])

        return k, d

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract KDJ state for rule evaluation."""
        k = current.get("k", 50)
        d = current.get("d", 50)
        j = current.get("j", 50)
        overbought = params["overbought"]
        oversold = params["oversold"]

        if pd.isna(k):
            return {"k": 50, "d": 50, "j": 50, "zone": "neutral"}

        if k >= overbought:
            zone = "overbought"
        elif k <= oversold:
            zone = "oversold"
        else:
            zone = "neutral"

        return {
            "k": float(k),
            "d": float(d),
            "j": float(j),
            "zone": zone,
        }
