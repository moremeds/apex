"""
Ultimate Oscillator Indicator.

Multi-timeframe momentum oscillator that uses three different periods
to reduce false signals.

Signals:
- Overbought: > 70
- Oversold: < 30
- Divergences: Price vs oscillator divergence
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


class UltimateOscillatorIndicator(IndicatorBase):
    """
    Ultimate Oscillator indicator.

    Default Parameters:
        period1: 7
        period2: 14
        period3: 28
        overbought: 70
        oversold: 30

    State Output:
        value: Ultimate Oscillator value (0-100)
        zone: "overbought", "oversold", or "neutral"
    """

    name = "ultimate"
    category = SignalCategory.MOMENTUM
    required_fields = ["high", "low", "close"]
    warmup_periods = 29

    _default_params = {
        "period1": 7,
        "period2": 14,
        "period3": 28,
        "overbought": 70,
        "oversold": 30,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Ultimate Oscillator values."""
        p1 = params["period1"]
        p2 = params["period2"]
        p3 = params["period3"]

        if len(data) == 0:
            return pd.DataFrame({"ultosc": pd.Series(dtype=float)}, index=data.index)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            ultosc = talib.ULTOSC(
                high, low, close,
                timeperiod1=p1, timeperiod2=p2, timeperiod3=p3
            )
        else:
            ultosc = self._calculate_manual(high, low, close, p1, p2, p3)

        return pd.DataFrame({"ultosc": ultosc}, index=data.index)

    def _calculate_manual(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
        p1: int, p2: int, p3: int
    ) -> np.ndarray:
        """Calculate Ultimate Oscillator without TA-Lib."""
        n = len(close)
        ultosc = np.full(n, np.nan, dtype=np.float64)

        # Buying pressure and true range
        bp = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            true_low = min(low[i], close[i - 1])
            true_high = max(high[i], close[i - 1])
            bp[i] = close[i] - true_low
            tr[i] = true_high - true_low

        # Calculate for each period
        max_period = max(p1, p2, p3)
        for i in range(max_period, n):
            bp1 = np.sum(bp[i - p1 + 1 : i + 1])
            tr1 = np.sum(tr[i - p1 + 1 : i + 1])
            bp2 = np.sum(bp[i - p2 + 1 : i + 1])
            tr2 = np.sum(tr[i - p2 + 1 : i + 1])
            bp3 = np.sum(bp[i - p3 + 1 : i + 1])
            tr3 = np.sum(tr[i - p3 + 1 : i + 1])

            avg1 = bp1 / tr1 if tr1 != 0 else 0
            avg2 = bp2 / tr2 if tr2 != 0 else 0
            avg3 = bp3 / tr3 if tr3 != 0 else 0

            # Weighted average (4, 2, 1)
            ultosc[i] = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7

        return ultosc

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Ultimate Oscillator state for rule evaluation."""
        ultosc = current.get("ultosc", 50)
        overbought = params["overbought"]
        oversold = params["oversold"]

        if pd.isna(ultosc):
            return {"value": 50, "zone": "neutral"}

        if ultosc >= overbought:
            zone = "overbought"
        elif ultosc <= oversold:
            zone = "oversold"
        else:
            zone = "neutral"

        return {"value": float(ultosc), "zone": zone}
