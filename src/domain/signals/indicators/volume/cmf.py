"""
CMF (Chaikin Money Flow) Indicator.

Measures the amount of Money Flow Volume over a period.
Oscillates between -1 and +1.

Signals:
- CMF > 0: Buying pressure
- CMF < 0: Selling pressure
- CMF > 0.25: Strong buying
- CMF < -0.25: Strong selling
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class CMFIndicator(IndicatorBase):
    """
    Chaikin Money Flow indicator.

    Default Parameters:
        period: 20
        strong_threshold: 0.25

    State Output:
        value: CMF value (-1 to +1)
        pressure: "strong_buying", "buying", "selling", "strong_selling", or "neutral"
        cross_zero: "bullish", "bearish", or None
    """

    name = "cmf"
    category = SignalCategory.VOLUME
    required_fields = ["high", "low", "close", "volume"]
    warmup_periods = 21

    _default_params = {
        "period": 20,
        "strong_threshold": 0.25,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate CMF values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"cmf": pd.Series(dtype=float)}, index=data.index)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)

        # Note: TA-Lib ADOSC is Chaikin Oscillator, not CMF - use manual calculation
        cmf = self._calculate_manual(high, low, close, volume, period)

        return pd.DataFrame({"cmf": cmf}, index=data.index)

    def _calculate_manual(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """Calculate CMF without TA-Lib."""
        n = len(close)
        cmf = np.full(n, np.nan, dtype=np.float64)

        # Money Flow Multiplier
        mfm = np.zeros(n, dtype=np.float64)
        for i in range(n):
            hl_range = high[i] - low[i]
            if hl_range != 0:
                mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range

        # Money Flow Volume
        mfv = mfm * volume

        # CMF = Sum(MFV) / Sum(Volume) over period
        for i in range(period - 1, n):
            sum_mfv = np.sum(mfv[i - period + 1 : i + 1])
            sum_vol = np.sum(volume[i - period + 1 : i + 1])
            if sum_vol != 0:
                cmf[i] = sum_mfv / sum_vol

        return cmf

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract CMF state for rule evaluation."""
        cmf = current.get("cmf", 0)
        strong = params["strong_threshold"]

        if pd.isna(cmf):
            return {"cmf": 0, "pressure": "neutral", "cross_zero": None}

        if cmf > strong:
            pressure = "strong_buying"
        elif cmf > 0:
            pressure = "buying"
        elif cmf < -strong:
            pressure = "strong_selling"
        elif cmf < 0:
            pressure = "selling"
        else:
            pressure = "neutral"

        cross_zero = None
        if previous is not None:
            prev_cmf = previous.get("cmf", 0)
            if not pd.isna(prev_cmf):
                if prev_cmf <= 0 and cmf > 0:
                    cross_zero = "bullish"
                elif prev_cmf >= 0 and cmf < 0:
                    cross_zero = "bearish"

        return {
            "value": float(cmf),
            "pressure": pressure,
            "cross_zero": cross_zero,
        }
