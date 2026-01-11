"""
TSI (True Strength Index) Indicator.

Double-smoothed momentum indicator that shows both trend direction
and overbought/oversold conditions.

Signals:
- Overbought: TSI > 25
- Oversold: TSI < -25
- Signal line crossover
- Zero line crossover
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class TSIIndicator(IndicatorBase):
    """
    True Strength Index indicator.

    Default Parameters:
        long_period: 25
        short_period: 13
        signal_period: 13
        overbought: 25
        oversold: -25

    State Output:
        tsi: TSI value (-100 to 100)
        signal: Signal line value
        zone: "overbought", "oversold", or "neutral"
    """

    name = "tsi"
    category = SignalCategory.MOMENTUM
    required_fields = ["close"]
    warmup_periods = 40

    _default_params = {
        "long_period": 25,
        "short_period": 13,
        "signal_period": 13,
        "overbought": 25,
        "oversold": -25,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate TSI values."""
        long_p = params["long_period"]
        short_p = params["short_period"]
        signal_p = params["signal_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"tsi": pd.Series(dtype=float), "tsi_signal": pd.Series(dtype=float)},
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)
        n = len(close)

        # Price change
        pc = np.zeros(n, dtype=np.float64)
        pc[1:] = close[1:] - close[:-1]

        # Double smoothed price change
        def ema(data: np.ndarray, period: int, start: int = 0) -> np.ndarray:
            result = np.full(n, np.nan, dtype=np.float64)
            alpha = 2.0 / (period + 1)
            # Find first valid value
            first_valid = start
            while first_valid < n and np.isnan(data[first_valid]):
                first_valid += 1
            if first_valid >= n:
                return result
            result[first_valid] = data[first_valid]
            for i in range(first_valid + 1, n):
                if not np.isnan(data[i]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
                else:
                    result[i] = result[i - 1]
            return result

        # First smoothing of PC
        pc_smooth1 = ema(pc, long_p, start=1)
        # Second smoothing
        pc_smooth2 = ema(pc_smooth1, short_p)

        # First smoothing of absolute PC
        abs_pc = np.abs(pc)
        abs_smooth1 = ema(abs_pc, long_p, start=1)
        # Second smoothing
        abs_smooth2 = ema(abs_smooth1, short_p)

        # TSI = 100 * (double smoothed PC / double smoothed abs PC)
        tsi = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            if not np.isnan(abs_smooth2[i]) and abs_smooth2[i] != 0:
                tsi[i] = 100 * pc_smooth2[i] / abs_smooth2[i]

        # Signal line
        tsi_signal = ema(tsi, signal_p)

        return pd.DataFrame({"tsi": tsi, "tsi_signal": tsi_signal}, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract TSI state for rule evaluation."""
        tsi = current.get("tsi", 0)
        tsi_signal = current.get("tsi_signal", 0)
        overbought = params["overbought"]
        oversold = params["oversold"]

        if pd.isna(tsi):
            return {"tsi": 0, "signal": 0, "zone": "neutral"}

        if tsi >= overbought:
            zone = "overbought"
        elif tsi <= oversold:
            zone = "oversold"
        else:
            zone = "neutral"

        return {
            "tsi": float(tsi),
            "signal": float(tsi_signal) if not pd.isna(tsi_signal) else 0,
            "zone": zone,
        }
