"""
CVD (Cumulative Volume Delta) Indicator.

Tracks the cumulative difference between buying and selling volume.
Positive CVD indicates buying pressure, negative indicates selling.

Note: True CVD requires tick data with bid/ask. This uses price-based
approximation where up closes add volume, down closes subtract.

Signals:
- Rising CVD: Buying pressure
- Falling CVD: Selling pressure
- Divergence with price: Potential reversal
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class CVDIndicator(IndicatorBase):
    """
    Cumulative Volume Delta indicator (price-based approximation).

    Default Parameters:
        signal_period: 14

    State Output:
        cvd: Cumulative Volume Delta value
        cvd_signal: Signal line (SMA)
        pressure: "buying", "selling", or "neutral"
    """

    name = "cvd"
    category = SignalCategory.VOLUME
    required_fields = ["high", "low", "close", "volume"]
    warmup_periods = 2

    _default_params = {
        "signal_period": 14,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate CVD values."""
        signal_period = params["signal_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"cvd": pd.Series(dtype=float), "cvd_signal": pd.Series(dtype=float)},
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)

        n = len(close)
        delta = np.zeros(n, dtype=np.float64)

        # Estimate volume delta using close position within range
        for i in range(n):
            hl_range = high[i] - low[i]
            if hl_range != 0:
                # Close location within range: 0 = bottom, 1 = top
                close_loc = (close[i] - low[i]) / hl_range
                # Delta: positive for bullish, negative for bearish
                delta[i] = volume[i] * (2 * close_loc - 1)
            else:
                delta[i] = 0

        # Cumulative
        cvd = np.cumsum(delta)

        # Signal line
        cvd_signal = np.full(n, np.nan, dtype=np.float64)
        for i in range(signal_period - 1, n):
            cvd_signal[i] = np.mean(cvd[i - signal_period + 1 : i + 1])

        return pd.DataFrame(
            {"cvd": cvd, "cvd_signal": cvd_signal}, index=data.index
        )

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract CVD state for rule evaluation."""
        cvd = current.get("cvd", 0)
        cvd_signal = current.get("cvd_signal", 0)

        if pd.isna(cvd):
            return {"cvd": 0, "cvd_signal": 0, "pressure": "neutral"}

        pressure = "neutral"
        if previous is not None:
            prev_cvd = previous.get("cvd", 0)
            if not pd.isna(prev_cvd):
                if cvd > prev_cvd:
                    pressure = "buying"
                elif cvd < prev_cvd:
                    pressure = "selling"

        return {
            "cvd": float(cvd),
            "cvd_signal": float(cvd_signal) if not pd.isna(cvd_signal) else 0,
            "pressure": pressure,
        }
