"""
VWAP (Volume Weighted Average Price) Indicator.

Benchmark price calculated by dividing cumulative typical price
times volume by cumulative volume. Resets at each trading session.

Signals:
- Price above VWAP: Bullish sentiment
- Price below VWAP: Bearish sentiment
- VWAP as support/resistance
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class VWAPIndicator(IndicatorBase):
    """
    Volume Weighted Average Price indicator.

    Default Parameters:
        reset_daily: True (reset VWAP at each new trading day)

    State Output:
        vwap: VWAP value
        deviation: Price deviation from VWAP (%)
        position: "above", "below", or "at"
    """

    name = "vwap"
    category = SignalCategory.VOLUME
    required_fields = ["high", "low", "close", "volume"]
    warmup_periods = 1

    _default_params: dict[str, Any] = {
        "reset_daily": True,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate VWAP values with optional daily reset."""
        if len(data) == 0:
            return pd.DataFrame(
                {"vwap": pd.Series(dtype=float), "vwap_close": pd.Series(dtype=float)},
                index=data.index,
            )

        reset_daily = params.get("reset_daily", True)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)

        # Typical price
        tp = (high + low + close) / 3

        n = len(close)
        vwap = np.full(n, np.nan, dtype=np.float64)

        if reset_daily and isinstance(data.index, pd.DatetimeIndex):
            # Reset VWAP at each new trading day
            dates = data.index.date
            cum_tp_vol = 0.0
            cum_vol = 0.0
            prev_date = None

            for i in range(n):
                current_date = dates[i]
                if prev_date is not None and current_date != prev_date:
                    # New trading day - reset accumulators
                    cum_tp_vol = 0.0
                    cum_vol = 0.0

                cum_tp_vol += tp[i] * volume[i]
                cum_vol += volume[i]

                if cum_vol != 0:
                    vwap[i] = cum_tp_vol / cum_vol

                prev_date = current_date
        else:
            # Single session / no reset - simple cumulative
            cum_tp_vol = np.cumsum(tp * volume)
            cum_vol = np.cumsum(volume)
            vwap = np.where(cum_vol != 0, cum_tp_vol / cum_vol, np.nan)

        return pd.DataFrame({"vwap": vwap, "vwap_close": close}, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract VWAP state for rule evaluation."""
        vwap = current.get("vwap", 0)
        close = current.get("vwap_close", np.nan)

        if pd.isna(vwap):
            return {"vwap": 0, "deviation": 0, "position": "at"}

        # Calculate deviation and position using close price
        if pd.isna(close) or vwap == 0:
            deviation = 0.0
            position = "at"
        else:
            deviation = ((close - vwap) / vwap) * 100
            if close > vwap * 1.001:  # Small threshold to avoid noise
                position = "above"
            elif close < vwap * 0.999:
                position = "below"
            else:
                position = "at"

        return {
            "vwap": float(vwap),
            "deviation": float(deviation),
            "position": position,
        }
