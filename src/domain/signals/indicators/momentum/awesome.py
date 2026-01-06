"""
Awesome Oscillator (AO) Indicator.

Bill Williams' Awesome Oscillator measures market momentum using
the difference between 5-period and 34-period simple moving averages
of the midpoint (H+L)/2.

Signals:
- Positive AO: Bullish momentum
- Negative AO: Bearish momentum
- Zero line cross: Trend change
- Twin peaks pattern
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class AwesomeOscillatorIndicator(IndicatorBase):
    """
    Awesome Oscillator indicator.

    Default Parameters:
        fast_period: 5
        slow_period: 34

    State Output:
        value: AO value
        direction: "bullish" if > 0, "bearish" if < 0
        color: "green" if rising, "red" if falling
    """

    name = "awesome"
    category = SignalCategory.MOMENTUM
    required_fields = ["high", "low"]
    warmup_periods = 35

    _default_params = {
        "fast_period": 5,
        "slow_period": 34,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Awesome Oscillator values."""
        fast = params["fast_period"]
        slow = params["slow_period"]

        if len(data) == 0:
            return pd.DataFrame({"ao": pd.Series(dtype=float)}, index=data.index)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)

        # Median price (midpoint)
        midpoint = (high + low) / 2

        n = len(midpoint)
        ao = np.full(n, np.nan, dtype=np.float64)

        # SMA of midpoint
        for i in range(slow - 1, n):
            fast_sma = np.mean(midpoint[i - fast + 1 : i + 1])
            slow_sma = np.mean(midpoint[i - slow + 1 : i + 1])
            ao[i] = fast_sma - slow_sma

        return pd.DataFrame({"ao": ao}, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract AO state for rule evaluation."""
        ao = current.get("ao", 0)

        if pd.isna(ao):
            return {"value": 0, "direction": "neutral", "color": "neutral"}

        direction = "bullish" if ao > 0 else "bearish" if ao < 0 else "neutral"

        # Color based on change
        if previous is not None:
            prev_ao = previous.get("ao", 0)
            if not pd.isna(prev_ao):
                color = "green" if ao > prev_ao else "red"
            else:
                color = "neutral"
        else:
            color = "neutral"

        return {"value": float(ao), "direction": direction, "color": color}
