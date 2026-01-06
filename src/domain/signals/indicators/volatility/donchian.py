"""
Donchian Channels Indicator.

Simple high/low channel over a lookback period.
Classic breakout indicator used in turtle trading.

Signals:
- Price breaks above upper: Bullish breakout
- Price breaks below lower: Bearish breakout
- Channel width: Volatility measure
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class DonchianChannelsIndicator(IndicatorBase):
    """
    Donchian Channels indicator.

    Default Parameters:
        period: 20

    State Output:
        upper: Highest high over period
        middle: Midpoint of channel
        lower: Lowest low over period
        width: Channel width in price units
        position: "above", "below", or "inside"
    """

    name = "donchian"
    category = SignalCategory.VOLATILITY
    required_fields = ["high", "low"]
    warmup_periods = 21

    _default_params = {
        "period": 20,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Donchian Channels values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "dc_upper": pd.Series(dtype=float),
                    "dc_middle": pd.Series(dtype=float),
                    "dc_lower": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(high)

        dc_upper = np.full(n, np.nan, dtype=np.float64)
        dc_middle = np.full(n, np.nan, dtype=np.float64)
        dc_lower = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            dc_upper[i] = np.max(high[i - period + 1 : i + 1])
            dc_lower[i] = np.min(low[i - period + 1 : i + 1])
            dc_middle[i] = (dc_upper[i] + dc_lower[i]) / 2

        return pd.DataFrame(
            {"dc_upper": dc_upper, "dc_middle": dc_middle, "dc_lower": dc_lower},
            index=data.index,
        )

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Donchian Channels state for rule evaluation."""
        upper = current.get("dc_upper", 0)
        middle = current.get("dc_middle", 0)
        lower = current.get("dc_lower", 0)

        if pd.isna(upper) or pd.isna(middle) or pd.isna(lower):
            return {
                "upper": 0,
                "middle": 0,
                "lower": 0,
                "width": 0,
                "position": "inside",
            }

        width = upper - lower

        return {
            "upper": float(upper),
            "middle": float(middle),
            "lower": float(lower),
            "width": float(width),
            "position": "inside",  # Determined by comparing with price in rules
        }
