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
        breakout: "upper", "lower", or "none"
    """

    name = "donchian"
    category = SignalCategory.VOLATILITY
    required_fields = ["high", "low", "close"]
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

        # Include close for breakout detection
        close = data["close"].values.astype(np.float64) if "close" in data.columns else None

        return pd.DataFrame(
            {
                "dc_upper": dc_upper,
                "dc_middle": dc_middle,
                "dc_lower": dc_lower,
                "dc_close": close if close is not None else np.full(n, np.nan),
            },
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
        close = current.get("dc_close", np.nan)

        if pd.isna(upper) or pd.isna(middle) or pd.isna(lower):
            return {
                "upper": 0,
                "middle": 0,
                "lower": 0,
                "width": 0,
                "breakout": "none",
            }

        width = upper - lower

        # Turtle-style breakout: compare current close with previous channel bounds
        # Upper breakout: close > previous upper (new high)
        # Lower breakout: close < previous lower (new low)
        breakout = "none"
        if previous is not None and not pd.isna(close):
            prev_upper = previous.get("dc_upper", np.nan)
            prev_lower = previous.get("dc_lower", np.nan)

            if not pd.isna(prev_upper) and close > prev_upper:
                breakout = "upper"
            elif not pd.isna(prev_lower) and close < prev_lower:
                breakout = "lower"

        return {
            "upper": float(upper),
            "middle": float(middle),
            "lower": float(lower),
            "width": float(width),
            "breakout": breakout,
        }
