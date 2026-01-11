"""
Zero-Lag EMA (ZLEMA) Indicator.

Modified EMA that reduces lag by incorporating price momentum.
Removes much of the inherent lag in standard moving averages.

Signals:
- Price above ZLEMA: Bullish
- Price below ZLEMA: Bearish
- ZLEMA slope: Trend direction
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class ZeroLagIndicator(IndicatorBase):
    """
    Zero-Lag EMA indicator.

    Default Parameters:
        period: 20

    State Output:
        zlema: Zero-lag EMA value
        slope: "rising", "falling", or "flat"
        change_rate: Rate of change in ZLEMA
    """

    name = "zerolag"
    category = SignalCategory.TREND
    required_fields = ["close"]
    warmup_periods = 21

    _default_params = {
        "period": 20,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Zero-Lag EMA values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"zlema": pd.Series(dtype=float)}, index=data.index)

        close = data["close"].values.astype(np.float64)
        n = len(close)

        # Zero-Lag EMA = EMA of (close + (close - close[lag]))
        # where lag = (period - 1) / 2
        lag = int((period - 1) / 2)
        zlema = np.full(n, np.nan, dtype=np.float64)

        if n < period + lag:
            return pd.DataFrame({"zlema": zlema}, index=data.index)

        # Create lag-adjusted close
        adjusted = np.zeros(n)
        for i in range(lag, n):
            adjusted[i] = close[i] + (close[i] - close[i - lag])

        # Apply EMA to adjusted close
        alpha = 2.0 / (period + 1)
        zlema[period + lag - 1] = np.mean(adjusted[lag : period + lag])

        for i in range(period + lag, n):
            zlema[i] = alpha * adjusted[i] + (1 - alpha) * zlema[i - 1]

        return pd.DataFrame({"zlema": zlema}, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Zero-Lag EMA state for rule evaluation."""
        zlema = current.get("zlema", 0)

        if pd.isna(zlema):
            return {"zlema": 0, "slope": "flat", "change_rate": 0}

        slope = "flat"
        change_rate = 0.0

        if previous is not None:
            prev_zlema = previous.get("zlema", 0)
            if not pd.isna(prev_zlema) and prev_zlema != 0:
                change_rate = ((zlema - prev_zlema) / prev_zlema) * 100
                if change_rate > 0.01:
                    slope = "rising"
                elif change_rate < -0.01:
                    slope = "falling"

        return {
            "zlema": float(zlema),
            "slope": slope,
            "change_rate": float(change_rate),
        }
