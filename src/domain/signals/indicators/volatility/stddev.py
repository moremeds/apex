"""
Standard Deviation Indicator.

Rolling standard deviation of price, measuring volatility.

Signals:
- High StdDev: High volatility
- Low StdDev: Low volatility, potential breakout
- StdDev spike: Sudden volatility increase
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


class StdDevIndicator(IndicatorBase):
    """
    Standard Deviation indicator.

    Default Parameters:
        period: 20

    State Output:
        stddev: Standard deviation value
        stddev_percent: StdDev as percentage of mean
        volatility: "high", "normal", or "low"
    """

    name = "stddev"
    category = SignalCategory.VOLATILITY
    required_fields = ["close"]
    warmup_periods = 21

    _default_params = {
        "period": 20,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Standard Deviation values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"stddev": pd.Series(dtype=float)}, index=data.index)

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            stddev = talib.STDDEV(close, timeperiod=period)
        else:
            stddev = self._calculate_manual(close, period)

        return pd.DataFrame({"stddev": stddev}, index=data.index)

    def _calculate_manual(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Standard Deviation without TA-Lib."""
        n = len(close)
        stddev = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            window = close[i - period + 1 : i + 1]
            stddev[i] = np.std(window, ddof=0)

        return stddev

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Standard Deviation state for rule evaluation."""
        stddev = current.get("stddev", 0)

        if pd.isna(stddev):
            return {"stddev": 0, "stddev_percent": 0, "volatility": "normal"}

        volatility = "normal"
        if previous is not None:
            prev_stddev = previous.get("stddev", 0)
            if not pd.isna(prev_stddev) and prev_stddev != 0:
                change = (stddev - prev_stddev) / prev_stddev
                if change > 0.2:
                    volatility = "high"
                elif change < -0.2:
                    volatility = "low"

        return {
            "stddev": float(stddev),
            "stddev_percent": 0,  # Would need price to calculate
            "volatility": volatility,
        }
