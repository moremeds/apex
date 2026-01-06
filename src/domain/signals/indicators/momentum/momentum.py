"""
Momentum Indicator.

Measures the absolute price change over n periods.
The simplest momentum indicator.

Signals:
- Positive: Bullish momentum
- Negative: Bearish momentum
- Accelerating/decelerating: Change in momentum
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


class MomentumIndicator(IndicatorBase):
    """
    Price momentum indicator.

    Default Parameters:
        period: 10

    State Output:
        value: Momentum value (price difference)
        direction: "bullish" if > 0, "bearish" if < 0
    """

    name = "momentum"
    category = SignalCategory.MOMENTUM
    required_fields = ["close"]
    warmup_periods = 11

    _default_params = {
        "period": 10,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Momentum values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"mom": pd.Series(dtype=float)}, index=data.index)

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            mom = talib.MOM(close, timeperiod=period)
        else:
            mom = self._calculate_manual(close, period)

        return pd.DataFrame({"mom": mom}, index=data.index)

    def _calculate_manual(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Momentum without TA-Lib."""
        n = len(close)
        mom = np.full(n, np.nan, dtype=np.float64)

        for i in range(period, n):
            mom[i] = close[i] - close[i - period]

        return mom

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Momentum state for rule evaluation."""
        mom = current.get("mom", 0)

        if pd.isna(mom):
            return {"value": 0, "direction": "neutral"}

        direction = "bullish" if mom > 0 else "bearish" if mom < 0 else "neutral"

        return {"value": float(mom), "direction": direction}
