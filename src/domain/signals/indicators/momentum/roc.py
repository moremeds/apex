"""
ROC (Rate of Change) Indicator.

Measures the percentage change between current price and price n periods ago.
Simple but effective momentum measurement.

Signals:
- Positive ROC: Bullish momentum
- Negative ROC: Bearish momentum
- Zero line crossover: Trend change
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


class ROCIndicator(IndicatorBase):
    """
    Rate of Change indicator.

    Default Parameters:
        period: 10

    State Output:
        value: ROC percentage value
        direction: "bullish" if > 0, "bearish" if < 0
    """

    name = "roc"
    category = SignalCategory.MOMENTUM
    required_fields = ["close"]
    warmup_periods = 11

    _default_params = {
        "period": 10,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate ROC values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"roc": pd.Series(dtype=float)}, index=data.index)

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            roc = talib.ROC(close, timeperiod=period)
        else:
            roc = self._calculate_manual(close, period)

        return pd.DataFrame({"roc": roc}, index=data.index)

    def _calculate_manual(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ROC without TA-Lib."""
        n = len(close)
        roc = np.full(n, np.nan, dtype=np.float64)

        for i in range(period, n):
            if close[i - period] != 0:
                roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100
            else:
                roc[i] = 0

        return roc

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract ROC state for rule evaluation."""
        roc = current.get("roc", 0)

        if pd.isna(roc):
            return {"value": 0, "direction": "neutral"}

        direction = "bullish" if roc > 0 else "bearish" if roc < 0 else "neutral"

        return {"value": float(roc), "direction": direction}
