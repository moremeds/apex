"""
Chaikin Volatility Indicator.

Measures the rate of change of the trading range (high - low)
to identify volatility expansion and contraction.

Signals:
- High Chaikin Volatility: Market tops or bottoms
- Low Chaikin Volatility: Consolidation
- Rising volatility: Increasing price movement
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


class ChaikinVolatilityIndicator(IndicatorBase):
    """
    Chaikin Volatility indicator.

    Default Parameters:
        ema_period: 10
        roc_period: 10

    State Output:
        chaikin_vol: Chaikin Volatility value
        direction: "expanding" or "contracting"
    """

    name = "chaikin_vol"
    category = SignalCategory.VOLATILITY
    required_fields = ["high", "low"]
    warmup_periods = 21

    _default_params = {
        "ema_period": 10,
        "roc_period": 10,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Chaikin Volatility values."""
        ema_period = params["ema_period"]
        roc_period = params["roc_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"chaikin_vol": pd.Series(dtype=float)}, index=data.index
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(high)

        # High-Low range
        hl_range = high - low

        # EMA of High-Low range
        if HAS_TALIB:
            ema_hl = talib.EMA(hl_range, timeperiod=ema_period)
        else:
            ema_hl = self._calculate_ema(hl_range, ema_period)

        # Rate of Change of EMA
        chaikin_vol = np.full(n, np.nan, dtype=np.float64)
        for i in range(ema_period + roc_period - 1, n):
            if not np.isnan(ema_hl[i]) and not np.isnan(ema_hl[i - roc_period]):
                if ema_hl[i - roc_period] != 0:
                    chaikin_vol[i] = (
                        (ema_hl[i] - ema_hl[i - roc_period]) / ema_hl[i - roc_period]
                    ) * 100

        return pd.DataFrame({"chaikin_vol": chaikin_vol}, index=data.index)

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        n = len(data)
        ema = np.full(n, np.nan, dtype=np.float64)

        if n < period:
            return ema

        alpha = 2.0 / (period + 1)
        ema[period - 1] = np.mean(data[:period])

        for i in range(period, n):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Chaikin Volatility state for rule evaluation."""
        chaikin_vol = current.get("chaikin_vol", 0)

        if pd.isna(chaikin_vol):
            return {"chaikin_vol": 0, "direction": "neutral"}

        direction = "expanding" if chaikin_vol > 0 else "contracting"

        return {
            "chaikin_vol": float(chaikin_vol),
            "direction": direction,
        }
