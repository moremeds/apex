"""
Vortex Indicator.

Identifies trend direction and strength by comparing positive
and negative trend movements.

Signals:
- VI+ > VI-: Bullish trend
- VI- > VI+: Bearish trend
- VI crossover: Trend reversal
- High VI+ or VI-: Strong trend
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class VortexIndicator(IndicatorBase):
    """
    Vortex indicator.

    Default Parameters:
        period: 14

    State Output:
        vi_plus: Positive vortex indicator value
        vi_minus: Negative vortex indicator value
        trend: "bullish" if VI+ > VI-, "bearish" otherwise
        cross: "bullish", "bearish", or None
        spread: VI+ - VI- (positive = bullish strength)
    """

    name = "vortex"
    category = SignalCategory.TREND
    required_fields = ["high", "low", "close"]
    warmup_periods = 15

    _default_params = {
        "period": 14,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Vortex Indicator values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame(
                {"vi_plus": pd.Series(dtype=float), "vi_minus": pd.Series(dtype=float)},
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        n = len(close)
        vi_plus = np.full(n, np.nan, dtype=np.float64)
        vi_minus = np.full(n, np.nan, dtype=np.float64)

        if n < period + 1:
            return pd.DataFrame({"vi_plus": vi_plus, "vi_minus": vi_minus}, index=data.index)

        # True Range
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Vortex Movement
        vm_plus = np.zeros(n)
        vm_minus = np.zeros(n)
        for i in range(1, n):
            vm_plus[i] = abs(high[i] - low[i - 1])
            vm_minus[i] = abs(low[i] - high[i - 1])

        # Sum over period
        for i in range(period, n):
            sum_tr = np.sum(tr[i - period + 1 : i + 1])
            sum_vm_plus = np.sum(vm_plus[i - period + 1 : i + 1])
            sum_vm_minus = np.sum(vm_minus[i - period + 1 : i + 1])

            if sum_tr != 0:
                vi_plus[i] = sum_vm_plus / sum_tr
                vi_minus[i] = sum_vm_minus / sum_tr

        return pd.DataFrame({"vi_plus": vi_plus, "vi_minus": vi_minus}, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Vortex state for rule evaluation."""
        vi_plus = current.get("vi_plus", 0)
        vi_minus = current.get("vi_minus", 0)

        if pd.isna(vi_plus) or pd.isna(vi_minus):
            return {
                "vi_plus": 0,
                "vi_minus": 0,
                "trend": "neutral",
                "cross": None,
                "spread": 0,
            }

        trend = "bullish" if vi_plus > vi_minus else "bearish"
        spread = vi_plus - vi_minus

        cross = None
        if previous is not None:
            prev_plus = previous.get("vi_plus", 0)
            prev_minus = previous.get("vi_minus", 0)
            if not pd.isna(prev_plus) and not pd.isna(prev_minus):
                if prev_plus <= prev_minus and vi_plus > vi_minus:
                    cross = "bullish"
                elif prev_plus >= prev_minus and vi_plus < vi_minus:
                    cross = "bearish"

        return {
            "vi_plus": float(vi_plus),
            "vi_minus": float(vi_minus),
            "trend": trend,
            "cross": cross,
            "spread": float(spread),
        }
