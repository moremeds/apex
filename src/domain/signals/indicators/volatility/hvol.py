"""
Historical Volatility (HV) Indicator.

Annualized standard deviation of returns, commonly used for
options pricing and risk assessment.

Signals:
- High HV: Elevated risk, potential for large moves
- Low HV: Calm market, potential for breakout
- HV percentile: Current volatility vs historical range
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class HistoricalVolatilityIndicator(IndicatorBase):
    """
    Historical Volatility indicator.

    Default Parameters:
        period: 20
        trading_days: 252  # For annualization

    State Output:
        hv: Historical volatility (annualized %)
        hv_rank: Percentile rank vs lookback period
        regime: "high_vol", "normal_vol", or "low_vol"
    """

    name = "hvol"
    category = SignalCategory.VOLATILITY
    required_fields = ["close"]
    warmup_periods = 21

    _default_params = {
        "period": 20,
        "trading_days": 252,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Historical Volatility values."""
        period = params["period"]
        trading_days = params["trading_days"]

        if len(data) == 0:
            return pd.DataFrame({"hvol": pd.Series(dtype=float)}, index=data.index)

        close = data["close"].values.astype(np.float64)
        n = len(close)

        # Log returns - skip non-positive values to avoid log(0) or log(negative)
        returns = np.full(n, np.nan, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                returns[i] = np.log(close[i] / close[i - 1])
            # else: returns[i] stays NaN (already initialized)

        # Rolling standard deviation of returns
        hvol = np.full(n, np.nan, dtype=np.float64)
        for i in range(period, n):
            window = returns[i - period + 1 : i + 1]
            std = np.nanstd(window, ddof=1)
            # Annualize
            hvol[i] = std * np.sqrt(trading_days) * 100

        return pd.DataFrame({"hvol": hvol}, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Historical Volatility state for rule evaluation."""
        hvol = current.get("hvol", 0)

        if pd.isna(hvol):
            return {"hv": 0, "hv_rank": 50, "regime": "normal_vol"}

        # Simple regime classification based on typical equity HV levels
        if hvol > 30:
            regime = "high_vol"
        elif hvol < 15:
            regime = "low_vol"
        else:
            regime = "normal_vol"

        return {
            "hv": float(hvol),
            "hv_rank": 50,  # Would need full history to calculate percentile
            "regime": regime,
        }
