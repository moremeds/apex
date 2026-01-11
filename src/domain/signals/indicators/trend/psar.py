"""
Parabolic SAR (Stop and Reverse) Indicator.

Welles Wilder's trend-following indicator that provides potential
entry/exit points and trailing stop levels.

Signals:
- SAR below price: Bullish trend
- SAR above price: Bearish trend
- SAR flip: Trend reversal signal
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


class PSARIndicator(IndicatorBase):
    """
    Parabolic SAR indicator.

    Default Parameters:
        acceleration: 0.02
        maximum: 0.2

    State Output:
        psar: Parabolic SAR value
        trend: "bullish" (SAR below) or "bearish" (SAR above)
        flip: True if trend just reversed, False otherwise
    """

    name = "psar"
    category = SignalCategory.TREND
    required_fields = ["high", "low", "close"]
    warmup_periods = 2

    _default_params = {
        "acceleration": 0.02,
        "maximum": 0.2,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Parabolic SAR values."""
        acceleration = params["acceleration"]
        maximum = params["maximum"]

        if len(data) == 0:
            return pd.DataFrame({"psar": pd.Series(dtype=float)}, index=data.index)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)

        if HAS_TALIB:
            psar = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
        else:
            psar = self._calculate_manual(high, low, acceleration, maximum)

        return pd.DataFrame({"psar": psar}, index=data.index)

    def _calculate_manual(
        self,
        high: np.ndarray,
        low: np.ndarray,
        acceleration: float,
        maximum: float,
    ) -> np.ndarray:
        """Calculate Parabolic SAR without TA-Lib."""
        n = len(high)
        psar = np.full(n, np.nan, dtype=np.float64)

        if n < 2:
            return psar

        af = acceleration
        uptrend = True
        ep = high[0]
        psar[0] = low[0]

        for i in range(1, n):
            if uptrend:
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                psar[i] = min(psar[i], low[i - 1])
                if i >= 2:
                    psar[i] = min(psar[i], low[i - 2])

                if low[i] < psar[i]:
                    uptrend = False
                    psar[i] = ep
                    ep = low[i]
                    af = acceleration
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
            else:
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                psar[i] = max(psar[i], high[i - 1])
                if i >= 2:
                    psar[i] = max(psar[i], high[i - 2])

                if high[i] > psar[i]:
                    uptrend = True
                    psar[i] = ep
                    ep = high[i]
                    af = acceleration
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)

        return psar

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract PSAR state for rule evaluation."""
        psar = current.get("psar", 0)

        if pd.isna(psar):
            return {"psar": 0, "trend": "neutral", "flip": False}

        # Need close price to determine trend, but we only have PSAR value
        # The trend is determined externally when used with price
        # For now, return the value and let rules compare with price
        flip = False
        if previous is not None:
            prev_psar = previous.get("psar", 0)
            if not pd.isna(prev_psar):
                # Flip detected by large jump in PSAR (crosses price)
                flip = abs(psar - prev_psar) / prev_psar > 0.05 if prev_psar != 0 else False

        return {
            "psar": float(psar),
            "trend": "neutral",  # Determined by comparing with price in rules
            "flip": bool(flip),  # Ensure JSON serializable
        }
