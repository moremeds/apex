"""
Force Index Indicator.

Elder's Force Index measures the power behind price movement
by combining price change, direction, and volume.

Signals:
- Positive Force: Bullish power
- Negative Force: Bearish power
- Zero line cross: Trend change
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


class ForceIndexIndicator(IndicatorBase):
    """
    Force Index indicator.

    Default Parameters:
        period: 13  # EMA smoothing period

    State Output:
        force: Force Index value
        direction: "bullish", "bearish", or "neutral"
        cross_zero: "bullish", "bearish", or None
    """

    name = "force"
    category = SignalCategory.VOLUME
    required_fields = ["close", "volume"]
    warmup_periods = 14

    _default_params = {
        "period": 13,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Force Index values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"force": pd.Series(dtype=float)}, index=data.index)

        close = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)
        n = len(close)

        # Raw Force Index
        raw_force = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            raw_force[i] = (close[i] - close[i - 1]) * volume[i]

        # Smooth with EMA
        if HAS_TALIB:
            force = talib.EMA(raw_force, timeperiod=period)
        else:
            force = self._calculate_ema(raw_force, period)

        return pd.DataFrame({"force": force}, index=data.index)

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
        """Extract Force Index state for rule evaluation."""
        force = current.get("force", 0)

        if pd.isna(force):
            return {"force": 0, "direction": "neutral", "cross_zero": None}

        direction = "bullish" if force > 0 else "bearish" if force < 0 else "neutral"

        cross_zero = None
        if previous is not None:
            prev_force = previous.get("force", 0)
            if not pd.isna(prev_force):
                if prev_force <= 0 and force > 0:
                    cross_zero = "bullish"
                elif prev_force >= 0 and force < 0:
                    cross_zero = "bearish"

        return {
            "force": float(force),
            "direction": direction,
            "cross_zero": cross_zero,
        }
