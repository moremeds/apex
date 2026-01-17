"""
Volume Ratio Indicator.

Compares current volume to average volume, identifying
unusual volume activity.

Signals:
- Ratio > 2: High volume, significant activity
- Ratio < 0.5: Low volume, lack of interest
- Volume spike: Potential trend confirmation or reversal
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class VolumeRatioIndicator(IndicatorBase):
    """
    Volume Ratio indicator (relative volume).

    Default Parameters:
        period: 20
        high_threshold: 2.0
        low_threshold: 0.5

    State Output:
        volume_ratio: Current volume / Average volume
        level: "high", "normal", or "low"
        spike: True if volume ratio > high_threshold
    """

    name = "volume_ratio"
    category = SignalCategory.VOLUME
    required_fields = ["volume"]
    warmup_periods = 21

    _default_params = {
        "period": 20,
        "high_threshold": 2.0,
        "low_threshold": 0.5,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Volume Ratio values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame({"volume_ratio": pd.Series(dtype=float)}, index=data.index)

        volume = data["volume"].values.astype(np.float64)
        n = len(volume)

        volume_ratio = np.full(n, np.nan, dtype=np.float64)

        for i in range(period, n):
            avg_vol = np.mean(volume[i - period : i])
            if avg_vol != 0:
                volume_ratio[i] = volume[i] / avg_vol

        return pd.DataFrame({"volume_ratio": volume_ratio}, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Volume Ratio state for rule evaluation."""
        volume_ratio = current.get("volume_ratio", 1)
        high_threshold = params["high_threshold"]
        low_threshold = params["low_threshold"]

        if pd.isna(volume_ratio):
            return {"volume_ratio": 1, "level": "normal", "spike": False}

        if volume_ratio >= high_threshold:
            level = "high"
        elif volume_ratio <= low_threshold:
            level = "low"
        else:
            level = "normal"

        # Convert to Python bool for JSON serialization
        spike = bool(volume_ratio >= high_threshold)

        return {
            "volume_ratio": float(volume_ratio),
            "level": level,
            "spike": spike,
        }
