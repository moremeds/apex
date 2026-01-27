"""
Volume Indicator.

Basic volume analysis with spike detection and relative volume classification.

Signals:
- Volume Spike: Unusually high volume (potential significant move)
- Volume Dry: Very low volume (potential consolidation)
- Relative Volume: Classification vs average (high/normal/low)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class VolumeIndicator(IndicatorBase):
    """
    Basic volume analysis indicator.

    Default Parameters:
        period: 20  # Lookback for average volume
        spike_threshold: 2.0  # Volume > 2x average = spike
        high_threshold: 1.5  # Volume > 1.5x average = high
        low_threshold: 0.5  # Volume < 0.5x average = low

    State Output:
        volume: Current volume
        avg_volume: Average volume over period
        ratio: Current volume / average volume
        spike: True if volume > spike_threshold * average
        relative: "high", "normal", "low", or "dry"
    """

    name = "volume"
    category = SignalCategory.VOLUME
    required_fields = ["volume"]
    warmup_periods = 21

    _default_params = {
        "period": 20,
        "spike_threshold": 2.0,  # 2x average = spike
        "high_threshold": 1.5,  # 1.5x average = high
        "low_threshold": 0.5,  # 0.5x average = low
        "dry_threshold": 0.25,  # 0.25x average = dry
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate volume analysis values."""
        period = params["period"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "vol_current": pd.Series(dtype=float),
                    "vol_avg": pd.Series(dtype=float),
                },
                index=data.index,
            )

        volume = data["volume"].values.astype(np.float64)
        n = len(volume)

        # Calculate rolling average volume
        vol_avg = np.full(n, np.nan, dtype=np.float64)
        for i in range(period - 1, n):
            vol_avg[i] = np.mean(volume[i - period + 1 : i + 1])

        return pd.DataFrame(
            {"vol_current": volume, "vol_avg": vol_avg},
            index=data.index,
        )

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract volume state for rule evaluation."""
        vol_current = current.get("vol_current", 0)
        vol_avg = current.get("vol_avg", 0)

        spike_threshold = params.get("spike_threshold", 2.0)
        high_threshold = params.get("high_threshold", 1.5)
        low_threshold = params.get("low_threshold", 0.5)
        dry_threshold = params.get("dry_threshold", 0.25)

        if pd.isna(vol_current) or pd.isna(vol_avg) or vol_avg == 0:
            return {
                "volume": 0,
                "avg_volume": 0,
                "ratio": 1.0,
                "spike": False,
                "relative": "normal",
            }

        ratio = vol_current / vol_avg

        # Spike detection
        spike = ratio >= spike_threshold

        # Relative volume classification
        if ratio >= high_threshold:
            relative = "high"
        elif ratio <= dry_threshold:
            relative = "dry"
        elif ratio <= low_threshold:
            relative = "low"
        else:
            relative = "normal"

        return {
            "volume": float(vol_current),
            "avg_volume": float(vol_avg),
            "ratio": float(ratio),
            "spike": spike,
            "relative": relative,
        }
