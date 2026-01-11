"""
VPVR (Volume Profile Visible Range) Indicator.

Analyzes volume distribution across price levels to identify
areas of high and low trading activity.

Note: Simplified implementation that creates price bins and
calculates volume at each level.

Signals:
- High Volume Node (HVN): Strong support/resistance
- Low Volume Node (LVN): Easy price movement areas
- Point of Control (POC): Price level with highest volume
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class VPVRIndicator(IndicatorBase):
    """
    Volume Profile indicator.

    Default Parameters:
        num_bins: 24  # Number of price levels
        lookback: 50  # Bars to analyze

    State Output:
        poc: Point of Control price level
        vah: Value Area High price
        val: Value Area Low price
        position: "above_poc", "below_poc", or "at_poc"
    """

    name = "vpvr"
    category = SignalCategory.VOLUME
    required_fields = ["high", "low", "close", "volume"]
    warmup_periods = 51

    _default_params = {
        "num_bins": 24,
        "lookback": 50,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate VPVR values."""
        num_bins = params["num_bins"]
        lookback = params["lookback"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "vpvr_poc": pd.Series(dtype=float),
                    "vpvr_vah": pd.Series(dtype=float),
                    "vpvr_val": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)
        n = len(close)

        poc = np.full(n, np.nan, dtype=np.float64)
        vah = np.full(n, np.nan, dtype=np.float64)
        val = np.full(n, np.nan, dtype=np.float64)

        for i in range(lookback, n):
            # Get price range
            window_high = high[i - lookback : i + 1]
            window_low = low[i - lookback : i + 1]
            window_close = close[i - lookback : i + 1]
            window_volume = volume[i - lookback : i + 1]

            price_high = np.max(window_high)
            price_low = np.min(window_low)

            if price_high == price_low:
                poc[i] = price_high
                vah[i] = price_high
                val[i] = price_low
                continue

            # Create price bins
            bin_edges = np.linspace(price_low, price_high, num_bins + 1)
            bin_volume = np.zeros(num_bins)

            # Distribute volume across bins (simplified - assign to close price bin)
            for j in range(len(window_close)):
                bin_idx = int(
                    (window_close[j] - price_low) / (price_high - price_low) * num_bins
                )
                bin_idx = min(bin_idx, num_bins - 1)
                bin_volume[bin_idx] += window_volume[j]

            # Point of Control - highest volume bin
            poc_bin = np.argmax(bin_volume)
            poc[i] = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2

            # Value Area (70% of volume)
            total_volume = np.sum(bin_volume)
            target_volume = total_volume * 0.7

            # Start from POC and expand outward
            accumulated = bin_volume[poc_bin]
            lower_idx = poc_bin
            upper_idx = poc_bin

            while accumulated < target_volume:
                lower_add = bin_volume[lower_idx - 1] if lower_idx > 0 else 0
                upper_add = bin_volume[upper_idx + 1] if upper_idx < num_bins - 1 else 0

                if lower_add >= upper_add and lower_idx > 0:
                    lower_idx -= 1
                    accumulated += bin_volume[lower_idx]
                elif upper_idx < num_bins - 1:
                    upper_idx += 1
                    accumulated += bin_volume[upper_idx]
                else:
                    break

            val[i] = bin_edges[lower_idx]
            vah[i] = bin_edges[upper_idx + 1]

        return pd.DataFrame(
            {"vpvr_poc": poc, "vpvr_vah": vah, "vpvr_val": val}, index=data.index
        )

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract VPVR state for rule evaluation."""
        poc = current.get("vpvr_poc", 0)
        vah = current.get("vpvr_vah", 0)
        val = current.get("vpvr_val", 0)

        if pd.isna(poc):
            return {
                "poc": 0,
                "vah": 0,
                "val": 0,
                "position": "at_poc",
            }

        return {
            "poc": float(poc),
            "vah": float(vah) if not pd.isna(vah) else 0,
            "val": float(val) if not pd.isna(val) else 0,
            "position": "at_poc",  # Determined by comparing with price in rules
        }
