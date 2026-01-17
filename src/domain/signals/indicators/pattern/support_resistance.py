"""
Support/Resistance Levels Indicator.

Identifies dynamic support and resistance levels based on
swing highs and swing lows in the price action.

Signals:
- Price near support: Potential bounce or breakdown
- Price near resistance: Potential rejection or breakout
- Level strength: Based on number of touches
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class SupportResistanceIndicator(IndicatorBase):
    """
    Support/Resistance Level detector.

    Default Parameters:
        lookback: 50  # Bars to analyze for levels
        swing_period: 5  # Period for swing detection
        proximity_pct: 1.0  # % threshold for "near" level

    State Output:
        nearest_support: Nearest support level price
        nearest_resistance: Nearest resistance level price
        support_distance_pct: Distance to support as %
        resistance_distance_pct: Distance to resistance as %
        position: "at_support", "at_resistance", "between", or "outside"
    """

    name = "support_resistance"
    category = SignalCategory.PATTERN
    required_fields = ["high", "low", "close"]
    warmup_periods = 51

    _default_params = {
        "lookback": 50,
        "swing_period": 5,
        "proximity_pct": 1.0,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate support/resistance levels."""
        lookback = params["lookback"]
        swing_period = params["swing_period"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "sr_support": pd.Series(dtype=float),
                    "sr_resistance": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        n = len(close)

        support = np.full(n, np.nan, dtype=np.float64)
        resistance = np.full(n, np.nan, dtype=np.float64)

        for i in range(lookback, n):
            window_high = high[i - lookback : i + 1]
            window_low = low[i - lookback : i + 1]
            window_close = close[i]

            # Find swing highs and lows
            swing_highs = []
            swing_lows = []

            for j in range(swing_period, lookback - swing_period):
                # Swing high: higher than neighbors
                if all(
                    window_high[j] >= window_high[j - k] for k in range(1, swing_period + 1)
                ) and all(window_high[j] >= window_high[j + k] for k in range(1, swing_period + 1)):
                    swing_highs.append(window_high[j])

                # Swing low: lower than neighbors
                if all(
                    window_low[j] <= window_low[j - k] for k in range(1, swing_period + 1)
                ) and all(window_low[j] <= window_low[j + k] for k in range(1, swing_period + 1)):
                    swing_lows.append(window_low[j])

            # Find nearest support (swing low below current price)
            supports_below = [s for s in swing_lows if s < window_close]
            if supports_below:
                support[i] = max(supports_below)

            # Find nearest resistance (swing high above current price)
            resistances_above = [r for r in swing_highs if r > window_close]
            if resistances_above:
                resistance[i] = min(resistances_above)

        return pd.DataFrame(
            {"sr_support": support, "sr_resistance": resistance, "sr_close": close},
            index=data.index,
        )

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract S/R state for rule evaluation."""
        support = current.get("sr_support", np.nan)
        resistance = current.get("sr_resistance", np.nan)
        close = current.get("sr_close", np.nan)
        proximity = params["proximity_pct"]

        if pd.isna(close) or (pd.isna(support) and pd.isna(resistance)):
            return {
                "nearest_support": 0,
                "nearest_resistance": 0,
                "support_distance_pct": 0,
                "resistance_distance_pct": 0,
                "position": "outside",
            }

        # Calculate distance percentages from close price
        support_dist_pct = 0.0
        resistance_dist_pct = 0.0

        if not pd.isna(support) and support > 0:
            support_dist_pct = ((close - support) / support) * 100

        if not pd.isna(resistance) and resistance > 0:
            resistance_dist_pct = ((resistance - close) / resistance) * 100

        # Determine position relative to levels
        position = "between"
        if not pd.isna(support) and support_dist_pct <= proximity:
            position = "at_support"
        elif not pd.isna(resistance) and resistance_dist_pct <= proximity:
            position = "at_resistance"
        elif pd.isna(support) and not pd.isna(resistance):
            position = "below_resistance"
        elif pd.isna(resistance) and not pd.isna(support):
            position = "above_support"

        return {
            "nearest_support": float(support) if not pd.isna(support) else 0,
            "nearest_resistance": float(resistance) if not pd.isna(resistance) else 0,
            "support_distance_pct": float(support_dist_pct),
            "resistance_distance_pct": float(resistance_dist_pct),
            "position": position,
        }
