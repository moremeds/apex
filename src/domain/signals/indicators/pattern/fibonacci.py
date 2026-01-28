"""
Fibonacci Retracement Indicator.

Calculates Fibonacci retracement levels based on recent swing
high and low points.

Levels:
- 0% (low): Support in uptrend
- 23.6%: First retracement level
- 38.2%: Common retracement level
- 50%: Half retracement
- 61.8%: Golden ratio retracement
- 78.6%: Deep retracement
- 100% (high): Resistance in uptrend

Signals:
- Price at key level: Potential bounce/break
- Confluence with other indicators
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


class FibonacciIndicator(IndicatorBase):
    """
    Fibonacci Retracement Level calculator.

    Default Parameters:
        lookback: 50  # Bars to find swing points

    State Output:
        swing_high: Recent swing high price
        swing_low: Recent swing low price
        fib_levels: Dict of level name to price
        nearest_level: Name of nearest Fibonacci level
        nearest_distance_pct: Distance to nearest level as %
    """

    name = "fibonacci"
    category = SignalCategory.PATTERN
    required_fields = ["high", "low", "close"]
    warmup_periods = 51

    _default_params = {
        "lookback": 50,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Fibonacci retracement levels."""
        lookback = params["lookback"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "fib_swing_high": pd.Series(dtype=float),
                    "fib_swing_low": pd.Series(dtype=float),
                    "fib_236": pd.Series(dtype=float),
                    "fib_382": pd.Series(dtype=float),
                    "fib_500": pd.Series(dtype=float),
                    "fib_618": pd.Series(dtype=float),
                    "fib_786": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(high)

        swing_high = np.full(n, np.nan, dtype=np.float64)
        swing_low = np.full(n, np.nan, dtype=np.float64)
        fib_236 = np.full(n, np.nan, dtype=np.float64)
        fib_382 = np.full(n, np.nan, dtype=np.float64)
        fib_500 = np.full(n, np.nan, dtype=np.float64)
        fib_618 = np.full(n, np.nan, dtype=np.float64)
        fib_786 = np.full(n, np.nan, dtype=np.float64)

        for i in range(lookback, n):
            window_high = high[i - lookback : i + 1]
            window_low = low[i - lookback : i + 1]

            sh = np.max(window_high)
            sl = np.min(window_low)
            swing_high[i] = sh
            swing_low[i] = sl

            # Calculate retracement levels
            # For downtrend: levels from high going down
            range_size = sh - sl
            fib_236[i] = sh - 0.236 * range_size
            fib_382[i] = sh - 0.382 * range_size
            fib_500[i] = sh - 0.500 * range_size
            fib_618[i] = sh - 0.618 * range_size
            fib_786[i] = sh - 0.786 * range_size

        close = data["close"].values.astype(np.float64)

        return pd.DataFrame(
            {
                "fib_swing_high": swing_high,
                "fib_swing_low": swing_low,
                "fib_236": fib_236,
                "fib_382": fib_382,
                "fib_500": fib_500,
                "fib_618": fib_618,
                "fib_786": fib_786,
                "fib_close": close,
            },
            index=data.index,
        )

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Fibonacci state for rule evaluation."""
        swing_high = current.get("fib_swing_high", 0)
        swing_low = current.get("fib_swing_low", 0)
        close = current.get("fib_close", np.nan)

        if pd.isna(swing_high) or pd.isna(swing_low):
            return {
                "swing_high": 0,
                "swing_low": 0,
                "fib_levels": {},
                "nearest_level": "none",
                "nearest_distance_pct": 0,
                "position": "outside",
            }

        fib_levels = {
            "0.0": float(swing_low),
            "0.236": float(current.get("fib_236", 0)) if not pd.isna(current.get("fib_236")) else 0,
            "0.382": float(current.get("fib_382", 0)) if not pd.isna(current.get("fib_382")) else 0,
            "0.5": float(current.get("fib_500", 0)) if not pd.isna(current.get("fib_500")) else 0,
            "0.618": float(current.get("fib_618", 0)) if not pd.isna(current.get("fib_618")) else 0,
            "0.786": float(current.get("fib_786", 0)) if not pd.isna(current.get("fib_786")) else 0,
            "1.0": float(swing_high),
        }

        # Find nearest level and calculate distance
        nearest_level = "none"
        nearest_distance_pct = 0.0
        position = "between"

        if not pd.isna(close) and close > 0:
            min_dist = float("inf")
            for level_name, level_price in fib_levels.items():
                if level_price > 0:
                    dist = abs(close - level_price) / close * 100
                    if dist < min_dist:
                        min_dist = dist
                        nearest_level = level_name
                        nearest_distance_pct = dist

            # Determine position: at_level (within 1%), above, or below
            if nearest_distance_pct <= 1.0:
                position = f"at_{nearest_level}"
            elif close > swing_high:
                position = "above_range"
            elif close < swing_low:
                position = "below_range"

        return {
            "swing_high": float(swing_high),
            "swing_low": float(swing_low),
            "fib_levels": fib_levels,
            "nearest_level": nearest_level,
            "nearest_distance_pct": float(nearest_distance_pct),
            "position": position,
        }
