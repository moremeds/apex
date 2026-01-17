"""
Pivot Points Indicator.

Calculates classic pivot points and support/resistance levels
based on prior period's high, low, and close.

Levels:
- PP (Pivot Point): Central level
- R1, R2, R3: Resistance levels
- S1, S2, S3: Support levels

Signals:
- Price above PP: Bullish bias
- Price below PP: Bearish bias
- Price at R/S levels: Potential reversal points
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class PivotPointsIndicator(IndicatorBase):
    """
    Pivot Points calculator.

    Default Parameters:
        method: "classic"  # classic, woodie, camarilla, fibonacci

    State Output:
        pivot: Pivot Point value
        r1, r2, r3: Resistance levels
        s1, s2, s3: Support levels
        position: "above_pivot" or "below_pivot"
    """

    name = "pivot"
    category = SignalCategory.PATTERN
    required_fields = ["high", "low", "close"]
    warmup_periods = 2

    _default_params = {
        "method": "classic",
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Pivot Points."""
        method = params["method"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "pivot_pp": pd.Series(dtype=float),
                    "pivot_r1": pd.Series(dtype=float),
                    "pivot_r2": pd.Series(dtype=float),
                    "pivot_r3": pd.Series(dtype=float),
                    "pivot_s1": pd.Series(dtype=float),
                    "pivot_s2": pd.Series(dtype=float),
                    "pivot_s3": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        n = len(close)

        pp = np.full(n, np.nan, dtype=np.float64)
        r1 = np.full(n, np.nan, dtype=np.float64)
        r2 = np.full(n, np.nan, dtype=np.float64)
        r3 = np.full(n, np.nan, dtype=np.float64)
        s1 = np.full(n, np.nan, dtype=np.float64)
        s2 = np.full(n, np.nan, dtype=np.float64)
        s3 = np.full(n, np.nan, dtype=np.float64)

        for i in range(1, n):
            prev_h = high[i - 1]
            prev_l = low[i - 1]
            prev_c = close[i - 1]

            if method == "classic":
                pp[i] = (prev_h + prev_l + prev_c) / 3
                r1[i] = 2 * pp[i] - prev_l
                s1[i] = 2 * pp[i] - prev_h
                r2[i] = pp[i] + (prev_h - prev_l)
                s2[i] = pp[i] - (prev_h - prev_l)
                r3[i] = pp[i] + 2 * (prev_h - prev_l)
                s3[i] = pp[i] - 2 * (prev_h - prev_l)

            elif method == "woodie":
                pp[i] = (prev_h + prev_l + 2 * prev_c) / 4
                r1[i] = 2 * pp[i] - prev_l
                s1[i] = 2 * pp[i] - prev_h
                r2[i] = pp[i] + (prev_h - prev_l)
                s2[i] = pp[i] - (prev_h - prev_l)
                r3[i] = prev_h + 2 * (pp[i] - prev_l)
                s3[i] = prev_l - 2 * (prev_h - pp[i])

            elif method == "camarilla":
                pp[i] = (prev_h + prev_l + prev_c) / 3
                diff = prev_h - prev_l
                r1[i] = prev_c + diff * 1.1 / 12
                s1[i] = prev_c - diff * 1.1 / 12
                r2[i] = prev_c + diff * 1.1 / 6
                s2[i] = prev_c - diff * 1.1 / 6
                r3[i] = prev_c + diff * 1.1 / 4
                s3[i] = prev_c - diff * 1.1 / 4

            elif method == "fibonacci":
                pp[i] = (prev_h + prev_l + prev_c) / 3
                diff = prev_h - prev_l
                r1[i] = pp[i] + 0.382 * diff
                s1[i] = pp[i] - 0.382 * diff
                r2[i] = pp[i] + 0.618 * diff
                s2[i] = pp[i] - 0.618 * diff
                r3[i] = pp[i] + diff
                s3[i] = pp[i] - diff

        return pd.DataFrame(
            {
                "pivot_pp": pp,
                "pivot_r1": r1,
                "pivot_r2": r2,
                "pivot_r3": r3,
                "pivot_s1": s1,
                "pivot_s2": s2,
                "pivot_s3": s3,
            },
            index=data.index,
        )

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Pivot Points state for rule evaluation."""
        pp = current.get("pivot_pp", 0)
        r1 = current.get("pivot_r1", 0)
        r2 = current.get("pivot_r2", 0)
        r3 = current.get("pivot_r3", 0)
        s1 = current.get("pivot_s1", 0)
        s2 = current.get("pivot_s2", 0)
        s3 = current.get("pivot_s3", 0)

        if pd.isna(pp):
            return {
                "pivot": 0,
                "r1": 0,
                "r2": 0,
                "r3": 0,
                "s1": 0,
                "s2": 0,
                "s3": 0,
                "position": "neutral",
            }

        return {
            "pivot": float(pp),
            "r1": float(r1) if not pd.isna(r1) else 0,
            "r2": float(r2) if not pd.isna(r2) else 0,
            "r3": float(r3) if not pd.isna(r3) else 0,
            "s1": float(s1) if not pd.isna(s1) else 0,
            "s2": float(s2) if not pd.isna(s2) else 0,
            "s3": float(s3) if not pd.isna(s3) else 0,
            "position": "neutral",  # Determined with price in rules
        }
