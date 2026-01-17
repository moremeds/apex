"""
Ichimoku Cloud Indicator.

Japanese technical analysis system with 5 lines:
- Tenkan-sen (Conversion Line): 9-period mid-price
- Kijun-sen (Base Line): 26-period mid-price
- Senkou Span A: Average of Tenkan and Kijun, plotted 26 periods ahead
- Senkou Span B: 52-period mid-price, plotted 26 periods ahead
- Chikou Span: Close price plotted 26 periods behind

Signals:
- Price above cloud: Bullish
- Price below cloud: Bearish
- TK Cross: Tenkan crosses Kijun
- Kumo twist: Span A crosses Span B

Implementation Notes:
- senkou_a, senkou_b: Displaced (shifted forward) - the cloud visible at current bar
- senkou_a_future, senkou_b_future: Undisplaced - the cloud being projected ahead
- chikou: Current close price (for Chikou span plotting)
- price_at_chikou: Close from `displacement` bars ago (for Chikou comparison)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...models import SignalCategory
from ..base import IndicatorBase


class IchimokuIndicator(IndicatorBase):
    """
    Ichimoku Cloud indicator with proper displacement.

    Default Parameters:
        tenkan_period: 9
        kijun_period: 26
        senkou_b_period: 52
        displacement: 26

    State Output:
        tenkan: Tenkan-sen value
        kijun: Kijun-sen value
        senkou_a: Senkou Span A (displaced - current visible cloud)
        senkou_b: Senkou Span B (displaced - current visible cloud)
        senkou_a_future: Senkou Span A (undisplaced - projected cloud)
        senkou_b_future: Senkou Span B (undisplaced - projected cloud)
        chikou: Current close (for Chikou span)
        price_at_chikou: Close from displacement bars ago
        cloud_color: "green" (bullish) or "red" (bearish)
        price_vs_cloud: "above", "below", or "inside"
        tk_cross: "bullish", "bearish", or None
        kumo_twist: "bullish", "bearish", or None
    """

    name = "ichimoku"
    category = SignalCategory.TREND
    required_fields = ["high", "low", "close"]
    warmup_periods = 78  # senkou_b_period (52) + displacement (26)

    _default_params = {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_b_period": 52,
        "displacement": 26,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Ichimoku values with proper displacement."""
        tenkan_p = params["tenkan_period"]
        kijun_p = params["kijun_period"]
        senkou_b_p = params["senkou_b_period"]
        displacement = params["displacement"]

        if len(data) == 0:
            return pd.DataFrame(
                {
                    "tenkan": pd.Series(dtype=float),
                    "kijun": pd.Series(dtype=float),
                    "senkou_a": pd.Series(dtype=float),
                    "senkou_b": pd.Series(dtype=float),
                    "senkou_a_future": pd.Series(dtype=float),
                    "senkou_b_future": pd.Series(dtype=float),
                    "chikou": pd.Series(dtype=float),
                    "price_at_chikou": pd.Series(dtype=float),
                },
                index=data.index,
            )

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)
        n = len(high)

        # Tenkan-sen (Conversion Line)
        tenkan = self._donchian_mid(high, low, tenkan_p)

        # Kijun-sen (Base Line)
        kijun = self._donchian_mid(high, low, kijun_p)

        # Senkou Span A/B raw (undisplaced) - the cloud being projected
        senkou_a_future = np.full(n, np.nan, dtype=np.float64)
        for i in range(kijun_p - 1, n):
            if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                senkou_a_future[i] = (tenkan[i] + kijun[i]) / 2

        senkou_b_future = self._donchian_mid(high, low, senkou_b_p)

        # Displaced Senkou spans - the cloud visible at current bar
        # At bar i, we see the cloud that was calculated at bar i - displacement
        senkou_a = np.full(n, np.nan, dtype=np.float64)
        senkou_b = np.full(n, np.nan, dtype=np.float64)
        for i in range(displacement, n):
            senkou_a[i] = senkou_a_future[i - displacement]
            senkou_b[i] = senkou_b_future[i - displacement]

        # Chikou span - current close (for plotting 26 periods behind)
        chikou = close.copy()

        # Price at chikou position - close from displacement bars ago
        # Used for comparing current close to historical price levels
        price_at_chikou = np.full(n, np.nan, dtype=np.float64)
        for i in range(displacement, n):
            price_at_chikou[i] = close[i - displacement]

        return pd.DataFrame(
            {
                "tenkan": tenkan,
                "kijun": kijun,
                "senkou_a": senkou_a,
                "senkou_b": senkou_b,
                "senkou_a_future": senkou_a_future,
                "senkou_b_future": senkou_b_future,
                "chikou": chikou,
                "price_at_chikou": price_at_chikou,
            },
            index=data.index,
        )

    def _donchian_mid(self, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """Calculate Donchian channel midpoint."""
        n = len(high)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(period - 1, n):
            highest = np.max(high[i - period + 1 : i + 1])
            lowest = np.min(low[i - period + 1 : i + 1])
            result[i] = (highest + lowest) / 2

        return result

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract Ichimoku state for rule evaluation."""
        tenkan = current.get("tenkan", np.nan)
        kijun = current.get("kijun", np.nan)
        senkou_a = current.get("senkou_a", np.nan)
        senkou_b = current.get("senkou_b", np.nan)
        senkou_a_future = current.get("senkou_a_future", np.nan)
        senkou_b_future = current.get("senkou_b_future", np.nan)
        chikou = current.get("chikou", np.nan)
        price_at_chikou = current.get("price_at_chikou", np.nan)

        # Check if we have minimum required values for state calculation
        if any(pd.isna(v) for v in [tenkan, kijun, chikou]):
            return {
                "tenkan": 0,
                "kijun": 0,
                "senkou_a": 0,
                "senkou_b": 0,
                "senkou_a_future": 0,
                "senkou_b_future": 0,
                "chikou": 0,
                "price_at_chikou": 0,
                "cloud_color": "neutral",
                "price_vs_cloud": "neutral",
                "tk_cross": None,
                "kumo_twist": None,
                "chikou_vs_price": "neutral",
            }

        # Cloud color (based on displaced cloud - the visible one)
        if pd.isna(senkou_a) or pd.isna(senkou_b):
            cloud_color = "neutral"
            price_vs_cloud = "neutral"
        else:
            cloud_color = "green" if senkou_a > senkou_b else "red"

            # Price vs cloud (use close price)
            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)

            if chikou > cloud_top:
                price_vs_cloud = "above"
            elif chikou < cloud_bottom:
                price_vs_cloud = "below"
            else:
                price_vs_cloud = "inside"

        # TK Cross detection
        tk_cross = None
        if previous is not None:
            prev_tenkan = previous.get("tenkan", np.nan)
            prev_kijun = previous.get("kijun", np.nan)
            if not pd.isna(prev_tenkan) and not pd.isna(prev_kijun):
                if prev_tenkan <= prev_kijun and tenkan > kijun:
                    tk_cross = "bullish"
                elif prev_tenkan >= prev_kijun and tenkan < kijun:
                    tk_cross = "bearish"

        # Kumo twist detection (Senkou A crosses Senkou B)
        kumo_twist = None
        if previous is not None:
            prev_senkou_a = previous.get("senkou_a", np.nan)
            prev_senkou_b = previous.get("senkou_b", np.nan)
            if not any(pd.isna(v) for v in [prev_senkou_a, prev_senkou_b, senkou_a, senkou_b]):
                if prev_senkou_a <= prev_senkou_b and senkou_a > senkou_b:
                    kumo_twist = "bullish"
                elif prev_senkou_a >= prev_senkou_b and senkou_a < senkou_b:
                    kumo_twist = "bearish"

        # Chikou vs historical price comparison
        if pd.isna(price_at_chikou):
            chikou_vs_price = "neutral"
        elif chikou > price_at_chikou:
            chikou_vs_price = "above"
        elif chikou < price_at_chikou:
            chikou_vs_price = "below"
        else:
            chikou_vs_price = "at"

        return {
            "tenkan": float(tenkan) if not pd.isna(tenkan) else 0,
            "kijun": float(kijun) if not pd.isna(kijun) else 0,
            "senkou_a": float(senkou_a) if not pd.isna(senkou_a) else 0,
            "senkou_b": float(senkou_b) if not pd.isna(senkou_b) else 0,
            "senkou_a_future": float(senkou_a_future) if not pd.isna(senkou_a_future) else 0,
            "senkou_b_future": float(senkou_b_future) if not pd.isna(senkou_b_future) else 0,
            "chikou": float(chikou) if not pd.isna(chikou) else 0,
            "price_at_chikou": float(price_at_chikou) if not pd.isna(price_at_chikou) else 0,
            "cloud_color": cloud_color,
            "price_vs_cloud": price_vs_cloud,
            "tk_cross": tk_cross,
            "kumo_twist": kumo_twist,
            "chikou_vs_price": chikou_vs_price,
        }
