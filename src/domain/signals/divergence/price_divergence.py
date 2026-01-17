"""
Price Divergence Detection.

Detects divergences between price action and indicator values:
- Bullish Divergence: Price makes lower low, indicator makes higher low
- Bearish Divergence: Price makes higher high, indicator makes lower high
- Hidden Bullish: Price makes higher low, indicator makes lower low
- Hidden Bearish: Price makes lower high, indicator makes higher high
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from src.utils.logging_setup import get_logger

from ..models import Divergence, DivergenceType

logger = get_logger(__name__)


class PriceDivergenceDetector:
    """
    Detects bullish/bearish divergences between price and indicator values.

    Divergences occur when price and indicator move in opposite directions,
    often signaling potential reversals.

    Example:
        detector = PriceDivergenceDetector(lookback=20, min_bars_apart=5)
        divergences = detector.detect(
            price=df['close'],
            indicator=df['rsi'],
            indicator_name='rsi',
            symbol='AAPL',
            timeframe='1h'
        )
    """

    def __init__(
        self,
        lookback: int = 20,
        min_bars_apart: int = 5,
        swing_window: int = 5,
    ) -> None:
        """
        Initialize the detector.

        Args:
            lookback: Number of bars to scan for divergence patterns
            min_bars_apart: Minimum distance between pivot points
            swing_window: Window size for swing high/low detection
        """
        self._lookback = lookback
        self._min_bars_apart = min_bars_apart
        self._swing_window = swing_window

    def detect(
        self,
        price: pd.Series,
        indicator: pd.Series,
        indicator_name: str,
        symbol: str,
        timeframe: str,
    ) -> List[Divergence]:
        """
        Detect all divergence types in the given data.

        Args:
            price: Price series (typically close prices)
            indicator: Indicator value series
            indicator_name: Name of the indicator
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            List of detected Divergence objects
        """
        if len(price) < self._lookback + self._swing_window:
            return []

        # Align indices
        price = price.iloc[-self._lookback :]
        indicator = indicator.iloc[-self._lookback :]

        divergences: List[Divergence] = []

        # Find swing points
        price_highs = self._find_swing_highs(price)
        price_lows = self._find_swing_lows(price)
        ind_highs = self._find_swing_highs(indicator)
        ind_lows = self._find_swing_lows(indicator)

        # Detect regular divergences
        divergences.extend(
            self._check_bearish_divergence(
                price_highs, ind_highs, price, indicator, indicator_name, symbol, timeframe
            )
        )
        divergences.extend(
            self._check_bullish_divergence(
                price_lows, ind_lows, price, indicator, indicator_name, symbol, timeframe
            )
        )

        # Detect hidden divergences
        divergences.extend(
            self._check_hidden_bearish_divergence(
                price_highs, ind_highs, price, indicator, indicator_name, symbol, timeframe
            )
        )
        divergences.extend(
            self._check_hidden_bullish_divergence(
                price_lows, ind_lows, price, indicator, indicator_name, symbol, timeframe
            )
        )

        return divergences

    def _find_swing_highs(self, series: pd.Series) -> List[int]:
        """Find local maxima indices."""
        highs: List[int] = []
        w = self._swing_window

        for i in range(w, len(series) - w):
            window = series.iloc[i - w : i + w + 1]
            if series.iloc[i] == window.max():
                highs.append(i)

        return highs

    def _find_swing_lows(self, series: pd.Series) -> List[int]:
        """Find local minima indices."""
        lows: List[int] = []
        w = self._swing_window

        for i in range(w, len(series) - w):
            window = series.iloc[i - w : i + w + 1]
            if series.iloc[i] == window.min():
                lows.append(i)

        return lows

    def _find_nearest(self, pivots: List[int], target: int, max_dist: int = 3) -> Optional[int]:
        """Find the nearest pivot to a target index within max_dist."""
        candidates = [p for p in pivots if abs(p - target) <= max_dist]
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(x - target))

    def _check_bullish_divergence(
        self,
        price_lows: List[int],
        ind_lows: List[int],
        price: pd.Series,
        indicator: pd.Series,
        indicator_name: str,
        symbol: str,
        timeframe: str,
    ) -> List[Divergence]:
        """
        Detect bullish divergence: price lower low + indicator higher low.
        """
        divergences: List[Divergence] = []

        for i in range(1, len(price_lows)):
            idx1, idx2 = price_lows[i - 1], price_lows[i]
            if idx2 - idx1 < self._min_bars_apart:
                continue

            # Price makes lower low
            if price.iloc[idx2] >= price.iloc[idx1]:
                continue

            # Find corresponding indicator lows
            ind_idx1 = self._find_nearest(ind_lows, idx1)
            ind_idx2 = self._find_nearest(ind_lows, idx2)

            if ind_idx1 is None or ind_idx2 is None:
                continue

            # Indicator makes higher low
            if indicator.iloc[ind_idx2] > indicator.iloc[ind_idx1]:
                strength = self._calculate_strength(
                    price.iloc[idx1],
                    price.iloc[idx2],
                    indicator.iloc[ind_idx1],
                    indicator.iloc[ind_idx2],
                )

                divergences.append(
                    Divergence(
                        type=DivergenceType.BULLISH,
                        indicator=indicator_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        price_point1=self._make_point(price, idx1),
                        price_point2=self._make_point(price, idx2),
                        indicator_point1=self._make_point(indicator, ind_idx1),
                        indicator_point2=self._make_point(indicator, ind_idx2),
                        strength=strength,
                        bars_apart=idx2 - idx1,
                    )
                )

        return divergences

    def _check_bearish_divergence(
        self,
        price_highs: List[int],
        ind_highs: List[int],
        price: pd.Series,
        indicator: pd.Series,
        indicator_name: str,
        symbol: str,
        timeframe: str,
    ) -> List[Divergence]:
        """
        Detect bearish divergence: price higher high + indicator lower high.
        """
        divergences: List[Divergence] = []

        for i in range(1, len(price_highs)):
            idx1, idx2 = price_highs[i - 1], price_highs[i]
            if idx2 - idx1 < self._min_bars_apart:
                continue

            # Price makes higher high
            if price.iloc[idx2] <= price.iloc[idx1]:
                continue

            # Find corresponding indicator highs
            ind_idx1 = self._find_nearest(ind_highs, idx1)
            ind_idx2 = self._find_nearest(ind_highs, idx2)

            if ind_idx1 is None or ind_idx2 is None:
                continue

            # Indicator makes lower high
            if indicator.iloc[ind_idx2] < indicator.iloc[ind_idx1]:
                strength = self._calculate_strength(
                    price.iloc[idx1],
                    price.iloc[idx2],
                    indicator.iloc[ind_idx1],
                    indicator.iloc[ind_idx2],
                )

                divergences.append(
                    Divergence(
                        type=DivergenceType.BEARISH,
                        indicator=indicator_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        price_point1=self._make_point(price, idx1),
                        price_point2=self._make_point(price, idx2),
                        indicator_point1=self._make_point(indicator, ind_idx1),
                        indicator_point2=self._make_point(indicator, ind_idx2),
                        strength=strength,
                        bars_apart=idx2 - idx1,
                    )
                )

        return divergences

    def _check_hidden_bullish_divergence(
        self,
        price_lows: List[int],
        ind_lows: List[int],
        price: pd.Series,
        indicator: pd.Series,
        indicator_name: str,
        symbol: str,
        timeframe: str,
    ) -> List[Divergence]:
        """
        Detect hidden bullish divergence: price higher low + indicator lower low.
        """
        divergences: List[Divergence] = []

        for i in range(1, len(price_lows)):
            idx1, idx2 = price_lows[i - 1], price_lows[i]
            if idx2 - idx1 < self._min_bars_apart:
                continue

            # Price makes higher low
            if price.iloc[idx2] <= price.iloc[idx1]:
                continue

            # Find corresponding indicator lows
            ind_idx1 = self._find_nearest(ind_lows, idx1)
            ind_idx2 = self._find_nearest(ind_lows, idx2)

            if ind_idx1 is None or ind_idx2 is None:
                continue

            # Indicator makes lower low
            if indicator.iloc[ind_idx2] < indicator.iloc[ind_idx1]:
                strength = self._calculate_strength(
                    price.iloc[idx1],
                    price.iloc[idx2],
                    indicator.iloc[ind_idx1],
                    indicator.iloc[ind_idx2],
                )

                divergences.append(
                    Divergence(
                        type=DivergenceType.HIDDEN_BULLISH,
                        indicator=indicator_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        price_point1=self._make_point(price, idx1),
                        price_point2=self._make_point(price, idx2),
                        indicator_point1=self._make_point(indicator, ind_idx1),
                        indicator_point2=self._make_point(indicator, ind_idx2),
                        strength=strength,
                        bars_apart=idx2 - idx1,
                    )
                )

        return divergences

    def _check_hidden_bearish_divergence(
        self,
        price_highs: List[int],
        ind_highs: List[int],
        price: pd.Series,
        indicator: pd.Series,
        indicator_name: str,
        symbol: str,
        timeframe: str,
    ) -> List[Divergence]:
        """
        Detect hidden bearish divergence: price lower high + indicator higher high.
        """
        divergences: List[Divergence] = []

        for i in range(1, len(price_highs)):
            idx1, idx2 = price_highs[i - 1], price_highs[i]
            if idx2 - idx1 < self._min_bars_apart:
                continue

            # Price makes lower high
            if price.iloc[idx2] >= price.iloc[idx1]:
                continue

            # Find corresponding indicator highs
            ind_idx1 = self._find_nearest(ind_highs, idx1)
            ind_idx2 = self._find_nearest(ind_highs, idx2)

            if ind_idx1 is None or ind_idx2 is None:
                continue

            # Indicator makes higher high
            if indicator.iloc[ind_idx2] > indicator.iloc[ind_idx1]:
                strength = self._calculate_strength(
                    price.iloc[idx1],
                    price.iloc[idx2],
                    indicator.iloc[ind_idx1],
                    indicator.iloc[ind_idx2],
                )

                divergences.append(
                    Divergence(
                        type=DivergenceType.HIDDEN_BEARISH,
                        indicator=indicator_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        price_point1=self._make_point(price, idx1),
                        price_point2=self._make_point(price, idx2),
                        indicator_point1=self._make_point(indicator, ind_idx1),
                        indicator_point2=self._make_point(indicator, ind_idx2),
                        strength=strength,
                        bars_apart=idx2 - idx1,
                    )
                )

        return divergences

    def _calculate_strength(
        self,
        price1: float,
        price2: float,
        ind1: float,
        ind2: float,
    ) -> int:
        """Calculate divergence strength based on magnitude difference."""
        if price1 == 0 or ind1 == 0:
            return 50

        price_change = (price2 - price1) / abs(price1)
        ind_change = (ind2 - ind1) / abs(ind1)

        # Divergence magnitude: larger opposite movements = stronger
        divergence_magnitude = abs(price_change - ind_change)

        # Scale to 0-100, capped
        return min(100, max(0, int(divergence_magnitude * 500)))

    @staticmethod
    def _make_point(series: pd.Series, idx: int) -> Tuple[datetime, float]:
        """Create a (timestamp, value) tuple for a point."""
        ts = series.index[idx]
        if not isinstance(ts, datetime):
            ts = datetime.utcnow()
        return (ts, float(series.iloc[idx]))
