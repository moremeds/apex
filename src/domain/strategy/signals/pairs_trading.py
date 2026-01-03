"""Pairs Trading SignalGenerator for VectorBT."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from .indicators import sma


class PairsTradingSignalGenerator:
    """
    Vectorized pairs trading signal generation using z-score of spread.

    Trades the spread between two correlated assets using statistical arbitrage:
    - Long spread (buy A, sell B) when z-score < -entry_zscore
    - Short spread (sell A, buy B) when z-score > +entry_zscore
    - Exit when |z-score| <= exit_zscore (mean reversion)

    Parameters:
        lookback: Rolling mean/std period (default 20)
        entry_zscore: Entry threshold (default 2.0)
        exit_zscore: Exit threshold (default 0.5)

    Note: This generator requires DataFrame with 'close_a' and 'close_b' columns
    representing the two symbols in the pair.
    """

    @property
    def warmup_bars(self) -> int:
        """Rolling mean/std needs lookback bars for valid values."""
        return 20  # Default lookback

    def generate(
        self, data: pd.DataFrame, params: dict[str, Any]
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate pairs trading entry/exit signals (standard protocol).

        Args:
            data: DataFrame with 'close_a' and 'close_b' columns.
            params: Strategy parameters.
                - lookback: Rolling window for mean/std
                - entry_zscore: Entry threshold
                - exit_zscore: Exit threshold

        Returns:
            (entries, exits): Boolean series for entry/exit signals.
            Note: For direction (long/short spread), use generate_with_direction().
        """
        entries, exits, _ = self.generate_with_direction(data, params)
        return entries, exits

    def generate_with_direction(
        self, data: pd.DataFrame, params: dict[str, Any]
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate pairs trading signals with direction information.

        Extended interface for VectorBT to handle long/short spread positions.

        Args:
            data: DataFrame with 'close_a' and 'close_b' columns.
            params: Strategy parameters.

        Returns:
            (entries, exits, direction): Where direction is:
                1 = long spread (buy A, sell B)
                -1 = short spread (sell A, buy B)
                0 = no entry signal
        """
        close_a = data["close_a"]
        close_b = data["close_b"]
        index = close_a.index

        lookback = int(params.get("lookback", 20))
        entry_zscore = float(params.get("entry_zscore", 2.0))
        exit_zscore = float(params.get("exit_zscore", 0.5))

        # Calculate spread as ratio
        spread = close_a / close_b

        # Rolling statistics for z-score
        # Use TA-Lib SMA for mean; pandas rolling for std (no TA-Lib equivalent)
        rolling_mean = sma(spread, lookback)
        rolling_std = spread.rolling(window=lookback).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        zscore = (spread - rolling_mean) / rolling_std

        # Entry signals by direction
        long_entries = zscore < -entry_zscore  # Spread low → buy A, sell B
        short_entries = zscore > entry_zscore  # Spread high → sell A, buy B

        # Combined entry signal
        entries = long_entries | short_entries

        # Exit signal: mean reversion
        exits = zscore.abs() <= exit_zscore

        # Direction: 1 for long spread, -1 for short spread, 0 otherwise
        direction = pd.Series(
            np.select(
                [long_entries, short_entries],
                [1, -1],
                default=0,
            ),
            index=index,
        )

        return entries.fillna(False), exits.fillna(False), direction
