"""Buy and Hold SignalGenerator for VectorBT."""

from __future__ import annotations

from typing import Any, Tuple

import pandas as pd


class BuyAndHoldSignalGenerator:
    """
    Vectorized buy-and-hold signal generation.

    Generates a single entry signal at the start (after warmup)
    and no exit signals (hold until end).
    """

    @property
    def warmup_bars(self) -> int:
        """No warmup needed for buy-and-hold."""
        return 1

    def generate(
        self, data: pd.DataFrame, params: dict[str, Any]
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate buy-and-hold signals.

        Args:
            data: OHLCV DataFrame.
            params: Not used for buy-and-hold.

        Returns:
            (entries, exits): Entry on first bar, no exits.
        """
        index = data.index

        # Enter on first bar only
        entries = pd.Series(False, index=index)
        entries.iloc[0] = True

        # Never exit
        exits = pd.Series(False, index=index)

        return entries, exits
