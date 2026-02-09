"""Moving Average Crossover SignalGenerator for VectorBT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..param_loader import get_strategy_params
from .indicators import sma

_DEFAULTS = get_strategy_params("ma_cross")


class MACrossSignalGenerator:
    """
    Vectorized MA crossover signal generation using TA-Lib.

    Generates entry signals when fast MA crosses above slow MA,
    and exit signals when fast MA crosses below slow MA.

    Parameters:
        short_window: Fast MA period (default 10)
        long_window: Slow MA period (default 50)

    Note: Legacy param names 'fast_period'/'slow_period' are also supported
    for backward compatibility with existing experiment configs.
    """

    @property
    def warmup_bars(self) -> int:
        """Max of fast/slow period for indicator warmup."""
        return 50  # Default max(short_window, long_window)

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate MA crossover signals.

        Args:
            data: OHLCV DataFrame with 'close' column.
            params: Strategy parameters.
                - short_window (or fast_period): Fast MA period
                - long_window (or slow_period): Slow MA period

        Returns:
            (entries, exits): Boolean series for entry/exit signals.
        """
        close = data["close"]

        effective = {**_DEFAULTS, **params}
        # Support both new and legacy param names
        short_window = effective.get("short_window", effective.get("fast_period", 10))
        long_window = effective.get("long_window", effective.get("slow_period", 50))

        # Calculate MAs using TA-Lib
        fast_ma = sma(close, int(short_window))
        slow_ma = sma(close, int(long_window))

        # Entry: fast crosses above slow
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))

        # Exit: fast crosses below slow
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        return entries.fillna(False), exits.fillna(False)
