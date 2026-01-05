"""SignalGenerator protocol for vectorized entry/exit signal generation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import pandas as pd


class SignalGenerator(Protocol):
    """
    Protocol for vectorized entry/exit signal generation.

    Implementations generate boolean entry/exit signals for VectorBT backtesting.
    The same parameters used here should produce identical signals to the
    event-driven Strategy.on_bar() implementation.

    Example:
        class MACrossSignalGenerator:
            @property
            def warmup_bars(self) -> int:
                return 50

            def generate(self, data: pd.DataFrame, params: dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
                from .indicators import sma
                close = data["close"]
                fast = sma(close, params.get("short_window", 10))
                slow = sma(close, params.get("long_window", 50))
                entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
                exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))
                return entries, exits

    For Multi-Timeframe (MTF) strategies:
        class MTFSignalGenerator:
            @property
            def warmup_bars(self) -> int:
                return 50

            def generate(
                self,
                data: pd.DataFrame,
                params: dict[str, Any],
                secondary_data: Optional[Dict[str, pd.DataFrame]] = None
            ) -> Tuple[pd.Series, pd.Series]:
                # data = primary timeframe (e.g., 1d)
                # secondary_data = {"1h": hourly_df, "4h": four_hour_df}
                hourly = secondary_data.get("1h") if secondary_data else None
                # ... use both timeframes for signal generation
    """

    @property
    def warmup_bars(self) -> int:
        """
        Number of bars required before signals are valid.

        Signals during the warmup period should be ignored by the engine.
        This is typically max(indicator_periods) to ensure all indicators
        have sufficient data.
        """
        ...

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals from OHLCV data.

        Args:
            data: DataFrame with columns: open, high, low, close, volume.
                  Index is datetime. This is the primary timeframe data.
            params: Strategy parameters (e.g., {"short_window": 10, "long_window": 50}).
            secondary_data: Optional dict of secondary timeframe data.
                  Keys are timeframe strings (e.g., "1h", "4h").
                  Values are DataFrames with same columns as data.
                  Used for multi-timeframe (MTF) strategies.

        Returns:
            Tuple of (entries, exits) boolean Series aligned to data.index.
            - entries: True where a long position should be opened
            - exits: True where a position should be closed
        """
        ...
