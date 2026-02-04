"""
Signal capture classes for parity testing.

Capture entry/exit signals from event-driven strategy execution for
comparison with vectorized signal generators.
"""

from datetime import datetime
from typing import List, Tuple

import pandas as pd


class SignalCapture:
    """
    Captures entry/exit signals from event-driven strategy execution.

    Attach to a strategy during backtest to record when orders are placed,
    then convert to boolean signal series for parity comparison.

    Uses shadow position tracking to correctly classify entry/exit across
    multiple orders in the same bar.

    Usage:
        capture = SignalCapture(data.index, symbol="AAPL")

        # During backtest, call on each order
        capture.record_order(timestamp, symbol, side, quantity)

        # After backtest, get signals
        entries, exits = capture.get_signals()

    Note:
        - For parity testing, timestamps must exactly match data.index
        - Long-only strategies only (short positions treated as flat)
    """

    def __init__(self, index: pd.DatetimeIndex, symbol: str = ""):
        """
        Initialize signal capture.

        Args:
            index: DatetimeIndex to align signals to (from OHLCV data)
            symbol: Symbol being tracked (for multi-symbol strategies)
        """
        self.index = index
        self.symbol = symbol
        self._entries: List[datetime] = []
        self._exits: List[datetime] = []
        self._shadow_position: float = 0.0  # Track position state per order
        self._unmatched_timestamps: List[datetime] = []  # For debugging

    def record_order(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: float,
    ) -> None:
        """
        Record an order request as entry/exit signal.

        Uses shadow position to track state across multiple orders in same bar.

        Args:
            timestamp: Bar timestamp when order was placed
            symbol: Symbol for the order
            side: "BUY" or "SELL"
            quantity: Order quantity (for shadow position update)
        """
        if self.symbol and symbol != self.symbol:
            return

        # Map order to entry/exit based on shadow position state
        if side.upper() == "BUY":
            if self._shadow_position == 0:
                # BUY while flat → entry
                self._entries.append(timestamp)
            self._shadow_position += quantity

        elif side.upper() == "SELL":
            if self._shadow_position > 0:
                # SELL while long → exit
                self._exits.append(timestamp)
            self._shadow_position = max(0, self._shadow_position - quantity)

    def get_signals(self) -> Tuple[pd.Series, pd.Series]:
        """
        Convert captured orders to boolean signal series.

        Returns:
            (entries, exits): Boolean series aligned to self.index

        Note:
            Timestamps must exactly match index. Unmatched timestamps are logged.
        """
        entries = pd.Series(False, index=self.index)
        exits = pd.Series(False, index=self.index)
        self._unmatched_timestamps.clear()

        # Mark entry timestamps (exact match only)
        for ts in self._entries:
            if ts in entries.index:
                entries.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        # Mark exit timestamps (exact match only)
        for ts in self._exits:
            if ts in exits.index:
                exits.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        return entries, exits

    @property
    def unmatched_count(self) -> int:
        """Number of timestamps that didn't match index."""
        return len(self._unmatched_timestamps)

    def reset(self) -> None:
        """Clear captured signals and reset shadow position."""
        self._entries.clear()
        self._exits.clear()
        self._shadow_position = 0.0
        self._unmatched_timestamps.clear()


class DirectionalSignalCapture:
    """
    Captures both long AND short entry/exit signals from event-driven strategy.

    Extends SignalCapture to track directional positions and generate
    4 signal series for parity comparison with DirectionalSignalGenerator.

    Position tracking:
    - Positive shadow_position = long
    - Negative shadow_position = short
    - Zero = flat

    Signal mapping:
    - BUY while flat → long_entry
    - BUY while short → short_exit (covering)
    - SELL while flat → short_entry
    - SELL while long → long_exit

    Usage:
        capture = DirectionalSignalCapture(data.index, symbol="AAPL")

        # During backtest, call on each order
        capture.record_order(timestamp, symbol, side, quantity)

        # After backtest, get all 4 signal series
        long_entries, long_exits, short_entries, short_exits = capture.get_directional_signals()
    """

    def __init__(self, index: pd.DatetimeIndex, symbol: str = ""):
        """
        Initialize directional signal capture.

        Args:
            index: DatetimeIndex to align signals to (from OHLCV data)
            symbol: Symbol being tracked (for multi-symbol strategies)
        """
        self.index = index
        self.symbol = symbol
        self._long_entries: List[datetime] = []
        self._long_exits: List[datetime] = []
        self._short_entries: List[datetime] = []
        self._short_exits: List[datetime] = []
        self._shadow_position: float = 0.0
        self._unmatched_timestamps: List[datetime] = []

    def record_order(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: float,
    ) -> None:
        """
        Record an order request as directional entry/exit signal.

        Handles reversal orders that cross through zero position:
        - BUY 200 when short 100 → short_exit AND long_entry
        - SELL 200 when long 100 → long_exit AND short_entry

        Args:
            timestamp: Bar timestamp when order was placed
            symbol: Symbol for the order
            side: "BUY" or "SELL"
            quantity: Order quantity
        """
        if self.symbol and symbol != self.symbol:
            return

        if side.upper() == "BUY":
            if self._shadow_position == 0:
                # BUY while flat → long entry
                self._long_entries.append(timestamp)
            elif self._shadow_position < 0:
                # BUY while short
                self._short_exits.append(timestamp)
                # Check for reversal: BUY quantity exceeds short position
                if quantity > abs(self._shadow_position):
                    # Reversal: also entering long
                    self._long_entries.append(timestamp)
            # else: adding to existing long, no signal
            self._shadow_position += quantity

        elif side.upper() == "SELL":
            if self._shadow_position == 0:
                # SELL while flat → short entry
                self._short_entries.append(timestamp)
            elif self._shadow_position > 0:
                # SELL while long
                self._long_exits.append(timestamp)
                # Check for reversal: SELL quantity exceeds long position
                if quantity > self._shadow_position:
                    # Reversal: also entering short
                    self._short_entries.append(timestamp)
            # else: adding to existing short, no signal
            self._shadow_position -= quantity

    def get_directional_signals(
        self,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Convert captured orders to 4 boolean signal series.

        Returns:
            (long_entries, long_exits, short_entries, short_exits):
            All are boolean Series aligned to self.index
        """
        long_entries = pd.Series(False, index=self.index)
        long_exits = pd.Series(False, index=self.index)
        short_entries = pd.Series(False, index=self.index)
        short_exits = pd.Series(False, index=self.index)
        self._unmatched_timestamps.clear()

        # Mark long entries
        for ts in self._long_entries:
            if ts in long_entries.index:
                long_entries.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        # Mark long exits
        for ts in self._long_exits:
            if ts in long_exits.index:
                long_exits.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        # Mark short entries
        for ts in self._short_entries:
            if ts in short_entries.index:
                short_entries.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        # Mark short exits
        for ts in self._short_exits:
            if ts in short_exits.index:
                short_exits.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        return long_entries, long_exits, short_entries, short_exits

    def get_signals(self) -> Tuple[pd.Series, pd.Series]:
        """
        Get combined entry/exit signals (backward compatibility).

        Returns all entries (long or short) as entries, all exits as exits.
        """
        long_e, long_x, short_e, short_x = self.get_directional_signals()
        entries = long_e | short_e
        exits = long_x | short_x
        return entries, exits

    @property
    def unmatched_count(self) -> int:
        """Number of timestamps that didn't match index."""
        return len(self._unmatched_timestamps)

    def reset(self) -> None:
        """Clear captured signals and reset shadow position."""
        self._long_entries.clear()
        self._long_exits.clear()
        self._short_entries.clear()
        self._short_exits.clear()
        self._shadow_position = 0.0
        self._unmatched_timestamps.clear()
