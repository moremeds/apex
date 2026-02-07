"""
Data models for backtest data feeds.

Contains:
- HistoricalBar: Internal bar representation for data loading
- AlignedBarBuffer: Multi-timeframe bar alignment tracker
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ....domain.events.domain_events import BarData, QuoteTick


@dataclass
class HistoricalBar:
    """Internal bar representation for data loading."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    bar_size: str = "1d"

    def to_bar_data(self, source: str = "historical") -> BarData:
        """Convert to BarData event."""
        from ....domain.events.domain_events import BarData as BarDataClass

        return BarDataClass(
            symbol=self.symbol,
            timeframe=self.bar_size,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=int(self.volume),
            bar_start=self.timestamp,
            bar_end=self.timestamp,
            timestamp=self.timestamp,
            source=source,
        )

    def to_quote_tick(self, source: str = "historical") -> QuoteTick:
        """Convert to QuoteTick event (using close price)."""
        from ....domain.events.domain_events import QuoteTick as QuoteTickClass

        return QuoteTickClass(
            symbol=self.symbol,
            bid=self.close,
            ask=self.close,
            last=self.close,
            volume=int(self.volume),
            timestamp=self.timestamp,
            source=source,
        )


@dataclass
class AlignedBarBuffer:
    """
    Track latest bars per symbol/timeframe and emit aligned data on primary bar close.

    For multi-timeframe strategies, this buffer accumulates bars from different
    timeframes and emits a Dict[timeframe, BarData] whenever a primary timeframe
    bar arrives. The strategy then receives aligned multi-timeframe data.

    Design:
    - Secondary timeframe bars are processed first (smaller timeframes)
    - When primary bar arrives, return aligned dict with latest from each timeframe
    - Memory: O(num_symbols * num_timeframes) - only latest bar per combination
    """

    primary_timeframe: str
    secondary_timeframes: List[str] = field(default_factory=list)
    _latest_by_symbol: Dict[str, Dict[str, BarData]] = field(default_factory=dict)

    def update(self, bar: BarData) -> Optional[Dict[str, BarData]]:
        """
        Store bar and return aligned bars when a primary bar closes.

        Args:
            bar: BarData with timeframe attribute

        Returns:
            Dict[timeframe, BarData] when primary bar arrives, None otherwise
        """
        symbol_bars = self._latest_by_symbol.setdefault(bar.symbol, {})
        symbol_bars[bar.timeframe] = bar

        # Only emit aligned data when primary timeframe bar arrives
        if bar.timeframe != self.primary_timeframe:
            return None

        # Build aligned dict: primary bar + latest secondary bars
        aligned: Dict[str, BarData] = {self.primary_timeframe: bar}
        for timeframe in self.secondary_timeframes:
            latest = symbol_bars.get(timeframe)
            if latest is not None:
                aligned[timeframe] = latest
        return aligned

    @staticmethod
    def timeframe_order(timeframe: str) -> int:
        """Get sort order for timeframe (smaller timeframes first)."""
        order = {
            "1s": 0,
            "5s": 1,
            "15s": 2,
            "30s": 3,
            "1m": 4,
            "5m": 5,
            "15m": 6,
            "30m": 7,
            "1h": 8,
            "2h": 9,
            "4h": 10,
            "1d": 11,
            "1w": 12,
            "1M": 13,
        }
        return order.get(timeframe, 99)

    @classmethod
    def sort_key(cls, bar: BarData, primary_timeframe: str) -> Tuple[datetime, int, int, str]:
        """
        Sort bars so secondary timeframes at same timestamp precede primary.

        Order: (timestamp, is_primary, timeframe_order, symbol)
        This ensures secondary bars are processed before primary, so they're
        available when the aligned dict is constructed.
        """
        timestamp = bar.timestamp or datetime.min
        # Normalize to naive UTC for consistent sorting across tz-aware/naive data
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)
        is_primary = 1 if bar.timeframe == primary_timeframe else 0
        return (timestamp, is_primary, cls.timeframe_order(bar.timeframe), bar.symbol)
