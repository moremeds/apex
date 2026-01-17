"""
BarBuilder - Constructs OHLCV bars from tick data.

Maintains a single bar window [bar_start, bar_end) and accumulates
tick data into OHLCV values. Thread-safe for single-bar updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.domain.events.domain_events import BarCloseEvent
from src.utils.timezone import UTC, to_utc

TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}


@dataclass
class BarBuilder:
    """
    Accumulates tick data into an OHLCV bar.

    Each BarBuilder handles one symbol/timeframe combination for a single
    bar period. When the bar is complete, call to_close_event() to emit
    and reset for the next period.
    """

    symbol: str
    timeframe: str
    bar_start: datetime
    bar_end: datetime

    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: float = 0.0
    trade_count: int = 0

    @property
    def is_empty(self) -> bool:
        """True if no ticks have been applied to this bar."""
        return self.trade_count == 0

    def update(self, price: float, volume: Optional[float], timestamp: datetime) -> None:
        """
        Update OHLCV with a single tick. O(1) operation.

        Args:
            price: Tick price (last, mid, or calculated)
            volume: Optional tick volume
            timestamp: Tick timestamp (for tracking, not bar boundary detection)
        """
        if self.trade_count == 0:
            self.open = price
            self.high = price
            self.low = price
        else:
            if self.high is None or price > self.high:
                self.high = price
            if self.low is None or price < self.low:
                self.low = price

        self.close = price

        if volume is not None:
            self.volume += float(volume)

        self.trade_count += 1

    def to_close_event(self) -> BarCloseEvent:
        """
        Create a BarCloseEvent for the completed bar.

        Returns:
            BarCloseEvent with OHLCV data and bar boundaries
        """
        return BarCloseEvent(
            timestamp=self.bar_end,
            symbol=self.symbol,
            timeframe=self.timeframe,
            open=self.open or 0.0,
            high=self.high or 0.0,
            low=self.low or 0.0,
            close=self.close or 0.0,
            volume=self.volume,
            bar_end=self.bar_end,
        )

    @staticmethod
    def compute_bounds(timeframe: str, timestamp: datetime) -> tuple[datetime, datetime]:
        """
        Compute bar_start and bar_end for a given timeframe and timestamp.

        Args:
            timeframe: Timeframe string (e.g., "1m", "5m", "1h")
            timestamp: Reference timestamp to compute bar boundaries

        Returns:
            Tuple of (bar_start, bar_end) datetimes

        Raises:
            ValueError: If timeframe is not supported
        """
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(TIMEFRAME_SECONDS.keys())}"
            )

        ts = to_utc(timestamp)

        if timeframe == "1w":
            # Weekly bars start on Monday 00:00 UTC
            days_since_monday = ts.weekday()
            bar_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            bar_start = bar_start - timedelta(days=days_since_monday)
            bar_end = bar_start + timedelta(weeks=1)
            return bar_start, bar_end

        if timeframe == "1d":
            bar_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            bar_end = bar_start + timedelta(days=1)
            return bar_start, bar_end

        interval = TIMEFRAME_SECONDS[timeframe]
        epoch = int(ts.timestamp())
        start_epoch = epoch - (epoch % interval)
        bar_start = datetime.fromtimestamp(start_epoch, tz=UTC)
        bar_end = bar_start + timedelta(seconds=interval)
        return bar_start, bar_end

    @classmethod
    def create_for_timestamp(
        cls, symbol: str, timeframe: str, timestamp: datetime
    ) -> "BarBuilder":
        """
        Factory method to create a BarBuilder for a given timestamp.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            timestamp: Reference timestamp

        Returns:
            New BarBuilder with computed bar boundaries
        """
        bar_start, bar_end = cls.compute_bounds(timeframe, timestamp)
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            bar_start=bar_start,
            bar_end=bar_end,
        )
