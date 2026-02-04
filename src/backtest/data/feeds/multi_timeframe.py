"""
Multi-timeframe data feed for backtesting.

Contains:
- MultiTimeframeDataFeed: Combines multiple feeds with different timeframes
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, List, Sequence

from ....domain.events.domain_events import BarData
from .base import DataFeed

logger = logging.getLogger(__name__)


class MultiTimeframeDataFeed(DataFeed):
    """
    Combines multiple data feeds with different timeframes.

    Merges bars from multiple timeframes (e.g., 1m, 1h, 1d) and streams
    them in chronological order. Strategy receives bars with their
    timeframe in bar.timeframe field.

    Usage:
        # Create individual feeds for each timeframe
        feed_1m = CsvDataFeed(csv_dir="data", symbols=["AAPL"], bar_size="1m")
        feed_1h = CsvDataFeed(csv_dir="data", symbols=["AAPL"], bar_size="1h")
        feed_1d = CsvDataFeed(csv_dir="data", symbols=["AAPL"], bar_size="1d")

        # Combine into multi-timeframe feed
        mtf_feed = MultiTimeframeDataFeed([feed_1m, feed_1h, feed_1d])
        await mtf_feed.load()

        # Stream all bars interleaved by timestamp
        async for bar in mtf_feed.stream_bars():
            if bar.timeframe == "1d":
                # Daily bar logic
            elif bar.timeframe == "1h":
                # Hourly bar logic
            elif bar.timeframe == "1m":
                # Minute bar logic
    """

    def __init__(self, feeds: Sequence[DataFeed]) -> None:
        """
        Initialize multi-timeframe data feed.

        Args:
            feeds: List of DataFeed instances, each with a different timeframe.
        """
        self._feeds = list(feeds)
        self._bars: List[BarData] = []
        self._symbols: List[str] = []
        self._loaded = False

    async def load(self) -> None:
        """Load data from all feeds."""
        self._bars.clear()
        symbols_set = set()

        for feed in self._feeds:
            await feed.load()
            symbols_set.update(feed.get_symbols())

            # Collect bars from each feed
            async for bar in feed.stream_bars():
                self._bars.append(bar)

        self._symbols = list(symbols_set)

        # Sort all bars by timestamp (stable sort preserves order for same timestamp)
        # For same timestamp, larger timeframes come after smaller ones
        # (so 1d bar at 00:00 comes after all 1m bars for that day)
        self._bars.sort(key=lambda b: (b.timestamp, self._timeframe_order(b.timeframe)))
        self._loaded = True

        logger.info(
            f"MultiTimeframeDataFeed loaded {len(self._bars)} bars "
            f"across {len(self._feeds)} timeframes for {len(self._symbols)} symbols"
        )

    @staticmethod
    def _timeframe_order(timeframe: str) -> int:
        """Get sort order for timeframe (smaller timeframes first)."""
        order = {
            "1m": 1,
            "5m": 2,
            "15m": 3,
            "30m": 4,
            "1h": 5,
            "2h": 6,
            "4h": 7,
            "1d": 8,
            "1w": 9,
            "1M": 10,
        }
        return order.get(timeframe, 99)

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order across all timeframes."""
        if not self._loaded:
            await self.load()

        for bar in self._bars:
            yield bar

    def get_symbols(self) -> List[str]:
        """Get list of symbols across all feeds."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """Get total bar count across all timeframes."""
        return len(self._bars)

    def get_timeframes(self) -> List[str]:
        """Get list of timeframes in this feed."""
        return list(set(bar.timeframe for bar in self._bars))
