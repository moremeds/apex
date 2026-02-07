"""
Memory-based data feeds for backtesting.

Contains:
- InMemoryDataFeed: Programmatic bar insertion for testing
- FixtureDataFeed: Load from JSON fixtures for unit tests
- CachedBarDataFeed: Pre-fetched bars from HistoricalDataService
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from ....domain.events.domain_events import BarData, QuoteTick
from .base import DataFeed
from .models import HistoricalBar

logger = logging.getLogger(__name__)


class InMemoryDataFeed(DataFeed):
    """
    In-memory data feed for programmatic testing.

    Allows adding bars directly without file I/O.
    """

    def __init__(self) -> None:
        """Initialize in-memory data feed."""
        self._bars: List[HistoricalBar] = []
        self._symbols: List[str] = []

    def add_bar(
        self,
        symbol: str,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> None:
        """Add a bar to the feed."""
        bar = HistoricalBar(
            symbol=symbol,
            timestamp=timestamp,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        self._bars.append(bar)
        if symbol not in self._symbols:
            self._symbols.append(symbol)

    def add_bars(self, bars: List[Dict[str, Any]]) -> None:
        """Add multiple bars from dictionaries."""
        for bar_dict in bars:
            self.add_bar(
                symbol=bar_dict["symbol"],
                timestamp=bar_dict["timestamp"],
                open=bar_dict["open"],
                high=bar_dict["high"],
                low=bar_dict["low"],
                close=bar_dict["close"],
                volume=bar_dict.get("volume", 0),
            )

    async def load(self) -> None:
        """Sort bars by timestamp."""
        self._bars.sort(key=lambda b: b.timestamp)

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        for bar in sorted(self._bars, key=lambda b: b.timestamp):
            yield bar.to_bar_data(source="memory")

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return len(self._bars)

    def clear(self) -> None:
        """Clear all bars."""
        self._bars.clear()
        self._symbols.clear()


class FixtureDataFeed(DataFeed):
    """
    Load data from JSON fixture files for unit testing.

    Fixture format:
    [
        {"symbol": "AAPL", "timestamp": "2024-01-01T09:30:00",
         "open": 150.0, "high": 151.0, "low": 149.0, "close": 150.5, "volume": 1000},
        ...
    ]
    """

    def __init__(
        self,
        fixture_path: str,
        symbols: Optional[List[str]] = None,
    ):
        """
        Initialize fixture data feed.

        Args:
            fixture_path: Path to JSON fixture file.
            symbols: Optional symbol filter.
        """
        self._fixture_path = Path(fixture_path)
        self._symbols_filter = symbols

        self._bars: List[HistoricalBar] = []
        self._symbols: List[str] = []
        self._loaded = False

    async def load(self) -> None:
        """Load data from fixture file."""
        with open(self._fixture_path, "r") as f:
            data = json.load(f)

        self._bars.clear()
        symbols_set = set()

        for item in data:
            symbol = item["symbol"]

            if self._symbols_filter and symbol not in self._symbols_filter:
                continue

            bar = HistoricalBar(
                symbol=symbol,
                timestamp=datetime.fromisoformat(item["timestamp"]),
                open=float(item.get("open", 0)),
                high=float(item.get("high", 0)),
                low=float(item.get("low", 0)),
                close=float(item.get("close", 0)),
                volume=float(item.get("volume", 0)),
                bar_size=item.get("bar_size", "1d"),
            )
            self._bars.append(bar)
            symbols_set.add(symbol)

        self._symbols = list(symbols_set)
        self._bars.sort(key=lambda b: b.timestamp)
        self._loaded = True

        logger.info(
            f"FixtureDataFeed loaded {len(self._bars)} bars " f"for {len(self._symbols)} symbols"
        )

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        if not self._loaded:
            await self.load()

        for bar in self._bars:
            yield bar.to_bar_data(source="fixture")

    async def stream_ticks(self) -> AsyncIterator[QuoteTick]:
        """Stream as ticks (using close price)."""
        if not self._loaded:
            await self.load()

        for bar in self._bars:
            yield bar.to_quote_tick(source="fixture")

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return len(self._bars)


class CachedBarDataFeed(DataFeed):
    """
    Data feed using pre-fetched bars from HistoricalDataService.

    For in-process backtests (Lab panel) where bars are already cached.
    No IB connection required - uses bars passed at construction.

    Usage:
        # Fetch bars from HistoricalDataService on main loop
        bars_by_symbol = await historical_service.fetch_bars_batch([...])

        # Pass to backtest (can run in separate thread)
        feed = CachedBarDataFeed(
            bars_by_symbol=bars_by_symbol,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        )
        await feed.load()  # Just filters and sorts, no I/O
    """

    def __init__(
        self,
        bars_by_symbol: Dict[str, List[BarData]],
        start_date: date,
        end_date: date,
    ):
        """
        Initialize cached bar data feed.

        Args:
            bars_by_symbol: Dict mapping symbol to list of BarData.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
        """
        self._bars_by_symbol = bars_by_symbol
        self._start_date = start_date
        self._end_date = end_date
        self._symbols = list(bars_by_symbol.keys())
        self._bars: List[BarData] = []
        self._loaded = False

    async def load(self) -> None:
        """Load and filter bars (no I/O - just filters pre-fetched data)."""
        if self._loaded:
            return

        start_dt = datetime.combine(self._start_date, datetime.min.time())
        end_dt = datetime.combine(self._end_date, datetime.max.time())

        all_bars = []
        for symbol, bars in self._bars_by_symbol.items():
            for bar in bars:
                if bar.timestamp and start_dt <= bar.timestamp <= end_dt:
                    all_bars.append(bar)

        # Sort by timestamp
        all_bars.sort(key=lambda b: (b.timestamp or datetime.min, b.symbol))
        self._bars = all_bars
        self._loaded = True

        logger.info(
            f"CachedBarDataFeed loaded {len(self._bars)} bars " f"for {len(self._symbols)} symbols"
        )

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        if not self._loaded:
            await self.load()

        for bar in self._bars:
            yield bar

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return len(self._bars)
