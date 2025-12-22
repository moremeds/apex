"""
Data feeds for backtesting.

Provides historical data loading from various sources:
- Bar cache service (production - IB-backed daemon)
- CSV files (for offline testing)
- Parquet files (for large datasets)
- JSON fixtures (for unit tests)
- Multi-timeframe (combines multiple feeds)

All feeds yield BarData or QuoteTick events in chronological order.

Usage:
    # Load from bar cache (recommended for real backtests)
    feed = BarCacheDataFeed(
        symbols=["AAPL", "MSFT"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        bar_size="1d",
    )
    await feed.load()

    # Iterate over bars
    async for bar in feed.stream_bars():
        print(bar)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, AsyncIterator, Dict, Any
import asyncio
import csv
import json
import logging

from ...domain.events.domain_events import QuoteTick, BarData
from ...services.bar_cache_service import BarPeriod

logger = logging.getLogger(__name__)


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
        return BarData(
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
        return QuoteTick(
            symbol=self.symbol,
            bid=self.close,
            ask=self.close,
            last=self.close,
            volume=int(self.volume),
            timestamp=self.timestamp,
            source=source,
        )


class DataFeed(ABC):
    """Abstract base class for data feeds."""

    @abstractmethod
    async def load(self) -> None:
        """Load data from source."""
        ...

    @abstractmethod
    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        ...

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get list of symbols in feed."""
        ...

    @property
    @abstractmethod
    def bar_count(self) -> int:
        """Get total number of bars loaded."""
        ...


class CsvDataFeed(DataFeed):
    """
    Load historical data from CSV files.

    Expected file format:
    - One file per symbol: {symbol}.csv
    - Columns: date,open,high,low,close,volume
    - Date format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS

    Example:
        feed = CsvDataFeed(
            csv_dir="data/historical",
            symbols=["AAPL", "MSFT"],
            start_date=date(2024, 1, 1),
        )
        await feed.load()
    """

    def __init__(
        self,
        csv_dir: str,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        date_column: str = "date",
        date_format: str = "%Y-%m-%d",
        bar_size: str = "1d",
    ):
        """
        Initialize CSV data feed.

        Args:
            csv_dir: Directory containing CSV files.
            symbols: List of symbols to load.
            start_date: Start date filter (inclusive).
            end_date: End date filter (inclusive).
            date_column: Name of date column.
            date_format: Date format string.
            bar_size: Bar size string (e.g., "1d", "1h").
        """
        self._csv_dir = Path(csv_dir)
        self._symbols = symbols
        self._start_date = start_date
        self._end_date = end_date
        self._date_column = date_column
        self._date_format = date_format
        self._bar_size = bar_size

        self._bars: List[HistoricalBar] = []
        self._loaded = False

    async def load(self) -> None:
        """Load data from CSV files."""
        self._bars.clear()

        for symbol in self._symbols:
            csv_path = self._csv_dir / f"{symbol}.csv"
            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                continue

            self._load_csv_file(csv_path, symbol)

        # Sort by timestamp
        self._bars.sort(key=lambda b: b.timestamp)
        self._loaded = True

        logger.info(
            f"CsvDataFeed loaded {len(self._bars)} bars "
            f"for {len(self._symbols)} symbols"
        )

    def _load_csv_file(self, path: Path, symbol: str) -> None:
        """Load a single CSV file."""
        with open(path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse date
                    date_str = row[self._date_column]
                    try:
                        timestamp = datetime.strptime(date_str, self._date_format)
                    except ValueError:
                        # Try with time
                        timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

                    # Apply date filter
                    if self._start_date and timestamp.date() < self._start_date:
                        continue
                    if self._end_date and timestamp.date() > self._end_date:
                        continue

                    # Create bar
                    bar = HistoricalBar(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(row.get("open", 0)),
                        high=float(row.get("high", 0)),
                        low=float(row.get("low", 0)),
                        close=float(row.get("close", 0)),
                        volume=float(row.get("volume", 0)),
                        bar_size=self._bar_size,
                    )
                    self._bars.append(bar)

                except (KeyError, ValueError) as e:
                    logger.warning(f"Error parsing row in {path}: {e}")
                    continue

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        if not self._loaded:
            await self.load()

        for bar in self._bars:
            yield bar.to_bar_data(source="csv")

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return len(self._bars)


class ParquetDataFeed(DataFeed):
    """
    Load historical data from Parquet files.

    Requires pyarrow or pandas with parquet support.

    Expected file format:
    - One file per symbol: {symbol}.parquet
    - Columns: timestamp, open, high, low, close, volume
    """

    def __init__(
        self,
        parquet_dir: str,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        bar_size: str = "1d",
    ):
        """
        Initialize Parquet data feed.

        Args:
            parquet_dir: Directory containing Parquet files.
            symbols: List of symbols to load.
            start_date: Start date filter.
            end_date: End date filter.
            bar_size: Bar size string.
        """
        self._parquet_dir = Path(parquet_dir)
        self._symbols = symbols
        self._start_date = start_date
        self._end_date = end_date
        self._bar_size = bar_size

        self._bars: List[HistoricalBar] = []
        self._loaded = False

    async def load(self) -> None:
        """Load data from Parquet files."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Parquet support: pip install pandas pyarrow")

        self._bars.clear()

        for symbol in self._symbols:
            parquet_path = self._parquet_dir / f"{symbol}.parquet"
            if not parquet_path.exists():
                logger.warning(f"Parquet file not found: {parquet_path}")
                continue

            self._load_parquet_file(parquet_path, symbol, pd)

        # Sort by timestamp
        self._bars.sort(key=lambda b: b.timestamp)
        self._loaded = True

        logger.info(
            f"ParquetDataFeed loaded {len(self._bars)} bars "
            f"for {len(self._symbols)} symbols"
        )

    def _load_parquet_file(self, path: Path, symbol: str, pd) -> None:
        """Load a single Parquet file."""
        df = pd.read_parquet(path)

        # Determine timestamp column
        if "timestamp" in df.columns:
            df.set_index("timestamp", inplace=True)
        elif "date" in df.columns:
            df.set_index("date", inplace=True)

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Apply date filter
        if self._start_date:
            df = df[df.index >= pd.Timestamp(self._start_date)]
        if self._end_date:
            df = df[df.index <= pd.Timestamp(self._end_date)]

        # Create bars
        for timestamp, row in df.iterrows():
            bar = HistoricalBar(
                symbol=symbol,
                timestamp=timestamp.to_pydatetime(),
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("volume", 0)),
                bar_size=self._bar_size,
            )
            self._bars.append(bar)

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        if not self._loaded:
            await self.load()

        for bar in self._bars:
            yield bar.to_bar_data(source="parquet")

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return len(self._bars)


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
            f"FixtureDataFeed loaded {len(self._bars)} bars "
            f"for {len(self._symbols)} symbols"
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


class InMemoryDataFeed(DataFeed):
    """
    In-memory data feed for programmatic testing.

    Allows adding bars directly without file I/O.
    """

    def __init__(self):
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

    def __init__(self, feeds: List[DataFeed]):
        """
        Initialize multi-timeframe data feed.

        Args:
            feeds: List of DataFeed instances, each with a different timeframe.
        """
        self._feeds = feeds
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
            "1m": 1, "5m": 2, "15m": 3, "30m": 4,
            "1h": 5, "2h": 6, "4h": 7,
            "1d": 8, "1w": 9, "1M": 10,
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


class BarCacheDataFeed(DataFeed):
    """
    Load historical data from IB for backtesting.

    For standalone backtests, this wraps IbHistoricalDataFeed.
    For in-process backtests (Lab panel), use HistoricalDataService (TODO).

    Note: This is a thin wrapper that delegates to IbHistoricalDataFeed.
    Standalone backtests run in their own process, so no event loop conflicts.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        bar_size: str = "1d",
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 10,
    ):
        """
        Initialize bar cache data feed.

        Args:
            symbols: List of symbols to load.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            bar_size: Bar size (1d, 4h, 1h, etc.).
            host: IB TWS/Gateway host.
            port: IB TWS/Gateway port.
            client_id: IB client ID for historical requests.
        """
        # Delegate to IbHistoricalDataFeed
        self._delegate = IbHistoricalDataFeed(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            bar_size=bar_size,
            host=host,
            port=port,
            client_id=client_id,
        )

    async def load(self) -> None:
        """Load data from IB."""
        await self._delegate.load()

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        async for bar in self._delegate.stream_bars():
            yield bar

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._delegate.get_symbols()

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return self._delegate.bar_count


class IbHistoricalDataFeed(DataFeed):
    """
    Load historical data from Interactive Brokers (legacy).

    Prefer BarCacheDataFeed for production backtests.
    Uses the IbHistoricalAdapter to fetch real market data.

    Requirements:
    - IB TWS or IB Gateway running
    - ib_async package installed
    - Market data subscription for requested symbols

    Usage:
        feed = IbHistoricalDataFeed(
            symbols=["AAPL", "MSFT"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            bar_size="1d",
            host="127.0.0.1",
            port=7497,  # TWS paper: 7497, TWS live: 7496, Gateway: 4001/4002
        )
        await feed.load()

        async for bar in feed.stream_bars():
            print(bar)
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        bar_size: str = "1d",
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 10,
    ):
        """
        Initialize IB historical data feed.

        Args:
            symbols: List of symbols to load.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            bar_size: Bar size (1m, 5m, 15m, 1h, 1d, etc.).
            host: IB TWS/Gateway host.
            port: IB TWS/Gateway port.
            client_id: IB client ID.
        """
        self._symbols = symbols
        self._start_date = start_date
        self._end_date = end_date
        self._bar_size = bar_size
        self._host = host
        self._port = port
        self._client_id = client_id

        self._bars: List[HistoricalBar] = []
        self._loaded = False
        self._adapter = None

    async def load(self) -> None:
        """
        Load historical data from IB.

        Connects to IB, fetches data for all symbols, then disconnects.
        """
        from ..adapters.ib.historical_adapter import IbHistoricalAdapter

        self._bars.clear()

        # Create and connect adapter
        self._adapter = IbHistoricalAdapter(
            host=self._host,
            port=self._port,
            client_id=self._client_id,
        )

        try:
            await self._adapter.connect()
            logger.info(f"Connected to IB at {self._host}:{self._port}")

            # Convert dates to datetime
            start_dt = datetime.combine(self._start_date, datetime.min.time())
            end_dt = datetime.combine(self._end_date, datetime.max.time())

            # Fetch data for each symbol
            for symbol in self._symbols:
                logger.info(f"Fetching {self._bar_size} bars for {symbol}...")

                try:
                    bars = await self._adapter.fetch_bars(
                        symbol=symbol,
                        timeframe=self._bar_size,
                        start=start_dt,
                        end=end_dt,
                    )

                    for bar in bars:
                        hist_bar = HistoricalBar(
                            symbol=symbol,
                            timestamp=bar.timestamp,
                            open=bar.open,
                            high=bar.high,
                            low=bar.low,
                            close=bar.close,
                            volume=bar.volume or 0,
                            bar_size=self._bar_size,
                        )
                        self._bars.append(hist_bar)

                    logger.info(f"Loaded {len(bars)} bars for {symbol}")

                except Exception as e:
                    logger.error(f"Failed to fetch bars for {symbol}: {e}")

            # Sort all bars by timestamp
            self._bars.sort(key=lambda b: b.timestamp)
            self._loaded = True

            logger.info(
                f"IbHistoricalDataFeed loaded {len(self._bars)} total bars "
                f"for {len(self._symbols)} symbols"
            )

        finally:
            # Always disconnect
            if self._adapter:
                await self._adapter.disconnect()
                logger.info("Disconnected from IB")

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        if not self._loaded:
            await self.load()

        for bar in self._bars:
            yield bar.to_bar_data(source="IB")

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return len(self._bars)


# Factory functions for multi-timeframe feeds


def create_csv_multi_timeframe_feed(
    csv_dir: str,
    symbols: List[str],
    timeframes: List[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> MultiTimeframeDataFeed:
    """
    Create a multi-timeframe feed from CSV files.

    Expects CSV files named {symbol}_{timeframe}.csv in the csv_dir.
    For example: AAPL_1m.csv, AAPL_1h.csv, AAPL_1d.csv

    Args:
        csv_dir: Directory containing CSV files.
        symbols: List of symbols to load.
        timeframes: List of timeframes (e.g., ["1m", "1h", "1d"]).
        start_date: Start date filter.
        end_date: End date filter.

    Returns:
        MultiTimeframeDataFeed combining all timeframes.

    Example:
        feed = create_csv_multi_timeframe_feed(
            csv_dir="data/historical",
            symbols=["AAPL", "MSFT"],
            timeframes=["1m", "1h", "1d"],
            start_date=date(2024, 1, 1),
        )
    """
    feeds = []
    for timeframe in timeframes:
        # Create a feed for each timeframe
        # Files should be named {symbol}_{timeframe}.csv
        feed = CsvDataFeed(
            csv_dir=csv_dir,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            bar_size=timeframe,
        )
        feeds.append(feed)

    return MultiTimeframeDataFeed(feeds)


def create_ib_multi_timeframe_feed(
    symbols: List[str],
    timeframes: List[str],
    start_date: date,
    end_date: date,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 10,
) -> MultiTimeframeDataFeed:
    """
    Create a multi-timeframe feed from IB historical data.

    Args:
        symbols: List of symbols to load.
        timeframes: List of timeframes (e.g., ["1m", "1h", "1d"]).
        start_date: Start date.
        end_date: End date.
        host: IB TWS/Gateway host.
        port: IB TWS/Gateway port.
        client_id: Base client ID (incremented for each timeframe).

    Returns:
        MultiTimeframeDataFeed combining all timeframes.

    Example:
        feed = create_ib_multi_timeframe_feed(
            symbols=["AAPL", "MSFT"],
            timeframes=["1m", "1h", "1d"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
        )
    """
    feeds = []
    for i, timeframe in enumerate(timeframes):
        feed = IbHistoricalDataFeed(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            bar_size=timeframe,
            host=host,
            port=port,
            client_id=client_id + i,  # Use different client IDs
        )
        feeds.append(feed)

    return MultiTimeframeDataFeed(feeds)
