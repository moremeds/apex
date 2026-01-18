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

import csv
import heapq
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, AsyncIterator, Dict, Generator, List, Optional, Sequence, Tuple

from ...domain.events.domain_events import BarData, QuoteTick

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
    def stream_bars(self) -> AsyncIterator[BarData]:
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
    _latest_by_symbol: Dict[str, Dict[str, "BarData"]] = field(default_factory=dict)

    def update(self, bar: "BarData") -> Optional[Dict[str, "BarData"]]:
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
        aligned: Dict[str, "BarData"] = {self.primary_timeframe: bar}
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
    def sort_key(cls, bar: "BarData", primary_timeframe: str) -> Tuple[datetime, int, int, str]:
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
            f"CsvDataFeed loaded {len(self._bars)} bars " f"for {len(self._symbols)} symbols"
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


class StreamingCsvDataFeed(DataFeed):
    """
    OPT-009: Streaming data feed that yields bars without loading all to memory.

    Uses heap-based k-way merge sort to combine multiple symbol streams
    in timestamp order. Memory usage is O(num_symbols) instead of O(total_bars).

    Key differences from CsvDataFeed:
    - Lazy file reading: Data is read on-demand during stream_bars()
    - Heap-based merge: Only one bar per symbol in memory at a time
    - bar_count is -1 until streaming completes (count unknown until end)

    Example:
        feed = StreamingCsvDataFeed(
            csv_dir="data/historical",
            symbols=["AAPL", "MSFT"],
            start_date=date(2024, 1, 1),
        )
        await feed.load()  # Just initializes readers, no data loaded

        # Bars yielded in timestamp order, memory-efficient
        async for bar in feed.stream_bars():
            print(bar)
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
        Initialize streaming CSV data feed.

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
        self._bar_count = -1  # Unknown until streaming completes
        self._loaded = False

    async def load(self) -> None:
        """
        Initialize readers (no data loaded yet).

        Unlike CsvDataFeed, this just prepares for streaming.
        Actual data reading happens during stream_bars().
        """
        # Verify files exist
        missing = []
        for symbol in self._symbols:
            csv_path = self._csv_dir / f"{symbol}.csv"
            if not csv_path.exists():
                missing.append(symbol)

        if missing:
            logger.warning(f"CSV files not found for: {missing}")

        self._loaded = True
        logger.info(
            f"StreamingCsvDataFeed initialized for {len(self._symbols)} symbols "
            f"(data will be streamed on demand)"
        )

    def _create_reader(self, symbol: str) -> Generator[HistoricalBar, None, None]:
        """
        Create a streaming reader generator for a symbol.

        Yields bars one at a time from the CSV file, filtering by date range.
        The generator keeps only one bar in memory at a time.
        """
        csv_path = self._csv_dir / f"{symbol}.csv"
        if not csv_path.exists():
            return

        try:
            with open(csv_path, "r") as f:
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

                        # Apply date filter early (before yielding)
                        if self._start_date and timestamp.date() < self._start_date:
                            continue
                        if self._end_date and timestamp.date() > self._end_date:
                            continue

                        # Yield bar
                        yield HistoricalBar(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=float(row.get("open", 0)),
                            high=float(row.get("high", 0)),
                            low=float(row.get("low", 0)),
                            close=float(row.get("close", 0)),
                            volume=float(row.get("volume", 0)),
                            bar_size=self._bar_size,
                        )

                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error parsing row in {csv_path}: {e}")
                        continue

        except IOError as e:
            logger.error(f"Error reading {csv_path}: {e}")

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """
        Stream bars in timestamp order using heap-based k-way merge.

        Memory usage: O(num_symbols) instead of O(total_bars).
        Each symbol's bars are read lazily from its generator.

        The heap contains tuples of (timestamp, symbol_index, bar, generator)
        to ensure correct ordering even when timestamps are equal.
        """
        if not self._loaded:
            await self.load()

        # Create generators for each symbol
        generators: List[Generator[HistoricalBar, None, None]] = [
            self._create_reader(symbol) for symbol in self._symbols
        ]

        # Initialize min-heap with first bar from each generator
        # Heap items: (timestamp, index, bar, generator)
        # Index is used as tiebreaker when timestamps are equal
        heap: List[Tuple[datetime, int, HistoricalBar, Generator]] = []

        for i, gen in enumerate(generators):
            try:
                bar = next(gen)
                heapq.heappush(heap, (bar.timestamp, i, bar, gen))
            except StopIteration:
                # Empty generator (no bars for this symbol in date range)
                continue

        # Stream bars in sorted order
        bar_count = 0
        while heap:
            timestamp, idx, bar, gen = heapq.heappop(heap)
            yield bar.to_bar_data(source="csv")
            bar_count += 1

            # Push next bar from same generator
            try:
                next_bar = next(gen)
                heapq.heappush(heap, (next_bar.timestamp, idx, next_bar, gen))
            except StopIteration:
                # This symbol's stream is exhausted
                continue

        self._bar_count = bar_count
        logger.info(f"StreamingCsvDataFeed streamed {bar_count} bars")

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """
        Get total bar count.

        Returns -1 if streaming hasn't completed yet (count unknown).
        """
        return self._bar_count


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
            f"ParquetDataFeed loaded {len(self._bars)} bars " f"for {len(self._symbols)} symbols"
        )

    def _load_parquet_file(self, path: Path, symbol: str, pd: ModuleType) -> None:
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


class HistoricalStoreDataFeed(DataFeed):
    """
    Load historical data from ParquetHistoricalStore (coverage-managed).

    Uses the coverage-managed directory layout:
        {base_dir}/{symbol}/{timeframe}.parquet

    This differs from ParquetDataFeed which uses flat file layout:
        {parquet_dir}/{symbol}.parquet

    Supports multi-timeframe loading for MTF strategies. When secondary_timeframes
    is provided, loads data from all timeframes and sorts for aligned delivery.

    Example:
        # Single timeframe
        feed = HistoricalStoreDataFeed(
            base_dir="data/historical",
            symbols=["AAPL", "MSFT"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            bar_size="1d",
        )

        # Multi-timeframe (for MTF strategies)
        feed = HistoricalStoreDataFeed(
            base_dir="data/historical",
            symbols=["AAPL"],
            bar_size="1d",  # Primary timeframe
            secondary_timeframes=["1h"],  # Additional timeframes
        )
        await feed.load()

        async for bar in feed.stream_bars():
            print(bar)  # Bars from all timeframes, sorted for alignment
    """

    def __init__(
        self,
        base_dir: str,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        bar_size: str = "1d",
        secondary_timeframes: Optional[List[str]] = None,
    ):
        """
        Initialize historical store data feed.

        Args:
            base_dir: Base directory for ParquetHistoricalStore files.
            symbols: List of symbols to load.
            start_date: Start date filter.
            end_date: End date filter.
            bar_size: Bar size/timeframe string (e.g., "1d", "1h", "5m").
            secondary_timeframes: Additional timeframes for MTF strategies.
        """
        self._base_dir = Path(base_dir)
        self._symbols = symbols
        self._start_date = start_date
        self._end_date = end_date
        self._bar_size = bar_size
        # Deduplicate and filter out primary timeframe from secondary list
        self._secondary_timeframes = [
            tf for tf in (secondary_timeframes or []) if tf and tf != bar_size
        ]
        self._secondary_timeframes = list(dict.fromkeys(self._secondary_timeframes))

        self._bars: List[BarData] = []
        self._loaded = False
        self._store: Optional[Any] = None

    def _get_store(self) -> Any:
        """Lazy initialization of ParquetHistoricalStore."""
        if self._store is None:
            from ...infrastructure.stores.parquet_historical_store import ParquetHistoricalStore

            self._store = ParquetHistoricalStore(base_dir=self._base_dir)
        return self._store

    async def load(self) -> None:
        """Load data from ParquetHistoricalStore for all configured timeframes."""
        self._bars.clear()

        start_dt = (
            datetime.combine(self._start_date, datetime.min.time()) if self._start_date else None
        )
        end_dt = datetime.combine(self._end_date, datetime.max.time()) if self._end_date else None

        store = self._get_store()
        timeframes = [self._bar_size] + self._secondary_timeframes

        for symbol in self._symbols:
            for timeframe in timeframes:
                file_path = store.get_file_path(symbol, timeframe)
                if not file_path.exists():
                    level = logging.WARNING if timeframe == self._bar_size else logging.DEBUG
                    logger.log(
                        level,
                        f"Historical store file not found: {file_path}. "
                        f"Run with --coverage-mode download to fetch missing data.",
                    )
                    continue

                bars = store.read_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_dt,
                    end=end_dt,
                )
                self._bars.extend(bars)

        # Sort for MTF alignment: secondary timeframes before primary at same timestamp
        if self._secondary_timeframes:
            self._bars.sort(key=lambda b: AlignedBarBuffer.sort_key(b, self._bar_size))
        else:
            # Single timeframe: simple timestamp + symbol sort
            self._bars.sort(key=lambda b: (b.timestamp or datetime.min, b.symbol))

        self._loaded = True

        timeframe_str = (
            f"[{', '.join(timeframes)}]" if self._secondary_timeframes else self._bar_size
        )
        logger.info(
            f"HistoricalStoreDataFeed loaded {len(self._bars)} bars "
            f"for {len(self._symbols)} symbols ({timeframe_str}) from {self._base_dir}"
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
    def secondary_timeframes(self) -> List[str]:
        """Get configured secondary timeframes for MTF strategies."""
        return list(self._secondary_timeframes)

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return len(self._bars)


class StreamingParquetDataFeed(DataFeed):
    """
    OPT-009: Streaming Parquet data feed using chunked reading.

    Uses PyArrow's batch iterator to read Parquet files in chunks,
    then applies heap-based merge for multi-symbol timestamp ordering.

    Key benefits:
    - Chunk-based reading: Only chunk_size rows in memory per symbol at a time
    - Columnar efficiency: Parquet's columnar format is read efficiently
    - Memory bounded: Total memory ~= chunk_size x num_symbols x row_size

    Example:
        feed = StreamingParquetDataFeed(
            parquet_dir="data/historical",
            symbols=["AAPL", "MSFT"],
            chunk_size=10000,
        )
        await feed.load()

        async for bar in feed.stream_bars():
            print(bar)
    """

    def __init__(
        self,
        parquet_dir: str,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        bar_size: str = "1d",
        chunk_size: int = 10000,
    ):
        """
        Initialize streaming Parquet data feed.

        Args:
            parquet_dir: Directory containing Parquet files.
            symbols: List of symbols to load.
            start_date: Start date filter.
            end_date: End date filter.
            bar_size: Bar size string.
            chunk_size: Number of rows to read per chunk (default 10000).
        """
        self._parquet_dir = Path(parquet_dir)
        self._symbols = symbols
        self._start_date = start_date
        self._end_date = end_date
        self._bar_size = bar_size
        self._chunk_size = chunk_size
        self._bar_count = -1  # Unknown until streaming completes
        self._loaded = False

    async def load(self) -> None:
        """
        Initialize readers (verify files exist, no data loaded).
        """
        # Check for pyarrow
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            raise ImportError("pyarrow required for Parquet streaming: pip install pyarrow")

        # Verify files exist
        missing = []
        for symbol in self._symbols:
            parquet_path = self._parquet_dir / f"{symbol}.parquet"
            if not parquet_path.exists():
                missing.append(symbol)

        if missing:
            logger.warning(f"Parquet files not found for: {missing}")

        self._loaded = True
        logger.info(
            f"StreamingParquetDataFeed initialized for {len(self._symbols)} symbols "
            f"(chunk_size={self._chunk_size})"
        )

    def _create_reader(self, symbol: str) -> Generator[HistoricalBar, None, None]:
        """
        Create a chunked streaming reader for a Parquet file.

        Uses PyArrow's iter_batches for memory-efficient reading.
        """
        import pyarrow.parquet as pq

        parquet_path = self._parquet_dir / f"{symbol}.parquet"
        if not parquet_path.exists():
            return

        try:
            parquet_file = pq.ParquetFile(parquet_path)

            for batch in parquet_file.iter_batches(batch_size=self._chunk_size):
                # Convert batch to pandas for easier row iteration
                df = batch.to_pandas()

                # Determine timestamp column
                if "timestamp" in df.columns:
                    ts_col = "timestamp"
                elif "date" in df.columns:
                    ts_col = "date"
                else:
                    ts_col = df.columns[0]  # Assume first column

                for idx, row in df.iterrows():
                    try:
                        # Parse timestamp
                        ts_val = row[ts_col]
                        if hasattr(ts_val, "to_pydatetime"):
                            timestamp = ts_val.to_pydatetime()
                        elif isinstance(ts_val, str):
                            timestamp = datetime.fromisoformat(ts_val)
                        else:
                            timestamp = ts_val

                        # Apply date filter
                        if self._start_date and timestamp.date() < self._start_date:
                            continue
                        if self._end_date and timestamp.date() > self._end_date:
                            continue

                        yield HistoricalBar(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=float(row.get("open", 0)),
                            high=float(row.get("high", 0)),
                            low=float(row.get("low", 0)),
                            close=float(row.get("close", 0)),
                            volume=float(row.get("volume", 0)),
                            bar_size=self._bar_size,
                        )

                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Error parsing row in {parquet_path}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error reading {parquet_path}: {e}")

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """
        Stream bars in timestamp order using heap-based merge.

        Same algorithm as StreamingCsvDataFeed but with Parquet sources.
        """
        if not self._loaded:
            await self.load()

        # Create generators for each symbol
        generators: List[Generator[HistoricalBar, None, None]] = [
            self._create_reader(symbol) for symbol in self._symbols
        ]

        # Initialize min-heap
        heap: List[Tuple[datetime, int, HistoricalBar, Generator]] = []

        for i, gen in enumerate(generators):
            try:
                bar = next(gen)
                heapq.heappush(heap, (bar.timestamp, i, bar, gen))
            except StopIteration:
                continue

        # Stream in sorted order
        bar_count = 0
        while heap:
            timestamp, idx, bar, gen = heapq.heappop(heap)
            yield bar.to_bar_data(source="parquet")
            bar_count += 1

            try:
                next_bar = next(gen)
                heapq.heappush(heap, (next_bar.timestamp, idx, next_bar, gen))
            except StopIteration:
                continue

        self._bar_count = bar_count
        logger.info(f"StreamingParquetDataFeed streamed {bar_count} bars")

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """Get total bar count (-1 if not yet streamed)."""
        return self._bar_count


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


class BarCacheDataFeed(DataFeed):
    """
    Load historical data from IB for backtesting.

    For standalone backtests, this wraps IbHistoricalDataFeed.
    For in-process backtests (Lab panel), use CachedBarDataFeed instead.

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
        self._adapter: Optional[Any] = None

    async def load(self) -> None:
        """
        Load historical data from IB.

        Connects to IB, fetches data for all symbols, then disconnects.
        """
        from ...infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

        self._bars.clear()

        # Create and connect adapter
        adapter = IbHistoricalAdapter(
            host=self._host,
            port=self._port,
            client_id=self._client_id,
        )
        self._adapter = adapter

        try:
            await adapter.connect()
            logger.info(f"Connected to IB at {self._host}:{self._port}")

            # Convert dates to datetime
            start_dt = datetime.combine(self._start_date, datetime.min.time())
            end_dt = datetime.combine(self._end_date, datetime.max.time())

            # Fetch data for each symbol
            for symbol in self._symbols:
                logger.info(f"Fetching {self._bar_size} bars for {symbol}...")

                try:
                    bars = await adapter.fetch_bars(
                        symbol=symbol,
                        timeframe=self._bar_size,
                        start=start_dt,
                        end=end_dt,
                    )

                    for bar in bars:
                        hist_bar = HistoricalBar(
                            symbol=symbol,
                            timestamp=bar.timestamp,
                            open=bar.open or 0.0,
                            high=bar.high or 0.0,
                            low=bar.low or 0.0,
                            close=bar.close or 0.0,
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
            await adapter.disconnect()
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
            end_date=date(2024, 6, 30)
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


def create_data_feed(
    source: str,
    symbols: List[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    streaming: bool = True,
    bar_size: str = "1d",
    **kwargs: Any,
) -> DataFeed:
    """
    OPT-009: Factory for creating data feeds with streaming support.

    Automatically selects between streaming and full-load implementations
    based on the source type and streaming parameter.

    Streaming feeds (when streaming=True):
    - Memory efficient: O(num_symbols) instead of O(total_bars)
    - Faster startup: No upfront loading, bars yielded on demand
    - bar_count returns -1 until streaming completes

    Full-load feeds (when streaming=False):
    - All bars loaded into memory at once
    - Faster random access (if needed)
    - bar_count known immediately after load()

    Args:
        source: Data source path. Can be:
            - Directory path for CSV files
            - Directory path for Parquet files (if contains .parquet files)
            - Single .csv file
            - Single .parquet file
        symbols: List of symbols to load.
        start_date: Start date filter (inclusive).
        end_date: End date filter (inclusive).
        streaming: Use streaming implementation (default True).
        bar_size: Bar size string (e.g., "1d", "1h", "1m").
        **kwargs: Additional arguments passed to feed constructor.

    Returns:
        DataFeed instance.

    Examples:
        # Streaming CSV (memory efficient)
        feed = create_data_feed(
            "data/historical",
            symbols=["AAPL", "MSFT"],
            streaming=True,
        )

        # Full-load CSV (for small datasets)
        feed = create_data_feed(
            "data/historical",
            symbols=["AAPL"],
            streaming=False,
        )

        # Streaming Parquet with custom chunk size
        feed = create_data_feed(
            "data/parquet",
            symbols=["AAPL", "MSFT"],
            streaming=True,
            chunk_size=50000,
        )
    """
    source_path = Path(source)

    # Detect source type
    is_parquet = False
    if source_path.is_file():
        is_parquet = source_path.suffix == ".parquet"
    elif source_path.is_dir():
        # Check if directory contains parquet files
        parquet_files = list(source_path.glob("*.parquet"))
        csv_files = list(source_path.glob("*.csv"))
        is_parquet = len(parquet_files) > len(csv_files)

    # Select feed implementation
    if is_parquet:
        if streaming:
            return StreamingParquetDataFeed(
                parquet_dir=str(source_path if source_path.is_dir() else source_path.parent),
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                bar_size=bar_size,
                chunk_size=kwargs.get("chunk_size", 10000),
            )
        else:
            return ParquetDataFeed(
                parquet_dir=str(source_path if source_path.is_dir() else source_path.parent),
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                bar_size=bar_size,
            )
    else:
        # CSV
        if streaming:
            return StreamingCsvDataFeed(
                csv_dir=str(source_path if source_path.is_dir() else source_path.parent),
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                bar_size=bar_size,
                date_column=kwargs.get("date_column", "date"),
                date_format=kwargs.get("date_format", "%Y-%m-%d"),
            )
        else:
            return CsvDataFeed(
                csv_dir=str(source_path if source_path.is_dir() else source_path.parent),
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                bar_size=bar_size,
                date_column=kwargs.get("date_column", "date"),
                date_format=kwargs.get("date_format", "%Y-%m-%d"),
            )
