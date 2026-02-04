"""
CSV-based data feeds for backtesting.

Contains:
- CsvDataFeed: Full-load CSV data feed
- StreamingCsvDataFeed: Memory-efficient streaming CSV feed (OPT-009)
"""

from __future__ import annotations

import csv
import heapq
import logging
from datetime import date, datetime
from pathlib import Path
from typing import AsyncIterator, Generator, List, Optional, Tuple

from ....domain.events.domain_events import BarData
from .base import DataFeed
from .models import HistoricalBar

logger = logging.getLogger(__name__)


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
