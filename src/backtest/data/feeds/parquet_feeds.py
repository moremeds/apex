"""
Parquet-based data feeds for backtesting.

Contains:
- ParquetDataFeed: Full-load Parquet data feed
- StreamingParquetDataFeed: Memory-efficient streaming Parquet feed (OPT-009)
"""

from __future__ import annotations

import heapq
import logging
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import AsyncIterator, Generator, List, Optional, Tuple

from ....domain.events.domain_events import BarData
from .base import DataFeed
from .models import HistoricalBar

logger = logging.getLogger(__name__)


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
