"""
Historical store data feed for backtesting.

Contains:
- HistoricalStoreDataFeed: Loads from ParquetHistoricalStore (coverage-managed)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from ....domain.events.domain_events import BarData
from .base import DataFeed
from .models import AlignedBarBuffer

logger = logging.getLogger(__name__)


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
            from ....infrastructure.stores.parquet_historical_store import ParquetHistoricalStore

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
