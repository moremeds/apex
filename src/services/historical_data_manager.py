"""
Historical Data Manager - Orchestrates historical data downloads and queries.

This service coordinates between:
- Coverage metadata (DuckDB) - knows what data we have
- Bar storage (Parquet) - stores the actual bars
- Data sources (Yahoo, IB) - fetches data from external sources
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..utils.logging_setup import get_logger
from ..domain.interfaces.historical_source import DateRange
from ..domain.events.domain_events import BarData
from ..infrastructure.stores.duckdb_coverage_store import DuckDBCoverageStore
from ..infrastructure.stores.parquet_historical_store import ParquetHistoricalStore
from ..infrastructure.adapters.yahoo.historical_adapter import YahooHistoricalAdapter

logger = get_logger(__name__)


@dataclass
class DownloadResult:
    """Result of a download operation."""

    symbol: str
    timeframe: str
    source: str
    bars_downloaded: int
    start: datetime
    end: datetime
    success: bool
    error: Optional[str] = None


class HistoricalDataManager:
    """
    Orchestrates historical data downloads and queries.

    Responsibilities:
    - Check coverage to find missing data ranges
    - Download missing data from appropriate source
    - Store data in Parquet format
    - Update coverage metadata
    - Provide unified query interface

    Source Priority: IB > Yahoo (IB data replaces Yahoo when available)
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        source_priority: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize historical data manager.

        Args:
            base_dir: Base directory for data storage.
            source_priority: List of source names in priority order.
        """
        self._base_dir = base_dir or Path("data/historical")
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._source_priority = source_priority or ["ib", "yahoo"]

        # Initialize stores
        self._coverage_store = DuckDBCoverageStore(
            db_path=self._base_dir / "_metadata.duckdb"
        )
        self._bar_store = ParquetHistoricalStore(base_dir=self._base_dir)

        # Initialize sources
        self._sources: Dict[str, Any] = {}
        self._init_sources()

    def _init_sources(self) -> None:
        """Initialize data sources."""
        # Yahoo is always available (no connection needed)
        try:
            self._sources["yahoo"] = YahooHistoricalAdapter()
            logger.info("Yahoo historical source initialized")
        except Exception as e:
            logger.warning(f"Failed to init Yahoo source: {e}")

        # IB source is optional and requires connection
        # It will be set externally via set_ib_source()

    def set_ib_source(self, ib_adapter: Any) -> None:
        """
        Set IB historical adapter.

        Args:
            ib_adapter: IbHistoricalAdapter instance.
        """
        self._sources["ib"] = ib_adapter
        logger.info("IB historical source registered")

    async def ensure_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        source_priority: Optional[List[str]] = None,
    ) -> List[BarData]:
        """
        Ensure data exists for the requested range, downloading if needed.

        This is the main entry point for backtests - it guarantees data
        availability by downloading missing ranges.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe ('5min', '1h', '1d').
            start: Start datetime (naive or aware).
            end: End datetime (naive or aware).
            source_priority: Optional override for source priority.

        Returns:
            List of BarData for the requested range.
        """
        priority = source_priority or self._source_priority

        # Normalize datetimes to naive for coverage store (stores naive UTC)
        # Coverage store internally uses naive datetimes for comparison
        start_naive = start.replace(tzinfo=None) if start.tzinfo else start
        end_naive = end.replace(tzinfo=None) if end.tzinfo else end

        # Find missing ranges
        missing = self._coverage_store.find_gaps(symbol, timeframe, start_naive, end_naive)

        if missing:
            logger.info(
                f"Found {len(missing)} gaps in {symbol}/{timeframe} coverage, "
                f"downloading..."
            )

            for gap in missing:
                await self._download_range(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=gap.start,
                    end=gap.end,
                    priority=priority,
                )

        # Return data from storage (use naive datetimes for Parquet filtering)
        return self.get_bars(symbol, timeframe, start_naive, end_naive)

    async def _download_range(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        priority: List[str],
    ) -> DownloadResult:
        """
        Download a specific range from available sources.

        Tries sources in priority order until one succeeds.
        """
        for source_name in priority:
            source = self._sources.get(source_name)
            if not source:
                continue

            if not source.supports_timeframe(timeframe):
                logger.debug(f"{source_name} doesn't support {timeframe}")
                continue

            try:
                logger.info(
                    f"Downloading {symbol}/{timeframe} from {source_name}: "
                    f"{start.date()} to {end.date()}"
                )

                bars = await source.fetch_bars(symbol, timeframe, start, end)

                if bars:
                    # Write to storage
                    self._bar_store.write_bars(symbol, timeframe, bars, mode="upsert")

                    # Update coverage
                    self._coverage_store.update_coverage(
                        symbol=symbol,
                        timeframe=timeframe,
                        source=source_name,
                        start=start,
                        end=end,
                        bar_count=len(bars),
                    )

                    return DownloadResult(
                        symbol=symbol,
                        timeframe=timeframe,
                        source=source_name,
                        bars_downloaded=len(bars),
                        start=start,
                        end=end,
                        success=True,
                    )
                else:
                    logger.warning(f"No data returned from {source_name} for {symbol}")

            except Exception as e:
                logger.error(f"Failed to download from {source_name}: {e}")
                continue

        # All sources failed
        self._coverage_store.record_gap(symbol, timeframe, start, end)

        return DownloadResult(
            symbol=symbol,
            timeframe=timeframe,
            source="none",
            bars_downloaded=0,
            start=start,
            end=end,
            success=False,
            error="All sources failed",
        )

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[BarData]:
        """
        Query bars from storage (no download).

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Optional start filter.
            end: Optional end filter.

        Returns:
            List of BarData sorted by timestamp.
        """
        return self._bar_store.read_bars(symbol, timeframe, start, end)

    def get_coverage(
        self,
        symbol: str,
        timeframe: str,
    ) -> List[DateRange]:
        """Get available data ranges for symbol/timeframe."""
        return self._coverage_store.get_coverage(symbol, timeframe)

    def has_complete_coverage(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        """Check if we have complete coverage for a range."""
        return self._coverage_store.has_complete_coverage(
            symbol, timeframe, start, end
        )

    def find_missing_ranges(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[DateRange]:
        """Find gaps in coverage for the requested range."""
        return self._coverage_store.find_gaps(symbol, timeframe, start, end)

    async def download_symbols(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        source: Optional[str] = None,
    ) -> List[DownloadResult]:
        """
        Download data for multiple symbols.

        Args:
            symbols: List of ticker symbols.
            timeframe: Bar timeframe.
            start: Start datetime.
            end: End datetime.
            source: Specific source to use (or use priority).

        Returns:
            List of DownloadResult for each symbol.
        """
        results = []
        priority = [source] if source else self._source_priority

        for symbol in symbols:
            # When a specific source is requested, only check coverage for that source
            # This allows re-downloading from IB even if Yahoo data exists
            missing = self._coverage_store.find_gaps(
                symbol, timeframe, start, end, source=source
            )

            if not missing:
                results.append(DownloadResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    source="cached",
                    bars_downloaded=0,
                    start=start,
                    end=end,
                    success=True,
                ))
                continue

            # Download each missing range
            total_bars = 0
            success = True
            last_source = ""

            for gap in missing:
                result = await self._download_range(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=gap.start,
                    end=gap.end,
                    priority=priority,
                )
                total_bars += result.bars_downloaded
                success = success and result.success
                last_source = result.source

            results.append(DownloadResult(
                symbol=symbol,
                timeframe=timeframe,
                source=last_source,
                bars_downloaded=total_bars,
                start=start,
                end=end,
                success=success,
            ))

        return results

    def get_coverage_summary(self, symbol: str) -> List[dict]:
        """Get coverage summary for a symbol across all timeframes."""
        return self._coverage_store.get_coverage_summary(symbol)

    def list_symbols(self) -> List[str]:
        """List all symbols with stored data."""
        return self._bar_store.list_symbols()

    def list_timeframes(self, symbol: str) -> List[str]:
        """List available timeframes for a symbol."""
        return self._bar_store.list_timeframes(symbol)

    def get_date_range(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[tuple[datetime, datetime]]:
        """Get earliest and latest dates for symbol/timeframe."""
        return self._bar_store.get_date_range(symbol, timeframe)

    def get_bar_count(self, symbol: str, timeframe: str) -> int:
        """Get total bar count for symbol/timeframe."""
        return self._bar_store.get_bar_count(symbol, timeframe)

    def delete_data(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
    ) -> bool:
        """
        Delete stored data and coverage.

        Args:
            symbol: Symbol to delete.
            timeframe: Optional specific timeframe.

        Returns:
            True if any data was deleted.
        """
        bar_deleted = self._bar_store.delete_data(symbol, timeframe)
        coverage_deleted = self._coverage_store.delete_coverage(symbol, timeframe) > 0
        return bar_deleted or coverage_deleted

    async def fill_gaps(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 100,
    ) -> List[DownloadResult]:
        """
        Fill known gaps in data coverage.

        Args:
            symbol: Optional specific symbol.
            timeframe: Optional specific timeframe.
            limit: Maximum gaps to fill.

        Returns:
            List of download results.
        """
        gaps = self._coverage_store.get_pending_gaps(symbol, timeframe, limit)
        results = []

        for gap_symbol, gap_timeframe, gap_start, gap_end in gaps:
            result = await self._download_range(
                symbol=gap_symbol,
                timeframe=gap_timeframe,
                start=gap_start,
                end=gap_end,
                priority=self._source_priority,
            )
            results.append(result)

            if result.success:
                self._coverage_store.mark_gap_filled(
                    gap_symbol, gap_timeframe, gap_start, gap_end
                )

        return results

    def close(self) -> None:
        """Close resources."""
        self._coverage_store.close()

    def __enter__(self) -> HistoricalDataManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
