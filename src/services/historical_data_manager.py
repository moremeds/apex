"""
Historical Data Manager - Orchestrates historical data downloads and queries.

This service coordinates between:
- Coverage metadata (DuckDB) - knows what data we have
- Bar storage (Parquet) - stores the actual bars
- Data sources (Yahoo, IB) - fetches data from external sources
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from ..utils.logging_setup import get_logger

# US market hours in Eastern Time
US_EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_OPEN_HOUR = 9   # 9:30 AM ET
MARKET_CLOSE_HOUR = 16  # 4:00 PM ET


def _to_comparable_naive(dt: datetime) -> datetime:
    """
    Convert datetime to naive UTC with truncated microseconds for comparison.

    This ensures consistent cache hit detection by:
    1. Converting to UTC if timezone-aware
    2. Stripping timezone info
    3. Truncating microseconds to avoid spurious gap detection
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    return dt.replace(tzinfo=None, microsecond=0)

# Intraday timeframes that only have data during market hours
INTRADAY_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "4h"}

# Expected bars per regular trading day (9:30-16:00 ET = 6.5 hours)
EXPECTED_BARS_PER_DAY = {
    "1m": 390,   # 6.5 hours * 60 minutes
    "5m": 78,    # 6.5 hours * 12 (5-min bars per hour)
    "15m": 26,   # 6.5 hours / 0.25 hours
    "30m": 13,   # 6.5 hours / 0.5 hours
    "1h": 7,     # 7 bars: 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30
    "4h": 2,     # 2 bars: 9:30, 13:30
}

# Timeframes where Yahoo provides correctly market-aligned bars
# IB returns clock-aligned bars (10:00, 11:00...) with partial first bar
# Yahoo returns market-aligned bars (09:30, 10:30, 11:30...)
YAHOO_PREFERRED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h"}
from collections import defaultdict
import pandas as pd

from ..domain.interfaces.historical_source import DateRange
from ..domain.events.domain_events import BarData


def resample_bars_to_4h(bars_1h: List[BarData], symbol: str) -> List[BarData]:
    """
    Resample 1h bars to 4h bars aligned to market open (09:30).

    4h bars for US market:
    - Bar 1: 09:30-13:30 (aggregates 09:30, 10:30, 11:30, 12:30)
    - Bar 2: 13:30-16:00 (aggregates 13:30, 14:30, 15:30, partial)

    Args:
        bars_1h: List of 1h BarData
        symbol: Symbol name for output bars

    Returns:
        List of resampled 4h BarData
    """
    if not bars_1h:
        return []

    # Convert to DataFrame for easier aggregation
    data = []
    for bar in bars_1h:
        if bar.bar_start:
            data.append({
                "timestamp": bar.bar_start,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume or 0,
            })

    if not data:
        return []

    df = pd.DataFrame(data)
    df = df.set_index("timestamp").sort_index()

    # Convert to ET for proper market hour grouping
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert(US_EASTERN)
    else:
        df.index = df.index.tz_localize(UTC).tz_convert(US_EASTERN)

    # Create 4h groups based on market hours
    # Group 1: 09:30-13:30 (hours 9, 10, 11, 12)
    # Group 2: 13:30-16:00 (hours 13, 14, 15)
    def get_4h_group(ts):
        hour = ts.hour
        minute = ts.minute
        date = ts.date()
        if hour < 13 or (hour == 13 and minute < 30):
            return pd.Timestamp(date.year, date.month, date.day, 9, 30, tz=US_EASTERN)
        else:
            return pd.Timestamp(date.year, date.month, date.day, 13, 30, tz=US_EASTERN)

    df["group"] = df.index.map(get_4h_group)

    # Aggregate OHLCV
    resampled = df.groupby("group").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # Convert back to BarData
    result = []
    for ts, row in resampled.iterrows():
        # Convert back to UTC for storage
        ts_utc = ts.astimezone(UTC)
        bar = BarData(
            symbol=symbol,
            timeframe="4h",
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=int(row["volume"]),
            bar_start=ts_utc,
            timestamp=ts_utc,
            source="resampled",
        )
        result.append(bar)

    return sorted(result, key=lambda b: b.bar_start)
from ..infrastructure.stores.duckdb_coverage_store import DuckDBCoverageStore
from ..infrastructure.stores.parquet_historical_store import ParquetHistoricalStore
from ..infrastructure.adapters.yahoo.historical_adapter import YahooHistoricalAdapter

logger = get_logger(__name__)


def validate_intraday_bars(
    bars: List[BarData], timeframe: str, symbol: str
) -> List[str]:
    """
    Validate bar counts per trading day. Returns list of warnings.

    For each day, logs: date, bar count, first bar time, last bar time.
    Warns if count differs from expected or first bar isn't at 9:30 ET.
    """
    expected = EXPECTED_BARS_PER_DAY.get(timeframe)
    if not expected or not bars:
        return []

    warnings = []
    bars_by_date: Dict[Any, List[BarData]] = defaultdict(list)

    # Group bars by trading date in Eastern Time
    for bar in bars:
        if bar.bar_start:
            # Convert to ET for grouping by trading day
            if bar.bar_start.tzinfo:
                et_time = bar.bar_start.astimezone(US_EASTERN)
            else:
                # Assume naive timestamps are UTC
                et_time = bar.bar_start.replace(tzinfo=UTC).astimezone(US_EASTERN)
            bars_by_date[et_time.date()].append(bar)

    for date, day_bars in sorted(bars_by_date.items()):
        actual = len(day_bars)

        # Sort bars by time to get first/last
        sorted_bars = sorted(day_bars, key=lambda b: b.bar_start)
        first_bar = sorted_bars[0].bar_start
        last_bar = sorted_bars[-1].bar_start

        # Convert to ET for display
        if first_bar.tzinfo:
            first_et = first_bar.astimezone(US_EASTERN)
            last_et = last_bar.astimezone(US_EASTERN)
        else:
            first_et = first_bar.replace(tzinfo=UTC).astimezone(US_EASTERN)
            last_et = last_bar.replace(tzinfo=UTC).astimezone(US_EASTERN)

        # Log info for each day
        logger.debug(
            f"{symbol}/{timeframe} {date}: {actual} bars, "
            f"first={first_et.strftime('%H:%M')}, last={last_et.strftime('%H:%M')}"
        )

        # Warn if bar count differs (allow fewer for short days)
        if actual > expected:
            warnings.append(
                f"{date}: {actual} bars (expected max {expected}), "
                f"first={first_et.strftime('%H:%M')}, last={last_et.strftime('%H:%M')}"
            )

        # Warn if first bar isn't at 9:30 ET (or close to it for aggregated bars)
        expected_first_hour = 9
        expected_first_minute = 30
        if first_et.hour != expected_first_hour or first_et.minute != expected_first_minute:
            # Allow 4h bars to start at 9:30 (first bar covers 9:30-13:30)
            if not (timeframe == "4h" and first_et.hour == 9 and first_et.minute == 30):
                warnings.append(
                    f"{date}: first bar at {first_et.strftime('%H:%M')}, expected 09:30"
                )

    return warnings


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

        # For intraday timeframes, prefer Yahoo which provides market-aligned bars
        # IB returns clock-aligned bars (10:00, 11:00) with partial first bar at 09:30
        # Yahoo returns proper market-aligned bars (09:30, 10:30, 11:30...)
        if timeframe in YAHOO_PREFERRED_TIMEFRAMES and source_priority is None:
            priority = ["yahoo", "ib"]

        # Special handling for 4h: resample from 1h bars
        # Neither IB nor Yahoo provide properly aligned 4h bars
        if timeframe == "4h":
            return await self._ensure_4h_from_1h(symbol, start, end)

        # Normalize datetimes to naive UTC with truncated microseconds
        # This ensures consistent cache hit detection across requests
        start_cmp = _to_comparable_naive(start)
        end_cmp = _to_comparable_naive(end)

        # Find missing ranges
        missing = self._coverage_store.find_gaps(symbol, timeframe, start_cmp, end_cmp)

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

        # Return data from storage (use comparable timestamps for Parquet filtering)
        return self.get_bars(symbol, timeframe, start_cmp, end_cmp)

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

                # Rate limiting for IB to avoid pacing violations
                if source_name == "ib":
                    await asyncio.sleep(0.5)  # 500ms delay between IB requests

                if bars:
                    # Validate bars have timestamps before processing
                    bar_starts = [b.bar_start for b in bars if b.bar_start]
                    if not bar_starts:
                        logger.error(
                            f"Downloaded bars have no valid timestamps for "
                            f"{symbol}/{timeframe} from {source_name}"
                        )
                        continue  # Try next source

                    # Write to storage with error handling
                    try:
                        self._bar_store.write_bars(symbol, timeframe, bars, mode="upsert")
                    except Exception as e:
                        logger.error(
                            f"Failed to write bars to Parquet for {symbol}/{timeframe}: {e}"
                        )
                        raise  # Don't update coverage if write failed

                    # Get actual bar count from parquet after upsert (deduplication)
                    # This is more accurate than len(bars) when merging overlapping ranges
                    actual_bar_count = self._bar_store.get_bar_count(symbol, timeframe)

                    # Update coverage using ACTUAL data range from parquet
                    date_range = self._bar_store.get_date_range(symbol, timeframe)
                    if date_range:
                        actual_start, actual_end = date_range
                    else:
                        actual_start = min(bar_starts)
                        actual_end = max(bar_starts)

                    self._coverage_store.update_coverage(
                        symbol=symbol,
                        timeframe=timeframe,
                        source=source_name,
                        start=actual_start,
                        end=actual_end,
                        bar_count=actual_bar_count,
                    )

                    return DownloadResult(
                        symbol=symbol,
                        timeframe=timeframe,
                        source=source_name,
                        bars_downloaded=len(bars),
                        start=actual_start,
                        end=actual_end,
                        success=True,
                    )
                else:
                    # Provide more context for intraday timeframes
                    if timeframe in INTRADAY_TIMEFRAMES:
                        logger.info(
                            f"No {timeframe} bars returned from {source_name} for {symbol} "
                            f"({start} to {end}) - likely outside market hours"
                        )
                    else:
                        logger.warning(f"No data returned from {source_name} for {symbol}")

            except Exception as e:
                logger.error(f"Failed to download from {source_name}: {e}")
                continue

        # All sources failed - check if this is expected for intraday outside market hours
        if timeframe in INTRADAY_TIMEFRAMES:
            # For intraday, don't record gaps or log errors for out-of-hours requests
            # These gaps will naturally be filled when market data becomes available
            logger.debug(
                f"No {timeframe} data for {symbol} from {start} to {end} - "
                f"this is normal outside market hours"
            )
            return DownloadResult(
                symbol=symbol,
                timeframe=timeframe,
                source="none",
                bars_downloaded=0,
                start=start,
                end=end,
                success=True,  # Not a failure - just no data during off-hours
                error=None,
            )

        # For daily/weekly timeframes, record gap if we have existing coverage
        # (otherwise, data likely doesn't exist for this range, e.g., pre-IPO)
        existing_coverage = self._coverage_store.get_coverage(symbol, timeframe)
        if existing_coverage:
            # Only record gap if it's within known data range
            earliest = min(c.start for c in existing_coverage)
            latest = max(c.end for c in existing_coverage)
            if start >= earliest and end <= latest:
                self._coverage_store.record_gap(symbol, timeframe, start, end)
                logger.debug(f"Recorded gap for {symbol}/{timeframe}: {start} to {end}")
            else:
                logger.debug(
                    f"Skipping gap record for {symbol}/{timeframe} - "
                    f"outside known data range ({earliest} to {latest})"
                )
        else:
            logger.debug(
                f"Skipping gap record for {symbol}/{timeframe} - "
                f"no existing coverage (likely pre-IPO/inception)"
            )

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

    async def _ensure_4h_from_1h(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        """
        Ensure 4h data by resampling from 1h bars.

        Neither IB nor Yahoo provide properly market-aligned 4h bars.
        This method fetches 1h bars (which Yahoo aligns correctly) and
        resamples them to 4h bars aligned to market open (09:30, 13:30).

        Args:
            symbol: Ticker symbol.
            start: Start datetime.
            end: End datetime.

        Returns:
            List of 4h BarData.
        """
        # First ensure we have 1h data
        bars_1h = await self.ensure_data(symbol, "1h", start, end)

        if not bars_1h:
            logger.warning(f"No 1h bars available for 4h resampling: {symbol}")
            return []

        # Resample to 4h
        bars_4h = resample_bars_to_4h(bars_1h, symbol)

        if bars_4h:
            # Store resampled bars
            self._bar_store.write_bars(symbol, "4h", bars_4h, mode="upsert")

            # Update coverage
            bar_starts = [b.bar_start for b in bars_4h if b.bar_start]
            if bar_starts:
                actual_start = min(bar_starts)
                actual_end = max(bar_starts)
                self._coverage_store.update_coverage(
                    symbol=symbol,
                    timeframe="4h",
                    source="resampled",
                    start=actual_start,
                    end=actual_end,
                    bar_count=len(bars_4h),
                )

            logger.info(
                f"Resampled {len(bars_1h)} 1h bars to {len(bars_4h)} 4h bars for {symbol}"
            )

        return bars_4h

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        validate: bool = True,
    ) -> List[BarData]:
        """
        Query bars from storage (no download).

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Optional start filter.
            end: Optional end filter.
            validate: Whether to validate intraday bar counts (default: True).

        Returns:
            List of BarData sorted by timestamp.
        """
        bars = self._bar_store.read_bars(symbol, timeframe, start, end)

        # Validate intraday bar counts
        if validate and timeframe in INTRADAY_TIMEFRAMES and bars:
            warnings = validate_intraday_bars(bars, timeframe, symbol)
            if warnings:
                logger.warning(
                    f"Bar validation warnings for {symbol}/{timeframe}:",
                    extra={"warnings": warnings[:5]},  # Limit to first 5
                )

        return bars

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

    def get_max_history_days(self, timeframe: str) -> int:
        """
        Get maximum history available for a timeframe from preferred source.

        Returns the max from the highest priority source that supports this timeframe.
        Used for preload to request full available history on startup.
        """
        for source_name in self._source_priority:
            source = self._sources.get(source_name)
            if source and source.supports_timeframe(timeframe):
                if hasattr(source, "get_max_history_days"):
                    return source.get_max_history_days(timeframe)
        # Fallback defaults if no source has the method
        return {"1m": 7, "5m": 60, "15m": 60, "1h": 730, "1d": 3650}.get(timeframe, 365)

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
