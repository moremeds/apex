"""
DuckDB-based coverage tracking for historical market data.

Tracks what data ranges we have for each symbol/timeframe combination,
enabling incremental downloads and gap detection.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import duckdb

from ...utils.logging_setup import get_logger
from ...domain.interfaces.historical_source import DateRange

logger = get_logger(__name__)


@dataclass
class CoverageRecord:
    """A single coverage record from the database."""

    symbol: str
    timeframe: str
    source: str
    start_ts: datetime
    end_ts: datetime
    bar_count: int
    quality: str
    last_updated: datetime


class DuckDBCoverageStore:
    """
    Tracks historical data coverage in DuckDB.

    Features:
    - Track what date ranges we have for each symbol/timeframe
    - Automatically merge overlapping ranges on insert
    - Find gaps in coverage for incremental downloads
    - Track data quality and source provenance
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        in_memory: bool = False,
    ) -> None:
        """
        Initialize coverage store.

        Args:
            db_path: Path to DuckDB file. Defaults to data/historical/_metadata.duckdb.
            in_memory: Use in-memory database (for testing).
        """
        if in_memory:
            self._conn = duckdb.connect(":memory:")
        else:
            self._db_path = db_path or Path("data/historical/_metadata.duckdb")
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self._db_path))

        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS data_coverage (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                source TEXT NOT NULL,
                start_ts TIMESTAMP NOT NULL,
                end_ts TIMESTAMP NOT NULL,
                bar_count INTEGER DEFAULT 0,
                quality TEXT DEFAULT 'complete',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timeframe, source, start_ts)
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS data_gaps (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                gap_start TIMESTAMP NOT NULL,
                gap_end TIMESTAMP NOT NULL,
                retry_count INTEGER DEFAULT 0,
                last_retry TIMESTAMP,
                status TEXT DEFAULT 'pending',
                PRIMARY KEY (symbol, timeframe, gap_start, gap_end)
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol_metadata (
                symbol TEXT PRIMARY KEY,
                yahoo_symbol TEXT,
                ib_conid INTEGER,
                exchange TEXT,
                asset_type TEXT DEFAULT 'stock',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for faster queries
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_coverage_symbol_tf
            ON data_coverage(symbol, timeframe)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_gaps_symbol_tf
            ON data_gaps(symbol, timeframe, status)
        """)

    def get_coverage(
        self,
        symbol: str,
        timeframe: str,
        source: Optional[str] = None,
    ) -> List[DateRange]:
        """
        Get all covered date ranges for a symbol/timeframe.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            source: Optional source filter.

        Returns:
            List of DateRange objects, sorted by start time.
        """
        if source:
            result = self._conn.execute("""
                SELECT start_ts, end_ts FROM data_coverage
                WHERE symbol = ? AND timeframe = ? AND source = ?
                ORDER BY start_ts
            """, [symbol, timeframe, source]).fetchall()
        else:
            result = self._conn.execute("""
                SELECT start_ts, end_ts FROM data_coverage
                WHERE symbol = ? AND timeframe = ?
                ORDER BY start_ts
            """, [symbol, timeframe]).fetchall()

        return [DateRange(start=row[0], end=row[1]) for row in result]

    def update_coverage(
        self,
        symbol: str,
        timeframe: str,
        source: str,
        start: datetime,
        end: datetime,
        bar_count: int,
        quality: str = "complete",
    ) -> None:
        """
        Update coverage after successful download.

        Automatically merges overlapping or adjacent ranges.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            source: Data source ('yahoo', 'ib').
            start: Range start.
            end: Range end.
            bar_count: Number of bars in this range.
            quality: Data quality ('complete', 'partial', 'gaps').
        """
        # Truncate microseconds for consistent comparison across requests
        start = start.replace(microsecond=0) if hasattr(start, 'replace') else start
        end = end.replace(microsecond=0) if hasattr(end, 'replace') else end

        new_range = DateRange(start=start, end=end)

        # Get existing ranges for this symbol/timeframe/source
        existing = self._conn.execute("""
            SELECT start_ts, end_ts, bar_count FROM data_coverage
            WHERE symbol = ? AND timeframe = ? AND source = ?
            ORDER BY start_ts
        """, [symbol, timeframe, source]).fetchall()

        if not existing:
            # No existing ranges, just insert
            self._conn.execute("""
                INSERT INTO data_coverage
                (symbol, timeframe, source, start_ts, end_ts, bar_count, quality, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [symbol, timeframe, source, start, end, bar_count, quality])
            return

        # Find ranges to merge
        merged_range = new_range
        merged_bar_count = bar_count
        ranges_to_delete = []

        for row in existing:
            existing_range = DateRange(start=row[0], end=row[1])
            existing_count = row[2] or 0

            # Check if overlapping or adjacent (within 1 day for daily bars)
            if self._should_merge(merged_range, existing_range, timeframe):
                # Only add existing bar_count if ranges are strictly adjacent (no overlap)
                # Overlapping ranges mean upsert will de-duplicate in Parquet,
                # so we shouldn't sum bar_counts (would inflate the count)
                if not new_range.overlaps(existing_range):
                    merged_bar_count += existing_count
                # If overlapping, the new bar_count already reflects the fresh download
                merged_range = merged_range.merge(existing_range)
                ranges_to_delete.append(existing_range.start)

        # Delete old ranges that were merged
        for old_start in ranges_to_delete:
            self._conn.execute("""
                DELETE FROM data_coverage
                WHERE symbol = ? AND timeframe = ? AND source = ? AND start_ts = ?
            """, [symbol, timeframe, source, old_start])

        # Insert merged range
        self._conn.execute("""
            INSERT OR REPLACE INTO data_coverage
            (symbol, timeframe, source, start_ts, end_ts, bar_count, quality, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [symbol, timeframe, source, merged_range.start, merged_range.end,
              merged_bar_count, quality])

        logger.debug(
            f"Updated coverage: {symbol}/{timeframe} from {source}: "
            f"{merged_range.start} to {merged_range.end} ({merged_bar_count} bars)"
        )

    def _should_merge(
        self,
        range1: DateRange,
        range2: DateRange,
        timeframe: str,
    ) -> bool:
        """Check if two ranges should be merged (overlapping or adjacent)."""
        if range1.overlaps(range2):
            return True

        # Check if adjacent based on timeframe
        gap_seconds = abs((range1.end - range2.start).total_seconds())

        # Allow small gaps based on timeframe
        max_gap_map = {
            "5min": 300,      # 5 minutes
            "1h": 3600,       # 1 hour
            "1d": 86400 * 3,  # 3 days (weekend gap)
        }
        max_gap = max_gap_map.get(timeframe, 86400)

        return gap_seconds <= max_gap

    def find_gaps(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        source: Optional[str] = None,
    ) -> List[DateRange]:
        """
        Find missing date ranges within the requested period.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Requested range start.
            end: Requested range end.
            source: Optional source filter (if set, only considers coverage from this source).

        Returns:
            List of DateRange representing gaps that need to be filled.
        """
        requested = DateRange(start=start, end=end)
        coverage = self.get_coverage(symbol, timeframe, source=source)

        if not coverage:
            # No data at all, entire range is a gap
            return [requested]

        gaps = [requested]

        for covered in coverage:
            new_gaps = []
            for gap in gaps:
                # Subtract covered range from each gap
                remaining = gap.subtract(covered)
                new_gaps.extend(remaining)
            gaps = new_gaps

        return sorted(gaps, key=lambda r: r.start)

    def has_complete_coverage(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        """
        Check if we have complete coverage for the requested range.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Range start.
            end: Range end.

        Returns:
            True if no gaps exist in the requested range.
        """
        gaps = self.find_gaps(symbol, timeframe, start, end)
        return len(gaps) == 0

    def record_gap(
        self,
        symbol: str,
        timeframe: str,
        gap_start: datetime,
        gap_end: datetime,
    ) -> None:
        """Record a known gap for later retry."""
        # Check if gap already exists
        existing = self._conn.execute("""
            SELECT retry_count FROM data_gaps
            WHERE symbol = ? AND timeframe = ? AND gap_start = ? AND gap_end = ?
        """, [symbol, timeframe, gap_start, gap_end]).fetchone()

        if existing:
            # Update retry count
            self._conn.execute("""
                UPDATE data_gaps
                SET retry_count = retry_count + 1, last_retry = CURRENT_TIMESTAMP
                WHERE symbol = ? AND timeframe = ? AND gap_start = ? AND gap_end = ?
            """, [symbol, timeframe, gap_start, gap_end])
        else:
            # Insert new gap
            self._conn.execute("""
                INSERT INTO data_gaps (symbol, timeframe, gap_start, gap_end, status)
                VALUES (?, ?, ?, ?, 'pending')
            """, [symbol, timeframe, gap_start, gap_end])

    def get_pending_gaps(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 100,
    ) -> List[tuple]:
        """Get gaps that need to be filled."""
        query = "SELECT symbol, timeframe, gap_start, gap_end FROM data_gaps WHERE status = 'pending'"
        params: list = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)

        query += " ORDER BY retry_count ASC, gap_start ASC LIMIT ?"
        params.append(limit)

        return self._conn.execute(query, params).fetchall()

    def mark_gap_filled(
        self,
        symbol: str,
        timeframe: str,
        gap_start: datetime,
        gap_end: datetime,
    ) -> None:
        """Mark a gap as filled."""
        self._conn.execute("""
            UPDATE data_gaps SET status = 'filled'
            WHERE symbol = ? AND timeframe = ? AND gap_start = ? AND gap_end = ?
        """, [symbol, timeframe, gap_start, gap_end])

    def get_all_symbols(self, timeframe: Optional[str] = None) -> List[str]:
        """Get all symbols that have coverage data."""
        if timeframe:
            result = self._conn.execute("""
                SELECT DISTINCT symbol FROM data_coverage WHERE timeframe = ?
                ORDER BY symbol
            """, [timeframe]).fetchall()
        else:
            result = self._conn.execute("""
                SELECT DISTINCT symbol FROM data_coverage ORDER BY symbol
            """).fetchall()

        return [row[0] for row in result]

    def get_coverage_summary(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
    ) -> List[dict]:
        """Get coverage summary for a symbol."""
        if timeframe:
            result = self._conn.execute("""
                SELECT timeframe, source, MIN(start_ts), MAX(end_ts), SUM(bar_count)
                FROM data_coverage
                WHERE symbol = ? AND timeframe = ?
                GROUP BY timeframe, source
            """, [symbol, timeframe]).fetchall()
        else:
            result = self._conn.execute("""
                SELECT timeframe, source, MIN(start_ts), MAX(end_ts), SUM(bar_count)
                FROM data_coverage
                WHERE symbol = ?
                GROUP BY timeframe, source
                ORDER BY timeframe, source
            """, [symbol]).fetchall()

        return [
            {
                "timeframe": row[0],
                "source": row[1],
                "earliest": row[2],
                "latest": row[3],
                "total_bars": row[4],
            }
            for row in result
        ]

    def get_all_coverage(self) -> Dict[str, List[dict]]:
        """
        Get all coverage data grouped by symbol in a single query.
        Merges data from all sources (ib, yahoo, etc.) per timeframe.

        Returns:
            Dict mapping symbol -> list of coverage records.
            Each record: {timeframe, earliest, latest, total_bars}
        """
        # Merge all sources per symbol/timeframe
        result = self._conn.execute("""
            SELECT symbol, timeframe, MIN(start_ts), MAX(end_ts), SUM(bar_count)
            FROM data_coverage
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """).fetchall()

        grouped: Dict[str, List[dict]] = {}
        for row in result:
            symbol = row[0]
            if symbol not in grouped:
                grouped[symbol] = []
            grouped[symbol].append({
                "timeframe": row[1],
                "earliest": row[2],
                "latest": row[3],
                "total_bars": row[4],
            })
        return grouped

    def delete_coverage(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        source: Optional[str] = None,
    ) -> int:
        """Delete coverage records. Returns number of rows deleted."""
        query = "DELETE FROM data_coverage WHERE symbol = ?"
        params = [symbol]

        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if source:
            query += " AND source = ?"
            params.append(source)

        result = self._conn.execute(query, params)
        return result.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> DuckDBCoverageStore:
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        self.close()
