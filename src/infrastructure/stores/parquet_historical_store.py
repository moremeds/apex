"""
Parquet-based historical bar data storage.

Stores OHLCV bar data in Parquet format for efficient compression
and columnar reads. Each symbol/timeframe combination is stored
in a separate file.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import pyarrow as pa
import pyarrow.parquet as pq

from ...utils.logging_setup import get_logger
from ...domain.events.domain_events import BarData

logger = get_logger(__name__)


# PyArrow schema for bar data
BAR_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
    ("vwap", pa.float64()),
    ("trade_count", pa.int64()),
    ("source", pa.string()),
    ("ingested_at", pa.timestamp("us", tz="UTC")),
])


class ParquetHistoricalStore:
    """
    Stores and retrieves historical bars in Parquet format.

    Directory structure:
        {base_dir}/{symbol}/{timeframe}.parquet

    Features:
    - Efficient compression (83% vs CSV)
    - Columnar format for fast reads
    - Upsert mode (replace bars by timestamp)
    - Date range filtering on read
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize Parquet store.

        Args:
            base_dir: Base directory for Parquet files.
                     Defaults to data/historical.
        """
        self._base_dir = base_dir or Path("data/historical")
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def get_file_path(self, symbol: str, timeframe: str) -> Path:
        """
        Get Parquet file path for symbol/timeframe.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.

        Returns:
            Path to Parquet file.
        """
        symbol_dir = self._base_dir / symbol.upper()
        return symbol_dir / f"{timeframe}.parquet"

    def read_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[BarData]:
        """
        Read bars from Parquet file.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Optional start datetime filter.
            end: Optional end datetime filter.

        Returns:
            List of BarData sorted by timestamp ascending.
            Empty list if file doesn't exist.
        """
        file_path = self.get_file_path(symbol, timeframe)

        if not file_path.exists():
            return []

        # Build row group filter for efficient reads
        filters = []
        if start:
            filters.append(("timestamp", ">=", start))
        if end:
            filters.append(("timestamp", "<=", end))

        try:
            table = pq.read_table(
                file_path,
                filters=filters if filters else None,
            )
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []

        return self._table_to_bars(table, symbol, timeframe)

    def write_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: List[BarData],
        mode: str = "upsert",
    ) -> int:
        """
        Write bars to Parquet file.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            bars: List of BarData to write.
            mode: Write mode:
                - 'upsert': Merge with existing, replacing by timestamp
                - 'append': Append to existing file
                - 'overwrite': Replace entire file

        Returns:
            Number of bars written.
        """
        if not bars:
            return 0

        file_path = self.get_file_path(symbol, timeframe)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        new_table = self._bars_to_table(bars)

        if mode == "overwrite" or not file_path.exists():
            pq.write_table(
                new_table,
                file_path,
                compression="snappy",
            )
            logger.info(f"Wrote {len(bars)} bars to {file_path}")
            return len(bars)

        # Read existing data
        existing_table = pq.read_table(file_path)

        if mode == "append":
            # Simple append
            merged = pa.concat_tables([existing_table, new_table])
            # Sort by timestamp
            merged = merged.sort_by("timestamp")
        else:
            # Upsert: merge and deduplicate by timestamp
            merged = self._merge_tables(existing_table, new_table)

        pq.write_table(
            merged,
            file_path,
            compression="snappy",
        )

        final_count = merged.num_rows
        logger.info(
            f"Upserted {len(bars)} bars into {file_path} "
            f"(total: {final_count} bars)"
        )
        return len(bars)

    def _merge_tables(
        self,
        existing: pa.Table,
        new: pa.Table,
    ) -> pa.Table:
        """
        Merge tables, with new data taking precedence for duplicate timestamps.

        Args:
            existing: Existing table.
            new: New table to merge.

        Returns:
            Merged and deduplicated table.
        """
        # Combine tables
        combined = pa.concat_tables([existing, new])

        # Convert to pandas for deduplication (PyArrow lacks native dedup)
        df = combined.to_pandas()

        # Keep last occurrence (new data) for duplicate timestamps
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Convert back to PyArrow
        return pa.Table.from_pandas(df, schema=BAR_SCHEMA)

    def _bars_to_table(self, bars: List[BarData]) -> pa.Table:
        """Convert BarData list to PyArrow table."""
        now = datetime.utcnow()

        data = {
            "timestamp": [b.bar_start or b.timestamp for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume or 0 for b in bars],
            "vwap": [b.vwap for b in bars],
            "trade_count": [b.trade_count for b in bars],
            "source": [b.source for b in bars],
            "ingested_at": [now for _ in bars],
        }

        return pa.Table.from_pydict(data, schema=BAR_SCHEMA)

    def _table_to_bars(
        self,
        table: pa.Table,
        symbol: str,
        timeframe: str,
    ) -> List[BarData]:
        """Convert PyArrow table to BarData list."""
        df = table.to_pandas()
        bars = []

        for _, row in df.iterrows():
            bar = BarData(
                symbol=symbol,
                timeframe=timeframe,
                open=row["open"] if not pd_isna(row["open"]) else None,
                high=row["high"] if not pd_isna(row["high"]) else None,
                low=row["low"] if not pd_isna(row["low"]) else None,
                close=row["close"] if not pd_isna(row["close"]) else None,
                volume=int(row["volume"]) if not pd_isna(row["volume"]) else None,
                vwap=row["vwap"] if not pd_isna(row["vwap"]) else None,
                trade_count=int(row["trade_count"]) if not pd_isna(row["trade_count"]) else None,
                bar_start=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
                source=row["source"],
                timestamp=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
            )
            bars.append(bar)

        return bars

    def get_date_range(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[tuple[datetime, datetime]]:
        """
        Get the date range of stored data.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.

        Returns:
            Tuple of (earliest, latest) timestamps, or None if no data.
        """
        file_path = self.get_file_path(symbol, timeframe)

        if not file_path.exists():
            return None

        try:
            # Read only timestamp column for efficiency
            table = pq.read_table(file_path, columns=["timestamp"])
            if table.num_rows == 0:
                return None

            timestamps = table["timestamp"].to_pandas()
            return (timestamps.min().to_pydatetime(), timestamps.max().to_pydatetime())
        except Exception as e:
            logger.error(f"Error reading date range from {file_path}: {e}")
            return None

    def get_bar_count(self, symbol: str, timeframe: str) -> int:
        """Get number of bars stored for symbol/timeframe."""
        file_path = self.get_file_path(symbol, timeframe)

        if not file_path.exists():
            return 0

        try:
            metadata = pq.read_metadata(file_path)
            return int(metadata.num_rows)
        except Exception as e:
            logger.error(f"Error reading metadata from {file_path}: {e}")
            return 0

    def delete_data(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
    ) -> bool:
        """
        Delete stored data.

        Args:
            symbol: Ticker symbol.
            timeframe: Optional specific timeframe. If None, deletes all.

        Returns:
            True if any data was deleted.
        """
        symbol_dir = self._base_dir / symbol.upper()

        if not symbol_dir.exists():
            return False

        if timeframe:
            file_path = self.get_file_path(symbol, timeframe)
            if file_path.exists():
                file_path.unlink()
                return True
            return False

        # Delete entire symbol directory
        import shutil
        shutil.rmtree(symbol_dir)
        return True

    def list_symbols(self) -> List[str]:
        """List all symbols with stored data."""
        if not self._base_dir.exists():
            return []

        return sorted([
            d.name for d in self._base_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ])

    def list_timeframes(self, symbol: str) -> List[str]:
        """List available timeframes for a symbol."""
        symbol_dir = self._base_dir / symbol.upper()

        if not symbol_dir.exists():
            return []

        return sorted([
            f.stem for f in symbol_dir.glob("*.parquet")
        ])


def pd_isna(value: object) -> bool:
    """Check if value is NaN/None, handling pandas types."""
    if value is None:
        return True
    try:
        import pandas as pd
        import numpy as np
        if isinstance(value, (float, np.floating)) and np.isnan(value):  # type: ignore[arg-type]
            return True
        return False
    except (ImportError, TypeError, ValueError):
        return False
