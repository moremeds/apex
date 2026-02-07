"""
Factory functions for creating data feeds.

Contains:
- create_data_feed: Universal factory with streaming support (OPT-009)
- create_csv_multi_timeframe_feed: MTF feed from CSV files
- create_ib_multi_timeframe_feed: MTF feed from IB historical data
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, List, Optional

from .base import DataFeed
from .csv_feeds import CsvDataFeed, StreamingCsvDataFeed
from .ib_feeds import IbHistoricalDataFeed
from .multi_timeframe import MultiTimeframeDataFeed
from .parquet_feeds import ParquetDataFeed, StreamingParquetDataFeed


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
