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

# Base classes
from .base import DataFeed

# CSV feeds
from .csv_feeds import CsvDataFeed, StreamingCsvDataFeed

# Factory functions
from .factory import (
    create_csv_multi_timeframe_feed,
    create_data_feed,
    create_ib_multi_timeframe_feed,
)

# Historical store feed
from .historical_feeds import HistoricalStoreDataFeed

# IB feeds
from .ib_feeds import BarCacheDataFeed, IbHistoricalDataFeed

# Memory-based feeds
from .memory_feeds import CachedBarDataFeed, FixtureDataFeed, InMemoryDataFeed

# Data models
from .models import AlignedBarBuffer, HistoricalBar

# Multi-timeframe feed
from .multi_timeframe import MultiTimeframeDataFeed

# Parquet feeds
from .parquet_feeds import ParquetDataFeed, StreamingParquetDataFeed

__all__ = [
    # Base
    "DataFeed",
    # Models
    "HistoricalBar",
    "AlignedBarBuffer",
    # CSV
    "CsvDataFeed",
    "StreamingCsvDataFeed",
    # Parquet
    "ParquetDataFeed",
    "StreamingParquetDataFeed",
    # Historical store
    "HistoricalStoreDataFeed",
    # IB
    "IbHistoricalDataFeed",
    "BarCacheDataFeed",
    # Memory
    "InMemoryDataFeed",
    "FixtureDataFeed",
    "CachedBarDataFeed",
    # Multi-timeframe
    "MultiTimeframeDataFeed",
    # Factory
    "create_data_feed",
    "create_csv_multi_timeframe_feed",
    "create_ib_multi_timeframe_feed",
]
