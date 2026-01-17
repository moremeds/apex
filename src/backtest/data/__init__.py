"""
Data management layer for systematic backtesting.

This module provides:
- Feeds: Data loading (CSV, Parquet, IB, multi-timeframe, streaming)
- Providers: Data sources (IB Historical, etc.)
- Splitters: Walk-Forward Optimization, Combinatorial Purged CV
- Storage: DuckDB persistence, repositories
- Calendar: Trading calendar abstractions (NYSE, weekday, etc.)
"""

from .calendar import (
    ExchangeCalendar,
    TradingCalendar,
    WeekdayCalendar,
    get_calendar,
    list_available_calendars,
)
from .feeds import (
    BarCacheDataFeed,
    CachedBarDataFeed,
    CsvDataFeed,
    DataFeed,
    FixtureDataFeed,
    HistoricalBar,
    HistoricalStoreDataFeed,
    IbHistoricalDataFeed,
    InMemoryDataFeed,
    MultiTimeframeDataFeed,
    ParquetDataFeed,
    StreamingCsvDataFeed,
    StreamingParquetDataFeed,
    create_csv_multi_timeframe_feed,
    create_data_feed,
    create_ib_multi_timeframe_feed,
)
from .providers import (
    IbBacktestDataProvider,
    create_backtest_provider,
)
from .splitters import (
    CPCVConfig,
    CPCVSplitter,
    SplitConfig,
    WalkForwardSplitter,
)
from .storage import (
    DatabaseManager,
    ExperimentRepository,
    RunRepository,
    TrialRepository,
)

__all__ = [
    # Feeds
    "DataFeed",
    "CsvDataFeed",
    "StreamingCsvDataFeed",
    "ParquetDataFeed",
    "StreamingParquetDataFeed",
    "HistoricalStoreDataFeed",
    "FixtureDataFeed",
    "InMemoryDataFeed",
    "MultiTimeframeDataFeed",
    "CachedBarDataFeed",
    "BarCacheDataFeed",
    "IbHistoricalDataFeed",
    "HistoricalBar",
    "create_data_feed",
    "create_csv_multi_timeframe_feed",
    "create_ib_multi_timeframe_feed",
    # Providers
    "IbBacktestDataProvider",
    "create_backtest_provider",
    # Splitters
    "WalkForwardSplitter",
    "CPCVSplitter",
    "SplitConfig",
    "CPCVConfig",
    # Storage
    "DatabaseManager",
    "ExperimentRepository",
    "TrialRepository",
    "RunRepository",
    # Calendar
    "TradingCalendar",
    "WeekdayCalendar",
    "ExchangeCalendar",
    "get_calendar",
    "list_available_calendars",
]
