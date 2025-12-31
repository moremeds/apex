"""
Backtest infrastructure components.

This module provides:
- SimulatedExecution: Simulated order matching for backtests
- DataFeeds: Historical data loading (CSV, Parquet)
- BacktestEngine: Orchestrates backtest execution
- BacktraderAdapter: Integration with Backtrader (optional)
"""

from .simulated_execution import SimulatedExecution, FillModel
from .data_feeds import (
    CsvDataFeed,
    ParquetDataFeed,
    FixtureDataFeed,
    InMemoryDataFeed,
    IbHistoricalDataFeed,
    MultiTimeframeDataFeed,
    create_csv_multi_timeframe_feed,
    create_ib_multi_timeframe_feed,
    # OPT-009: Streaming data feeds
    StreamingCsvDataFeed,
    StreamingParquetDataFeed,
    create_data_feed,
)
from .backtest_engine import BacktestEngine, BacktestConfig
from .backtrader_adapter import ApexStrategyWrapper, run_backtest_with_backtrader

__all__ = [
    # Simulated execution
    "SimulatedExecution",
    "FillModel",
    # Data feeds
    "IbHistoricalDataFeed",  # Primary - uses real IB data
    "CsvDataFeed",
    "ParquetDataFeed",
    "FixtureDataFeed",
    "InMemoryDataFeed",
    "MultiTimeframeDataFeed",  # Combine multiple timeframes
    "create_csv_multi_timeframe_feed",
    "create_ib_multi_timeframe_feed",
    # OPT-009: Streaming data feeds (memory efficient)
    "StreamingCsvDataFeed",
    "StreamingParquetDataFeed",
    "create_data_feed",  # Factory function
    # Backtest engine
    "BacktestEngine",
    "BacktestConfig",
    # Backtrader integration
    "ApexStrategyWrapper",
    "run_backtest_with_backtrader",
]
