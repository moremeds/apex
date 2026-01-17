"""
Data providers for backtesting.

Provides data sources for backtest engines:
- IbBacktestDataProvider: IB Historical data via existing adapter
"""

from .ib_provider import IbBacktestDataProvider, create_backtest_provider

__all__ = [
    "IbBacktestDataProvider",
    "create_backtest_provider",
]
