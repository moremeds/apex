"""Base engine interface for Apex backtesting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

import polars as pl
from pydantic import BaseModel

from apex.core.types import MarketDataFrame


class StrategyProtocol(Protocol):
    """Protocol defining the interface for trading strategies."""

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate entry/exit signals based on market data.
        
        Analyzes market data to produce trading signals indicating when
        to enter or exit positions. The returned DataFrame must contain
        'datetime', 'entry', and 'exit' columns with boolean signals.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            pl.DataFrame: DataFrame with entry/exit signals
            
        Raises:
            ValueError: If input data is invalid or missing required columns
        """
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters and configuration.
        
        Provides access to the strategy's current parameter values,
        which may be used for portfolio construction, risk management,
        or optimization purposes.
        
        Returns:
            Dict[str, Any]: Dictionary of parameter names to values
            
        Examples:
            >>> strategy.get_parameters()
            {'fast_period': 20, 'slow_period': 50, 'risk_level': 0.02}
        """
        ...

    def validate_data(self, data: pl.DataFrame) -> bool:
        """Validate that data meets strategy requirements.
        
        Checks if the provided market data is suitable for this strategy's
        analysis. This includes validating required columns, data quality,
        minimum history requirements, and any strategy-specific needs.
        
        Args:
            data: Market data DataFrame to validate
            
        Returns:
            bool: True if data is valid for this strategy
            
        Examples:
            >>> strategy.validate_data(market_data)
            True
        """
        ...


class BacktestResult(BaseModel):
    """Results from a backtest run."""

    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_cash: float
    final_value: float
    total_return: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Raw data for further analysis
    portfolio_value: Optional[pl.DataFrame] = None
    trades: Optional[pl.DataFrame] = None
    positions: Optional[pl.DataFrame] = None
    
    class Config:
        arbitrary_types_allowed = True


class EngineConfig(BaseModel):
    """Configuration for backtesting engines."""

    initial_cash: float = 100000.0
    commission: float = 0.001  # 0.1% default commission
    slippage: float = 0.0001  # 0.01% default slippage
    margin: float = 1.0  # No margin by default
    
    # Risk management
    max_position_size: Optional[float] = None  # As fraction of portfolio
    stop_loss: Optional[float] = None  # As fraction (e.g., 0.02 = 2%)
    take_profit: Optional[float] = None  # As fraction
    
    # Execution settings
    fill_on_next_bar: bool = True  # Fill orders on next bar
    auto_close_positions: bool = True  # Close positions at end


class BaseEngine(ABC):
    """Base class for all backtesting engines."""

    def __init__(self, config: EngineConfig) -> None:
        """Initialize engine with configuration."""
        self.config = config

    @abstractmethod
    async def run_backtest(
        self, 
        strategy: StrategyProtocol, 
        data: MarketDataFrame
    ) -> BacktestResult:
        """Run backtest with given strategy and data."""
        pass

    @abstractmethod
    def calculate_metrics(self, result: BacktestResult) -> BacktestResult:
        """Calculate comprehensive performance metrics for backtest results.
        
        Computes risk-adjusted returns, drawdown analysis, trade statistics,
        and other key performance indicators. Updates the BacktestResult
        object with calculated metrics.
        
        Args:
            result: BacktestResult with basic performance data
            
        Returns:
            BacktestResult: Enhanced result with comprehensive metrics
            
        Raises:
            ValueError: If result data is invalid or incomplete
        """
        pass

    def validate_strategy(self, strategy: StrategyProtocol, data: MarketDataFrame) -> bool:
        """Validate strategy compatibility with market data.
        
        Ensures the strategy can process the provided market data by
        checking data requirements, column availability, and minimum
        history requirements.
        
        Args:
            strategy: Strategy to validate
            data: Market data to check against strategy requirements
            
        Returns:
            bool: True if strategy is compatible with the data
        """
        return strategy.validate_data(data.data)