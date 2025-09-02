"""Vectorbt-based backtesting engine implementation."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Dict

import polars as pl
import structlog
import vectorbt as vbt
import numpy as np

from apex.core.types import MarketDataFrame
from apex.engine.base import BaseEngine, BacktestResult, EngineConfig, StrategyProtocol
from apex.engine.data_extraction import VectorbtDataExtractor
from apex.engine.portfolio import PortfolioMetrics
from apex.engine.signal_processor import SignalProcessor

logger = structlog.get_logger(__name__)

# Suppress vectorbt warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="vectorbt")

# Import pandas for vectorbt compatibility
try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required for vectorbt integration. "
        "Install with: pip install pandas"
    )


class VectorbtEngine(BaseEngine):
    """Vectorbt-based backtesting engine for high-performance strategy testing.
    
    This engine provides vectorized backtesting capabilities using the vectorbt
    library, optimized for speed and handling large datasets with comprehensive
    portfolio management and performance metrics calculation.
    
    Attributes:
        config: Engine configuration with cash, fees, and risk management settings
        signal_processor: Handles signal processing and portfolio creation
        data_extractor: Extracts data from vectorbt portfolio objects
        metrics_calculator: Calculates comprehensive performance metrics
    """

    def __init__(self, config: EngineConfig) -> None:
        """Initialize the vectorbt engine with configuration.
        
        Sets up the engine with proper vectorbt configuration and initializes
        the component processors for signals, data extraction, and metrics.
        
        Args:
            config: EngineConfig containing backtesting parameters
        """
        super().__init__(config)
        self._setup_vectorbt_settings()
        
        # Initialize component processors
        self.signal_processor = SignalProcessor(config)
        self.data_extractor = VectorbtDataExtractor()
        self.metrics_calculator = PortfolioMetrics()

    def _setup_vectorbt_settings(self) -> None:
        """Configure vectorbt settings for optimal performance.
        
        Sets global vectorbt configuration for array wrapper frequency,
        portfolio initialization cash, fee structure, and performance optimizations.
        """
        # Basic configuration
        vbt.settings.array_wrapper['freq'] = 'D'  # Daily frequency by default
        vbt.settings.portfolio['init_cash'] = self.config.initial_cash
        vbt.settings.portfolio['fees'] = self.config.commission
        
        # Performance optimizations (only valid settings)
        try:
            vbt.settings.caching['enabled'] = True  # Enable built-in caching if available
        except (KeyError, AttributeError):
            pass  # Setting not available in this vectorbt version
            
        try:
            vbt.settings.array_wrapper['check_freq'] = False  # Skip frequency checks for speed
        except (KeyError, AttributeError):
            pass  # Setting not available in this vectorbt version
        
        logger.debug(
            "Vectorbt settings configured for performance",
            initial_cash=self.config.initial_cash,
            fees=self.config.commission
        )
        
    async def run_backtest(
        self, 
        strategy: StrategyProtocol, 
        data: MarketDataFrame
    ) -> BacktestResult:
        """Run backtest using vectorbt engine.
        
        Executes a complete backtesting workflow including strategy signal generation,
        portfolio simulation, performance metrics calculation, and result compilation.
        
        Args:
            strategy: Strategy object implementing StrategyProtocol
            data: Market data containing OHLCV information
            
        Returns:
            BacktestResult: Complete backtesting results with performance metrics
            
        Raises:
            ValueError: If strategy validation fails or data is incompatible
        """
        logger.info(
            "Starting vectorbt backtest",
            strategy=strategy.__class__.__name__,
            symbol=data.symbol,
            start_date=data.start_date,
            end_date=data.end_date,
            data_rows=len(data.data)
        )

        try:
            # Validate strategy and data compatibility
            if not self.validate_strategy(strategy, data):
                logger.error("Strategy validation failed", 
                            strategy=strategy.__class__.__name__,
                            data_columns=list(data.data.columns),
                            data_shape=data.data.shape)
                raise ValueError("Strategy validation failed for provided data")

            # Process signals and create portfolio with error context
            portfolio = await self._process_signals_and_create_portfolio(strategy, data)
            
            if portfolio is None:
                logger.error("Portfolio creation returned None",
                            strategy=strategy.__class__.__name__)
                raise ValueError("Portfolio creation failed: returned None")
            
            # Extract basic portfolio metrics with validation
            total_return = portfolio.total_return()
            if pd.isna(total_return) or not np.isfinite(total_return):
                logger.warning("Invalid total return calculated", 
                              total_return=total_return,
                              strategy=strategy.__class__.__name__)
                total_return = 0.0
                
            portfolio_values = portfolio.value()
            if portfolio_values.empty:
                logger.error("Empty portfolio values", 
                            strategy=strategy.__class__.__name__)
                raise ValueError("Portfolio values are empty")
                
            final_value = portfolio_values.iloc[-1]
            if pd.isna(final_value) or not np.isfinite(final_value):
                logger.warning("Invalid final portfolio value", 
                              final_value=final_value,
                              strategy=strategy.__class__.__name__)
                final_value = self.config.initial_cash
                
        except ValueError as e:
            logger.error("Validation error in backtest", 
                        error=str(e), 
                        strategy=strategy.__class__.__name__)
            raise
        except Exception as e:
            logger.error("Unexpected error in backtest execution", 
                        error=str(e), 
                        error_type=type(e).__name__,
                        strategy=strategy.__class__.__name__,
                        symbol=data.symbol)
            raise ValueError(f"Backtest execution failed: {str(e)}") from e
        
        # Create and populate result object
        result = self._create_backtest_result(
            strategy=strategy,
            data=data,
            portfolio=portfolio,
            total_return=total_return,
            final_value=final_value
        )

        # Calculate comprehensive performance metrics
        result = self.metrics_calculator.calculate_metrics(result)
        
        logger.info(
            "Vectorbt backtest completed successfully",
            strategy=strategy.__class__.__name__,
            symbol=data.symbol,
            total_return=f"{total_return:.2%}",
            final_value=f"${final_value:,.2f}",
            total_trades=result.total_trades,
            sharpe_ratio=result.sharpe_ratio
        )

        return result

    async def _process_signals_and_create_portfolio(
        self,
        strategy: StrategyProtocol,
        data: MarketDataFrame
    ) -> vbt.Portfolio:
        """Process strategy signals and create vectorbt portfolio.
        
        Handles the complete signal processing workflow including strategy
        signal generation, data merging, format conversion, and portfolio creation.
        
        Args:
            strategy: Strategy object for signal generation
            data: Market data for backtesting
            
        Returns:
            vbt.Portfolio: Configured vectorbt portfolio ready for analysis
        """
        # Generate signals using the strategy (avoid unnecessary clone)
        signals_df = strategy.generate_signals(data.data)
        
        # Merge signals with price data efficiently
        merged_df = data.data.join(signals_df, on="datetime", how="left")
        
        # Convert to pandas for vectorbt compatibility with optimization
        merged_pd = merged_df.to_pandas(use_pyarrow_extension_array=True).set_index("datetime")
        
        # Extract and validate signals
        close_prices = merged_pd["close"]
        entries = merged_pd.get("entry", pd.Series(False, index=close_prices.index))
        exits = merged_pd.get("exit", pd.Series(False, index=close_prices.index))
        
        # Validate signal format
        entries, exits = SignalProcessor.validate_signals(entries, exits)

        # Create portfolio using signal processor
        portfolio = self.signal_processor.create_portfolio(
            close_prices=close_prices,
            entries=entries,
            exits=exits,
            strategy_params=strategy.get_parameters()
        )

        return portfolio

    def _create_backtest_result(
        self,
        strategy: StrategyProtocol,
        data: MarketDataFrame,
        portfolio: vbt.Portfolio,
        total_return: float,
        final_value: float
    ) -> BacktestResult:
        """Create and populate BacktestResult object with portfolio data.
        
        Constructs a complete BacktestResult with basic metrics and detailed
        portfolio, trades, and positions data extracted from the vectorbt portfolio.
        
        Args:
            strategy: Strategy used for backtesting
            data: Original market data
            portfolio: Vectorbt portfolio object
            total_return: Total portfolio return
            final_value: Final portfolio value
            
        Returns:
            BacktestResult: Populated result object ready for metrics calculation
        """
        # Create base result object
        result = BacktestResult(
            strategy_name=strategy.__class__.__name__,
            symbol=data.symbol,
            start_date=data.start_date,
            end_date=data.end_date,
            initial_cash=self.config.initial_cash,
            final_value=float(final_value),
            total_return=float(total_return),
            total_trades=portfolio.orders.count(),
        )

        # Extract detailed portfolio data
        result.portfolio_value = self.data_extractor.convert_to_polars(
            portfolio.value().reset_index(), 
            ["datetime", "portfolio_value"]
        )
        
        result.trades = self.data_extractor.extract_trades_data(portfolio)
        result.positions = self.data_extractor.extract_positions_data(portfolio)

        return result

    def calculate_metrics(self, result: BacktestResult) -> BacktestResult:
        """Calculate comprehensive performance metrics.
        
        Delegates to PortfolioMetrics for comprehensive performance analysis.
        This method maintains backward compatibility with the original API.
        
        Args:
            result: BacktestResult to analyze
            
        Returns:
            BacktestResult: Result with calculated performance metrics
        """
        return self.metrics_calculator.calculate_metrics(result)