# VectorbtEngine Split Implementation Details

## Current File Analysis

The `vectorbt.py` file (314 lines) contains:

### Core Engine Methods (Lines 1-125):
- Imports and setup (1-21)
- `VectorbtEngine` class definition (23-29)
- `_setup_vectorbt_settings` (31-36)
- `run_backtest` method (38-123)

### Portfolio Management (Lines 125-175):
- `_create_portfolio` method (125-158)
- `_calculate_position_sizes` method (160-174)

### Metrics Calculation (Lines 176-232):
- `calculate_metrics` method (176-232)

### Data Extraction (Lines 234-314):
- `_extract_trades_data` method (234-265)
- `_extract_positions_data` method (266-293)
- `_convert_to_polars` method (294-314)

## Proposed Split Structure

### 1. engine.py (Core orchestration)
```python
"""
Vectorbt engine core implementation.

Contains the main VectorbtEngine class that orchestrates backtesting operations.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

import structlog
import vectorbt as vbt

from apex.core.types import MarketDataFrame
from apex.engine.base import BaseEngine, BacktestResult, EngineConfig, StrategyProtocol
from .portfolio import PortfolioManager
from .metrics import MetricsCalculator
from .data_extraction import DataExtractor

logger = structlog.get_logger(__name__)

# Suppress vectorbt warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="vectorbt")


class VectorbtEngine(BaseEngine):
    """Vectorbt-based backtesting engine for high-performance strategy testing."""

    def __init__(self, config: EngineConfig) -> None:
        """Initialize the vectorbt engine with specialized managers."""
        super().__init__(config)
        self._setup_vectorbt_settings()
        
        # Initialize specialized managers
        self._portfolio_manager = PortfolioManager(config)
        self._metrics_calculator = MetricsCalculator()
        self._data_extractor = DataExtractor()

    def _setup_vectorbt_settings(self) -> None:
        """Configure vectorbt settings for optimal performance.
        
        Sets up global vectorbt configuration including default frequency,
        initial cash, and commission rates for consistent backtesting.
        """
        vbt.settings.array_wrapper['freq'] = 'D'  # Daily frequency by default
        vbt.settings.portfolio['init_cash'] = self.config.initial_cash
        vbt.settings.portfolio['fees'] = self.config.commission
        
    async def run_backtest(
        self, 
        strategy: StrategyProtocol, 
        data: MarketDataFrame
    ) -> BacktestResult:
        """Run backtest using vectorbt engine with delegated processing."""
        # Main orchestration logic here...
        # Delegates to specialized managers for specific operations
        pass

    # Delegation methods
    def _create_portfolio(self, *args, **kwargs):
        """Delegate portfolio creation to PortfolioManager."""
        return self._portfolio_manager.create_portfolio(*args, **kwargs)
    
    def calculate_metrics(self, result: BacktestResult) -> BacktestResult:
        """Delegate metrics calculation to MetricsCalculator."""
        return self._metrics_calculator.calculate_metrics(result)
    
    def _extract_trades_data(self, *args, **kwargs):
        """Delegate trade data extraction to DataExtractor."""
        return self._data_extractor.extract_trades_data(*args, **kwargs)
    
    def _extract_positions_data(self, *args, **kwargs):
        """Delegate position data extraction to DataExtractor."""
        return self._data_extractor.extract_positions_data(*args, **kwargs)
```

### 2. portfolio.py (Portfolio management)
```python
"""
Portfolio management for vectorbt engine.

Handles portfolio creation, position sizing, and risk management configuration.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt

from apex.engine.base import EngineConfig


class PortfolioManager:
    """Manages portfolio creation and position sizing for vectorbt engine."""
    
    def __init__(self, config: EngineConfig) -> None:
        """Initialize portfolio manager with engine configuration.
        
        Args:
            config: Engine configuration containing cash, commission, and risk parameters
        """
        self.config = config
    
    def create_portfolio(
        self,
        close_prices: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        strategy_params: Dict[str, Any]
    ) -> vbt.Portfolio:
        """Create vectorbt portfolio from signals.
        
        Args:
            close_prices: Series of closing prices
            entries: Boolean series indicating entry signals
            exits: Boolean series indicating exit signals
            strategy_params: Additional strategy parameters
            
        Returns:
            Configured vectorbt Portfolio object
        """
        # Portfolio configuration
        portfolio_kwargs = {
            "close": close_prices,
            "entries": entries,
            "exits": exits,
            "init_cash": self.config.initial_cash,
            "fees": self.config.commission,
            "slippage": self.config.slippage,
            "freq": "D"
        }
        
        # Add position sizing if specified
        if self.config.max_position_size:
            position_sizes = self._calculate_position_sizes(close_prices, entries)
            portfolio_kwargs["size"] = position_sizes
        
        # Add risk management if specified
        if self.config.stop_loss:
            portfolio_kwargs["sl_stop"] = self.config.stop_loss
            
        if self.config.take_profit:
            portfolio_kwargs["tp_stop"] = self.config.take_profit

        return vbt.Portfolio.from_signals(**portfolio_kwargs)

    def _calculate_position_sizes(
        self, 
        close_prices: pd.Series, 
        entries: pd.Series
    ) -> pd.Series:
        """Calculate position sizes based on max position size constraint.
        
        Args:
            close_prices: Series of closing prices
            entries: Boolean series indicating entry signals
            
        Returns:
            Series with calculated position sizes for each entry signal
        """
        position_sizes = pd.Series(np.nan, index=close_prices.index)
        
        if self.config.max_position_size:
            # Calculate shares based on max position size
            max_investment = self.config.initial_cash * self.config.max_position_size
            shares = max_investment / close_prices
            position_sizes[entries] = shares[entries]
            
        return position_sizes
```

### 3. metrics.py (Performance calculations)
```python
"""
Performance metrics calculation for vectorbt engine.

Calculates comprehensive performance metrics including Sharpe ratio,
maximum drawdown, win rate, and profit factor.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
import structlog

from apex.engine.base import BacktestResult

logger = structlog.get_logger(__name__)


class MetricsCalculator:
    """Calculates comprehensive performance metrics for backtest results."""
    
    def calculate_metrics(self, result: BacktestResult) -> BacktestResult:
        """Calculate comprehensive performance metrics.
        
        Args:
            result: BacktestResult object to enhance with metrics
            
        Returns:
            BacktestResult with calculated performance metrics
        """
        if result.portfolio_value is None or len(result.portfolio_value) == 0:
            logger.warning("No portfolio data available for metrics calculation")
            return result

        # Convert portfolio values to numpy for calculations
        portfolio_values = result.portfolio_value.select("portfolio_value").to_numpy().flatten()
        
        if len(portfolio_values) < 2:
            return result

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return result

        # Sharpe Ratio (assuming daily data, annualized)
        if np.std(returns) > 0:
            result.sharpe_ratio = float(np.sqrt(252) * np.mean(returns) / np.std(returns))

        # Maximum Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        result.max_drawdown = float(np.min(drawdown))

        # Trade statistics
        self._calculate_trade_statistics(result)

        return result
    
    def _calculate_trade_statistics(self, result: BacktestResult) -> None:
        """Calculate trade-based statistics.
        
        Args:
            result: BacktestResult object to update with trade statistics
        """
        if result.trades is None or len(result.trades) == 0:
            return
            
        trades_df = result.trades
        
        if "pnl" not in trades_df.columns:
            return
            
        winning_trades = trades_df.filter(pl.col("pnl") > 0)
        losing_trades = trades_df.filter(pl.col("pnl") < 0)
        
        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        result.total_trades = len(trades_df)
        
        if result.total_trades > 0:
            result.win_rate = float(result.winning_trades / result.total_trades)
        
        if len(winning_trades) > 0:
            result.avg_win = float(winning_trades.select("pnl").mean().item())
        
        if len(losing_trades) > 0:
            result.avg_loss = float(losing_trades.select("pnl").mean().item())
        
        # Profit Factor
        if result.avg_loss and result.avg_loss < 0:
            total_wins = result.winning_trades * (result.avg_win or 0)
            total_losses = abs(result.losing_trades * (result.avg_loss or 0))
            if total_losses > 0:
                result.profit_factor = float(total_wins / total_losses)
```

### 4. data_extraction.py (Data processing)
```python
"""
Data extraction utilities for vectorbt engine.

Handles extraction and conversion of trade and position data from
vectorbt portfolios to polars DataFrames.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import polars as pl
import structlog
import vectorbt as vbt

logger = structlog.get_logger(__name__)


class DataExtractor:
    """Extracts and converts data from vectorbt portfolios."""
    
    def extract_trades_data(self, portfolio: vbt.Portfolio) -> Optional[pl.DataFrame]:
        """Extract trades data from vectorbt portfolio.
        
        Args:
            portfolio: Vectorbt Portfolio object
            
        Returns:
            Polars DataFrame with trade data or None if no trades
        """
        try:
            if portfolio.orders.count() == 0:
                return None
                
            # Get trades from portfolio
            trades = portfolio.trades
            if trades.count() == 0:
                return None
            
            # Convert to pandas DataFrame first
            trades_df = trades.records_readable
            
            if trades_df.empty:
                return None
            
            # Define expected columns for trades
            expected_columns = [
                "entry_timestamp", "exit_timestamp", "size", 
                "entry_price", "exit_price", "pnl", "return"
            ]
            
            return self._convert_to_polars(trades_df, expected_columns)
            
        except Exception as e:
            logger.warning(f"Failed to extract trades data: {e}")
            return None
    
    def extract_positions_data(self, portfolio: vbt.Portfolio) -> Optional[pl.DataFrame]:
        """Extract positions data from vectorbt portfolio.
        
        Args:
            portfolio: Vectorbt Portfolio object
            
        Returns:
            Polars DataFrame with position data or None if no positions
        """
        try:
            # Get positions from portfolio
            positions = portfolio.positions
            if positions.count() == 0:
                return None
            
            # Convert to pandas DataFrame first
            positions_df = positions.records_readable
            
            if positions_df.empty:
                return None
            
            # Define expected columns for positions
            expected_columns = [
                "entry_timestamp", "exit_timestamp", "size",
                "entry_price", "exit_price", "pnl"
            ]
            
            return self._convert_to_polars(positions_df, expected_columns)
            
        except Exception as e:
            logger.warning(f"Failed to extract positions data: {e}")
            return None
    
    def _convert_to_polars(
        self, 
        df: pd.DataFrame, 
        expected_columns: List[str]
    ) -> pl.DataFrame:
        """Convert pandas DataFrame to polars with specified column names.
        
        Args:
            df: Pandas DataFrame to convert
            expected_columns: List of expected column names
            
        Returns:
            Polars DataFrame with standardized columns
        """
        # Reset index to ensure we have all data
        df = df.reset_index(drop=True)
        
        # Convert to polars
        polars_df = pl.from_pandas(df)
        
        # Ensure we have the expected columns (create with null if missing)
        current_columns = polars_df.columns
        
        for col in expected_columns:
            if col not in current_columns:
                # Add missing column with null values
                polars_df = polars_df.with_columns(pl.lit(None).alias(col))
        
        # Select only expected columns in the right order
        return polars_df.select(expected_columns)
```

### 5. __init__.py (Public API)
```python
"""
Vectorbt engine module.

Provides high-performance backtesting using the vectorbt library.
"""
from .engine import VectorbtEngine

__all__ = ['VectorbtEngine']
```

## Migration Strategy

1. **Create directory structure**: `mkdir -p src/apex/engine/vectorbt`
2. **Create files in order**: Start with data_extraction.py (no deps), then metrics.py, portfolio.py, and finally engine.py
3. **Update imports**: Modify existing import statements to use new module structure
4. **Run tests incrementally**: Test after each file creation
5. **Remove original**: Only after all tests pass, remove original vectorbt.py

## Testing Verification

After split, these imports must still work:
```python
# Existing test imports should continue to work
from apex.engine.vectorbt import VectorbtEngine

# Engine should behave identically
engine = VectorbtEngine(config)
result = await engine.run_backtest(strategy, data)
```

## Quality Assurance

- Each new file under 200 lines ✓
- All methods have docstrings ✓
- Type hints preserved ✓
- Error handling maintained ✓
- Performance characteristics unchanged ✓