"""Signal processing utilities for vectorbt backtesting engine."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import structlog
import vectorbt as vbt

logger = structlog.get_logger(__name__)

# Import pandas for vectorbt compatibility
try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required for vectorbt integration. "
        "Install with: pip install pandas"
    )


class SignalProcessor:
    """Processes trading signals and creates vectorbt portfolio objects."""

    def __init__(self, config):
        """Initialize signal processor with engine configuration.
        
        Args:
            config: EngineConfig object containing backtesting parameters
        """
        self.config = config

    def create_portfolio(
        self,
        close_prices: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        strategy_params: Dict[str, Any]
    ) -> vbt.Portfolio:
        """Create vectorbt portfolio from trading signals.
        
        Constructs a vectorbt Portfolio object with proper configuration for
        backtesting including position sizing, risk management, and fees.
        
        Args:
            close_prices: Series of closing prices indexed by datetime
            entries: Series of boolean entry signals
            exits: Series of boolean exit signals
            strategy_params: Dictionary of strategy-specific parameters
            
        Returns:
            vbt.Portfolio: Configured vectorbt portfolio for backtesting
            
        Raises:
            ValueError: If portfolio creation fails due to invalid parameters
        """
        try:
            # Base portfolio configuration
            portfolio_kwargs = {
                "close": close_prices,
                "entries": entries,
                "exits": exits,
                "init_cash": self.config.initial_cash,
                "fees": self.config.commission,
                "slippage": self.config.slippage,
                "freq": "D"  # Daily frequency
            }
            
            # Add position sizing if specified
            if self.config.max_position_size:
                position_sizes = self._calculate_position_sizes(close_prices, entries)
                portfolio_kwargs["size"] = position_sizes
                logger.debug("Added position sizing constraints to portfolio")
            
            # Add risk management if specified
            if self.config.stop_loss:
                portfolio_kwargs["sl_stop"] = self.config.stop_loss
                logger.debug(f"Added stop loss at {self.config.stop_loss:.2%}")
                
            if self.config.take_profit:
                portfolio_kwargs["tp_stop"] = self.config.take_profit
                logger.debug(f"Added take profit at {self.config.take_profit:.2%}")

            # Create portfolio
            portfolio = vbt.Portfolio.from_signals(**portfolio_kwargs)
            
            logger.info(
                "Portfolio created successfully",
                initial_cash=self.config.initial_cash,
                commission=self.config.commission,
                slippage=self.config.slippage,
                orders_count=portfolio.orders.count()
            )
            
            return portfolio
            
        except KeyError as e:
            logger.error("Missing required data for portfolio creation", 
                        missing_key=str(e), 
                        available_columns=list(close_prices.index) if hasattr(close_prices, 'index') else None)
            raise ValueError(f"Missing required data for portfolio: {str(e)}") from e
        except TypeError as e:
            logger.error("Invalid data type for portfolio creation", 
                        error=str(e),
                        entries_type=type(entries).__name__,
                        exits_type=type(exits).__name__)
            raise ValueError(f"Invalid data types for portfolio creation: {str(e)}") from e
        except Exception as e:
            logger.error("Unexpected error in portfolio creation", 
                        error=str(e), 
                        error_type=type(e).__name__,
                        config_cash=self.config.initial_cash,
                        config_commission=self.config.commission)
            raise ValueError(f"Portfolio creation failed: {str(e)}") from e

    def _calculate_position_sizes(
        self, 
        close_prices: pd.Series, 
        entries: pd.Series
    ) -> pd.Series:
        """Calculate position sizes based on maximum position size constraint.
        
        Determines the number of shares to buy for each entry signal based on
        the maximum position size as a fraction of total portfolio value.
        
        Args:
            close_prices: Series of closing prices
            entries: Series of boolean entry signals
            
        Returns:
            pd.Series: Position sizes (number of shares) for each entry
            
        Raises:
            ValueError: If invalid position sizing configuration is detected
        """
        if not self.config.max_position_size:
            logger.warning("max_position_size not configured")
            return pd.Series(np.nan, index=close_prices.index)
            
        if not (0 < self.config.max_position_size <= 1):
            raise ValueError(
                f"max_position_size must be between 0 and 1, got {self.config.max_position_size}"
            )

        try:
            position_sizes = pd.Series(np.nan, index=close_prices.index)
            
            # Calculate maximum investment per position
            max_investment = self.config.initial_cash * self.config.max_position_size
            
            # Calculate shares based on price at entry points
            entry_prices = close_prices[entries]
            if len(entry_prices) > 0:
                shares = max_investment / entry_prices
                position_sizes[entries] = shares
                
                logger.debug(
                    f"Calculated position sizes for {len(entry_prices)} entries",
                    max_position_pct=f"{self.config.max_position_size:.1%}",
                    max_investment=f"${max_investment:,.2f}",
                    avg_shares=f"{shares.mean():.1f}"
                )
            else:
                logger.warning("No entry signals found for position sizing")
            
            return position_sizes
            
        except Exception as e:
            logger.error("Failed to calculate position sizes", error=str(e))
            raise ValueError(f"Position sizing calculation failed: {str(e)}") from e

    @staticmethod
    def validate_signals(entries: pd.Series, exits: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Validate and normalize trading signals.
        
        Ensures that entry and exit signals are properly formatted as boolean
        Series and handles different input formats gracefully.
        
        Args:
            entries: Entry signals (various formats)
            exits: Exit signals (various formats)
            
        Returns:
            tuple[pd.Series, pd.Series]: Validated boolean signal series
            
        Raises:
            ValueError: If signals cannot be converted to boolean format
        """
        try:
            # Normalize entry signals
            if entries.dtype != bool:
                entries = entries.fillna(False).astype(bool)
                logger.debug("Converted entry signals to boolean format")
                
            # Normalize exit signals  
            if exits.dtype != bool:
                exits = exits.fillna(False).astype(bool)
                logger.debug("Converted exit signals to boolean format")
            
            # Basic validation
            if len(entries) != len(exits):
                raise ValueError("Entry and exit signals must have same length")
                
            if len(entries) == 0:
                raise ValueError("Signal series cannot be empty")
            
            entry_count = entries.sum()
            exit_count = exits.sum()
            
            logger.info(
                "Signals validated",
                entries=int(entry_count),
                exits=int(exit_count),
                total_periods=len(entries)
            )
            
            return entries, exits
            
        except Exception as e:
            logger.error("Signal validation failed", error=str(e))
            raise ValueError(f"Invalid signal format: {str(e)}") from e