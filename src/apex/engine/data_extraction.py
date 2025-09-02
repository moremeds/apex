"""Data extraction utilities for vectorbt backtesting engine."""

from __future__ import annotations

from typing import Optional

import polars as pl
import structlog
import vectorbt as vbt

logger = structlog.get_logger(__name__)


class VectorbtDataExtractor:
    """Extracts and converts data from vectorbt portfolio objects."""

    @staticmethod
    def extract_trades_data(portfolio: vbt.Portfolio) -> Optional[pl.DataFrame]:
        """Extract trades data from vectorbt portfolio.
        
        Converts vectorbt portfolio trades into a polars DataFrame with
        standardized column names and proper type conversion.
        
        Args:
            portfolio: Vectorbt Portfolio object containing trade data
            
        Returns:
            Optional[pl.DataFrame]: DataFrame with trade data or None if no trades
            
        Raises:
            None: All exceptions are handled and logged gracefully
        """
        try:
            if portfolio.orders.count() == 0:
                logger.debug("No orders found in portfolio")
                return None
                
            trades = portfolio.trades.records_readable
            if trades.empty:
                logger.debug("No trades found in portfolio")
                return None
                
            # Convert to polars-compatible format
            trades_data = []
            for _, trade in trades.iterrows():
                trade_record = {
                    "entry_time": trade.get("Entry Timestamp"),
                    "exit_time": trade.get("Exit Timestamp"),
                    "entry_price": float(trade.get("Entry Price", 0)),
                    "exit_price": float(trade.get("Exit Price", 0)),
                    "size": float(trade.get("Size", 0)),
                    "pnl": float(trade.get("PnL", 0)),
                    "return": float(trade.get("Return", 0))
                }
                trades_data.append(trade_record)
            
            if not trades_data:
                logger.debug("No trade records to process")
                return None
                
            result_df = pl.DataFrame(trades_data)
            logger.debug(f"Extracted {len(result_df)} trade records")
            return result_df
            
        except Exception as e:
            logger.warning("Failed to extract trades data", error=str(e))
            return None

    @staticmethod
    def extract_positions_data(portfolio: vbt.Portfolio) -> Optional[pl.DataFrame]:
        """Extract positions data from vectorbt portfolio.
        
        Converts vectorbt portfolio positions into a polars DataFrame with
        standardized column names and proper type conversion.
        
        Args:
            portfolio: Vectorbt Portfolio object containing position data
            
        Returns:
            Optional[pl.DataFrame]: DataFrame with position data or None if no positions
            
        Raises:
            None: All exceptions are handled and logged gracefully
        """
        try:
            positions = portfolio.positions.records_readable
            if positions.empty:
                logger.debug("No positions found in portfolio")
                return None
                
            # Convert to polars-compatible format
            positions_data = []
            for _, position in positions.iterrows():
                position_record = {
                    "start_time": position.get("Entry Timestamp"),
                    "end_time": position.get("Exit Timestamp"), 
                    "size": float(position.get("Size", 0)),
                    "entry_price": float(position.get("Entry Price", 0)),
                    "exit_price": float(position.get("Exit Price", 0)),
                    "pnl": float(position.get("PnL", 0))
                }
                positions_data.append(position_record)
            
            if not positions_data:
                logger.debug("No position records to process")
                return None
                
            result_df = pl.DataFrame(positions_data)
            logger.debug(f"Extracted {len(result_df)} position records")
            return result_df
            
        except Exception as e:
            logger.warning("Failed to extract positions data", error=str(e))
            return None

    @staticmethod
    def convert_to_polars(
        pandas_df, 
        column_names: list[str]
    ) -> pl.DataFrame:
        """Convert pandas DataFrame to polars with specified column names.
        
        Safely converts pandas DataFrames to polars format while ensuring
        proper column naming and handling empty DataFrames.
        
        Args:
            pandas_df: Pandas DataFrame to convert
            column_names: List of desired column names
            
        Returns:
            pl.DataFrame: Converted polars DataFrame
            
        Raises:
            None: Handles conversion errors gracefully
        """
        try:
            if pandas_df.empty:
                logger.debug("Creating empty DataFrame with specified columns")
                return pl.DataFrame({name: [] for name in column_names})
            
            # Ensure we have the right column names
            pandas_df = pandas_df.copy()  # Avoid modifying original
            pandas_df.columns = column_names[:len(pandas_df.columns)]
            
            result = pl.from_pandas(pandas_df)
            logger.debug(
                f"Converted pandas DataFrame to polars",
                rows=len(result),
                columns=len(result.columns)
            )
            return result
            
        except Exception as e:
            logger.error("Failed to convert pandas DataFrame to polars", error=str(e))
            # Return empty DataFrame as fallback
            return pl.DataFrame({name: [] for name in column_names})