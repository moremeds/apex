"""Portfolio management and metrics calculation for vectorbt engine."""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
import structlog

from apex.engine.base import BacktestResult

logger = structlog.get_logger(__name__)


class PortfolioMetrics:
    """Portfolio performance metrics calculator for vectorbt backtesting."""

    @staticmethod
    def calculate_metrics(result: BacktestResult) -> BacktestResult:
        """Calculate comprehensive performance metrics for backtest result.
        
        Computes various financial metrics including Sharpe ratio, maximum drawdown,
        win rate, profit factor, and trade statistics based on portfolio performance.
        
        Args:
            result: BacktestResult object containing portfolio value and trades data
            
        Returns:
            BacktestResult: Updated result with calculated performance metrics
            
        Raises:
            None: All exceptions are handled gracefully with appropriate logging
        """
        if result.portfolio_value is None or len(result.portfolio_value) == 0:
            logger.warning("No portfolio data available for metrics calculation")
            return result

        # Convert portfolio values to numpy for calculations
        portfolio_values = result.portfolio_value.select("portfolio_value").to_numpy().flatten()
        
        if len(portfolio_values) < 2:
            logger.warning("Insufficient portfolio data for metrics calculation")
            return result

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            logger.warning("No valid returns data for metrics calculation")
            return result

        # Calculate risk-adjusted returns
        result = PortfolioMetrics._calculate_risk_metrics(result, returns)
        
        # Calculate drawdown metrics
        result = PortfolioMetrics._calculate_drawdown_metrics(result, portfolio_values)
        
        # Calculate trade statistics
        result = PortfolioMetrics._calculate_trade_statistics(result)

        return result

    @staticmethod
    def _calculate_risk_metrics(result: BacktestResult, returns: np.ndarray) -> BacktestResult:
        """Calculate risk-adjusted performance metrics.
        
        Args:
            result: BacktestResult to update with risk metrics
            returns: Array of portfolio returns
            
        Returns:
            BacktestResult: Updated result with Sharpe ratio and other risk metrics
        """
        # Sharpe Ratio (assuming daily data, annualized)
        if np.std(returns) > 0:
            result.sharpe_ratio = float(np.sqrt(252) * np.mean(returns) / np.std(returns))
        else:
            logger.warning("Zero volatility detected, cannot calculate Sharpe ratio")
            result.sharpe_ratio = None
            
        return result

    @staticmethod
    def _calculate_drawdown_metrics(result: BacktestResult, portfolio_values: np.ndarray) -> BacktestResult:
        """Calculate drawdown-related metrics.
        
        Args:
            result: BacktestResult to update with drawdown metrics
            portfolio_values: Array of portfolio values over time
            
        Returns:
            BacktestResult: Updated result with maximum drawdown
        """
        # Maximum Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        result.max_drawdown = float(np.min(drawdown))
        
        return result

    @staticmethod
    def _calculate_trade_statistics(result: BacktestResult) -> BacktestResult:
        """Calculate trade-based performance statistics.
        
        Args:
            result: BacktestResult to update with trade statistics
            
        Returns:
            BacktestResult: Updated result with win rate, profit factor, etc.
        """
        if result.trades is None or len(result.trades) == 0:
            logger.info("No trades data available for trade statistics calculation")
            return result

        trades_df = result.trades
        
        if "pnl" not in trades_df.columns:
            logger.warning("No PnL data available in trades for statistics calculation")
            return result

        # Filter winning and losing trades
        winning_trades = trades_df.filter(pl.col("pnl") > 0)
        losing_trades = trades_df.filter(pl.col("pnl") < 0)
        
        # Basic trade counts
        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        result.total_trades = len(trades_df)
        
        # Win rate calculation
        if result.total_trades > 0:
            result.win_rate = float(result.winning_trades / result.total_trades)
        
        # Average win/loss calculations
        if len(winning_trades) > 0:
            result.avg_win = float(winning_trades.select("pnl").mean().item())
        
        if len(losing_trades) > 0:
            result.avg_loss = float(losing_trades.select("pnl").mean().item())
        
        # Profit Factor calculation
        if result.avg_loss and result.avg_loss < 0 and result.avg_win:
            total_wins = result.winning_trades * result.avg_win
            total_losses = abs(result.losing_trades * result.avg_loss)
            if total_losses > 0:
                result.profit_factor = float(total_wins / total_losses)
        
        logger.debug(
            "Trade statistics calculated",
            total_trades=result.total_trades,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor
        )

        return result