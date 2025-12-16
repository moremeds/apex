"""
BacktestResult: Formal backtest result schema with standard metrics.

Every backtest produces a BacktestResult with standardized metrics. This enables:
1. Consistent comparison across strategies
2. Persistence to database (BacktestRepository)
3. Report generation
4. CI/CD pass/fail thresholds

Standard Metrics:
- Performance: Total return, CAGR, best/worst day
- Risk: Sharpe, Sortino, Max Drawdown, Volatility
- Trades: Win rate, profit factor, average trade
- Costs: Commission, slippage, cost analysis
- Exposure: Time in market, position count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance/returns metrics."""

    total_return: float = 0.0  # (final - initial) / initial
    total_return_pct: float = 0.0  # total_return * 100
    cagr: float = 0.0  # Compound Annual Growth Rate
    annualized_return: float = 0.0  # Geometric mean
    best_day: float = 0.0  # Largest single-day gain (%)
    worst_day: float = 0.0  # Largest single-day loss (%)
    best_month: float = 0.0  # Largest monthly gain (%)
    worst_month: float = 0.0  # Largest monthly loss (%)

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "cagr": self.cagr,
            "annualized_return": self.annualized_return,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "best_month": self.best_month,
            "worst_month": self.worst_month,
        }


@dataclass
class RiskMetrics:
    """Risk and volatility metrics."""

    sharpe_ratio: float = 0.0  # Risk-adjusted return
    sortino_ratio: float = 0.0  # Downside risk-adjusted return
    calmar_ratio: float = 0.0  # CAGR / Max Drawdown
    volatility: float = 0.0  # Annualized std dev (%)
    downside_volatility: float = 0.0  # Std dev of negative returns
    max_drawdown: float = 0.0  # Peak-to-trough decline (0.20 = 20%)
    max_drawdown_duration_days: int = 0  # Days in max drawdown
    avg_drawdown: float = 0.0  # Mean drawdown
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "volatility": self.volatility,
            "downside_volatility": self.downside_volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "avg_drawdown": self.avg_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }


@dataclass
class TradeMetrics:
    """Trade-level statistics."""

    total_trades: int = 0  # Number of round-trips
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0  # winning / total (%)
    profit_factor: float = 0.0  # gross profit / gross loss
    avg_win: float = 0.0  # Mean profit on winners
    avg_loss: float = 0.0  # Mean loss on losers (negative)
    avg_trade: float = 0.0  # Mean P&L per trade
    largest_win: float = 0.0
    largest_loss: float = 0.0  # Negative
    avg_trade_duration_seconds: float = 0.0  # Mean holding period
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_trade": self.avg_trade,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_trade_duration_seconds": self.avg_trade_duration_seconds,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
        }


@dataclass
class CostMetrics:
    """Transaction cost analysis."""

    total_commission: float = 0.0  # Sum of all commissions
    total_slippage: float = 0.0  # Estimated slippage impact
    total_costs: float = 0.0  # commission + slippage
    cost_pct_of_capital: float = 0.0  # total_costs / initial_capital (%)
    cost_pct_of_gross_profit: float = 0.0  # total_costs / gross_profit (%)
    avg_commission_per_trade: float = 0.0
    avg_slippage_per_trade: float = 0.0
    commission_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "total_costs": self.total_costs,
            "cost_pct_of_capital": self.cost_pct_of_capital,
            "cost_pct_of_gross_profit": self.cost_pct_of_gross_profit,
            "avg_commission_per_trade": self.avg_commission_per_trade,
            "avg_slippage_per_trade": self.avg_slippage_per_trade,
            "commission_breakdown": self.commission_breakdown,
        }


@dataclass
class ExposureMetrics:
    """Portfolio exposure analysis."""

    avg_exposure: float = 0.0  # Mean portfolio utilization (%)
    max_exposure: float = 0.0  # Peak utilization (%)
    min_exposure: float = 0.0  # Min utilization (%)
    time_in_market: float = 0.0  # % of time with any position
    avg_position_count: float = 0.0  # Mean number of positions
    max_position_count: int = 0  # Peak number of positions
    exposure_by_symbol: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_exposure": self.avg_exposure,
            "max_exposure": self.max_exposure,
            "min_exposure": self.min_exposure,
            "time_in_market": self.time_in_market,
            "avg_position_count": self.avg_position_count,
            "max_position_count": self.max_position_count,
            "exposure_by_symbol": self.exposure_by_symbol,
        }


@dataclass
class TradeRecord:
    """Individual trade record for trade log."""

    trade_id: str = ""
    symbol: str = ""
    side: str = ""  # "LONG" or "SHORT"
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0  # Net P&L after costs
    pnl_pct: float = 0.0  # P&L as % of entry value
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def duration(self) -> timedelta:
        """Calculate trade duration."""
        if self.entry_time and self.exit_time:
            return self.exit_time - self.entry_time
        return timedelta(0)

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "commission": self.commission,
            "slippage": self.slippage,
            "duration_seconds": self.duration.total_seconds(),
        }


@dataclass
class BacktestResult:
    """
    Complete backtest result with all metrics.

    This is the standardized output from any backtest run.
    Can be persisted to database via BacktestRepository.
    """

    # Identity
    backtest_id: UUID = field(default_factory=uuid4)
    strategy_name: str = ""
    strategy_id: str = ""

    # Spec reference (for reproducibility)
    spec_file: Optional[str] = None
    spec_hash: Optional[str] = None  # SHA256 of spec for verification

    # Period
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    trading_days: int = 0

    # Capital
    initial_capital: float = 0.0
    final_capital: float = 0.0
    currency: str = "USD"

    # Universe
    symbols: List[str] = field(default_factory=list)

    # Reality model used
    reality_model_config: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    performance: Optional[PerformanceMetrics] = None
    risk: Optional[RiskMetrics] = None
    trades: Optional[TradeMetrics] = None
    costs: Optional[CostMetrics] = None
    exposure: Optional[ExposureMetrics] = None

    # Time series (for equity curve plotting)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{"date": "2024-01-02", "equity": 100500.0, "drawdown": 0.005}, ...]

    # Trade log
    trade_log: List[TradeRecord] = field(default_factory=list)

    # Metadata
    run_timestamp: datetime = field(default_factory=datetime.now)
    engine: str = "apex"  # "apex" | "backtrader"
    run_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization and database storage."""
        return {
            "backtest_id": str(self.backtest_id),
            "strategy_name": self.strategy_name,
            "strategy_id": self.strategy_id,
            "spec_file": self.spec_file,
            "spec_hash": self.spec_hash,
            "start_date": str(self.start_date) if self.start_date else None,
            "end_date": str(self.end_date) if self.end_date else None,
            "trading_days": self.trading_days,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "currency": self.currency,
            "symbols": self.symbols,
            "reality_model_config": self.reality_model_config,
            "performance": self.performance.to_dict() if self.performance else None,
            "risk": self.risk.to_dict() if self.risk else None,
            "trades": self.trades.to_dict() if self.trades else None,
            "costs": self.costs.to_dict() if self.costs else None,
            "exposure": self.exposure.to_dict() if self.exposure else None,
            "equity_curve": self.equity_curve,
            "trade_log": [t.to_dict() for t in self.trade_log],
            "run_timestamp": self.run_timestamp.isoformat(),
            "engine": self.engine,
            "run_duration_seconds": self.run_duration_seconds,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"{'=' * 60}",
            f"BACKTEST RESULT: {self.strategy_name}",
            f"{'=' * 60}",
            f"Period:           {self.start_date} to {self.end_date} ({self.trading_days} days)",
            f"Symbols:          {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}",
            f"Initial Capital:  ${self.initial_capital:,.2f}",
            f"Final Capital:    ${self.final_capital:,.2f}",
            "",
        ]

        if self.performance:
            lines.extend(
                [
                    "--- PERFORMANCE ---",
                    f"Total Return:     {self.performance.total_return_pct:+.2f}%",
                    f"CAGR:             {self.performance.cagr:.2f}%",
                    f"Best Day:         {self.performance.best_day:+.2f}%",
                    f"Worst Day:        {self.performance.worst_day:+.2f}%",
                    "",
                ]
            )

        if self.risk:
            lines.extend(
                [
                    "--- RISK ---",
                    f"Sharpe Ratio:     {self.risk.sharpe_ratio:.2f}",
                    f"Sortino Ratio:    {self.risk.sortino_ratio:.2f}",
                    f"Max Drawdown:     {self.risk.max_drawdown:.2f}%",
                    f"Volatility:       {self.risk.volatility:.2f}%",
                    "",
                ]
            )

        if self.trades:
            lines.extend(
                [
                    "--- TRADES ---",
                    f"Total Trades:     {self.trades.total_trades}",
                    f"Win Rate:         {self.trades.win_rate:.1f}%",
                    f"Profit Factor:    {self.trades.profit_factor:.2f}",
                    f"Avg Trade:        ${self.trades.avg_trade:,.2f}",
                    "",
                ]
            )

        if self.costs:
            lines.extend(
                [
                    "--- COSTS ---",
                    f"Total Commission: ${self.costs.total_commission:,.2f}",
                    f"Total Slippage:   ${self.costs.total_slippage:,.2f}",
                    f"Cost % of Capital:{self.costs.cost_pct_of_capital:.2f}%",
                    "",
                ]
            )

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to console."""
        print(self.summary())

    @property
    def is_profitable(self) -> bool:
        """Check if backtest was profitable."""
        return self.final_capital > self.initial_capital

    @property
    def total_return_pct(self) -> float:
        """Calculate total return percentage."""
        if self.initial_capital > 0:
            return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100
        return 0.0
