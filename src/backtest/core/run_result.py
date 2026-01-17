"""
Run result - outcome of a single backtest execution.

Contains all metrics calculated from a single run, including:
- Performance metrics (return, CAGR, Sharpe)
- Risk metrics (drawdown, volatility)
- Trade metrics (win rate, profit factor)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RunStatus(str, Enum):
    """Status of a backtest run."""

    SUCCESS = "success"
    FAIL_DATA = "fail_data"  # Missing/bad data
    FAIL_STRATEGY = "fail_strategy"  # Strategy error
    FAIL_EXECUTION = "fail_execution"  # Execution error
    SKIPPED = "skipped"  # Skipped (e.g., duplicate)
    TIMEOUT = "timeout"  # Exceeded time limit


@dataclass
class RunMetrics:
    """
    Comprehensive metrics from a single backtest run.

    All metrics are calculated from the test period only
    to ensure proper out-of-sample evaluation.

    Metric Categories:
    - Returns: total_return, cagr, annualized_return
    - Risk-adjusted: sharpe, sortino, calmar
    - Drawdown: max_drawdown, avg_drawdown, max_dd_duration_days
    - Trade: win_rate, profit_factor, expectancy, sqn
    - Tail Risk: var_95, cvar_95, skewness, kurtosis
    - Stability: ulcer_index, pain_index, recovery_factor
    - Statistical: returns_tstat, jarque_bera_stat
    - Time-based: best/worst month, rolling sharpe
    - Benchmark: alpha, beta, information_ratio
    - Trading Extended: max consecutive wins/losses, edge_ratio
    """

    # =========================================================================
    # RETURN METRICS
    # =========================================================================
    total_return: float = 0.0  # (final - initial) / initial
    cagr: float = 0.0  # Compound annual growth rate
    annualized_return: float = 0.0  # Simple annualized return

    # =========================================================================
    # RISK-ADJUSTED METRICS
    # =========================================================================
    sharpe: float = 0.0  # Sharpe ratio (annualized)
    sortino: float = 0.0  # Sortino ratio (downside deviation)
    calmar: float = 0.0  # CAGR / max drawdown

    # =========================================================================
    # DRAWDOWN METRICS
    # =========================================================================
    max_drawdown: float = 0.0  # Maximum drawdown (as positive fraction)
    avg_drawdown: float = 0.0  # Average drawdown
    max_dd_duration_days: int = 0  # Max drawdown duration in days

    # =========================================================================
    # TRADE METRICS
    # =========================================================================
    total_trades: int = 0
    win_rate: float = 0.0  # Winning trades / total trades
    profit_factor: float = 0.0  # Gross profit / gross loss
    expectancy: float = 0.0  # Average P&L per trade
    sqn: float = 0.0  # System Quality Number

    # Trade details
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    # =========================================================================
    # EXPOSURE METRICS
    # =========================================================================
    exposure_pct: float = 0.0  # Time in market
    avg_trade_duration_days: float = 0.0

    # =========================================================================
    # COST METRICS
    # =========================================================================
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0

    # =========================================================================
    # TAIL RISK METRICS (NEW)
    # =========================================================================
    var_95: float = 0.0  # Value at Risk (95%) - daily
    var_99: float = 0.0  # Value at Risk (99%) - daily
    cvar_95: float = 0.0  # Conditional VaR / Expected Shortfall (95%)
    cvar_99: float = 0.0  # Conditional VaR / Expected Shortfall (99%)
    tail_ratio: float = 0.0  # P95/P5 ratio (right tail vs left tail)
    skewness: float = 0.0  # Return distribution skewness
    kurtosis: float = 0.0  # Return distribution excess kurtosis

    # =========================================================================
    # STABILITY METRICS (NEW)
    # =========================================================================
    ulcer_index: float = 0.0  # sqrt(mean(drawdown^2)) - penalizes deep/long DDs
    pain_index: float = 0.0  # mean(drawdown) - average pain experienced
    recovery_factor: float = 0.0  # total_return / max_drawdown
    serenity_index: float = 0.0  # (return - rf) / ulcer_index

    # =========================================================================
    # STATISTICAL METRICS (NEW)
    # =========================================================================
    returns_tstat: float = 0.0  # t-statistic on mean return
    returns_pvalue: float = 1.0  # p-value for returns != 0
    jarque_bera_stat: float = 0.0  # Normality test statistic
    jarque_bera_pvalue: float = 1.0  # Normality test p-value
    autocorr_lag1: float = 0.0  # Return autocorrelation at lag 1

    # =========================================================================
    # TIME-BASED METRICS (NEW)
    # =========================================================================
    best_month_return: float = 0.0
    worst_month_return: float = 0.0
    best_year_return: float = 0.0
    worst_year_return: float = 0.0
    monthly_win_rate: float = 0.0  # % of positive months
    yearly_win_rate: float = 0.0  # % of positive years
    rolling_sharpe_min: float = 0.0  # Min 6-month rolling Sharpe
    rolling_sharpe_max: float = 0.0  # Max 6-month rolling Sharpe
    rolling_sharpe_std: float = 0.0  # Std of 6-month rolling Sharpe

    # =========================================================================
    # BENCHMARK METRICS (NEW) - computed if benchmark provided
    # =========================================================================
    alpha: float = 0.0  # Jensen's alpha (annualized)
    beta: float = 0.0  # Market beta
    information_ratio: float = 0.0  # (return - benchmark) / tracking_error
    tracking_error: float = 0.0  # Std(return - benchmark)
    up_capture: float = 0.0  # Up market capture ratio
    down_capture: float = 0.0  # Down market capture ratio

    # =========================================================================
    # TRADING EXTENDED METRICS (NEW)
    # =========================================================================
    avg_bars_in_trade: float = 0.0
    max_bars_in_trade: int = 0
    min_bars_in_trade: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_mae_pct: float = 0.0  # Avg Maximum Adverse Excursion
    avg_mfe_pct: float = 0.0  # Avg Maximum Favorable Excursion
    edge_ratio: float = 0.0  # MFE / MAE ratio (>1 is good)

    # =========================================================================
    # REGIME SENSITIVITY METRICS (Phase 2)
    # =========================================================================
    regime_transition_rate: float = 0.0  # Regime switches per 100 bars (lower = more stable)
    regime_switch_lag: float = 0.0  # Median bars delay vs ground truth (lower = more responsive)
    regime_time_in_r0: float = 0.0  # % time in R0 (Healthy Uptrend)
    regime_time_in_r1: float = 0.0  # % time in R1 (Choppy/Extended)
    regime_time_in_r2: float = 0.0  # % time in R2 (Risk-Off)
    regime_time_in_r3: float = 0.0  # % time in R3 (Rebound)
    regime_avg_duration_bars: float = 0.0  # Average bars per regime stint

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            # Returns
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annualized_return": self.annualized_return,
            # Risk-adjusted
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            # Drawdown
            "max_drawdown": self.max_drawdown,
            "avg_drawdown": self.avg_drawdown,
            "max_dd_duration_days": self.max_dd_duration_days,
            # Trade
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "sqn": self.sqn,
            "best_trade_pct": self.best_trade_pct,
            "worst_trade_pct": self.worst_trade_pct,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            # Exposure
            "exposure_pct": self.exposure_pct,
            "avg_trade_duration_days": self.avg_trade_duration_days,
            # Costs
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "total_costs": self.total_costs,
            # Tail Risk (NEW)
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "tail_ratio": self.tail_ratio,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            # Stability (NEW)
            "ulcer_index": self.ulcer_index,
            "pain_index": self.pain_index,
            "recovery_factor": self.recovery_factor,
            "serenity_index": self.serenity_index,
            # Statistical (NEW)
            "returns_tstat": self.returns_tstat,
            "returns_pvalue": self.returns_pvalue,
            "jarque_bera_stat": self.jarque_bera_stat,
            "jarque_bera_pvalue": self.jarque_bera_pvalue,
            "autocorr_lag1": self.autocorr_lag1,
            # Time-based (NEW)
            "best_month_return": self.best_month_return,
            "worst_month_return": self.worst_month_return,
            "best_year_return": self.best_year_return,
            "worst_year_return": self.worst_year_return,
            "monthly_win_rate": self.monthly_win_rate,
            "yearly_win_rate": self.yearly_win_rate,
            "rolling_sharpe_min": self.rolling_sharpe_min,
            "rolling_sharpe_max": self.rolling_sharpe_max,
            "rolling_sharpe_std": self.rolling_sharpe_std,
            # Benchmark (NEW)
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
            "tracking_error": self.tracking_error,
            "up_capture": self.up_capture,
            "down_capture": self.down_capture,
            # Trading Extended (NEW)
            "avg_bars_in_trade": self.avg_bars_in_trade,
            "max_bars_in_trade": self.max_bars_in_trade,
            "min_bars_in_trade": self.min_bars_in_trade,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_mae_pct": self.avg_mae_pct,
            "avg_mfe_pct": self.avg_mfe_pct,
            "edge_ratio": self.edge_ratio,
            # Regime Sensitivity (Phase 2)
            "regime_transition_rate": self.regime_transition_rate,
            "regime_switch_lag": self.regime_switch_lag,
            "regime_time_in_r0": self.regime_time_in_r0,
            "regime_time_in_r1": self.regime_time_in_r1,
            "regime_time_in_r2": self.regime_time_in_r2,
            "regime_time_in_r3": self.regime_time_in_r3,
            "regime_avg_duration_bars": self.regime_avg_duration_bars,
        }

    @classmethod
    def from_backtest_result(cls, result: Any) -> "RunMetrics":
        """Create RunMetrics from BacktestResult."""
        return cls(
            total_return=result.performance.total_return,
            cagr=result.performance.cagr / 100,  # Convert from percentage
            annualized_return=result.performance.annualized_return,
            sharpe=result.risk.sharpe_ratio,
            max_drawdown=result.risk.max_drawdown / 100,  # Convert from percentage
            max_dd_duration_days=result.risk.max_drawdown_duration_days,
            total_trades=result.trades.total_trades,
            win_rate=result.trades.win_rate,
            profit_factor=result.trades.profit_factor,
            exposure_pct=result.trades.exposure_pct,
            total_commission=result.costs.total_commission,
            total_slippage=result.costs.total_slippage,
            total_costs=result.costs.total_costs,
        )


@dataclass
class RunResult:
    """
    Complete result from a single backtest run.

    Includes:
    - Run identification (run_id, trial_id, experiment_id)
    - Execution metadata (timing, status)
    - Metrics (performance, risk, trade)
    - Optional equity curve and trade log
    """

    # Identification
    run_id: str
    trial_id: str
    experiment_id: str
    symbol: str
    window_id: str

    # Versioning
    profile_version: str = ""
    data_version: str = ""

    # Status
    status: RunStatus = RunStatus.SUCCESS
    error: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Metrics
    metrics: RunMetrics = field(default_factory=RunMetrics)

    # Flags for IS/OOS separation
    is_train: bool = True  # True = in-sample, False = out-of-sample
    is_oos: bool = False  # Explicitly out-of-sample

    # Optional detailed data (may be omitted for memory efficiency)
    equity_curve: Optional[List[Dict[str, Any]]] = None
    trade_log: Optional[List[Dict[str, Any]]] = None
    params: Optional[Dict[str, Any]] = None

    @property
    def is_success(self) -> bool:
        """Check if run completed successfully."""
        return self.status == RunStatus.SUCCESS

    @property
    def sharpe(self) -> float:
        """Quick access to Sharpe ratio."""
        return self.metrics.sharpe

    @property
    def total_return(self) -> float:
        """Quick access to total return."""
        return self.metrics.total_return

    @property
    def max_drawdown(self) -> float:
        """Quick access to max drawdown."""
        return self.metrics.max_drawdown

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate that the result is meaningful and not corrupted.

        HIGH-013: Prevents false-success persistence of invalid results.

        Returns:
            Tuple of (is_valid, error_message)
        """
        import math

        # Check critical metrics for NaN
        critical_metrics = [
            ("sharpe", self.metrics.sharpe),
            ("total_return", self.metrics.total_return),
            ("max_drawdown", self.metrics.max_drawdown),
        ]

        for name, value in critical_metrics:
            if math.isnan(value) or math.isinf(value):
                return False, f"Invalid {name}: {value}"

        # Check equity curve if status is SUCCESS
        if self.status == RunStatus.SUCCESS:
            if self.equity_curve is not None and len(self.equity_curve) == 0:
                return False, "Empty equity curve for successful run"

        return True, None

    def to_dict(self, include_curves: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "symbol": self.symbol,
            "window_id": self.window_id,
            "profile_version": self.profile_version,
            "data_version": self.data_version,
            "status": self.status.value,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "is_train": self.is_train,
            "is_oos": self.is_oos,
            "params": self.params,
            **self.metrics.to_dict(),
        }

        if include_curves:
            result["equity_curve"] = self.equity_curve
            result["trade_log"] = self.trade_log

        return result
