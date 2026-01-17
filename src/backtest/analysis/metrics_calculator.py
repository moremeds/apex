"""
Metrics Calculator for computing comprehensive backtest metrics.

Computes all extended metrics from returns and trade data:
- Tail Risk: VaR, CVaR, skewness, kurtosis
- Stability: Ulcer index, pain index, recovery factor
- Statistical: t-stat, Jarque-Bera normality test
- Time-based: Monthly/yearly aggregates, rolling Sharpe
- Benchmark: Alpha, beta, information ratio
- Trading Extended: Consecutive wins/losses, MAE/MFE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging_setup import get_logger

from ..core.run_result import RunMetrics

logger = get_logger(__name__)


# Annualization constants
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12


@dataclass
class Trade:
    """Simplified trade record for metrics calculation."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    return_pct: float
    bars_held: int
    mae_pct: float = 0.0  # Maximum Adverse Excursion
    mfe_pct: float = 0.0  # Maximum Favorable Excursion


class MetricsCalculator:
    """
    Compute comprehensive metrics from returns and trade data.

    Example:
        calc = MetricsCalculator(risk_free_rate=0.02)
        metrics = calc.compute_all(
            returns=daily_returns_series,
            trades=list_of_trades,
            benchmark_returns=spy_returns,  # optional
        )
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        rolling_window: int = 126,  # ~6 months for rolling metrics
    ):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 0)
            rolling_window: Window for rolling metrics (default 126 days)
        """
        self.risk_free_rate = risk_free_rate
        self.rolling_window = rolling_window
        self._daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    def compute_all(
        self,
        returns: pd.Series,
        trades: Optional[List[Trade]] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> RunMetrics:
        """
        Compute all metrics from returns and optional trade data.

        Args:
            returns: Daily returns series (indexed by date)
            trades: Optional list of Trade objects
            benchmark_returns: Optional benchmark returns for alpha/beta

        Returns:
            RunMetrics with all fields populated
        """
        metrics = RunMetrics()

        # Clean returns
        returns = returns.dropna()
        if len(returns) < 2:
            logger.warning("Insufficient data for metrics calculation")
            return metrics

        # Basic return metrics
        self._compute_return_metrics(metrics, returns)

        # Risk-adjusted metrics
        self._compute_risk_adjusted_metrics(metrics, returns)

        # Drawdown metrics
        self._compute_drawdown_metrics(metrics, returns)

        # Tail risk metrics
        self._compute_tail_risk_metrics(metrics, returns)

        # Stability metrics
        self._compute_stability_metrics(metrics, returns)

        # Statistical metrics
        self._compute_statistical_metrics(metrics, returns)

        # Time-based metrics
        self._compute_time_based_metrics(metrics, returns)

        # Benchmark metrics (if benchmark provided)
        if benchmark_returns is not None:
            self._compute_benchmark_metrics(metrics, returns, benchmark_returns)

        # Trade metrics (if trades provided)
        if trades:
            self._compute_trade_metrics(metrics, trades)

        return metrics

    # =========================================================================
    # RETURN METRICS
    # =========================================================================

    def _compute_return_metrics(self, metrics: RunMetrics, returns: pd.Series) -> None:
        """Compute basic return metrics."""
        # Total return (compounded)
        cumulative = (1 + returns).cumprod()
        metrics.total_return = cumulative.iloc[-1] - 1

        # Annualized return
        n_days = len(returns)
        if n_days > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (
                TRADING_DAYS_PER_YEAR / n_days
            ) - 1

        # CAGR
        n_years = n_days / TRADING_DAYS_PER_YEAR
        if n_years > 0:
            metrics.cagr = (1 + metrics.total_return) ** (1 / n_years) - 1

    # =========================================================================
    # RISK-ADJUSTED METRICS
    # =========================================================================

    def _compute_risk_adjusted_metrics(self, metrics: RunMetrics, returns: pd.Series) -> None:
        """Compute Sharpe, Sortino, Calmar."""
        excess_returns = returns - self._daily_rf

        # Sharpe ratio (annualized)
        if returns.std() > 0:
            metrics.sharpe = excess_returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            downside_std = downside_returns.std()
            metrics.sortino = excess_returns.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Calmar ratio (CAGR / max drawdown)
        if metrics.max_drawdown > 0:
            metrics.calmar = metrics.cagr / metrics.max_drawdown

    # =========================================================================
    # DRAWDOWN METRICS
    # =========================================================================

    def _compute_drawdown_metrics(self, metrics: RunMetrics, returns: pd.Series) -> None:
        """Compute drawdown-related metrics."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        # Max drawdown (as positive fraction)
        metrics.max_drawdown = abs(drawdown.min())

        # Average drawdown
        metrics.avg_drawdown = abs(drawdown.mean())

        # Max drawdown duration
        is_underwater = drawdown < 0
        if is_underwater.any():
            # Find consecutive underwater periods
            underwater_groups = (~is_underwater).cumsum().where(is_underwater)
            if underwater_groups.notna().any():
                durations = underwater_groups.groupby(underwater_groups).count()
                metrics.max_dd_duration_days = int(durations.max())

    # =========================================================================
    # TAIL RISK METRICS
    # =========================================================================

    def _compute_tail_risk_metrics(self, metrics: RunMetrics, returns: pd.Series) -> None:
        """Compute VaR, CVaR, skewness, kurtosis."""
        # VaR (Value at Risk) - loss at given percentile
        metrics.var_95 = abs(np.percentile(returns, 5))
        metrics.var_99 = abs(np.percentile(returns, 1))

        # CVaR (Conditional VaR / Expected Shortfall)
        var_95_threshold = np.percentile(returns, 5)
        var_99_threshold = np.percentile(returns, 1)
        tail_95 = returns[returns <= var_95_threshold]
        tail_99 = returns[returns <= var_99_threshold]

        if len(tail_95) > 0:
            metrics.cvar_95 = abs(tail_95.mean())
        if len(tail_99) > 0:
            metrics.cvar_99 = abs(tail_99.mean())

        # Tail ratio (right tail / left tail)
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        if abs(p5) > 0:
            metrics.tail_ratio = abs(p95 / p5)

        # Skewness and kurtosis
        metrics.skewness = float(returns.skew())
        metrics.kurtosis = float(returns.kurtosis())  # Excess kurtosis

    # =========================================================================
    # STABILITY METRICS
    # =========================================================================

    def _compute_stability_metrics(self, metrics: RunMetrics, returns: pd.Series) -> None:
        """Compute Ulcer index, pain index, recovery factor."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        # Ulcer index: sqrt(mean(drawdown^2))
        # Penalizes both depth and duration of drawdowns
        metrics.ulcer_index = np.sqrt((drawdown**2).mean())

        # Pain index: mean(|drawdown|)
        metrics.pain_index = abs(drawdown).mean()

        # Recovery factor: total return / max drawdown
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = metrics.total_return / metrics.max_drawdown

        # Serenity index: (annualized return - rf) / ulcer index
        if metrics.ulcer_index > 0:
            excess_ann_return = metrics.annualized_return - self.risk_free_rate
            metrics.serenity_index = excess_ann_return / (
                metrics.ulcer_index * np.sqrt(TRADING_DAYS_PER_YEAR)
            )

    # =========================================================================
    # STATISTICAL METRICS
    # =========================================================================

    def _compute_statistical_metrics(self, metrics: RunMetrics, returns: pd.Series) -> None:
        """Compute t-stat, p-value, Jarque-Bera, autocorrelation."""
        # t-test on mean return (H0: mean = 0)
        if len(returns) > 1:
            tstat, pvalue = stats.ttest_1samp(returns, 0)
            metrics.returns_tstat = float(tstat)
            metrics.returns_pvalue = float(pvalue)

        # Jarque-Bera normality test
        if len(returns) >= 8:  # Need minimum samples
            try:
                jb_stat, jb_pvalue = stats.jarque_bera(returns)
                metrics.jarque_bera_stat = float(jb_stat)
                metrics.jarque_bera_pvalue = float(jb_pvalue)
            except Exception:
                pass  # Keep defaults if test fails

        # Autocorrelation at lag 1
        clean_returns = returns.dropna()
        if len(clean_returns) > 1:
            shifted = clean_returns.shift(1)
            aligned = pd.concat([clean_returns, shifted], axis=1).dropna()
            if len(aligned) > 1:
                x = aligned.iloc[:, 0]
                y = aligned.iloc[:, 1]
                if x.std() > 0 and y.std() > 0:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        corr = np.corrcoef(x, y)[0, 1]
                    if np.isfinite(corr):
                        metrics.autocorr_lag1 = float(corr)

    # =========================================================================
    # TIME-BASED METRICS
    # =========================================================================

    def _compute_time_based_metrics(self, metrics: RunMetrics, returns: pd.Series) -> None:
        """Compute monthly/yearly aggregates and rolling Sharpe."""
        # Monthly returns
        if hasattr(returns.index, "to_period"):
            try:
                monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
                if len(monthly) > 0:
                    metrics.best_month_return = float(monthly.max())
                    metrics.worst_month_return = float(monthly.min())
                    metrics.monthly_win_rate = float((monthly > 0).mean())
            except Exception:
                pass

            # Yearly returns
            try:
                yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
                if len(yearly) > 0:
                    metrics.best_year_return = float(yearly.max())
                    metrics.worst_year_return = float(yearly.min())
                    metrics.yearly_win_rate = float((yearly > 0).mean())
            except Exception:
                pass

        # Rolling Sharpe ratio
        if len(returns) >= self.rolling_window:
            rolling_mean = returns.rolling(self.rolling_window).mean()
            rolling_std = returns.rolling(self.rolling_window).std()
            rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)).dropna()

            if len(rolling_sharpe) > 0:
                metrics.rolling_sharpe_min = float(rolling_sharpe.min())
                metrics.rolling_sharpe_max = float(rolling_sharpe.max())
                metrics.rolling_sharpe_std = float(rolling_sharpe.std())

    # =========================================================================
    # BENCHMARK METRICS
    # =========================================================================

    def _compute_benchmark_metrics(
        self,
        metrics: RunMetrics,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> None:
        """Compute alpha, beta, information ratio, tracking error."""
        # Align returns
        aligned = pd.DataFrame(
            {
                "strategy": returns,
                "benchmark": benchmark_returns,
            }
        ).dropna()

        if len(aligned) < 10:
            return

        strat = aligned["strategy"]
        bench = aligned["benchmark"]

        # Beta (regression slope)
        if bench.std() > 0:
            covariance = np.cov(strat, bench)[0, 1]
            metrics.beta = covariance / bench.var()

        # Alpha (Jensen's alpha, annualized)
        expected_return = self._daily_rf + metrics.beta * (bench.mean() - self._daily_rf)
        daily_alpha = strat.mean() - expected_return
        metrics.alpha = daily_alpha * TRADING_DAYS_PER_YEAR

        # Tracking error and Information Ratio
        active_returns = strat - bench
        metrics.tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        if metrics.tracking_error > 0:
            excess_return = (strat.mean() - bench.mean()) * TRADING_DAYS_PER_YEAR
            metrics.information_ratio = excess_return / metrics.tracking_error

        # Up/Down capture ratios
        up_days = bench > 0
        down_days = bench < 0

        if up_days.sum() > 0:
            up_strat = (1 + strat[up_days]).prod() - 1
            up_bench = (1 + bench[up_days]).prod() - 1
            if up_bench > 0:
                metrics.up_capture = up_strat / up_bench

        if down_days.sum() > 0:
            down_strat = (1 + strat[down_days]).prod() - 1
            down_bench = (1 + bench[down_days]).prod() - 1
            if down_bench < 0:
                metrics.down_capture = down_strat / down_bench

    # =========================================================================
    # TRADE METRICS
    # =========================================================================

    def _compute_trade_metrics(self, metrics: RunMetrics, trades: List[Trade]) -> None:
        """Compute trade-level metrics."""
        if not trades:
            return

        returns = [t.return_pct for t in trades]
        bars = [t.bars_held for t in trades]
        maes = [t.mae_pct for t in trades]
        mfes = [t.mfe_pct for t in trades]

        # Basic trade stats
        metrics.total_trades = len(trades)
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r <= 0]

        if metrics.total_trades > 0:
            metrics.win_rate = len(winners) / metrics.total_trades

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss

        # Expectancy
        metrics.expectancy = np.mean(returns)

        # SQN (System Quality Number)
        if len(returns) > 1:
            std_returns = np.std(returns)
            if std_returns > 0:
                metrics.sqn = (np.mean(returns) / std_returns) * np.sqrt(len(returns))

        # Best/worst trades
        metrics.best_trade_pct = max(returns)
        metrics.worst_trade_pct = min(returns)

        # Average win/loss
        if winners:
            metrics.avg_win_pct = np.mean(winners)
        if losers:
            metrics.avg_loss_pct = np.mean(losers)

        # Trade duration
        if bars:
            metrics.avg_bars_in_trade = np.mean(bars)
            metrics.max_bars_in_trade = max(bars)
            metrics.min_bars_in_trade = min(bars)
            metrics.avg_trade_duration_days = metrics.avg_bars_in_trade  # Assuming daily bars

        # MAE/MFE
        if any(m != 0 for m in maes):
            metrics.avg_mae_pct = np.mean(maes)
        if any(m != 0 for m in mfes):
            metrics.avg_mfe_pct = np.mean(mfes)

        # Edge ratio (MFE / MAE)
        if metrics.avg_mae_pct > 0:
            metrics.edge_ratio = metrics.avg_mfe_pct / metrics.avg_mae_pct

        # Consecutive wins/losses
        metrics.max_consecutive_wins = self._max_consecutive(returns, winning=True)
        metrics.max_consecutive_losses = self._max_consecutive(returns, winning=False)

    def _max_consecutive(self, returns: List[float], winning: bool) -> int:
        """Calculate max consecutive wins or losses."""
        max_streak = 0
        current_streak = 0

        for r in returns:
            is_win = r > 0
            if is_win == winning:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    # =========================================================================
    # REGIME SENSITIVITY METRICS (Phase 2)
    # =========================================================================

    def compute_regime_metrics(
        self,
        metrics: RunMetrics,
        regime_series: pd.Series,
    ) -> None:
        """
        Compute regime sensitivity metrics from a series of regime classifications.

        Args:
            metrics: RunMetrics object to populate
            regime_series: Series of regime values (R0, R1, R2, R3) indexed by bar timestamp

        Metrics computed:
            - regime_transition_rate: Regime switches per 100 bars (lower = more stable)
            - regime_time_in_r0/r1/r2/r3: % time in each regime
            - regime_avg_duration_bars: Average bars per regime stint
        """
        if regime_series is None or len(regime_series) < 2:
            return

        # Convert to string values for consistent comparison
        regimes = regime_series.astype(str)
        n_bars = len(regimes)

        # Count transitions (regime changes)
        transitions = (regimes != regimes.shift(1)).sum() - 1  # Subtract 1 for first bar
        transitions = max(0, transitions)

        # Transition rate per 100 bars
        if n_bars > 0:
            metrics.regime_transition_rate = (transitions / n_bars) * 100

        # Time in each regime
        regime_counts = regimes.value_counts(normalize=True)
        metrics.regime_time_in_r0 = regime_counts.get("R0", 0.0) * 100
        metrics.regime_time_in_r1 = regime_counts.get("R1", 0.0) * 100
        metrics.regime_time_in_r2 = regime_counts.get("R2", 0.0) * 100
        metrics.regime_time_in_r3 = regime_counts.get("R3", 0.0) * 100

        # Average regime duration
        # Count number of regime stints (consecutive periods in same regime)
        regime_change_mask = regimes != regimes.shift(1)
        n_stints = regime_change_mask.sum()
        if n_stints > 0:
            metrics.regime_avg_duration_bars = n_bars / n_stints

    def compute_regime_switch_lag(
        self,
        predicted_regimes: pd.Series,
        ground_truth_regimes: pd.Series,
    ) -> float:
        """
        Compute median lag between predicted and ground truth regime changes.

        This measures detection delay - how many bars after the ground truth
        regime changed did the predicted regime catch up.

        Args:
            predicted_regimes: Series of predicted regime values
            ground_truth_regimes: Series of "true" regime values (e.g., from lookback analysis)

        Returns:
            Median switch lag in bars (lower = more responsive)
        """
        if predicted_regimes is None or ground_truth_regimes is None:
            return 0.0

        # Align series
        common_idx = predicted_regimes.index.intersection(ground_truth_regimes.index)
        if len(common_idx) < 2:
            return 0.0

        pred = predicted_regimes.loc[common_idx].astype(str)
        truth = ground_truth_regimes.loc[common_idx].astype(str)

        # Find ground truth regime changes
        truth_changes = truth != truth.shift(1)
        change_indices = truth_changes[truth_changes].index.tolist()

        if not change_indices:
            return 0.0

        # For each ground truth change, find when predicted caught up
        lags = []
        for change_idx in change_indices:
            new_regime = truth.loc[change_idx]
            # Look forward from change point to find when predicted matches
            future_pred = pred.loc[change_idx:]
            match_mask = future_pred == new_regime
            if match_mask.any():
                match_idx = match_mask.idxmax()
                lag_bars = pred.index.get_loc(match_idx) - pred.index.get_loc(change_idx)
                lags.append(lag_bars)

        if lags:
            return float(np.median(lags))
        return 0.0

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def returns_from_equity(equity: pd.Series) -> pd.Series:
        """Convert equity curve to returns series."""
        return equity.pct_change().dropna()

    @staticmethod
    def returns_from_prices(prices: pd.Series) -> pd.Series:
        """Convert price series to returns."""
        return prices.pct_change().dropna()
