"""
Statistical validation methods for detecting overfitting.

References:
- Lopez de Prado, "The Probability of Backtest Overfitting"
- Bailey & Lopez de Prado, "The Deflated Sharpe Ratio"
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math

import numpy as np
import pandas as pd


@dataclass
class PBOCalculator:
    """
    Calculator for Probability of Backtest Overfit (PBO).

    PBO measures the probability that the best in-sample strategy
    will underperform the median out-of-sample.

    A high PBO (> 0.5) suggests significant overfitting risk.

    Reference: Bailey et al. (2014), "The Probability of Backtest Overfitting"
    """

    n_simulations: int = 1000
    random_seed: Optional[int] = 42

    def calculate(
        self,
        is_sharpes: List[float],
        oos_sharpes: List[float],
    ) -> float:
        """
        Calculate Probability of Backtest Overfit.

        Uses CSCV (Combinatorially Symmetric Cross-Validation) approach:
        For each partition, check if IS-optimal underperforms OOS median.

        Args:
            is_sharpes: In-sample Sharpe ratios for each strategy
            oos_sharpes: Out-of-sample Sharpe ratios for same strategies

        Returns:
            PBO value between 0 and 1 (higher = more overfitting)
        """
        if len(is_sharpes) != len(oos_sharpes) or len(is_sharpes) < 2:
            return 0.0

        n = len(is_sharpes)

        # Find best IS performer
        best_is_idx = np.argmax(is_sharpes)
        best_is_oos = oos_sharpes[best_is_idx]

        # Compare to OOS median
        oos_median = np.median(oos_sharpes)

        # Simple PBO: probability best IS underperforms OOS median
        # Using simulation for robustness
        rng = np.random.default_rng(self.random_seed)

        underperform_count = 0
        for _ in range(self.n_simulations):
            # Bootstrap resample
            indices = rng.choice(n, size=n, replace=True)
            resampled_is = [is_sharpes[i] for i in indices]
            resampled_oos = [oos_sharpes[i] for i in indices]

            # Find best in resampled IS
            best_idx = np.argmax(resampled_is)
            best_oos = resampled_oos[best_idx]

            # Check if underperforms median
            if best_oos < np.median(resampled_oos):
                underperform_count += 1

        return underperform_count / self.n_simulations


@dataclass
class DSRCalculator:
    """
    Calculator for Deflated Sharpe Ratio (DSR).

    DSR adjusts the Sharpe ratio for multiple testing bias:
    - More strategies tested → higher bar for significance
    - Shorter track record → higher bar for significance
    - Non-normal returns → adjust variance estimate

    Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
    """

    def expected_max_sharpe(self, n_trials: int, sharpe_std: float = 1.0) -> float:
        """
        Calculate expected maximum Sharpe ratio under null hypothesis.

        Per Bailey & Lopez de Prado (2014):
        E[max(Z)] = (1-γ)*Φ^-1(1-1/N) + γ*Φ^-1(1-1/(N*e))

        where γ is the Euler-Mascheroni constant (~0.5772).

        Args:
            n_trials: Number of independent strategies tested
            sharpe_std: Standard deviation of Sharpe estimates (default 1 for standardized)

        Returns:
            Expected maximum Sharpe under null (no skill)
        """
        from scipy import stats

        if n_trials <= 1:
            return 0.0

        euler = 0.5772156649015329  # Euler-Mascheroni constant

        # Expected max of n standard normal variates
        expected_max = (
            (1 - euler) * stats.norm.ppf(1 - 1 / n_trials)
            + euler * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )

        return expected_max * sharpe_std

    def sharpe_variance(
        self,
        sharpe: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Calculate variance of Sharpe ratio estimate.

        Per Bailey & Lopez de Prado (2014):
        Var(SR) = (1 - γ₃*SR + (κ-1)/4 * SR²) / T

        where T is number of observations, γ₃ is skewness, κ is kurtosis.

        Args:
            sharpe: Sharpe ratio estimate
            n_observations: Number of return observations
            skewness: Return skewness (γ₃, default 0 for normal)
            kurtosis: Return kurtosis (κ, default 3 for normal)

        Returns:
            Variance of the Sharpe ratio estimator
        """
        if n_observations <= 1:
            return float('inf')

        # Correct formula per Bailey & Lopez de Prado
        # Note: (kurtosis - 1)/4 accounts for excess kurtosis adjustment
        variance = (
            1
            - skewness * sharpe
            + ((kurtosis - 1) / 4) * sharpe**2
        ) / n_observations

        return max(variance, 1e-10)  # Ensure positive

    def calculate(
        self,
        observed_sharpe: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> Tuple[float, float]:
        """
        Calculate Deflated Sharpe Ratio.

        DSR tests whether the observed Sharpe ratio is statistically
        significant after accounting for multiple testing.

        Args:
            observed_sharpe: Best observed Sharpe ratio (annualized)
            n_trials: Number of strategies/configurations tested
            n_observations: Number of return observations (e.g., daily returns)
            skewness: Return skewness (default 0, normal distribution)
            kurtosis: Return kurtosis (default 3, normal distribution)

        Returns:
            Tuple of (DSR probability, p-value)
            - DSR: Probability the strategy has genuine skill (0-1)
            - p-value: Probability of observing this SR by chance
        """
        from scipy import stats

        # Calculate variance of Sharpe ratio estimate
        sharpe_var = self.sharpe_variance(
            sharpe=observed_sharpe,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis,
        )
        sharpe_se = np.sqrt(sharpe_var)

        # Expected maximum Sharpe under null (strategies have no skill)
        expected_max = self.expected_max_sharpe(n_trials, sharpe_std=sharpe_se)

        # DSR statistic: how many standard errors above expected max?
        if sharpe_se <= 0:
            return 0.0, 1.0

        dsr_statistic = (observed_sharpe - expected_max) / sharpe_se

        # P-value: probability of observing this SR by chance
        p_value = 1 - stats.norm.cdf(dsr_statistic)

        # DSR: probability the strategy has genuine skill
        dsr = stats.norm.cdf(dsr_statistic)

        return dsr, p_value

    def calculate_from_returns(
        self,
        returns: np.ndarray,
        n_trials: int,
        annualization_factor: float = 252,
    ) -> Tuple[float, float]:
        """
        Calculate DSR directly from return series.

        Convenience method that computes Sharpe, skewness, kurtosis from returns.

        Args:
            returns: Array of returns (daily, monthly, etc.)
            n_trials: Number of strategies tested
            annualization_factor: Factor to annualize Sharpe (252 for daily)

        Returns:
            Tuple of (DSR probability, p-value)
        """
        from scipy import stats as sp_stats

        if len(returns) < 2:
            return 0.0, 1.0

        # Calculate Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0, 1.0

        sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)

        # Calculate higher moments
        skewness = sp_stats.skew(returns)
        kurtosis = sp_stats.kurtosis(returns, fisher=False)  # Excess kurtosis + 3

        return self.calculate(
            observed_sharpe=sharpe,
            n_trials=n_trials,
            n_observations=len(returns),
            skewness=skewness,
            kurtosis=kurtosis,
        )


@dataclass
class MonteCarloSimulator:
    """
    Monte Carlo simulator for trade reshuffling analysis.

    Tests whether equity curve could have arisen by chance by
    randomly reshuffling trades and comparing results.
    """

    n_simulations: int = 1000
    seed: Optional[int] = 42

    def reshuffle_trades(
        self, trades: pd.DataFrame, initial_equity: float = 10000
    ) -> "MonteCarloResult":
        """
        Reshuffle trades to generate confidence bands.

        Args:
            trades: DataFrame with 'pnl' column
            initial_equity: Starting equity

        Returns:
            MonteCarloResult with simulation statistics
        """
        rng = np.random.default_rng(self.seed)
        pnls = trades["pnl"].values

        if len(pnls) == 0:
            return MonteCarloResult(
                original_total_return=0,
                p_value=1.0,
                equity_p5=np.array([initial_equity]),
                equity_p50=np.array([initial_equity]),
                equity_p95=np.array([initial_equity]),
            )

        # Original equity curve
        original_equity = initial_equity + np.cumsum(pnls)
        original_return = (original_equity[-1] - initial_equity) / initial_equity

        # Simulate reshuffled curves
        all_curves = []
        for _ in range(self.n_simulations):
            shuffled = rng.permutation(pnls)
            curve = initial_equity + np.cumsum(shuffled)
            all_curves.append(curve)

        all_curves = np.array(all_curves)

        # Calculate percentiles at each point
        equity_p5 = np.percentile(all_curves, 5, axis=0)
        equity_p50 = np.percentile(all_curves, 50, axis=0)
        equity_p95 = np.percentile(all_curves, 95, axis=0)

        # P-value: fraction of simulations with better return
        final_returns = (all_curves[:, -1] - initial_equity) / initial_equity
        p_value = np.mean(final_returns >= original_return)

        return MonteCarloResult(
            original_total_return=original_return,
            p_value=p_value,
            equity_p5=equity_p5,
            equity_p50=equity_p50,
            equity_p95=equity_p95,
        )


@dataclass
class MonteCarloResult:
    """Result from Monte Carlo simulation."""

    original_total_return: float
    p_value: float
    equity_p5: np.ndarray
    equity_p50: np.ndarray
    equity_p95: np.ndarray

    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant at 5% level."""
        return self.p_value < 0.05
