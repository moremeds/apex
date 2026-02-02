"""
Aggregator for computing trial-level statistics from run results.

Uses robust statistics:
- Median for central tendency (resistant to outliers)
- MAD (Median Absolute Deviation) for dispersion
- Percentiles for constraint checking
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field

from ..core import RunResult, RunStatus, TrialAggregates, TrialResult


class AggregationConfig(BaseModel):
    """Configuration for result aggregation."""

    min_runs_for_aggregation: int = Field(
        default=3, description="Minimum runs needed for aggregation"
    )
    require_oos: bool = Field(default=True, description="Require OOS runs for degradation ratio")


@dataclass
class Aggregator:
    """
    Aggregates run results to produce trial-level statistics.

    Uses robust statistics:
    - Median instead of mean (resistant to outliers)
    - MAD instead of std (robust dispersion measure)
    - Percentiles for risk-adjusted constraint checking
    """

    config: Optional[AggregationConfig] = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = AggregationConfig()

    @staticmethod
    def median(values: List[float]) -> float:
        """Calculate median of a list."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 1:
            return sorted_vals[n // 2]
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    @staticmethod
    def mad(values: List[float]) -> float:
        """
        Calculate Median Absolute Deviation.

        MAD = median(|x_i - median(x)|)

        More robust than standard deviation for measuring dispersion.
        """
        if len(values) < 2:
            return 0.0
        med = Aggregator.median(values)
        abs_devs = [abs(v - med) for v in values]
        return Aggregator.median(abs_devs)

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        """
        Calculate percentile using linear interpolation.

        Args:
            values: List of values
            p: Percentile (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        if n == 1:
            return sorted_vals[0]

        # Linear interpolation
        k = (n - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_vals[int(k)]

        return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)

    def aggregate(self, runs: List[RunResult]) -> TrialAggregates:
        """
        Aggregate run results into trial-level statistics.

        Args:
            runs: List of run results

        Returns:
            TrialAggregates with robust statistics
        """
        # Separate successful runs
        successful_runs = [r for r in runs if r.status == RunStatus.SUCCESS]
        failed_runs = [r for r in runs if r.status != RunStatus.SUCCESS]

        # Separate IS and OOS runs
        is_runs = [r for r in successful_runs if r.is_train and not r.is_oos]
        oos_runs = [r for r in successful_runs if r.is_oos]

        # If we don't have explicit IS/OOS distinction, use all as OOS
        if not is_runs and not oos_runs:
            oos_runs = successful_runs

        agg = TrialAggregates(
            total_runs=len(runs),
            successful_runs=len(successful_runs),
            failed_runs=len(failed_runs),
        )

        if not successful_runs:
            return agg

        # Extract metrics
        sharpes = [r.metrics.sharpe for r in successful_runs]
        returns = [r.metrics.total_return for r in successful_runs]
        max_dds = [r.metrics.max_drawdown for r in successful_runs]
        win_rates = [r.metrics.win_rate for r in successful_runs]
        profit_factors = [r.metrics.profit_factor for r in successful_runs]

        # Central tendency (median)
        agg.median_sharpe = self.median(sharpes)
        agg.median_return = self.median(returns)
        agg.median_max_dd = self.median(max_dds)
        agg.median_win_rate = self.median(win_rates)
        agg.median_profit_factor = self.median(profit_factors)

        # Dispersion (MAD)
        agg.mad_sharpe = self.mad(sharpes)
        agg.mad_return = self.mad(returns)
        agg.mad_max_dd = self.mad(max_dds)

        # Percentiles for constraint checking
        agg.p10_sharpe = self.percentile(sharpes, 10)
        agg.p90_sharpe = self.percentile(sharpes, 90)
        agg.p10_max_dd = self.percentile(max_dds, 10)
        agg.p90_max_dd = self.percentile(max_dds, 90)

        # IS/OOS metrics
        if is_runs:
            is_sharpes = [r.metrics.sharpe for r in is_runs]
            agg.is_median_sharpe = self.median(is_sharpes)

        if oos_runs:
            oos_sharpes = [r.metrics.sharpe for r in oos_runs]
            agg.oos_median_sharpe = self.median(oos_sharpes)

        # Degradation ratio (IS to OOS performance drop)
        if agg.is_median_sharpe != 0:
            agg.degradation_ratio = 1 - (agg.oos_median_sharpe / agg.is_median_sharpe)
        else:
            agg.degradation_ratio = 0.0

        # Stability score: consistency across windows
        # Lower MAD/Median ratio = more stable
        if agg.median_sharpe != 0:
            stability = 1 - min(1.0, abs(agg.mad_sharpe / agg.median_sharpe))
            agg.stability_score = max(0, stability)
        else:
            agg.stability_score = 0.0

        return agg

    def compute_trial_score(
        self,
        aggregates: TrialAggregates,
        weights: Optional[dict] = None,
        runs: Optional[List[RunResult]] = None,
    ) -> float:
        """
        Compute composite trial score for ranking.

        If `runs` is provided, uses symbol-equal-weight + fold-equal-weight
        scoring with concentration penalty to prevent single-symbol dominance.
        Otherwise falls back to aggregate-based scoring.

        Scoring approach (when runs provided):
        1. Group OOS runs by symbol → compute per-symbol median Sharpe
        2. Equal-weight across symbols (each symbol counts the same)
        3. Penalize if top-1 symbol contributes >50% of total positive Sharpe
        4. Penalize high drawdown and reward stability

        Args:
            aggregates: Aggregated statistics
            weights: Optional custom weights
            runs: Optional list of run results for per-symbol scoring

        Returns:
            Composite score (higher = better)
        """
        if weights is None:
            weights = {
                "oos_sharpe": 0.50,
                "stability": 0.25,
                "drawdown_protection": 0.25,
            }

        # If runs provided, use symbol-equal-weight scoring
        if runs:
            score = self._compute_robust_score(runs, aggregates, weights)
            return float(score)

        # Fallback: aggregate-based scoring
        oos_sharpe = aggregates.oos_median_sharpe or aggregates.median_sharpe
        sharpe_score = max(0, oos_sharpe)

        stability_score = aggregates.stability_score

        dd_score = max(0, 1 - aggregates.median_max_dd)

        score = (
            weights["oos_sharpe"] * sharpe_score
            + weights["stability"] * stability_score
            + weights["drawdown_protection"] * dd_score
        )

        return float(score)

    def _compute_robust_score(
        self,
        runs: List[RunResult],
        aggregates: TrialAggregates,
        weights: dict,
    ) -> float:
        """
        Symbol-equal-weight, fold-equal-weight scoring with concentration penalty.

        Steps:
        1. Group OOS runs by (symbol, fold) → per-cell Sharpe
        2. Per-symbol: median across folds
        3. Equal-weight mean across symbols
        4. Concentration penalty if top symbol dominates
        5. Combine with stability + DD protection
        """
        from collections import defaultdict

        # Group OOS runs by symbol
        oos_runs = [r for r in runs if r.is_oos]
        if not oos_runs:
            oos_runs = [r for r in runs if not r.is_train]
        if not oos_runs:
            oos_runs = runs  # fallback

        symbol_sharpes: dict[str, list[float]] = defaultdict(list)
        for r in oos_runs:
            if r.status == RunStatus.SUCCESS:
                symbol_sharpes[r.symbol].append(r.metrics.sharpe)

        if not symbol_sharpes:
            return 0.0

        # Per-symbol median Sharpe (fold-equal-weight within symbol)
        per_symbol_sharpe = {sym: self.median(sharpes) for sym, sharpes in symbol_sharpes.items()}

        # Equal-weight mean across symbols
        sym_values = list(per_symbol_sharpe.values())
        n_symbols = len(sym_values)
        mean_sharpe = sum(sym_values) / n_symbols if n_symbols > 0 else 0.0

        # Concentration penalty: if top symbol contributes >50% of positive Sharpe sum
        positive_sharpes = [s for s in sym_values if s > 0]
        total_positive = sum(positive_sharpes) if positive_sharpes else 0.0
        concentration_penalty = 0.0
        if total_positive > 0 and positive_sharpes:
            max_contrib = max(positive_sharpes) / total_positive
            if max_contrib > 0.5:
                # Penalty scales with dominance: 50% → 0, 100% → 0.5
                concentration_penalty = (max_contrib - 0.5) * 1.0

        # Breadth bonus: fraction of symbols with positive Sharpe
        breadth = len(positive_sharpes) / n_symbols if n_symbols > 0 else 0.0

        # Sharpe component (can be negative, but we clip at -0.5)
        sharpe_score = max(-0.5, mean_sharpe) - concentration_penalty

        # Stability and DD from aggregates
        stability_score = aggregates.stability_score
        dd_score = max(0, 1 - aggregates.median_max_dd)

        # Composite with breadth bonus
        score = (
            weights["oos_sharpe"] * sharpe_score
            + weights["stability"] * stability_score
            + weights["drawdown_protection"] * dd_score
            + 0.10 * breadth  # bonus for broad edge
        )

        return float(score)

    def aggregate_trial(
        self,
        trial_id: str,
        experiment_id: str,
        params: dict,
        runs: List[RunResult],
    ) -> TrialResult:
        """
        Create a complete TrialResult with aggregated statistics.

        Args:
            trial_id: Trial identifier
            experiment_id: Parent experiment ID
            params: Trial parameters
            runs: List of run results

        Returns:
            Complete TrialResult
        """
        aggregates = self.aggregate(runs)
        trial_score = self.compute_trial_score(aggregates, runs=runs)

        return TrialResult(
            trial_id=trial_id,
            experiment_id=experiment_id,
            params=params,
            aggregates=aggregates,
            trial_score=trial_score,
            runs=runs,
        )
