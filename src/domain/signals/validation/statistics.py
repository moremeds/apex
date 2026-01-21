"""
Symbol-Level Statistics with Block Bootstrap.

Statistical validation using symbol-level aggregation (n ~ 140) instead of
bar-level (n ~ 1000s), providing actual statistical power for hypothesis testing.

Block bootstrap preserves temporal correlation for proper confidence intervals.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import stats


@dataclass
class StatisticalResult:
    """Statistical validation result with proper power."""

    # Sample sizes
    n_trending_symbols: int
    n_choppy_symbols: int

    # Per-symbol rates
    trending_r0_rates: List[float]
    choppy_r0_rates: List[float]

    # T-test results
    t_statistic: float
    p_value: float
    effect_size_cohens_d: float

    # Bootstrap confidence intervals
    trending_mean: float
    trending_ci_lower: float
    trending_ci_upper: float
    choppy_mean: float
    choppy_ci_lower: float
    choppy_ci_upper: float

    # Bootstrap details
    n_bootstrap_samples: int
    block_size_bars: int

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "n_trending_symbols": self.n_trending_symbols,
            "n_choppy_symbols": self.n_choppy_symbols,
            "trending_r0_rates": self.trending_r0_rates,
            "choppy_r0_rates": self.choppy_r0_rates,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "effect_size_cohens_d": self.effect_size_cohens_d,
            "trending_mean": self.trending_mean,
            "trending_ci_lower": self.trending_ci_lower,
            "trending_ci_upper": self.trending_ci_upper,
            "choppy_mean": self.choppy_mean,
            "choppy_ci_lower": self.choppy_ci_lower,
            "choppy_ci_upper": self.choppy_ci_upper,
            "n_bootstrap_samples": self.n_bootstrap_samples,
            "block_size_bars": self.block_size_bars,
        }

    def passes_gates(
        self,
        min_cohens_d: float = 0.8,
        max_p_value: float = 0.01,
        min_trending_ci_lower: float = 0.60,
        max_choppy_ci_upper: float = 0.25,
    ) -> Tuple[bool, List[str]]:
        """
        Check if result passes all statistical gates.

        Returns:
            Tuple of (passes, list_of_failures)
        """
        failures = []

        if self.effect_size_cohens_d < min_cohens_d:
            failures.append(f"Cohen's d ({self.effect_size_cohens_d:.3f}) < {min_cohens_d}")

        if self.p_value > max_p_value:
            failures.append(f"p-value ({self.p_value:.4f}) > {max_p_value}")

        if self.trending_ci_lower < min_trending_ci_lower:
            failures.append(
                f"trending_ci_lower ({self.trending_ci_lower:.3f}) < {min_trending_ci_lower}"
            )

        if self.choppy_ci_upper > max_choppy_ci_upper:
            failures.append(f"choppy_ci_upper ({self.choppy_ci_upper:.3f}) > {max_choppy_ci_upper}")

        return len(failures) == 0, failures


@dataclass
class SymbolMetrics:
    """Metrics for a single symbol in one fold."""

    symbol: str
    label_type: str  # "TRENDING" or "CHOPPY"
    r0_rate: float  # Rate of R0 detections
    total_bars: int
    r0_bars: int


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.

    Cohen's d = (mean1 - mean2) / pooled_std

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def block_bootstrap_ci(
    data: List[float],
    block_size: int = 20,
    n_samples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Block bootstrap confidence interval.

    Preserves temporal correlation by resampling in blocks.

    Args:
        data: List of values to bootstrap
        block_size: Size of blocks for resampling
        n_samples: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95 for 95%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower, upper) CI bounds
    """
    if len(data) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(data)
    data_arr = np.array(data)

    # Handle case where data is smaller than block size
    if n <= block_size:
        # Fall back to standard bootstrap
        bootstrap_means = []
        for _ in range(n_samples):
            indices = rng.choice(n, size=n, replace=True)
            bootstrap_means.append(np.mean(data_arr[indices]))
    else:
        n_blocks = n // block_size
        bootstrap_means = []

        for _ in range(n_samples):
            # Sample blocks with replacement
            block_indices = rng.choice(n_blocks, size=n_blocks, replace=True)
            sample_values: list[float] = []

            for bi in block_indices:
                start = bi * block_size
                end = min(start + block_size, n)
                sample_values.extend(data_arr[start:end])

            if len(sample_values) > 0:
                bootstrap_means.append(np.mean(sample_values))

    if len(bootstrap_means) == 0:
        return (np.mean(data_arr), np.mean(data_arr))

    alpha = 1 - ci_level
    lower = float(np.percentile(bootstrap_means, alpha / 2 * 100))
    upper = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))

    return (lower, upper)


def compute_symbol_level_stats(
    symbol_metrics: List[SymbolMetrics],
    block_size: int = 20,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> StatisticalResult:
    """
    Compute statistics at SYMBOL level (not fold level).

    This gives n ~ 140 (training universe size), providing
    actual statistical power for hypothesis testing.

    Args:
        symbol_metrics: List of per-symbol metrics
        block_size: Block size for bootstrap
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        StatisticalResult with proper statistics
    """
    # Separate by label type
    trending_rates = [m.r0_rate for m in symbol_metrics if m.label_type == "TRENDING"]
    choppy_rates = [m.r0_rate for m in symbol_metrics if m.label_type == "CHOPPY"]

    # Handle empty groups
    if len(trending_rates) == 0 or len(choppy_rates) == 0:
        return StatisticalResult(
            n_trending_symbols=len(trending_rates),
            n_choppy_symbols=len(choppy_rates),
            trending_r0_rates=trending_rates,
            choppy_r0_rates=choppy_rates,
            t_statistic=0.0,
            p_value=1.0,
            effect_size_cohens_d=0.0,
            trending_mean=float(np.mean(trending_rates)) if trending_rates else 0.0,
            trending_ci_lower=0.0,
            trending_ci_upper=0.0,
            choppy_mean=float(np.mean(choppy_rates)) if choppy_rates else 0.0,
            choppy_ci_lower=0.0,
            choppy_ci_upper=0.0,
            n_bootstrap_samples=n_bootstrap,
            block_size_bars=block_size,
        )

    # T-test (independent samples)
    t_stat, p_value = stats.ttest_ind(trending_rates, choppy_rates)

    # Cohen's d effect size
    cohens_d = compute_cohens_d(trending_rates, choppy_rates)

    # Block bootstrap CIs
    trending_ci = block_bootstrap_ci(trending_rates, block_size, n_bootstrap, 0.95, seed)
    choppy_ci = block_bootstrap_ci(choppy_rates, block_size, n_bootstrap, 0.95, seed)

    return StatisticalResult(
        n_trending_symbols=len(trending_rates),
        n_choppy_symbols=len(choppy_rates),
        trending_r0_rates=trending_rates,
        choppy_r0_rates=choppy_rates,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        effect_size_cohens_d=float(cohens_d),
        trending_mean=float(np.mean(trending_rates)),
        trending_ci_lower=trending_ci[0],
        trending_ci_upper=trending_ci[1],
        choppy_mean=float(np.mean(choppy_rates)),
        choppy_ci_lower=choppy_ci[0],
        choppy_ci_upper=choppy_ci[1],
        n_bootstrap_samples=n_bootstrap,
        block_size_bars=block_size,
    )


def aggregate_symbol_metrics_across_folds(
    fold_metrics: List[List[SymbolMetrics]],
) -> List[SymbolMetrics]:
    """
    Aggregate per-symbol metrics across folds.

    For each symbol, average the R0 rate across all folds where it appears.

    Args:
        fold_metrics: List of [fold1_metrics, fold2_metrics, ...]

    Returns:
        Aggregated SymbolMetrics (one per symbol)
    """
    from collections import defaultdict

    # Accumulate per-symbol
    symbol_data: dict = defaultdict(lambda: {"rates": [], "bars": 0, "r0": 0, "type": None})

    for fold in fold_metrics:
        for m in fold:
            symbol_data[m.symbol]["rates"].append(m.r0_rate)
            symbol_data[m.symbol]["bars"] += m.total_bars
            symbol_data[m.symbol]["r0"] += m.r0_bars
            symbol_data[m.symbol]["type"] = m.label_type

    # Create aggregated metrics
    aggregated = []
    for symbol, data in symbol_data.items():
        if len(data["rates"]) > 0 and data["type"]:
            aggregated.append(
                SymbolMetrics(
                    symbol=symbol,
                    label_type=data["type"],
                    r0_rate=float(np.mean(data["rates"])),
                    total_bars=data["bars"],
                    r0_bars=data["r0"],
                )
            )

    return aggregated
