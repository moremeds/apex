"""Tests for Symbol-Level Statistics and Block Bootstrap."""

import numpy as np
import pytest

from src.domain.signals.validation.statistics import (
    StatisticalResult,
    SymbolMetrics,
    aggregate_symbol_metrics_across_folds,
    block_bootstrap_ci,
    compute_cohens_d,
    compute_symbol_level_stats,
)


class TestBlockBootstrapCI:
    """Tests for block_bootstrap_ci function."""

    def test_basic_ci(self):
        """Test basic confidence interval calculation."""
        np.random.seed(42)
        data = list(np.random.randn(100) + 5)  # Mean ~5

        lower, upper = block_bootstrap_ci(data, block_size=10, n_samples=500, seed=42)

        assert lower < upper
        assert lower > 4.0  # Mean should be around 5
        assert upper < 6.0

    def test_ci_contains_mean(self):
        """Test that CI typically contains the sample mean."""
        # Use data with some variance
        np.random.seed(42)
        data = list(np.random.randn(100) + 10)  # Mean ~10 with variance

        lower, upper = block_bootstrap_ci(data, block_size=5, n_samples=1000, seed=42)

        mean = np.mean(data)
        # Mean should be within CI (with some tolerance for boundary)
        assert lower <= mean <= upper

    def test_empty_data(self):
        """Test handling of empty data."""
        lower, upper = block_bootstrap_ci([], block_size=10, n_samples=100)
        assert lower == 0.0
        assert upper == 0.0

    def test_small_data_fallback(self):
        """Test fallback for data smaller than block size."""
        data = [1.0, 2.0, 3.0]  # Less than default block size

        lower, upper = block_bootstrap_ci(data, block_size=20, n_samples=100, seed=42)

        assert lower < upper
        assert lower > 0

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        data = list(np.random.randn(50))

        ci1 = block_bootstrap_ci(data, seed=42)
        ci2 = block_bootstrap_ci(data, seed=42)

        assert ci1 == ci2

    def test_different_seeds_differ(self):
        """Test that different seeds give different results."""
        data = list(np.random.randn(100))

        ci1 = block_bootstrap_ci(data, seed=42)
        ci2 = block_bootstrap_ci(data, seed=123)

        # Very unlikely to be identical
        assert ci1 != ci2


class TestCohensD:
    """Tests for compute_cohens_d function."""

    def test_known_effect_size(self):
        """Test with known effect size."""
        # Two groups with clear separation
        group1 = [10.0] * 100
        group2 = [8.0] * 100

        d = compute_cohens_d(group1, group2)

        # With zero variance, this is undefined, but we handle it
        # Actually both have zero variance, so pooled_std=0
        assert d == 0.0  # Division by zero protection

    def test_moderate_effect(self):
        """Test moderate effect size."""
        np.random.seed(42)
        group1 = list(np.random.randn(100) + 0.5)  # Mean ~0.5
        group2 = list(np.random.randn(100))  # Mean ~0

        d = compute_cohens_d(group1, group2)

        # Effect size should be around 0.5 (moderate)
        assert 0.3 < d < 0.8

    def test_large_effect(self):
        """Test large effect size."""
        np.random.seed(42)
        group1 = list(np.random.randn(100) + 1.0)  # Mean ~1
        group2 = list(np.random.randn(100))  # Mean ~0

        d = compute_cohens_d(group1, group2)

        # Effect size should be around 1.0 (large)
        assert d > 0.7

    def test_empty_groups(self):
        """Test handling of empty groups."""
        assert compute_cohens_d([], [1, 2, 3]) == 0.0
        assert compute_cohens_d([1, 2, 3], []) == 0.0
        assert compute_cohens_d([], []) == 0.0


class TestSymbolMetrics:
    """Tests for SymbolMetrics dataclass."""

    def test_creation(self):
        """Test basic creation."""
        m = SymbolMetrics(
            symbol="AAPL",
            label_type="TRENDING",
            r0_rate=0.75,
            total_bars=100,
            r0_bars=75,
        )

        assert m.symbol == "AAPL"
        assert m.label_type == "TRENDING"
        assert m.r0_rate == 0.75
        assert m.total_bars == 100
        assert m.r0_bars == 75


class TestComputeSymbolLevelStats:
    """Tests for compute_symbol_level_stats function."""

    @pytest.fixture
    def trending_metrics(self):
        """Create metrics for trending symbols."""
        return [
            SymbolMetrics("AAPL", "TRENDING", 0.80, 100, 80),
            SymbolMetrics("MSFT", "TRENDING", 0.75, 100, 75),
            SymbolMetrics("GOOGL", "TRENDING", 0.82, 100, 82),
            SymbolMetrics("AMZN", "TRENDING", 0.78, 100, 78),
            SymbolMetrics("META", "TRENDING", 0.85, 100, 85),
        ]

    @pytest.fixture
    def choppy_metrics(self):
        """Create metrics for choppy symbols."""
        return [
            SymbolMetrics("GME", "CHOPPY", 0.20, 100, 20),
            SymbolMetrics("AMC", "CHOPPY", 0.15, 100, 15),
            SymbolMetrics("SPCE", "CHOPPY", 0.25, 100, 25),
            SymbolMetrics("BBBY", "CHOPPY", 0.18, 100, 18),
            SymbolMetrics("WISH", "CHOPPY", 0.22, 100, 22),
        ]

    def test_basic_stats(self, trending_metrics, choppy_metrics):
        """Test basic statistical computation."""
        all_metrics = trending_metrics + choppy_metrics

        result = compute_symbol_level_stats(all_metrics, block_size=2, n_bootstrap=100)

        assert result.n_trending_symbols == 5
        assert result.n_choppy_symbols == 5
        assert result.trending_mean > 0.7
        assert result.choppy_mean < 0.3
        assert result.t_statistic > 0  # Trending > Choppy
        assert result.p_value < 0.05  # Significant difference
        assert result.effect_size_cohens_d > 0.8  # Large effect

    def test_ci_ordering(self, trending_metrics, choppy_metrics):
        """Test that CIs are properly ordered."""
        all_metrics = trending_metrics + choppy_metrics

        result = compute_symbol_level_stats(all_metrics, n_bootstrap=500)

        assert result.trending_ci_lower < result.trending_ci_upper
        assert result.choppy_ci_lower < result.choppy_ci_upper

    def test_empty_group(self):
        """Test handling of empty groups."""
        metrics = [SymbolMetrics("AAPL", "TRENDING", 0.80, 100, 80)]

        result = compute_symbol_level_stats(metrics)

        assert result.n_trending_symbols == 1
        assert result.n_choppy_symbols == 0
        assert result.p_value == 1.0  # No comparison possible

    def test_to_dict(self, trending_metrics, choppy_metrics):
        """Test serialization to dict."""
        all_metrics = trending_metrics + choppy_metrics
        result = compute_symbol_level_stats(all_metrics, n_bootstrap=100)

        d = result.to_dict()

        assert "n_trending_symbols" in d
        assert "t_statistic" in d
        assert "p_value" in d
        assert "effect_size_cohens_d" in d
        assert "trending_ci_lower" in d


class TestStatisticalResultGates:
    """Tests for StatisticalResult gate checking."""

    def test_passes_all_gates(self):
        """Test result that passes all gates."""
        result = StatisticalResult(
            n_trending_symbols=70,
            n_choppy_symbols=70,
            trending_r0_rates=[0.8] * 70,
            choppy_r0_rates=[0.2] * 70,
            t_statistic=15.0,
            p_value=0.001,
            effect_size_cohens_d=1.5,
            trending_mean=0.80,
            trending_ci_lower=0.75,
            trending_ci_upper=0.85,
            choppy_mean=0.20,
            choppy_ci_lower=0.15,
            choppy_ci_upper=0.22,
            n_bootstrap_samples=1000,
            block_size_bars=20,
        )

        passes, failures = result.passes_gates()
        assert passes
        assert len(failures) == 0

    def test_fails_cohens_d(self):
        """Test failure on Cohen's d."""
        result = StatisticalResult(
            n_trending_symbols=70,
            n_choppy_symbols=70,
            trending_r0_rates=[0.5] * 70,
            choppy_r0_rates=[0.4] * 70,
            t_statistic=2.0,
            p_value=0.001,
            effect_size_cohens_d=0.5,  # Too low
            trending_mean=0.50,
            trending_ci_lower=0.65,
            trending_ci_upper=0.75,
            choppy_mean=0.40,
            choppy_ci_lower=0.15,
            choppy_ci_upper=0.22,
            n_bootstrap_samples=1000,
            block_size_bars=20,
        )

        passes, failures = result.passes_gates()
        assert not passes
        assert any("Cohen's d" in f for f in failures)

    def test_fails_p_value(self):
        """Test failure on p-value."""
        result = StatisticalResult(
            n_trending_symbols=70,
            n_choppy_symbols=70,
            trending_r0_rates=[0.8] * 70,
            choppy_r0_rates=[0.2] * 70,
            t_statistic=1.5,
            p_value=0.05,  # Too high
            effect_size_cohens_d=1.0,
            trending_mean=0.80,
            trending_ci_lower=0.65,
            trending_ci_upper=0.85,
            choppy_mean=0.20,
            choppy_ci_lower=0.15,
            choppy_ci_upper=0.22,
            n_bootstrap_samples=1000,
            block_size_bars=20,
        )

        passes, failures = result.passes_gates()
        assert not passes
        assert any("p-value" in f for f in failures)

    def test_fails_trending_ci_lower(self):
        """Test failure on trending CI lower bound."""
        result = StatisticalResult(
            n_trending_symbols=70,
            n_choppy_symbols=70,
            trending_r0_rates=[0.55] * 70,
            choppy_r0_rates=[0.2] * 70,
            t_statistic=10.0,
            p_value=0.001,
            effect_size_cohens_d=1.0,
            trending_mean=0.55,
            trending_ci_lower=0.50,  # Too low
            trending_ci_upper=0.60,
            choppy_mean=0.20,
            choppy_ci_lower=0.15,
            choppy_ci_upper=0.22,
            n_bootstrap_samples=1000,
            block_size_bars=20,
        )

        passes, failures = result.passes_gates()
        assert not passes
        assert any("trending_ci_lower" in f for f in failures)

    def test_fails_choppy_ci_upper(self):
        """Test failure on choppy CI upper bound."""
        result = StatisticalResult(
            n_trending_symbols=70,
            n_choppy_symbols=70,
            trending_r0_rates=[0.8] * 70,
            choppy_r0_rates=[0.3] * 70,
            t_statistic=10.0,
            p_value=0.001,
            effect_size_cohens_d=1.0,
            trending_mean=0.80,
            trending_ci_lower=0.75,
            trending_ci_upper=0.85,
            choppy_mean=0.30,
            choppy_ci_lower=0.25,
            choppy_ci_upper=0.35,  # Too high
            n_bootstrap_samples=1000,
            block_size_bars=20,
        )

        passes, failures = result.passes_gates()
        assert not passes
        assert any("choppy_ci_upper" in f for f in failures)


class TestAggregateSymbolMetrics:
    """Tests for aggregate_symbol_metrics_across_folds function."""

    def test_aggregate_single_fold(self):
        """Test aggregation with single fold."""
        fold1 = [
            SymbolMetrics("AAPL", "TRENDING", 0.80, 100, 80),
            SymbolMetrics("GME", "CHOPPY", 0.20, 100, 20),
        ]

        aggregated = aggregate_symbol_metrics_across_folds([fold1])

        assert len(aggregated) == 2
        aapl = next(m for m in aggregated if m.symbol == "AAPL")
        assert aapl.r0_rate == 0.80

    def test_aggregate_multiple_folds(self):
        """Test aggregation across multiple folds."""
        fold1 = [
            SymbolMetrics("AAPL", "TRENDING", 0.80, 100, 80),
            SymbolMetrics("GME", "CHOPPY", 0.20, 100, 20),
        ]
        fold2 = [
            SymbolMetrics("AAPL", "TRENDING", 0.70, 100, 70),
            SymbolMetrics("GME", "CHOPPY", 0.30, 100, 30),
        ]

        aggregated = aggregate_symbol_metrics_across_folds([fold1, fold2])

        assert len(aggregated) == 2
        aapl = next(m for m in aggregated if m.symbol == "AAPL")
        assert aapl.r0_rate == 0.75  # Average of 0.80 and 0.70
        assert aapl.total_bars == 200  # Sum

        gme = next(m for m in aggregated if m.symbol == "GME")
        assert gme.r0_rate == 0.25  # Average of 0.20 and 0.30

    def test_empty_folds(self):
        """Test handling of empty folds."""
        aggregated = aggregate_symbol_metrics_across_folds([])
        assert len(aggregated) == 0

        aggregated = aggregate_symbol_metrics_across_folds([[]])
        assert len(aggregated) == 0
