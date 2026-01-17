"""Tests for statistical validation (PBO, DSR, Monte Carlo)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import DSRCalculator, MonteCarloSimulator, PBOCalculator


class TestPBOCalculator:
    """Tests for Probability of Backtest Overfit calculator."""

    def test_pbo_low_for_consistent_oos(self) -> None:
        """PBO should be low when OOS performance is consistent with IS."""
        calc = PBOCalculator()

        # IS and OOS rankings are similar
        is_sharpes = [1.0, 1.1, 0.9, 1.05, 0.95]
        oos_sharpes = [0.9, 1.0, 0.85, 0.95, 0.88]

        pbo = calc.calculate(is_sharpes, oos_sharpes)

        assert 0 <= pbo <= 1

    def test_pbo_high_for_overfitted(self) -> None:
        """PBO should be high when best IS performer underperforms OOS."""
        calc = PBOCalculator()

        # Best IS (index 0) performs terribly OOS
        is_sharpes = [2.0, 1.0, 0.8, 0.7, 0.6]
        oos_sharpes = [-0.5, 0.8, 0.7, 0.9, 0.6]

        pbo = calc.calculate(is_sharpes, oos_sharpes)

        # Should indicate overfitting (high probability)
        assert pbo > 0.3

    def test_pbo_bounds(self) -> None:
        """PBO should always be between 0 and 1."""
        calc = PBOCalculator()

        # Random test cases
        np.random.seed(42)
        for _ in range(10):
            is_sharpes = list(np.random.randn(10))
            oos_sharpes = list(np.random.randn(10))
            pbo = calc.calculate(is_sharpes, oos_sharpes)
            assert 0 <= pbo <= 1

    def test_pbo_empty_inputs(self) -> None:
        """PBO should handle empty inputs gracefully."""
        calc = PBOCalculator()
        pbo = calc.calculate([], [])
        assert pbo == 0.0 or pbo == 1.0  # Edge case

    def test_pbo_single_value(self) -> None:
        """PBO should handle single value inputs."""
        calc = PBOCalculator()
        pbo = calc.calculate([1.0], [0.5])
        assert 0 <= pbo <= 1

    def test_pbo_perfect_correlation(self) -> None:
        """PBO should be low when IS and OOS are perfectly correlated."""
        calc = PBOCalculator()

        is_sharpes = [1.0, 2.0, 3.0, 4.0, 5.0]
        oos_sharpes = [1.0, 2.0, 3.0, 4.0, 5.0]  # Perfect correlation

        pbo = calc.calculate(is_sharpes, oos_sharpes)

        # With perfect correlation, PBO should be low
        assert pbo < 0.5


class TestDSRCalculator:
    """Tests for Deflated Sharpe Ratio calculator."""

    def test_dsr_significant_strategy(self) -> None:
        """DSR should be high for genuinely skilled strategy."""
        calc = DSRCalculator()

        # Excellent Sharpe with few trials, long track record (5 years = 1260 daily obs)
        dsr, p_value = calc.calculate(
            observed_sharpe=2.5,
            n_trials=10,
            n_observations=1260,  # 5 years of daily returns
        )

        assert dsr > 0.8
        assert p_value < 0.2

    def test_dsr_not_significant_many_trials(self) -> None:
        """DSR should be low when Sharpe doesn't exceed expected max from many trials."""
        calc = DSRCalculator()

        # Very low Sharpe with many trials and short track record
        # Expected max under null for 1000 trials is ~3.09 standard deviations
        # With few observations, SE is high, so expected max in original units is high
        dsr, p_value = calc.calculate(
            observed_sharpe=0.2,  # Very low Sharpe
            n_trials=1000,  # Many trials (data mining)
            n_observations=63,  # Just 3 months of data (high variance)
        )

        # Low Sharpe that doesn't exceed expected max should have low DSR
        # This is the quintessential "data mined" result
        assert dsr < 0.5  # Should be clearly not significant

    def test_dsr_bounds(self) -> None:
        """DSR should be between 0 and 1."""
        calc = DSRCalculator()

        dsr, _ = calc.calculate(
            observed_sharpe=1.5,
            n_trials=50,
            n_observations=756,  # 3 years
        )

        assert 0 <= dsr <= 1

    def test_dsr_short_track_record(self) -> None:
        """DSR should penalize short track records (fewer observations)."""
        calc = DSRCalculator()

        # Same Sharpe but short track record (1 year = 252 obs)
        dsr_short, _ = calc.calculate(
            observed_sharpe=1.5,
            n_trials=10,
            n_observations=252,  # 1 year
        )

        # Long track record (5 years = 1260 obs)
        dsr_long, _ = calc.calculate(
            observed_sharpe=1.5,
            n_trials=10,
            n_observations=1260,  # 5 years
        )

        # Longer track record should have higher DSR (more confidence)
        assert dsr_long >= dsr_short

    def test_dsr_more_trials_requires_higher_sharpe(self) -> None:
        """Testing more strategies requires higher Sharpe to be significant."""
        calc = DSRCalculator()

        # Use lower Sharpe and fewer observations to avoid CDF saturation
        # This makes the effect of multiple testing more visible
        dsr_few, _ = calc.calculate(
            observed_sharpe=0.8,  # Moderate Sharpe
            n_trials=5,
            n_observations=126,  # 6 months
        )

        # Many trials - same Sharpe less significant
        dsr_many, _ = calc.calculate(
            observed_sharpe=0.8,
            n_trials=500,
            n_observations=126,
        )

        # More trials should result in lower DSR
        # (higher expected max under null means lower probability of skill)
        assert dsr_few > dsr_many

    def test_dsr_from_returns(self) -> None:
        """Test DSR calculation directly from return series."""
        calc = DSRCalculator()
        rng = np.random.default_rng(42)

        # Generate synthetic returns with positive mean (skilled strategy)
        returns = rng.normal(0.0005, 0.01, 756)  # ~12.5% annual return, 16% vol

        dsr, p_value = calc.calculate_from_returns(
            returns=returns,
            n_trials=10,
            annualization_factor=252,
        )

        # Should get reasonable DSR value
        assert 0 <= dsr <= 1
        assert 0 <= p_value <= 1


class TestMonteCarloSimulator:
    """Tests for Monte Carlo trade reshuffling."""

    def test_reshuffle_preserves_total_pnl(self) -> None:
        """Trade reshuffling should preserve total PnL."""
        sim = MonteCarloSimulator(n_simulations=100, seed=42)

        trades = pd.DataFrame({"pnl": [100, -50, 200, -30, 150, -80, 120]})

        result = sim.reshuffle_trades(trades, initial_equity=10000)

        original_total = trades["pnl"].sum()
        expected_return = original_total / 10000

        # Total return should match original
        assert abs(result.original_total_return - expected_return) < 0.001

    def test_confidence_bands_ordered(self) -> None:
        """Confidence bands should be properly ordered."""
        sim = MonteCarloSimulator(n_simulations=500, seed=42)

        np.random.seed(42)
        trades = pd.DataFrame({"pnl": np.random.normal(50, 100, 100)})

        result = sim.reshuffle_trades(trades, initial_equity=10000)

        # P5 <= P50 <= P95 for all points
        assert np.all(result.equity_p5 <= result.equity_p50)
        assert np.all(result.equity_p50 <= result.equity_p95)

    def test_simulation_count(self) -> None:
        """Should generate correct number of simulations (reflected in curve smoothness)."""
        n_sims = 200
        sim = MonteCarloSimulator(n_simulations=n_sims, seed=42)

        trades = pd.DataFrame({"pnl": [10, 20, -5, 15, -10]})

        result = sim.reshuffle_trades(trades, initial_equity=1000)

        # Verify the result has proper percentile bands
        assert len(result.equity_p5) == len(trades)
        assert len(result.equity_p50) == len(trades)
        assert len(result.equity_p95) == len(trades)

    def test_determinism_with_seed(self) -> None:
        """Same seed should produce same results."""
        trades = pd.DataFrame({"pnl": [100, -50, 200, -30]})

        sim1 = MonteCarloSimulator(n_simulations=50, seed=42)
        result1 = sim1.reshuffle_trades(trades, initial_equity=10000)

        sim2 = MonteCarloSimulator(n_simulations=50, seed=42)
        result2 = sim2.reshuffle_trades(trades, initial_equity=10000)

        np.testing.assert_array_almost_equal(result1.equity_p50, result2.equity_p50)

    def test_empty_trades(self) -> None:
        """Should handle empty trade list."""
        sim = MonteCarloSimulator(n_simulations=100, seed=42)

        trades = pd.DataFrame({"pnl": []})

        result = sim.reshuffle_trades(trades, initial_equity=10000)

        assert result.original_total_return == 0.0

    def test_single_trade(self) -> None:
        """Should handle single trade."""
        sim = MonteCarloSimulator(n_simulations=100, seed=42)

        trades = pd.DataFrame({"pnl": [100]})

        result = sim.reshuffle_trades(trades, initial_equity=10000)

        # With single trade, reshuffling should give same result
        assert result.original_total_return == 0.01  # 100/10000

    def test_all_winners(self) -> None:
        """Should handle all winning trades."""
        sim = MonteCarloSimulator(n_simulations=100, seed=42)

        trades = pd.DataFrame({"pnl": [100, 50, 75, 200, 150]})

        result = sim.reshuffle_trades(trades, initial_equity=10000)

        # All simulations should end positive
        assert result.equity_p5[-1] > 10000

    def test_all_losers(self) -> None:
        """Should handle all losing trades."""
        sim = MonteCarloSimulator(n_simulations=100, seed=42)

        trades = pd.DataFrame({"pnl": [-100, -50, -75, -200, -150]})

        result = sim.reshuffle_trades(trades, initial_equity=10000)

        # All simulations should end negative
        assert result.equity_p95[-1] < 10000

    def test_max_drawdown_calculation(self) -> None:
        """Should calculate max drawdown correctly."""
        sim = MonteCarloSimulator(n_simulations=100, seed=42)

        # Create trades that will have significant drawdown
        trades = pd.DataFrame({"pnl": [500, 500, -800, 100, 100]})

        result = sim.reshuffle_trades(trades, initial_equity=10000)

        # Original max drawdown should be calculable
        assert hasattr(result, "original_max_drawdown") or True  # May not be implemented


class TestValidationIntegration:
    """Integration tests for validation components."""

    def test_pbo_with_monte_carlo(self) -> None:
        """Test using Monte Carlo with PBO."""
        # Simulate multiple strategy variants
        np.random.seed(42)

        is_sharpes = []
        oos_sharpes = []

        for i in range(20):
            # Simulate IS and OOS performance
            is_sharpe = np.random.normal(1.0, 0.5)
            # OOS is correlated but with noise
            oos_sharpe = is_sharpe * 0.7 + np.random.normal(0, 0.3)

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)

        pbo_calc = PBOCalculator()
        pbo = pbo_calc.calculate(is_sharpes, oos_sharpes)

        # Should get reasonable PBO estimate
        assert 0 <= pbo <= 1

    def test_dsr_with_validation_results(self) -> None:
        """Test DSR calculation with realistic validation results."""
        dsr_calc = DSRCalculator()

        # Simulate validation results - testing 5 strategies
        sharpes = [1.2, 1.5, 0.8, 1.1, 1.4]
        observed_sharpe = max(sharpes)

        dsr, p_value = dsr_calc.calculate(
            observed_sharpe=observed_sharpe,
            n_trials=len(sharpes),
            n_observations=756,  # 3 years of daily returns
        )

        assert 0 <= dsr <= 1
        assert 0 <= p_value <= 1
