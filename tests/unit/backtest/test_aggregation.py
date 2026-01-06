"""Tests for trial aggregation with robust statistics."""

from __future__ import annotations

from typing import List

import pytest

from src.backtest import AggregationConfig, Aggregator
from src.backtest import RunMetrics, RunResult, RunStatus


class TestAggregatorStatistics:
    """Tests for basic statistical functions."""

    def test_median_odd_count(self) -> None:
        """Test median with odd number of values."""
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        assert Aggregator.median(values) == 5.0

    def test_median_even_count(self) -> None:
        """Test median with even number of values."""
        values = [1.0, 3.0, 5.0, 7.0]
        assert Aggregator.median(values) == 4.0  # Average of 3 and 5

    def test_median_single_value(self) -> None:
        """Test median with single value."""
        assert Aggregator.median([42.0]) == 42.0

    def test_median_empty_list(self) -> None:
        """Test median with empty list."""
        assert Aggregator.median([]) == 0.0

    def test_mad_calculation(self) -> None:
        """Test MAD (Median Absolute Deviation) calculation."""
        # Values: 1, 2, 3, 4, 5
        # Median: 3
        # Absolute deviations: 2, 1, 0, 1, 2
        # MAD (median of deviations): 1
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert Aggregator.mad(values) == 1.0

    def test_mad_single_value(self) -> None:
        """Test MAD with single value."""
        assert Aggregator.mad([42.0]) == 0.0

    def test_mad_empty_list(self) -> None:
        """Test MAD with empty list."""
        assert Aggregator.mad([]) == 0.0

    def test_percentile(self) -> None:
        """Test percentile calculation."""
        values = list(range(1, 101))  # 1 to 100
        assert Aggregator.percentile(values, 10) == pytest.approx(10.9, rel=0.1)
        assert Aggregator.percentile(values, 50) == pytest.approx(50.5, rel=0.1)
        assert Aggregator.percentile(values, 90) == pytest.approx(90.1, rel=0.1)

    def test_percentile_empty_list(self) -> None:
        """Test percentile with empty list."""
        assert Aggregator.percentile([], 50) == 0.0


class TestAggregatorTrialAggregation:
    """Tests for trial aggregation."""

    def _create_run_result(
        self,
        run_id: str,
        sharpe: float,
        total_return: float,
        max_drawdown: float,
        is_oos: bool = True,
    ) -> RunResult:
        """Helper to create run results."""
        return RunResult(
            run_id=run_id,
            trial_id="trial_test",
            experiment_id="exp_test",
            symbol="AAPL",
            window_id="window_1",
            profile_version="v1",
            data_version="v1",
            status=RunStatus.SUCCESS,
            is_train=not is_oos,
            is_oos=is_oos,
            metrics=RunMetrics(
                sharpe=sharpe,
                total_return=total_return,
                max_drawdown=max_drawdown,
                total_trades=50,
                win_rate=0.55,
                profit_factor=1.5,
            ),
        )

    def test_aggregate_trial_basic(self) -> None:
        """Test basic trial aggregation."""
        aggregator = Aggregator()

        runs = [
            self._create_run_result("run_1", sharpe=1.0, total_return=0.10, max_drawdown=-0.05),
            self._create_run_result("run_2", sharpe=1.5, total_return=0.15, max_drawdown=-0.08),
            self._create_run_result("run_3", sharpe=2.0, total_return=0.20, max_drawdown=-0.10),
        ]

        result = aggregator.aggregate_trial(
            trial_id="trial_test",
            experiment_id="exp_test",
            params={"fast": 10},
            runs=runs,
        )

        # Median of [1.0, 1.5, 2.0] = 1.5
        assert result.aggregates.median_sharpe == 1.5
        # Median of [0.10, 0.15, 0.20] = 0.15
        assert result.aggregates.median_return == 0.15
        # Median of [-0.05, -0.08, -0.10] = -0.08
        assert result.aggregates.median_max_dd == -0.08

    def test_aggregate_trial_with_failures(self) -> None:
        """Test aggregation with some failed runs."""
        aggregator = Aggregator()

        runs = [
            self._create_run_result("run_1", sharpe=1.0, total_return=0.10, max_drawdown=-0.05),
            RunResult(
                run_id="run_2",
                trial_id="trial_test",
                experiment_id="exp_test",
                symbol="AAPL",
                window_id="window_2",
                profile_version="v1",
                data_version="v1",
                status=RunStatus.FAIL_DATA,
                is_train=False,
                is_oos=True,
                error="No data",
            ),
            self._create_run_result("run_3", sharpe=2.0, total_return=0.20, max_drawdown=-0.10),
        ]

        result = aggregator.aggregate_trial(
            trial_id="trial_test",
            experiment_id="exp_test",
            params={"fast": 10},
            runs=runs,
        )

        # Only successful runs should be included
        assert result.aggregates.successful_runs == 2
        assert result.aggregates.failed_runs == 1
        # Median of [1.0, 2.0] = 1.5
        assert result.aggregates.median_sharpe == 1.5

    def test_aggregate_trial_is_vs_oos(self) -> None:
        """Test aggregation separates IS and OOS metrics."""
        aggregator = Aggregator()

        runs = [
            self._create_run_result(
                "run_1", sharpe=2.0, total_return=0.20, max_drawdown=-0.05, is_oos=False
            ),  # IS
            self._create_run_result(
                "run_2", sharpe=1.0, total_return=0.10, max_drawdown=-0.10, is_oos=True
            ),  # OOS
            self._create_run_result(
                "run_3", sharpe=1.5, total_return=0.15, max_drawdown=-0.08, is_oos=True
            ),  # OOS
        ]

        result = aggregator.aggregate_trial(
            trial_id="trial_test",
            experiment_id="exp_test",
            params={"fast": 10},
            runs=runs,
        )

        # IS median Sharpe: 2.0 (only one IS run)
        assert result.aggregates.is_median_sharpe == 2.0
        # OOS median Sharpe: median of [1.0, 1.5] = 1.25
        assert result.aggregates.oos_median_sharpe == 1.25

    def test_aggregate_trial_percentiles(self) -> None:
        """Test that percentiles are calculated correctly."""
        aggregator = Aggregator()

        # Create 10 runs with varying Sharpe ratios
        runs = [
            self._create_run_result(
                f"run_{i}",
                sharpe=float(i),
                total_return=0.1,
                max_drawdown=-0.1,
            )
            for i in range(1, 11)
        ]

        result = aggregator.aggregate_trial(
            trial_id="trial_test",
            experiment_id="exp_test",
            params={"fast": 10},
            runs=runs,
        )

        # P10 should be around 1.9, P90 should be around 9.1
        assert result.aggregates.p10_sharpe is not None
        assert result.aggregates.p90_sharpe is not None
        assert result.aggregates.p10_sharpe < result.aggregates.p90_sharpe

    def test_aggregate_trial_all_failed(self) -> None:
        """Test aggregation when all runs fail."""
        aggregator = Aggregator()

        runs = [
            RunResult(
                run_id=f"run_{i}",
                trial_id="trial_test",
                experiment_id="exp_test",
                symbol="AAPL",
                window_id=f"window_{i}",
                profile_version="v1",
                data_version="v1",
                status=RunStatus.FAIL_STRATEGY,
                is_train=False,
                is_oos=True,
                error="Strategy error",
            )
            for i in range(3)
        ]

        result = aggregator.aggregate_trial(
            trial_id="trial_test",
            experiment_id="exp_test",
            params={"fast": 10},
            runs=runs,
        )

        assert result.aggregates.successful_runs == 0
        assert result.aggregates.failed_runs == 3
        assert result.aggregates.median_sharpe == 0.0

    def test_stability_score(self) -> None:
        """Test stability score calculation."""
        aggregator = Aggregator()

        # Create runs with consistent performance (low MAD)
        consistent_runs = [
            self._create_run_result(
                f"run_{i}",
                sharpe=1.0 + 0.01 * i,  # Very small variance
                total_return=0.1,
                max_drawdown=-0.1,
            )
            for i in range(5)
        ]

        result = aggregator.aggregate_trial(
            trial_id="trial_test",
            experiment_id="exp_test",
            params={"fast": 10},
            runs=consistent_runs,
        )

        # With low variance, stability score should be high
        assert result.aggregates.stability_score > 0.5

    def test_trial_score(self) -> None:
        """Test that trial score is calculated."""
        aggregator = Aggregator()

        runs = [
            self._create_run_result("run_1", sharpe=1.5, total_return=0.15, max_drawdown=-0.05),
            self._create_run_result("run_2", sharpe=1.5, total_return=0.15, max_drawdown=-0.05),
        ]

        result = aggregator.aggregate_trial(
            trial_id="trial_test",
            experiment_id="exp_test",
            params={"fast": 10},
            runs=runs,
        )

        # Trial score should be non-zero for successful trials
        assert result.trial_score != 0.0
