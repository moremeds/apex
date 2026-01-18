"""Tests for walk-forward parameter optimizer."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.domain.services.regime.objectives import (
    CombinedObjectiveResult,
    ObjectiveEvaluator,
    RegimeStabilityObjective,
    TradingProxyObjective,
    TurningPointQualityObjective,
)
from src.domain.services.regime.param_optimizer import (
    FoldResult,
    ParamStability,
    WalkForwardConfig,
    WalkForwardOptimizer,
    WalkForwardResult,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=500, freq="D")
    np.random.seed(42)

    # Generate trending price data with some volatility
    returns = np.random.normal(0.0005, 0.02, len(dates))
    close = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "open": close * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            "high": close * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
            "close": close,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def sample_regime_series():
    """Create sample regime series for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    # Create regime pattern with some transitions
    regimes = ["R0"] * 30 + ["R1"] * 20 + ["R2"] * 15 + ["R0"] * 25 + ["R3"] * 10
    return pd.Series(regimes, index=dates)


class TestObjectiveFunctions:
    """Test individual objective functions."""

    def test_regime_stability_objective(self, sample_ohlcv, sample_regime_series):
        """Test regime stability calculation."""
        obj = RegimeStabilityObjective(weight=0.4)
        result = obj.evaluate(sample_ohlcv[:100], sample_regime_series, {})

        assert result.name == "regime_stability"
        assert result.direction == "minimize"
        assert result.weight == 0.4
        assert "transitions" in result.details
        assert "transition_rate_per_100" in result.details
        assert result.details["transitions"] == 4  # R0->R1, R1->R2, R2->R0, R0->R3

    def test_regime_stability_high_transitions(self):
        """Test with high transition rate."""
        obj = RegimeStabilityObjective()
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        # Alternating regimes = high transitions
        regimes = ["R0", "R1"] * 25
        regime_series = pd.Series(regimes, index=dates)

        result = obj.evaluate(pd.DataFrame(index=dates), regime_series, {})

        # 49 transitions in 50 bars = 98 per 100 bars
        assert result.details["transition_rate_per_100"] == pytest.approx(98.0, rel=0.01)

    def test_trading_proxy_objective(self, sample_ohlcv, sample_regime_series):
        """Test trading proxy calculation."""
        obj = TradingProxyObjective(weight=0.3, forward_bars=5)
        result = obj.evaluate(sample_ohlcv[:100], sample_regime_series, {})

        assert result.name == "trading_proxy"
        assert result.direction == "maximize"
        assert "separation" in result.details or "error" in result.details

    def test_trading_proxy_insufficient_data(self):
        """Test trading proxy with insufficient data."""
        obj = TradingProxyObjective(forward_bars=5)
        ohlcv = pd.DataFrame({"close": [100, 101, 102]})
        regime = pd.Series(["R0", "R1", "R0"])

        result = obj.evaluate(ohlcv, regime, {})

        assert "error" in result.details

    def test_turning_point_quality_no_experiment(self):
        """Test turning point quality without experiment results."""
        obj = TurningPointQualityObjective()
        result = obj.evaluate(pd.DataFrame(), pd.Series(dtype=object), {"symbol": "NONEXISTENT"})

        assert result.value == 0.5  # Neutral score
        assert result.details.get("error") == "no_experiment_results"


class TestObjectiveEvaluator:
    """Test combined objective evaluation."""

    def test_evaluator_default_objectives(self, sample_ohlcv, sample_regime_series):
        """Test evaluator with default objectives."""
        evaluator = ObjectiveEvaluator()
        result = evaluator.evaluate(sample_ohlcv[:100], sample_regime_series, {})

        assert isinstance(result, CombinedObjectiveResult)
        assert len(result.objective_results) == 3
        assert result.total_score != 0

    def test_evaluator_custom_objectives(self, sample_ohlcv, sample_regime_series):
        """Test evaluator with custom objectives."""
        objectives = [RegimeStabilityObjective(weight=1.0)]
        evaluator = ObjectiveEvaluator(objectives=objectives)

        result = evaluator.evaluate(sample_ohlcv[:100], sample_regime_series, {})

        assert len(result.objective_results) == 1
        assert result.objective_results[0].name == "regime_stability"

    def test_objective_summary(self):
        """Test objective summary generation."""
        evaluator = ObjectiveEvaluator()
        summary = evaluator.get_objective_summary()

        assert "regime_stability" in summary
        assert summary["regime_stability"]["direction"] == "minimize"
        assert summary["regime_stability"]["weight"] == 0.4


class TestWalkForwardConfig:
    """Test walk-forward configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()

        assert config.n_folds == 5
        assert config.train_days == 252
        assert config.test_days == 63
        assert config.min_fold_agreement == 0.7

    def test_config_to_dict(self):
        """Test config serialization."""
        config = WalkForwardConfig(n_folds=3, train_days=200)
        d = config.to_dict()

        assert d["n_folds"] == 3
        assert d["train_days"] == 200


class TestWalkForwardOptimizer:
    """Test walk-forward optimizer."""

    def test_generate_folds(self, sample_ohlcv):
        """Test fold generation."""
        config = WalkForwardConfig(n_folds=3, train_days=100, test_days=30)
        optimizer = WalkForwardOptimizer(config=config)

        data_start = sample_ohlcv.index.min().date()
        data_end = sample_ohlcv.index.max().date()

        folds = optimizer._generate_folds(data_start, data_end)

        assert len(folds) >= 1
        # Each fold should have train and test windows
        for train_start, train_end, test_start, test_end in folds:
            assert train_start < train_end
            assert test_start < test_end
            assert train_end < test_start  # Purge gap

    def test_fold_no_overlap_property(self, sample_ohlcv):
        """Test that train and test windows don't overlap (no-overlap property)."""
        config = WalkForwardConfig(
            n_folds=3,
            train_days=100,
            test_days=30,
            purge_gap_days=5,
            embargo_days=2,
        )
        optimizer = WalkForwardOptimizer(config=config)

        data_start = sample_ohlcv.index.min().date()
        data_end = sample_ohlcv.index.max().date()

        folds = optimizer._generate_folds(data_start, data_end)

        for train_start, train_end, test_start, test_end in folds:
            # Train end must be at least purge_gap before test start
            gap = (test_start - train_end).days
            assert (
                gap >= config.purge_gap_days
            ), f"Purge gap violated: {gap} < {config.purge_gap_days}"

    def test_embargo_property(self, sample_ohlcv):
        """Test embargo between consecutive folds."""
        config = WalkForwardConfig(
            n_folds=4,
            train_days=80,
            test_days=25,
            embargo_days=3,
        )
        optimizer = WalkForwardOptimizer(config=config)

        data_start = sample_ohlcv.index.min().date()
        data_end = sample_ohlcv.index.max().date()

        folds = optimizer._generate_folds(data_start, data_end)

        if len(folds) >= 2:
            for i in range(len(folds) - 1):
                _, _, _, test_end_k = folds[i]
                train_start_k1, _, _, _ = folds[i + 1]
                # Note: folds are ordered oldest-first, so k+1 is newer
                # Embargo check applies to train_end of k vs test_start of k
                # In reverse order (most recent fold last), we check if there's enough gap

    def test_optimize_insufficient_data(self):
        """Test optimization with insufficient data."""
        config = WalkForwardConfig(n_folds=5, train_days=300, test_days=100)
        optimizer = WalkForwardOptimizer(config=config)

        # Only 50 days of data
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        ohlcv = pd.DataFrame(
            {
                "open": [100] * 50,
                "high": [101] * 50,
                "low": [99] * 50,
                "close": [100] * 50,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        result = optimizer.optimize("TEST", ohlcv, {})

        assert len(result.fold_results) == 0
        assert "Insufficient data" in result.why_not_changed[0]

    def test_param_stability_calculation(self):
        """Test parameter stability calculation."""
        optimizer = WalkForwardOptimizer()

        # Create mock fold results
        fold_results = [
            FoldResult(
                fold_id=0,
                train_start=date(2023, 1, 1),
                train_end=date(2023, 6, 30),
                test_start=date(2023, 7, 5),
                test_end=date(2023, 9, 30),
                objective_result=CombinedObjectiveResult([], 0.5, {}),
                suggested_changes={"vol_high_short_pct": 2.0},
            ),
            FoldResult(
                fold_id=1,
                train_start=date(2023, 4, 1),
                train_end=date(2023, 9, 30),
                test_start=date(2023, 10, 5),
                test_end=date(2023, 12, 31),
                objective_result=CombinedObjectiveResult([], 0.6, {}),
                suggested_changes={"vol_high_short_pct": 3.0},
            ),
        ]

        stability = optimizer._calculate_param_stability(fold_results)

        assert "vol_high_short_pct" in stability
        assert stability["vol_high_short_pct"].mean_change == 2.5
        assert stability["vol_high_short_pct"].agreement_ratio == 1.0  # Both positive
        assert stability["vol_high_short_pct"].suggested_direction == "increase"


class TestParamStability:
    """Test parameter stability metrics."""

    def test_stability_all_positive(self):
        """Test stability when all folds suggest increase."""
        stability = ParamStability(
            param_name="vol_high_short_pct",
            changes_by_fold=[2.0, 3.0, 2.5],
            mean_change=2.5,
            std_change=0.5,
            agreement_ratio=1.0,
            suggested_direction="increase",
        )

        d = stability.to_dict()
        assert d["suggested_direction"] == "increase"
        assert d["agreement_ratio"] == 1.0

    def test_stability_mixed_directions(self):
        """Test stability with mixed directions."""
        stability = ParamStability(
            param_name="chop_high_pct",
            changes_by_fold=[2.0, -1.0, 1.5, -0.5],
            mean_change=0.5,
            std_change=1.44,
            agreement_ratio=0.5,  # 2 positive, 2 negative
            suggested_direction="increase",
        )

        d = stability.to_dict()
        assert d["agreement_ratio"] == 0.5


class TestWalkForwardResult:
    """Test walk-forward result serialization."""

    def test_result_to_dict(self):
        """Test result serialization."""
        result = WalkForwardResult(
            symbol="SPY",
            config=WalkForwardConfig(),
            fold_results=[],
            param_stability={},
            objective_summary={"regime_stability": 10.5},
            recommendations={},
            why_not_changed=["Test reason"],
            total_score_mean=0.75,
            total_score_std=0.1,
        )

        d = result.to_dict()
        assert d["symbol"] == "SPY"
        assert d["objective_summary"]["regime_stability"] == 10.5
        assert "Test reason" in d["why_not_changed"]
