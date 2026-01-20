"""Tests for Nested Walk-Forward Cross-Validation Framework."""

from datetime import date

import pytest

from src.domain.signals.validation.nested_cv import (
    NestedCVConfig,
    NestedCVResult,
    NestedWalkForwardCV,
    OuterFold,
    OuterFoldResult,
    TimeWindow,
    create_default_param_space,
)
from src.domain.signals.validation.statistics import SymbolMetrics
from src.domain.signals.validation.time_units import ValidationTimeConfig


class TestTimeWindow:
    """Tests for TimeWindow dataclass."""

    def test_days_calculation(self):
        """Test days property calculation."""
        window = TimeWindow(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        assert window.days == 30

    def test_frozen(self):
        """Test that TimeWindow is frozen."""
        window = TimeWindow(date(2024, 1, 1), date(2024, 1, 31))
        with pytest.raises(AttributeError):
            window.start_date = date(2024, 2, 1)  # type: ignore


class TestNestedCVConfig:
    """Tests for NestedCVConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NestedCVConfig()

        assert config.outer_folds == 5
        assert config.inner_folds == 3
        assert config.outer_train_pct == 0.7
        assert config.inner_max_trials == 20

    def test_validate_outer_folds(self):
        """Test validation rejects outer_folds < 2."""
        config = NestedCVConfig(outer_folds=1)
        with pytest.raises(ValueError, match="outer_folds"):
            config.validate()

    def test_validate_inner_folds(self):
        """Test validation rejects inner_folds < 2."""
        config = NestedCVConfig(inner_folds=1)
        with pytest.raises(ValueError, match="inner_folds"):
            config.validate()

    def test_validate_train_pct(self):
        """Test validation rejects invalid train_pct."""
        config = NestedCVConfig(outer_train_pct=0.1)
        with pytest.raises(ValueError, match="outer_train_pct"):
            config.validate()

        config = NestedCVConfig(outer_train_pct=0.95)
        with pytest.raises(ValueError, match="outer_train_pct"):
            config.validate()


class TestNestedWalkForwardCV:
    """Tests for NestedWalkForwardCV class."""

    @pytest.fixture
    def cv(self):
        """Create nested CV instance."""
        config = NestedCVConfig(
            outer_folds=3,
            inner_folds=2,
            outer_train_pct=0.7,
        )
        return NestedWalkForwardCV(config)

    def test_generate_outer_splits(self, cv):
        """Test outer split generation."""
        start = date(2023, 1, 1)
        end = date(2024, 1, 1)  # 365 days

        splits = list(cv.generate_outer_splits(start, end))

        assert len(splits) > 0
        for split in splits:
            assert isinstance(split, OuterFold)
            assert split.train_window.start_date >= start
            assert split.test_window.end_date <= end
            # Train should end before test starts
            assert split.train_window.end_date < split.test_window.start_date

    def test_outer_splits_no_overlap(self, cv):
        """Test that outer train doesn't overlap with test (with purge)."""
        start = date(2023, 1, 1)
        end = date(2024, 1, 1)

        for split in cv.generate_outer_splits(start, end):
            purge_days = cv.config.time_config.purge_days
            gap_days = (split.test_window.start_date - split.train_window.end_date).days
            assert gap_days >= purge_days, "Purge gap not respected"

    def test_generate_inner_splits(self, cv):
        """Test inner split generation."""
        # Use larger window to ensure enough days for splits
        train_window = TimeWindow(date(2023, 1, 1), date(2023, 12, 31))

        splits = list(cv.generate_inner_splits(train_window))

        assert len(splits) > 0, f"Expected splits but got none for {train_window.days} day window"
        for inner_train, inner_test in splits:
            assert inner_train.start_date >= train_window.start_date
            assert inner_test.end_date <= train_window.end_date
            assert inner_train.end_date < inner_test.start_date


class TestNestedCVIntegration:
    """Integration tests for nested CV."""

    @pytest.fixture
    def simple_cv(self):
        """Create simple CV for testing."""
        config = NestedCVConfig(
            outer_folds=2,
            inner_folds=2,
            inner_max_trials=3,  # Few trials for speed
        )
        return NestedWalkForwardCV(config)

    def test_run_with_mock_functions(self, simple_cv):
        """Test full run with mock objective and evaluation functions."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start = date(2023, 1, 1)
        end = date(2024, 1, 1)

        # Mock objective function (returns fixed score)
        def mock_objective(params, train_window, test_window):
            return 0.75

        # Mock evaluation function
        def mock_evaluate(symbol, window, params):
            return SymbolMetrics(
                symbol=symbol,
                label_type="TRENDING",
                r0_rate=0.8,
                total_bars=100,
                r0_bars=80,
            )

        # Simple param space
        param_space = {
            "threshold": {"type": "float", "low": 0.5, "high": 1.0},
        }

        result = simple_cv.run(
            symbols=symbols,
            start_date=start,
            end_date=end,
            objective_fn=mock_objective,
            evaluate_symbol_fn=mock_evaluate,
            param_space=param_space,
        )

        assert isinstance(result, NestedCVResult)
        assert len(result.outer_results) > 0
        assert len(result.aggregated_metrics) > 0

    def test_run_aggregates_metrics(self, simple_cv):
        """Test that metrics are properly aggregated across folds."""
        symbols = ["AAPL", "GME"]
        start = date(2023, 1, 1)
        end = date(2024, 1, 1)

        fold_counter = [0]

        def mock_objective(params, train_window, test_window):
            return 0.7

        def mock_evaluate(symbol, window, params):
            # Different rates per fold to verify aggregation
            fold_counter[0] += 1
            rate = 0.8 if fold_counter[0] % 2 == 0 else 0.6
            return SymbolMetrics(
                symbol=symbol,
                label_type="TRENDING" if symbol == "AAPL" else "CHOPPY",
                r0_rate=rate,
                total_bars=100,
                r0_bars=int(rate * 100),
            )

        result = simple_cv.run(
            symbols=symbols,
            start_date=start,
            end_date=end,
            objective_fn=mock_objective,
            evaluate_symbol_fn=mock_evaluate,
            param_space={"threshold": {"type": "float", "low": 0.5, "high": 1.0}},
        )

        # Should have aggregated metrics for both symbols
        assert len(result.aggregated_metrics) == 2


class TestOuterFoldResult:
    """Tests for OuterFoldResult."""

    def test_structure(self):
        """Test OuterFoldResult structure."""
        result = OuterFoldResult(
            fold_id=0,
            best_params={"threshold": 0.7},
            symbol_metrics=[
                SymbolMetrics("AAPL", "TRENDING", 0.8, 100, 80),
            ],
            inner_cv_score=0.75,
        )

        assert result.fold_id == 0
        assert result.best_params["threshold"] == 0.7
        assert len(result.symbol_metrics) == 1
        assert result.inner_cv_score == 0.75


class TestNestedCVResult:
    """Tests for NestedCVResult."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = NestedCVResult(
            outer_results=[
                OuterFoldResult(
                    fold_id=0,
                    best_params={"a": 1},
                    symbol_metrics=[
                        SymbolMetrics("AAPL", "TRENDING", 0.8, 100, 80),
                    ],
                    inner_cv_score=0.7,
                ),
            ],
            aggregated_metrics=[
                SymbolMetrics("AAPL", "TRENDING", 0.8, 100, 80),
            ],
        )

        d = result.to_dict()

        assert d["n_outer_folds"] == 1
        assert d["n_aggregated_symbols"] == 1
        assert d["outer_folds"][0]["fold_id"] == 0
        assert d["outer_folds"][0]["inner_cv_score"] == 0.7


class TestCreateDefaultParamSpace:
    """Tests for create_default_param_space function."""

    def test_has_expected_params(self):
        """Test that default param space has expected parameters."""
        space = create_default_param_space()

        assert "ma50_period" in space
        assert "ma200_period" in space
        assert "atr_period" in space

    def test_valid_param_structure(self):
        """Test that each param has valid structure."""
        space = create_default_param_space()

        for name, spec in space.items():
            assert "type" in spec
            assert spec["type"] in ("int", "float", "categorical")
            if spec["type"] in ("int", "float"):
                assert "low" in spec
                assert "high" in spec
                assert spec["low"] < spec["high"]
