"""Tests for Pydantic specification models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.backtest import (
    ExperimentSpec,
    OptimizationConfig,
    ParameterDef,
    ProfileConfig,
    RunSpec,
    TemporalConfig,
    TimeWindow,
    TrialSpec,
    UniverseConfig,
)


class TestParameterDef:
    """Tests for ParameterDef model."""

    def test_range_parameter(self) -> None:
        """Test range parameter creation and expansion."""
        param = ParameterDef(name="fast", type="range", min=10, max=30, step=10)
        values = param.expand()
        assert values == [10, 20, 30]

    def test_range_with_float_step(self) -> None:
        """Test range with float step."""
        param = ParameterDef(name="ratio", type="range", min=0.1, max=0.3, step=0.1)
        values = param.expand()
        # Float arithmetic can cause precision issues, so check approx values
        assert len(values) >= 2  # At least 0.1 and 0.2
        assert abs(values[0] - 0.1) < 0.01
        if len(values) >= 2:
            assert abs(values[1] - 0.2) < 0.01

    def test_categorical_parameter(self) -> None:
        """Test categorical parameter creation and expansion."""
        param = ParameterDef(name="method", type="categorical", values=["sma", "ema"])
        values = param.expand()
        assert values == ["sma", "ema"]

    def test_fixed_parameter(self) -> None:
        """Test fixed parameter creation and expansion."""
        param = ParameterDef(name="seed", type="fixed", value=42)
        values = param.expand()
        assert values == [42]

    def test_range_requires_min_max(self) -> None:
        """Test that range parameters require min and max."""
        with pytest.raises(ValueError, match="require min and max"):
            ParameterDef(name="fast", type="range", min=10)

    def test_categorical_requires_values(self) -> None:
        """Test that categorical parameters require values."""
        with pytest.raises(ValueError, match="require values list"):
            ParameterDef(name="method", type="categorical")

    def test_fixed_requires_value(self) -> None:
        """Test that fixed parameters require a value."""
        with pytest.raises(ValueError, match="require a value"):
            ParameterDef(name="seed", type="fixed")


class TestUniverseConfig:
    """Tests for UniverseConfig model."""

    def test_static_universe(self) -> None:
        """Test static universe creation."""
        universe = UniverseConfig(type="static", symbols=["AAPL", "MSFT"])
        assert universe.type == "static"
        assert universe.symbols == ["AAPL", "MSFT"]

    def test_static_requires_symbols(self) -> None:
        """Test that static universe requires symbols."""
        with pytest.raises(ValueError, match="requires symbols list"):
            UniverseConfig(type="static")

    def test_dynamic_requires_rules(self) -> None:
        """Test that dynamic universe requires rules."""
        with pytest.raises(ValueError, match="requires rules"):
            UniverseConfig(type="dynamic")

    def test_index_requires_index(self) -> None:
        """Test that index universe requires index name."""
        with pytest.raises(ValueError, match="requires index name"):
            UniverseConfig(type="index")


class TestTemporalConfig:
    """Tests for TemporalConfig model."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = TemporalConfig()
        assert config.primary_method == "walk_forward"
        assert config.train_days == 252
        assert config.test_days == 63
        assert config.folds == 5
        assert config.purge_days == 5
        assert config.embargo_days == 2

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = TemporalConfig(
            primary_method="sliding",
            train_days=504,
            test_days=126,
            folds=10,
            purge_days=10,
            embargo_days=5,
        )
        assert config.primary_method == "sliding"
        assert config.train_days == 504
        assert config.folds == 10


class TestTimeWindow:
    """Tests for TimeWindow model."""

    def test_window_creation(self) -> None:
        """Test time window creation."""
        window = TimeWindow(
            window_id="fold_1",
            fold_index=0,
            train_start="2020-01-01",
            train_end="2020-12-31",
            test_start="2021-01-05",
            test_end="2021-03-31",
            is_train=False,
            is_oos=True,
        )
        assert window.window_id == "fold_1"
        assert window.is_oos is True


class TestExperimentSpec:
    """Tests for ExperimentSpec model."""

    def test_basic_creation(self, sample_experiment_spec: ExperimentSpec) -> None:
        """Test basic experiment spec creation."""
        assert sample_experiment_spec.name == "Test_Experiment"
        assert sample_experiment_spec.strategy == "ma_cross"
        assert sample_experiment_spec.experiment_id is not None
        assert sample_experiment_spec.experiment_id.startswith("exp_")

    def test_parameter_grid_expansion(
        self, sample_experiment_spec: ExperimentSpec
    ) -> None:
        """Test parameter grid expansion."""
        grid = sample_experiment_spec.expand_parameter_grid()
        # 3 fast_period values × 3 slow_period values = 9 combinations
        assert len(grid) == 9

    def test_get_symbols(self, sample_experiment_spec: ExperimentSpec) -> None:
        """Test symbol retrieval."""
        symbols = sample_experiment_spec.get_symbols()
        assert symbols == ["AAPL", "MSFT"]

    def test_id_determinism(self) -> None:
        """Test that experiment ID is deterministic."""
        spec1 = ExperimentSpec(
            name="Test",
            strategy="ma_cross",
            parameters={"fast": {"type": "fixed", "value": 10}},
            universe={"type": "static", "symbols": ["AAPL"]},
            temporal={"folds": 5},
            reproducibility={"random_seed": 42, "data_version": "v1"},
        )
        spec2 = ExperimentSpec(
            name="Test",
            strategy="ma_cross",
            parameters={"fast": {"type": "fixed", "value": 10}},
            universe={"type": "static", "symbols": ["AAPL"]},
            temporal={"folds": 5},
            reproducibility={"random_seed": 42, "data_version": "v1"},
        )
        assert spec1.experiment_id == spec2.experiment_id

    def test_yaml_roundtrip(self, sample_experiment_spec: ExperimentSpec) -> None:
        """Test YAML save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.yaml"
            sample_experiment_spec.to_yaml(path)

            loaded = ExperimentSpec.from_yaml(path)
            assert loaded.name == sample_experiment_spec.name
            assert loaded.strategy == sample_experiment_spec.strategy


class TestTrialSpec:
    """Tests for TrialSpec model."""

    def test_basic_creation(self) -> None:
        """Test trial spec creation."""
        spec = TrialSpec(
            experiment_id="exp_abc123",
            params={"fast": 10, "slow": 50},
            trial_index=0,
        )
        assert spec.trial_id.startswith("trial_")
        assert spec.trial_index == 0

    def test_id_determinism(self) -> None:
        """Test that trial ID is deterministic."""
        spec1 = TrialSpec(
            experiment_id="exp_abc123",
            params={"fast": 10, "slow": 50},
        )
        spec2 = TrialSpec(
            experiment_id="exp_abc123",
            params={"fast": 10, "slow": 50},
        )
        assert spec1.trial_id == spec2.trial_id


class TestRunSpec:
    """Tests for RunSpec model."""

    def test_basic_creation(self) -> None:
        """Test run spec creation."""
        window = TimeWindow(
            window_id="fold_1",
            fold_index=0,
            train_start="2020-01-01",
            train_end="2020-12-31",
            test_start="2021-01-05",
            test_end="2021-03-31",
            is_train=False,
            is_oos=True,
        )
        spec = RunSpec(
            trial_id="trial_abc123",
            symbol="AAPL",
            window=window,
            profile_version="v1",
            data_version="test_v1",
            params={"fast": 10},
            experiment_id="exp_abc123",
        )
        assert spec.run_id.startswith("run_")
        assert spec.symbol == "AAPL"

    def test_id_determinism(self) -> None:
        """Test that run ID is deterministic."""
        window = TimeWindow(
            window_id="fold_1",
            fold_index=0,
            train_start="2020-01-01",
            train_end="2020-12-31",
            test_start="2021-01-05",
            test_end="2021-03-31",
        )
        spec1 = RunSpec(
            trial_id="trial_abc123",
            symbol="AAPL",
            window=window,
            profile_version="v1",
            data_version="v1",
            params={"fast": 10},
            experiment_id="exp_abc123",
        )
        spec2 = RunSpec(
            trial_id="trial_abc123",
            symbol="AAPL",
            window=window,
            profile_version="v1",
            data_version="v1",
            params={"fast": 10},
            experiment_id="exp_abc123",
        )
        assert spec1.run_id == spec2.run_id
