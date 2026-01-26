"""Tests for Output JSON Schemas."""

from typing import Any

import pytest

from src.domain.signals.validation.confirmation import ConfirmationResult, StrategyMetrics
from src.domain.signals.validation.earliness import EarlinessResult
from src.domain.signals.validation.schemas import (
    BarValidation,
    GateResult,
    HorizonConfig,
    LabelerThreshold,
    SplitConfig,
    ValidationOutput,
    create_fast_validation_output,
    create_full_validation_output,
)
from src.domain.signals.validation.statistics import StatisticalResult


class TestHorizonConfig:
    """Tests for HorizonConfig."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        cfg = HorizonConfig(
            horizon_calendar_days=20,
            bars_per_day={"1d": 1.0, "4h": 1.625},
            horizon_bars_by_tf={"1d": 20, "4h": 32},
        )

        d = cfg.to_dict()

        assert d["horizon_calendar_days"] == 20
        assert d["bars_per_day"]["1d"] == 1.0
        assert d["horizon_bars_by_tf"]["4h"] == 32


class TestLabelerThreshold:
    """Tests for LabelerThreshold."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        t = LabelerThreshold(
            version="v1.0",
            trending_forward_return_min=0.10,
            trending_sharpe_min=1.0,
            choppy_volatility_min=0.25,
            choppy_drawdown_max=-0.10,
        )

        d = t.to_dict()

        assert d["version"] == "v1.0"
        assert d["trending_forward_return_min"] == 0.10


class TestSplitConfig:
    """Tests for SplitConfig."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        cfg = SplitConfig(
            outer_folds=5,
            inner_folds=3,
            purge_bars_by_tf={"1d": 20, "4h": 32},
            embargo_bars_by_tf={"1d": 10, "4h": 16},
        )

        d = cfg.to_dict()

        assert d["outer_folds"] == 5
        assert d["purge_bars_by_tf"]["4h"] == 32


class TestBarValidation:
    """Tests for BarValidation."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        bv = BarValidation(
            requested_bars=350,
            loaded_bars=348,
            usable_bars=340,
            validated_bars=88,
            max_lookback_indicator="SMA_200",
            reasons=["WEEKEND_GAP", "NAN_DROP"],
        )

        d = bv.to_dict()

        assert d["requested_bars"] == 350
        assert "SMA_200" in d["max_lookback_indicator"]
        assert len(d["reasons"]) == 2


class TestGateResult:
    """Tests for GateResult."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        g = GateResult(
            gate_name="trending_r0",
            passed=True,
            value=0.75,
            threshold=0.50,
            message="Trending R0 rate: 0.750",
        )

        d = g.to_dict()

        assert d["gate_name"] == "trending_r0"
        assert d["passed"] is True
        assert d["value"] == 0.75


class TestValidationOutput:
    """Tests for ValidationOutput."""

    def test_default_values(self) -> None:
        """Test default values."""
        vo = ValidationOutput()

        assert vo.version == "m2_v2.0"
        assert vo.mode == "full"
        assert vo.all_gates_passed is False
        assert len(vo.training_symbols) == 0

    def test_to_dict_minimal(self) -> None:
        """Test serialization with minimal data."""
        vo = ValidationOutput(mode="fast")

        d = vo.to_dict()

        assert d["version"] == "m2_v2.0"
        assert d["mode"] == "fast"
        assert "generated_at" in d
        assert "universe" in d

    def test_to_dict_with_gates(self) -> None:
        """Test serialization with gate results."""
        vo = ValidationOutput(
            mode="fast",
            gate_results=[
                GateResult("test", True, 0.75, 0.50, "Test passed"),
            ],
            all_gates_passed=True,
        )

        d = vo.to_dict()

        assert len(d["gate_results"]) == 1
        assert d["all_gates_passed"] is True


class TestCreateFastValidationOutput:
    """Tests for create_fast_validation_output."""

    def test_all_gates_pass(self) -> None:
        """Test when all gates pass."""
        vo = create_fast_validation_output(
            trending_r0_rate=0.75,
            choppy_r0_rate=0.20,
            causality_passed=True,
            symbols=["AAPL", "MSFT"],
        )

        assert vo.mode == "fast"
        assert vo.all_gates_passed is True
        assert len(vo.gate_results) == 3
        assert vo.training_symbols == ["AAPL", "MSFT"]

    def test_trending_r0_fails(self) -> None:
        """Test when trending R0 gate fails."""
        vo = create_fast_validation_output(
            trending_r0_rate=0.40,  # Below 0.50
            choppy_r0_rate=0.20,
            causality_passed=True,
            symbols=["AAPL"],
        )

        assert vo.all_gates_passed is False
        trending_gate = next(g for g in vo.gate_results if g.gate_name == "trending_r0")
        assert trending_gate.passed is False

    def test_choppy_r0_fails(self) -> None:
        """Test when choppy R0 gate fails."""
        vo = create_fast_validation_output(
            trending_r0_rate=0.75,
            choppy_r0_rate=0.50,  # Above 0.40
            causality_passed=True,
            symbols=["AAPL"],
        )

        assert vo.all_gates_passed is False
        choppy_gate = next(g for g in vo.gate_results if g.gate_name == "choppy_r0")
        assert choppy_gate.passed is False

    def test_causality_fails(self) -> None:
        """Test when causality gate fails."""
        vo = create_fast_validation_output(
            trending_r0_rate=0.75,
            choppy_r0_rate=0.20,
            causality_passed=False,
            symbols=["AAPL"],
        )

        assert vo.all_gates_passed is False
        causality_gate = next(g for g in vo.gate_results if g.gate_name == "causality_g7")
        assert causality_gate.passed is False


class TestCreateFullValidationOutput:
    """Tests for create_full_validation_output."""

    @pytest.fixture
    def good_statistical_result(self):
        """Create good statistical result."""
        return StatisticalResult(
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

    @pytest.fixture
    def horizon_config(self):
        """Create horizon config."""
        return HorizonConfig(
            horizon_calendar_days=20,
            bars_per_day={"1d": 1.0, "4h": 1.625},
            horizon_bars_by_tf={"1d": 20, "4h": 32},
        )

    @pytest.fixture
    def split_config(self):
        """Create split config."""
        return SplitConfig(
            outer_folds=5,
            inner_folds=3,
            purge_bars_by_tf={"1d": 20},
            embargo_bars_by_tf={"1d": 10},
        )

    def test_all_gates_pass(
        self, good_statistical_result: Any, horizon_config: Any, split_config: Any
    ) -> None:
        """Test when all gates pass."""
        earliness = {
            "4h_vs_1d": EarlinessResult(
                tf_pair="4h_vs_1d",
                median_earliness_days=1.5,
                p75_earliness_days=2.5,
                pct_earlier_than_baseline=0.70,
                ci_95=(1.0, 2.0),
                n_episodes=30,
                earliness_values=[1.5] * 30,
            )
        }

        s1 = StrategyMetrics(
            "1d_only",
            precision=0.70,
            recall=0.80,
            false_positive_rate=0.20,
            true_positives=70,
            false_positives=20,
            true_negatives=80,
            false_negatives=30,
            total_samples=200,
        )
        s2 = StrategyMetrics(
            "1d_and_4h",
            precision=0.71,
            recall=0.75,
            false_positive_rate=0.10,
            true_positives=65,
            false_positives=10,
            true_negatives=90,
            false_negatives=35,
            total_samples=200,
        )
        confirmation = ConfirmationResult(
            s1=s1,
            s2=s2,
            s1_ci_precision=(0.65, 0.75),
            s1_ci_fp_rate=(0.15, 0.25),
            s2_ci_precision=(0.66, 0.76),
            s2_ci_fp_rate=(0.05, 0.15),
            delta_precision=0.01,
            delta_fp_rate=-0.10,
            confirmation_value="POSITIVE",
        )

        vo = create_full_validation_output(
            statistical_result=good_statistical_result,
            earliness_by_tf=earliness,
            confirmation_result=confirmation,
            horizon_config=horizon_config,
            split_config=split_config,
            labeler_thresholds={"1d": LabelerThreshold("v1.0", 0.10, 1.0, 0.25, -0.10)},
            training_symbols=["AAPL", "MSFT"],
            holdout_symbols=["GME"],
        )

        assert vo.mode == "full"
        assert vo.all_gates_passed is True
        assert len(vo.gate_results) >= 6  # At least 4 statistical + 2 earliness + 2 confirmation

    def test_statistical_failure(self, horizon_config: Any, split_config: Any) -> None:
        """Test when statistical gates fail."""
        bad_stats = StatisticalResult(
            n_trending_symbols=70,
            n_choppy_symbols=70,
            trending_r0_rates=[0.5] * 70,
            choppy_r0_rates=[0.4] * 70,
            t_statistic=2.0,
            p_value=0.10,  # Too high
            effect_size_cohens_d=0.3,  # Too low
            trending_mean=0.50,
            trending_ci_lower=0.45,  # Too low
            trending_ci_upper=0.55,
            choppy_mean=0.40,
            choppy_ci_lower=0.35,
            choppy_ci_upper=0.45,  # Too high
            n_bootstrap_samples=1000,
            block_size_bars=20,
        )

        vo = create_full_validation_output(
            statistical_result=bad_stats,
            earliness_by_tf={},
            confirmation_result=None,
            horizon_config=horizon_config,
            split_config=split_config,
            labeler_thresholds={},
            training_symbols=[],
            holdout_symbols=[],
        )

        assert vo.all_gates_passed is False
        # Check individual gates failed
        cohens_gate = next(g for g in vo.gate_results if g.gate_name == "cohens_d")
        assert cohens_gate.passed is False
