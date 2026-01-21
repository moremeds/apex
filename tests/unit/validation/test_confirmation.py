"""Tests for Multi-Timeframe Confirmation Analysis."""

import pytest

from src.domain.signals.validation.confirmation import (
    ConfirmationResult,
    StrategyMetrics,
    apply_and_rule,
    apply_majority_vote,
    compare_strategies,
    compute_strategy_metrics,
)


class TestStrategyMetrics:
    """Tests for StrategyMetrics dataclass."""

    def test_f1_score(self):
        """Test F1 score calculation."""
        # precision=0.8, recall=0.6
        m = StrategyMetrics(
            strategy_name="test",
            precision=0.8,
            recall=0.6,
            false_positive_rate=0.2,
            true_positives=6,
            false_positives=2,  # precision = 6/(6+2) = 0.75 ish
            true_negatives=8,
            false_negatives=4,
            total_samples=20,
        )

        # F1 = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 0.685...
        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert abs(m.f1_score - expected_f1) < 0.001

    def test_f1_score_zero(self):
        """Test F1 score when precision and recall are zero."""
        m = StrategyMetrics(
            strategy_name="test",
            precision=0.0,
            recall=0.0,
            false_positive_rate=0.0,
            true_positives=0,
            false_positives=0,
            true_negatives=10,
            false_negatives=0,
            total_samples=10,
        )
        assert m.f1_score == 0.0

    def test_to_dict(self):
        """Test serialization."""
        m = StrategyMetrics(
            strategy_name="test",
            precision=0.8,
            recall=0.6,
            false_positive_rate=0.2,
            true_positives=6,
            false_positives=2,
            true_negatives=8,
            false_negatives=4,
            total_samples=20,
        )

        d = m.to_dict()

        assert d["strategy_name"] == "test"
        assert d["precision"] == 0.8
        assert "f1_score" in d


class TestComputeStrategyMetrics:
    """Tests for compute_strategy_metrics function."""

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        predictions = [True, True, False, False]
        actuals = [True, True, False, False]

        m = compute_strategy_metrics(predictions, actuals, "test")

        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.false_positive_rate == 0.0
        assert m.true_positives == 2
        assert m.true_negatives == 2

    def test_all_false_positives(self):
        """Test with all false positives."""
        predictions = [True, True, True, True]
        actuals = [False, False, False, False]

        m = compute_strategy_metrics(predictions, actuals, "test")

        assert m.precision == 0.0
        assert m.false_positive_rate == 1.0
        assert m.false_positives == 4

    def test_mixed_results(self):
        """Test with mixed results."""
        # 3 TP, 2 FP, 2 TN, 1 FN
        predictions = [True, True, True, True, True, False, False, False]
        actuals = [True, True, True, False, False, False, False, True]

        m = compute_strategy_metrics(predictions, actuals, "test")

        assert m.true_positives == 3
        assert m.false_positives == 2
        assert m.true_negatives == 2
        assert m.false_negatives == 1
        assert m.precision == 3 / 5  # 3 / (3+2)
        assert m.recall == 3 / 4  # 3 / (3+1)
        assert m.false_positive_rate == 2 / 4  # 2 / (2+2)

    def test_empty_inputs(self):
        """Test with empty inputs."""
        m = compute_strategy_metrics([], [], "test")

        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.total_samples == 0

    def test_length_mismatch(self):
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError, match="same length"):
            compute_strategy_metrics([True], [True, False], "test")


class TestCompareStrategies:
    """Tests for compare_strategies function."""

    def test_s2_better(self):
        """Test when S2 is clearly better (lower FP rate)."""
        # S1: predicts True often (more FPs)
        # S2: more conservative (fewer FPs)
        actuals = [True, True, False, False, False, False, False, False]
        s1_preds = [True, True, True, True, True, False, False, False]
        s2_preds = [True, True, False, False, False, False, False, False]

        result = compare_strategies(
            s1_preds, s2_preds, actuals, n_bootstrap=100, seed=42
        )

        # S2 should have lower FP rate
        assert result.s2.false_positive_rate < result.s1.false_positive_rate
        assert result.delta_fp_rate < 0

    def test_s2_worse(self):
        """Test when S2 is worse (misses too much)."""
        actuals = [True, True, True, True, False, False]
        s1_preds = [True, True, True, True, False, False]  # Perfect
        s2_preds = [True, True, False, False, False, False]  # Misses 2

        result = compare_strategies(
            s1_preds, s2_preds, actuals, n_bootstrap=100, seed=42
        )

        # S2 has lower recall/precision
        assert result.s2.recall < result.s1.recall

    def test_result_structure(self):
        """Test that result has all required fields."""
        actuals = [True, True, False, False]
        s1_preds = [True, True, True, False]
        s2_preds = [True, False, False, False]

        result = compare_strategies(
            s1_preds, s2_preds, actuals, n_bootstrap=50
        )

        assert hasattr(result, "s1")
        assert hasattr(result, "s2")
        assert hasattr(result, "delta_precision")
        assert hasattr(result, "delta_fp_rate")
        assert hasattr(result, "confirmation_value")
        assert result.confirmation_value in ["POSITIVE", "NEGATIVE", "NEUTRAL"]


class TestConfirmationResult:
    """Tests for ConfirmationResult."""

    def test_passes_gates_success(self):
        """Test when gates pass."""
        s1 = StrategyMetrics(
            "s1", precision=0.70, recall=0.80,
            false_positive_rate=0.20, true_positives=70,
            false_positives=20, true_negatives=80,
            false_negatives=30, total_samples=200,
        )
        s2 = StrategyMetrics(
            "s2", precision=0.72, recall=0.75,
            false_positive_rate=0.10, true_positives=65,
            false_positives=10, true_negatives=90,
            false_negatives=35, total_samples=200,
        )

        result = ConfirmationResult(
            s1=s1,
            s2=s2,
            s1_ci_precision=(0.65, 0.75),
            s1_ci_fp_rate=(0.15, 0.25),
            s2_ci_precision=(0.68, 0.76),
            s2_ci_fp_rate=(0.05, 0.15),
            delta_precision=0.02,
            delta_fp_rate=-0.10,  # S2 is 10% better
            confirmation_value="POSITIVE",
        )

        passes, failures = result.passes_gates(
            min_fp_reduction=0.05,
            max_precision_drop=0.02,
        )

        assert passes
        assert len(failures) == 0

    def test_passes_gates_fails_fp_reduction(self):
        """Test when FP reduction gate fails."""
        s1 = StrategyMetrics(
            "s1", precision=0.70, recall=0.80,
            false_positive_rate=0.20, true_positives=70,
            false_positives=20, true_negatives=80,
            false_negatives=30, total_samples=200,
        )
        s2 = StrategyMetrics(
            "s2", precision=0.70, recall=0.75,
            false_positive_rate=0.18, true_positives=65,
            false_positives=18, true_negatives=82,
            false_negatives=35, total_samples=200,
        )

        result = ConfirmationResult(
            s1=s1, s2=s2,
            s1_ci_precision=(0.65, 0.75),
            s1_ci_fp_rate=(0.15, 0.25),
            s2_ci_precision=(0.65, 0.75),
            s2_ci_fp_rate=(0.13, 0.23),
            delta_precision=0.0,
            delta_fp_rate=-0.02,  # Only 2% reduction
            confirmation_value="NEUTRAL",
        )

        passes, failures = result.passes_gates(min_fp_reduction=0.05)

        assert not passes
        assert any("FP rate reduction" in f for f in failures)

    def test_to_dict(self):
        """Test serialization."""
        s1 = StrategyMetrics(
            "1d_only", precision=0.70, recall=0.80,
            false_positive_rate=0.20, true_positives=70,
            false_positives=20, true_negatives=80,
            false_negatives=30, total_samples=200,
        )
        s2 = StrategyMetrics(
            "1d_and_4h", precision=0.72, recall=0.75,
            false_positive_rate=0.10, true_positives=65,
            false_positives=10, true_negatives=90,
            false_negatives=35, total_samples=200,
        )

        result = ConfirmationResult(
            s1=s1, s2=s2,
            s1_ci_precision=(0.65, 0.75),
            s1_ci_fp_rate=(0.15, 0.25),
            s2_ci_precision=(0.68, 0.76),
            s2_ci_fp_rate=(0.05, 0.15),
            delta_precision=0.02,
            delta_fp_rate=-0.10,
            confirmation_value="POSITIVE",
        )

        d = result.to_dict()

        assert "strategy_comparison" in d
        assert "S1_1d_only" in d["strategy_comparison"]
        assert "S2_1d_and_4h" in d["strategy_comparison"]
        assert "delta_fp_rate" in d["strategy_comparison"]


class TestApplyAndRule:
    """Tests for apply_and_rule function."""

    def test_and_logic(self):
        """Test basic AND logic."""
        tf1 = [True, True, False, False]
        tf2 = [True, False, True, False]

        result = apply_and_rule(tf1, tf2)

        assert result == [True, False, False, False]

    def test_length_mismatch(self):
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError, match="same length"):
            apply_and_rule([True], [True, False])


class TestApplyMajorityVote:
    """Tests for apply_majority_vote function."""

    def test_majority_3_of_3(self):
        """Test majority vote with 3 TFs."""
        tf1 = [True, True, True, False]
        tf2 = [True, True, False, False]
        tf3 = [True, False, False, False]

        result = apply_majority_vote([tf1, tf2, tf3])

        # Majority (2 of 3) needed
        assert result == [True, True, False, False]

    def test_custom_min_agree(self):
        """Test with custom min_agree."""
        tf1 = [True, True, True]
        tf2 = [True, True, False]
        tf3 = [True, False, False]

        # Require all 3
        result = apply_majority_vote([tf1, tf2, tf3], min_agree=3)
        assert result == [True, False, False]

        # Require at least 1
        result = apply_majority_vote([tf1, tf2, tf3], min_agree=1)
        assert result == [True, True, True]

    def test_empty_input(self):
        """Test with empty input."""
        result = apply_majority_vote([])
        assert result == []

    def test_length_mismatch(self):
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError, match="same length"):
            apply_majority_vote([[True], [True, False]])
