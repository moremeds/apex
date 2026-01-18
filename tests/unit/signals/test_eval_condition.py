"""
Tests for eval_condition() and EvalResult with Hypothesis property tests.

Phase 1: Mathematical Integrity Tests

These tests verify the core invariants:
- Invariant A: passed=True ⇒ gap ≤ 0, passed=False ⇒ gap > 0
- Invariant B: direction ↔ operator (structural, not text-based)
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.domain.signals.indicators.regime.rule_trace import (
    EvalResult,
    RuleTrace,
    ThresholdInfo,
    eval_condition,
    generate_counterfactual,
    generate_counterfactual_v2,
)

# =============================================================================
# SAFE FLOAT STRATEGY (no NaN, no inf, reasonable range)
# =============================================================================

# Safe float strategy: no NaN, no inf, reasonable range for financial metrics
safe_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


# =============================================================================
# UNIT TESTS FOR eval_condition()
# =============================================================================


class TestEvalConditionBasic:
    """Basic unit tests for eval_condition()."""

    def test_greater_than_passed(self):
        """Test > operator when condition passes."""
        result = eval_condition("close", 80.0, 70.0, ">", "$")
        assert result.passed is True
        assert result.gap == -10.0  # 70 - 80 = -10 (negative = already crossed)
        assert result.direction == "increase"
        assert result.rendered_op == "above"

    def test_greater_than_failed(self):
        """Test > operator when condition fails."""
        result = eval_condition("close", 60.0, 70.0, ">", "$")
        assert result.passed is False
        assert result.gap == 10.0  # 70 - 60 = 10 (positive = need to increase)
        assert result.direction == "increase"

    def test_greater_equal_at_threshold(self):
        """Test >= operator at exactly threshold."""
        result = eval_condition("pctile", 70.0, 70.0, ">=", "%")
        assert result.passed is True
        assert result.gap == 0.0  # At threshold
        assert result.direction == "increase"

    def test_less_than_passed(self):
        """Test < operator when condition passes."""
        result = eval_condition("chop_pct", 60.0, 70.0, "<", "%")
        assert result.passed is True
        assert result.gap == -10.0  # 60 - 70 = -10 (negative = already crossed)
        assert result.direction == "decrease"
        assert result.rendered_op == "below"

    def test_less_than_failed(self):
        """Test < operator when condition fails."""
        result = eval_condition("chop_pct", 80.0, 70.0, "<", "%")
        assert result.passed is False
        assert result.gap == 10.0  # 80 - 70 = 10 (positive = need to decrease)
        assert result.direction == "decrease"

    def test_less_equal_at_threshold(self):
        """Test <= operator at exactly threshold."""
        result = eval_condition("ext", -2.0, -2.0, "<=", " ATR")
        assert result.passed is True
        assert result.gap == 0.0  # At threshold
        assert result.direction == "decrease"

    def test_equality_operator_rejected(self):
        """Test that == operator is explicitly rejected."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            eval_condition("metric", 70.0, 70.0, "==")

    def test_invalid_operator_rejected(self):
        """Test that invalid operators are rejected."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            eval_condition("metric", 70.0, 70.0, "!=")


class TestEvalConditionTestMatrix:
    """
    Test matrix from the plan:

    | current | threshold | op   | passed | gap      | direction  | invariant_a |
    |---------|-----------|------|--------|----------|------------|-------------|
    | 80      | 70        | >    | True   | -10      | increase   | ✓ (≤0)      |
    | 60      | 70        | >    | False  | +10      | increase   | ✓ (>0)      |
    | 70      | 70        | >=   | True   | 0        | increase   | ✓ (≤0)      |
    | 60      | 70        | <    | True   | -10      | decrease   | ✓ (≤0)      |
    | 80      | 70        | <    | False  | +10      | decrease   | ✓ (>0)      |
    """

    @pytest.mark.parametrize(
        "current,threshold,operator,expected_passed,expected_gap,expected_direction",
        [
            (80, 70, ">", True, -10, "increase"),
            (60, 70, ">", False, 10, "increase"),
            (70, 70, ">=", True, 0, "increase"),
            (60, 70, "<", True, -10, "decrease"),
            (80, 70, "<", False, 10, "decrease"),
            (70, 70, "<=", True, 0, "decrease"),  # Edge: equal, <= passes
        ],
    )
    def test_matrix(
        self, current, threshold, operator, expected_passed, expected_gap, expected_direction
    ):
        """Test the complete matrix from the plan."""
        result = eval_condition("metric", current, threshold, operator)
        assert result.passed == expected_passed
        assert result.gap == expected_gap
        assert result.direction == expected_direction

        # Verify Invariant A
        if expected_passed:
            assert result.gap <= 0, f"Invariant A violated: passed=True but gap={result.gap}"
        else:
            assert result.gap > 0, f"Invariant A violated: passed=False but gap={result.gap}"

    def test_strict_inequality_at_threshold(self):
        """
        Edge case: current == threshold with strict operators (> or <).

        Gap should be tiny epsilon (1e-10) to maintain Invariant A.
        """
        # Test > at threshold
        result_gt = eval_condition("metric", 70, 70, ">")
        assert result_gt.passed is False
        assert result_gt.gap == pytest.approx(1e-10)  # Tiny epsilon, not 0
        assert result_gt.direction == "increase"

        # Test < at threshold
        result_lt = eval_condition("metric", 70, 70, "<")
        assert result_lt.passed is False
        assert result_lt.gap == pytest.approx(1e-10)  # Tiny epsilon, not 0
        assert result_lt.direction == "decrease"


# =============================================================================
# HYPOTHESIS PROPERTY TESTS FOR INVARIANTS
# =============================================================================


class TestInvariantAProperty:
    """
    Invariant A: passed ↔ gap sign consistency

    - passed=True ⇒ gap ≤ 0
    - passed=False ⇒ gap > 0
    """

    @given(current=safe_floats, threshold=safe_floats)
    @settings(max_examples=200)
    def test_invariant_a_for_greater_than(self, current, threshold):
        """Invariant A: passed ↔ gap sign consistency for > operator."""
        result = eval_condition("metric", current, threshold, ">")
        if result.passed:
            assert result.gap <= 0, f"passed=True but gap={result.gap} > 0"
        else:
            assert result.gap > 0, f"passed=False but gap={result.gap} <= 0"

    @given(current=safe_floats, threshold=safe_floats)
    @settings(max_examples=200)
    def test_invariant_a_for_greater_equal(self, current, threshold):
        """Invariant A: passed ↔ gap sign consistency for >= operator."""
        result = eval_condition("metric", current, threshold, ">=")
        if result.passed:
            assert result.gap <= 0, f"passed=True but gap={result.gap} > 0"
        else:
            assert result.gap > 0, f"passed=False but gap={result.gap} <= 0"

    @given(current=safe_floats, threshold=safe_floats)
    @settings(max_examples=200)
    def test_invariant_a_for_less_than(self, current, threshold):
        """Invariant A: passed ↔ gap sign consistency for < operator."""
        result = eval_condition("metric", current, threshold, "<")
        if result.passed:
            assert result.gap <= 0, f"passed=True but gap={result.gap} > 0"
        else:
            assert result.gap > 0, f"passed=False but gap={result.gap} <= 0"

    @given(current=safe_floats, threshold=safe_floats)
    @settings(max_examples=200)
    def test_invariant_a_for_less_equal(self, current, threshold):
        """Invariant A: passed ↔ gap sign consistency for <= operator."""
        result = eval_condition("metric", current, threshold, "<=")
        if result.passed:
            assert result.gap <= 0, f"passed=True but gap={result.gap} > 0"
        else:
            assert result.gap > 0, f"passed=False but gap={result.gap} <= 0"


class TestInvariantBProperty:
    """
    Invariant B: direction ↔ operator (structural, not text-based)

    - For > or >=: direction = "increase"
    - For < or <=: direction = "decrease"
    """

    @given(
        current=safe_floats,
        threshold=safe_floats,
        op=st.sampled_from(["<", "<=", ">", ">="]),
    )
    @settings(max_examples=300)
    def test_invariant_b_direction_matches_operator(self, current, threshold, op):
        """Invariant B: direction ↔ operator (structural)."""
        result = eval_condition("metric", current, threshold, op)
        if op in (">", ">="):
            assert (
                result.direction == "increase"
            ), f"Expected direction='increase' for operator {op}"
        else:
            assert (
                result.direction == "decrease"
            ), f"Expected direction='decrease' for operator {op}"


class TestGapSemantics:
    """Test gap calculation semantics."""

    @given(current=safe_floats, threshold=safe_floats)
    @settings(max_examples=200)
    def test_gap_formula_for_greater_operators(self, current, threshold):
        """
        For > or >=: gap = threshold - current.

        Edge case: when current == threshold with >, gap is epsilon (1e-10)
        instead of 0 to maintain Invariant A.
        """
        for op in [">", ">="]:
            result = eval_condition("metric", current, threshold, op)
            expected_gap = threshold - current

            # Handle edge case: strict inequality at threshold
            if op == ">" and current == threshold:
                expected_gap = 1e-10  # Epsilon to maintain invariant

            assert result.gap == pytest.approx(
                expected_gap
            ), f"Expected gap={expected_gap}, got {result.gap} for {op}"

    @given(current=safe_floats, threshold=safe_floats)
    @settings(max_examples=200)
    def test_gap_formula_for_less_operators(self, current, threshold):
        """
        For < or <=: gap = current - threshold.

        Edge case: when current == threshold with <, gap is epsilon (1e-10)
        instead of 0 to maintain Invariant A.
        """
        for op in ["<", "<="]:
            result = eval_condition("metric", current, threshold, op)
            expected_gap = current - threshold

            # Handle edge case: strict inequality at threshold
            if op == "<" and current == threshold:
                expected_gap = 1e-10  # Epsilon to maintain invariant

            assert result.gap == pytest.approx(
                expected_gap
            ), f"Expected gap={expected_gap}, got {result.gap} for {op}"


# =============================================================================
# EVAL RESULT TESTS
# =============================================================================


class TestEvalResultConstruction:
    """Test EvalResult invariant enforcement on construction."""

    def test_valid_passed_with_negative_gap(self):
        """Valid: passed=True with gap <= 0."""
        result = EvalResult(
            passed=True,
            gap=-10.0,
            direction="increase",
            rendered_op="above",
            metric_name="close",
            current_value=80.0,
            threshold=70.0,
        )
        assert result.passed is True
        assert result.gap == -10.0

    def test_valid_passed_with_zero_gap(self):
        """Valid: passed=True with gap = 0 (at threshold for >= or <=)."""
        result = EvalResult(
            passed=True,
            gap=0.0,
            direction="increase",
            rendered_op="at least",
            metric_name="pctile",
            current_value=70.0,
            threshold=70.0,
        )
        assert result.passed is True
        assert result.gap == 0.0

    def test_valid_failed_with_positive_gap(self):
        """Valid: passed=False with gap > 0."""
        result = EvalResult(
            passed=False,
            gap=10.0,
            direction="increase",
            rendered_op="above",
            metric_name="close",
            current_value=60.0,
            threshold=70.0,
        )
        assert result.passed is False
        assert result.gap == 10.0

    def test_invalid_passed_with_positive_gap_raises(self):
        """Invalid: passed=True with gap > 0 should raise."""
        with pytest.raises(ValueError, match="Invariant A violated"):
            EvalResult(
                passed=True,
                gap=10.0,  # Invalid: positive gap with passed=True
                direction="increase",
                rendered_op="above",
                metric_name="close",
                current_value=60.0,
                threshold=70.0,
            )

    def test_invalid_failed_with_negative_gap_raises(self):
        """Invalid: passed=False with gap < 0 should raise."""
        with pytest.raises(ValueError, match="Invariant A violated"):
            EvalResult(
                passed=False,
                gap=-10.0,  # Invalid: negative gap with passed=False
                direction="increase",
                rendered_op="above",
                metric_name="close",
                current_value=80.0,
                threshold=70.0,
            )

    def test_invalid_failed_with_zero_gap_raises(self):
        """Invalid: passed=False with gap = 0 should raise."""
        with pytest.raises(ValueError, match="Invariant A violated"):
            EvalResult(
                passed=False,
                gap=0.0,  # Invalid: zero gap with passed=False
                direction="increase",
                rendered_op="above",
                metric_name="close",
                current_value=70.0,
                threshold=70.0,
            )


class TestEvalResultFormatting:
    """Test EvalResult formatting methods."""

    def test_format_comparison_pass(self):
        """Test format_comparison for passing condition."""
        result = eval_condition("close", 80.0, 70.0, ">", "$")
        formatted = result.format_comparison()
        assert "close" in formatted
        assert "80.00$" in formatted
        assert "above" in formatted
        assert "70.00$" in formatted
        assert "PASS" in formatted

    def test_format_comparison_fail(self):
        """Test format_comparison for failing condition."""
        result = eval_condition("close", 60.0, 70.0, ">", "$")
        formatted = result.format_comparison()
        assert "FAIL" in formatted

    def test_format_counterfactual_for_failed(self):
        """Test format_counterfactual for failed condition."""
        result = eval_condition("close", 60.0, 70.0, ">", "$")
        formatted = result.format_counterfactual()
        assert "increase" in formatted
        assert "10.00$" in formatted

    def test_format_counterfactual_for_passed(self):
        """Test format_counterfactual for passed condition."""
        result = eval_condition("close", 80.0, 70.0, ">", "$")
        formatted = result.format_counterfactual()
        assert "already satisfied" in formatted


# =============================================================================
# THRESHOLD INFO TESTS
# =============================================================================


class TestThresholdInfoFromEvalResult:
    """Test ThresholdInfo.from_eval_result() conversion."""

    def test_from_eval_result_preserves_fields(self):
        """Test that from_eval_result preserves all fields."""
        eval_result = eval_condition("atr_pct", 60.0, 80.0, ">=", "%")
        ti = ThresholdInfo.from_eval_result(eval_result)

        assert ti.metric_name == "atr_pct"
        assert ti.current_value == 60.0
        assert ti.threshold == 80.0
        assert ti.operator == ">="
        assert ti.gap == eval_result.gap
        assert ti.unit == "%"
        assert ti.direction == "increase"
        assert ti.passed is False


# =============================================================================
# COUNTERFACTUAL GENERATION TESTS
# =============================================================================


class TestCounterfactualGeneration:
    """Test counterfactual generation functions."""

    def test_generate_counterfactual_filters_by_passed(self):
        """Test that generate_counterfactual only includes failed conditions."""
        # Create rule traces with mixed passed/failed
        passed_result = eval_condition("metric1", 80.0, 70.0, ">")
        failed_result = eval_condition("metric2", 60.0, 70.0, ">")

        rules = [
            RuleTrace(
                rule_id="r1",
                description="Test rule 1",
                passed=True,
                evidence={},
                regime_target="R0",
                category="test",
                priority=1,
                eval_result=passed_result,
            ),
            RuleTrace(
                rule_id="r2",
                description="Test rule 2",
                passed=False,
                evidence={},
                regime_target="R0",
                category="test",
                priority=1,
                eval_result=failed_result,
            ),
        ]

        result = generate_counterfactual(rules, "R0")
        assert len(result) == 1
        assert result[0].metric_name == "metric2"

    def test_generate_counterfactual_v2_returns_eval_results(self):
        """Test that generate_counterfactual_v2 returns EvalResult objects."""
        failed_result = eval_condition("metric", 60.0, 70.0, ">")

        rules = [
            RuleTrace(
                rule_id="r1",
                description="Test rule",
                passed=False,
                evidence={},
                regime_target="R0",
                category="test",
                priority=1,
                eval_result=failed_result,
            ),
        ]

        result = generate_counterfactual_v2(rules, "R0")
        assert len(result) == 1
        assert isinstance(result[0], EvalResult)
        assert result[0].metric_name == "metric"

    def test_generate_counterfactual_sorts_by_gap(self):
        """Test that results are sorted by gap (smallest first)."""
        result1 = eval_condition("metric1", 60.0, 70.0, ">")  # gap = 10
        result2 = eval_condition("metric2", 65.0, 70.0, ">")  # gap = 5
        result3 = eval_condition("metric3", 50.0, 70.0, ">")  # gap = 20

        rules = [
            RuleTrace(
                rule_id="r1",
                description="Rule 1",
                passed=False,
                evidence={},
                regime_target="R0",
                category="test",
                priority=1,
                eval_result=result1,
            ),
            RuleTrace(
                rule_id="r2",
                description="Rule 2",
                passed=False,
                evidence={},
                regime_target="R0",
                category="test",
                priority=1,
                eval_result=result2,
            ),
            RuleTrace(
                rule_id="r3",
                description="Rule 3",
                passed=False,
                evidence={},
                regime_target="R0",
                category="test",
                priority=1,
                eval_result=result3,
            ),
        ]

        result = generate_counterfactual(rules, "R0")
        assert len(result) == 2
        assert result[0].metric_name == "metric2"  # gap = 5, smallest
        assert result[1].metric_name == "metric1"  # gap = 10, second smallest
