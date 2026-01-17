"""
Rule Trace Models for Regime Classification Explainability.

Provides structured tracing of rule evaluations to enable:
- Auditability: Every regime label traces back to concrete inputs + rules
- Counterfactuals: Show users what would need to change to reach a different regime
- Debugging: Identify which conditions passed/failed and why

Mathematical Invariants (Phase 1):
- Invariant A: passed=True ⇒ gap ≤ 0, passed=False ⇒ gap > 0
- Invariant B: direction field matches operator (structural, not text-based)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# =============================================================================
# EVAL RESULT & EVAL CONDITION (Phase 1 - Mathematical Integrity)
# =============================================================================


@dataclass(frozen=True)
class EvalResult:
    """
    Unified evaluation result with consistent semantics.

    This dataclass represents the result of evaluating a single condition
    with mathematically guaranteed invariants.

    Gap Convention:
    - For `>` or `>=`: gap = threshold - current (positive = need to increase)
    - For `<` or `<=`: gap = current - threshold (positive = need to decrease)

    Mathematical Invariants (REQUIRED - enforced via property tests):
    - Invariant A: passed=True ⇒ gap ≤ 0, passed=False ⇒ gap > 0
    - Invariant B: direction matches operator (structural field, not text)

    Attributes:
        passed: Whether the condition was satisfied
        gap: Distance to threshold (positive = needs to move toward threshold)
        direction: Structural direction field ("increase" or "decrease")
        rendered_op: Human-readable operator ("above", "below", "at least", "at most")
        metric_name: Name of the metric being evaluated
        current_value: The observed value
        threshold: The threshold value for comparison
        unit: Unit of measurement (e.g., "%", "ATR", "$")
    """

    passed: bool
    gap: float
    direction: Literal["increase", "decrease"]
    rendered_op: str
    metric_name: str
    current_value: float
    threshold: float
    unit: str = ""

    def __post_init__(self) -> None:
        """Validate invariants on construction."""
        # Invariant A check (allow tiny tolerance for floating point)
        if self.passed and self.gap > 1e-10:
            raise ValueError(
                f"Invariant A violated: passed=True but gap={self.gap} > 0 "
                f"for {self.metric_name}"
            )
        if not self.passed and self.gap <= 0:
            raise ValueError(
                f"Invariant A violated: passed=False but gap={self.gap} <= 0 "
                f"for {self.metric_name}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "passed": self.passed,
            "gap": self.gap,
            "direction": self.direction,
            "rendered_op": self.rendered_op,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "unit": self.unit,
        }

    def format_comparison(self) -> str:
        """Format as human-readable comparison string."""
        unit_str = self.unit if self.unit else ""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.metric_name}: {self.current_value:.2f}{unit_str} "
            f"{self.rendered_op} {self.threshold:.2f}{unit_str} → {status}"
        )

    def format_counterfactual(self) -> str:
        """
        Format counterfactual advice based on direction.

        Uses the structural direction field, not text matching.
        """
        if self.passed:
            return f"{self.metric_name}: condition already satisfied"

        verb = "increase" if self.direction == "increase" else "decrease"
        unit_str = self.unit if self.unit else ""
        return f"Need to {verb} {self.metric_name} by {abs(self.gap):.2f}{unit_str}"


# Operator rendering map
_OPERATOR_RENDER: Dict[str, str] = {
    ">": "above",
    ">=": "at least",
    "<": "below",
    "<=": "at most",
}


def eval_condition(
    metric_name: str,
    current_value: float,
    threshold: float,
    operator: Literal["<", "<=", ">", ">="],
    unit: str = "",
) -> EvalResult:
    """
    Evaluate a threshold condition with consistent gap semantics.

    This is the SINGLE SOURCE OF TRUTH for condition evaluation.
    All rule evaluations in the decision tree MUST use this function.

    Gap Convention:
    - For `>` or `>=`: gap = threshold - current (positive = need to increase)
                       direction = "increase"
    - For `<` or `<=`: gap = current - threshold (positive = need to decrease)
                       direction = "decrease"

    IMPORTANT: "==" operator is NOT supported.
    Financial metrics should never use exact equality due to floating-point noise.
    Use threshold bands (>, <) instead. If approximate equality is needed,
    use explicit epsilon: abs(current - threshold) <= eps.

    Args:
        metric_name: Name of the metric being compared
        current_value: The actual observed value
        threshold: The threshold value for comparison
        operator: Comparison operator ("<", "<=", ">", ">=")
        unit: Unit of measurement for display

    Returns:
        EvalResult with consistent gap/direction/passed semantics

    Raises:
        ValueError: If operator is unsupported (e.g., "==")
    """
    if operator not in ("<", "<=", ">", ">="):
        raise ValueError(
            f"Unsupported operator '{operator}'. "
            "Use <, <=, >, >= only. Equality (==) is not supported for floats."
        )

    # Evaluate the condition
    if operator == ">":
        passed = current_value > threshold
    elif operator == ">=":
        passed = current_value >= threshold
    elif operator == "<":
        passed = current_value < threshold
    else:  # operator == "<="
        passed = current_value <= threshold

    # Calculate gap based on operator
    # For > or >=: gap = threshold - current (positive means need to increase)
    # For < or <=: gap = current - threshold (positive means need to decrease)
    if operator in (">", ">="):
        gap = threshold - current_value
        direction: Literal["increase", "decrease"] = "increase"
    else:  # operator in ("<", "<=")
        gap = current_value - threshold
        direction = "decrease"

    # Handle edge case: when current == threshold with strict operators,
    # gap = 0 but passed = False. This violates Invariant A.
    # Use a small epsilon to ensure gap > 0 in this case.
    # This represents "infinitesimally close to threshold but still failing".
    if not passed and gap == 0:
        gap = 1e-10  # Tiny positive value to maintain invariant

    rendered_op = _OPERATOR_RENDER[operator]

    return EvalResult(
        passed=passed,
        gap=gap,
        direction=direction,
        rendered_op=rendered_op,
        metric_name=metric_name,
        current_value=current_value,
        threshold=threshold,
        unit=unit,
    )


@dataclass
class ThresholdInfo:
    """
    Details for threshold-based comparisons in rule evaluation.

    Used both for documenting threshold crossings and for counterfactual generation
    (showing users how far they are from triggering a different regime).

    NOTE: For new code, prefer using EvalResult directly. ThresholdInfo exists
    for backward compatibility and integration with RuleTrace.

    Attributes:
        metric_name: Name of the metric being compared (e.g., "atr_pctile_short")
        current_value: The actual value observed
        threshold: The threshold value for comparison
        operator: Comparison operator (">" or "<" or ">=" or "<=")
        gap: Distance to threshold (positive = how far from triggering)
             For categorical values, gap may be 0 or float('inf')
        unit: Unit of measurement ("%" or "ATR" or "" for dimensionless)
        direction: Structural direction ("increase" or "decrease") - Phase 1 addition
        passed: Whether the condition was satisfied - Phase 1 addition
    """

    metric_name: str
    current_value: float
    threshold: float
    operator: str  # ">" or "<" or ">=" or "<="
    gap: float  # Distance to threshold (positive = how far from triggering)
    unit: str = ""  # "%" or "ATR" or ""
    direction: Optional[str] = None  # "increase" or "decrease" - Phase 1 addition
    passed: Optional[bool] = None  # Whether condition passed - Phase 1 addition

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "operator": self.operator,
            "gap": self.gap,
            "unit": self.unit,
        }
        if self.direction is not None:
            result["direction"] = self.direction
        if self.passed is not None:
            result["passed"] = self.passed
        return result

    def format_comparison(self) -> str:
        """Format as human-readable comparison string."""
        unit_str = self.unit if self.unit else ""
        status_str = ""
        if self.passed is not None:
            status_str = " → PASS" if self.passed else " → FAIL"
        return (
            f"{self.metric_name}: {self.current_value:.2f}{unit_str} "
            f"{self.operator} {self.threshold:.2f}{unit_str}{status_str}"
        )

    def format_gap(self) -> str:
        """Format the gap for display using structural direction."""
        if self.gap == float("inf"):
            return "N/A (categorical)"
        unit_str = self.unit if self.unit else ""

        # Use direction field if available (Phase 1), otherwise infer from operator
        if self.direction:
            verb = "increase" if self.direction == "increase" else "decrease"
        else:
            verb = "increase" if self.operator in (">", ">=") else "decrease"

        return f"Need to {verb} by {abs(self.gap):.2f}{unit_str}"

    @classmethod
    def from_eval_result(cls, result: EvalResult) -> "ThresholdInfo":
        """
        Create ThresholdInfo from EvalResult for backward compatibility.

        This bridges the new eval_condition() system with existing code.
        """
        return cls(
            metric_name=result.metric_name,
            current_value=result.current_value,
            threshold=result.threshold,
            operator={"above": ">", "at least": ">=", "below": "<", "at most": "<="}[
                result.rendered_op
            ],
            gap=result.gap,
            unit=result.unit,
            direction=result.direction,
            passed=result.passed,
        )


@dataclass
class RuleTrace:
    """
    Trace of a single rule evaluation in the regime decision tree.

    Captures everything needed for audit trail and counterfactual generation:
    - What rule was evaluated
    - Whether it passed or failed
    - The evidence (input values) used
    - Which regime this rule supports
    - Structured failure info for counterfactuals

    Attributes:
        rule_id: Unique identifier for the rule (e.g., "r2_trend_down")
        description: Human-readable description (e.g., "R2: Trend is DOWN")
        passed: Whether the rule condition was satisfied
        evidence: Dict of input values used in evaluation
        regime_target: Which regime this rule supports ("R2", "R3", "R1", "R0")
        category: Component category ("trend" | "vol" | "chop" | "ext" | "iv" | "hysteresis")
        priority: Lower = higher priority in decision tree (R2=1, R3=2, R1=3, R0=4)
        threshold_info: For threshold-based rules, the comparison details
        failed_conditions: List of conditions that failed (empty if passed)
        eval_result: Phase 1 - EvalResult from eval_condition() for math integrity
    """

    rule_id: str
    description: str
    passed: bool
    evidence: Dict[str, Any]

    # For easy grouping and counterfactual generation
    regime_target: str  # "R2", "R3", "R1", "R0"
    category: str  # "trend" | "vol" | "chop" | "ext" | "iv" | "hysteresis"
    priority: int  # Lower = higher priority in decision tree

    # Structured failure info for counterfactuals
    threshold_info: Optional[ThresholdInfo] = None  # For threshold-based rules
    failed_conditions: List[ThresholdInfo] = field(default_factory=list)  # Empty if passed

    # Phase 1: EvalResult for mathematical integrity
    eval_result: Optional[EvalResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "rule_id": self.rule_id,
            "description": self.description,
            "passed": self.passed,
            "evidence": self.evidence,
            "regime_target": self.regime_target,
            "category": self.category,
            "priority": self.priority,
            "threshold_info": (self.threshold_info.to_dict() if self.threshold_info else None),
            "failed_conditions": [c.to_dict() for c in self.failed_conditions],
        }
        if self.eval_result:
            result["eval_result"] = self.eval_result.to_dict()
        return result


def generate_counterfactual(
    rules_fired: List[RuleTrace],
    target_regime: str,
) -> List[ThresholdInfo]:
    """
    Get the smallest-gap failed conditions to reach target regime.

    This enables showing users exactly what would need to change to reach
    a different regime classification.

    Phase 1 Implementation:
    1. ONLY include conditions where passed=False (drives inclusion)
    2. Gap is ONLY used for sorting (Invariant A guarantees gap > 0 for passed=False)
    3. Sort by gap ascending (smallest positive gap = closest to triggering first)

    NOTE: Step 2 filtering "gap <= 0" should NEVER trigger if Invariant A holds.
    If it does, the eval is broken.

    Args:
        rules_fired: All rules evaluated in the decision tree
        target_regime: The regime we want to know how to reach (e.g., "R0")

    Returns:
        Top 2 failed conditions with smallest gaps (closest to triggering)
    """
    # Filter rules that support target regime and didn't pass
    target_rules = [r for r in rules_fired if r.regime_target == target_regime and not r.passed]

    # Collect all failed conditions
    all_failures: List[ThresholdInfo] = []
    for rule in target_rules:
        # Phase 1: Prefer eval_result if available (has invariant guarantees)
        if rule.eval_result and not rule.eval_result.passed:
            all_failures.append(ThresholdInfo.from_eval_result(rule.eval_result))
        elif rule.failed_conditions:
            # Legacy path: use failed_conditions
            for fc in rule.failed_conditions:
                # Only include if passed=False (Phase 1 semantics)
                if fc.passed is None or not fc.passed:
                    all_failures.append(fc)

    # Filter out any with gap <= 0 (should not happen if Invariant A holds)
    # This is a safety check, not the primary filter
    valid_failures = [f for f in all_failures if f.gap > 0 or f.gap == float("inf")]

    # Sort by gap ascending (smallest positive gap = closest to triggering)
    valid_failures.sort(key=lambda x: x.gap if x.gap != float("inf") else float("inf"))

    return valid_failures[:2]  # Top 2 smallest gaps


def generate_counterfactual_v2(
    rules_fired: List[RuleTrace],
    target_regime: str,
) -> List[EvalResult]:
    """
    Phase 1: Get failed EvalResults for counterfactual generation.

    This is the preferred method when rules use eval_condition().
    Returns EvalResults directly for better type safety.

    Args:
        rules_fired: All rules evaluated in the decision tree
        target_regime: The regime we want to know how to reach (e.g., "R0")

    Returns:
        Top 2 failed EvalResults with smallest gaps (closest to triggering)
    """
    # Collect all failed eval_results for target regime
    all_failures: List[EvalResult] = []
    for rule in rules_fired:
        if rule.regime_target == target_regime and not rule.passed:
            if rule.eval_result and not rule.eval_result.passed:
                all_failures.append(rule.eval_result)

    # Sort by gap ascending (smallest positive gap first)
    all_failures.sort(key=lambda x: x.gap)

    return all_failures[:2]


def format_rule_result(trace: RuleTrace) -> str:
    """
    Format a rule trace as a single-line result for display.

    Returns a string like:
    "r2_trend_down  Trend = DOWN?  ── FAIL (trend_state=UP)"
    """
    status = "PASS" if trace.passed else "FAIL"

    # Build evidence summary
    evidence_items = []
    for key, value in trace.evidence.items():
        if isinstance(value, float):
            evidence_items.append(f"{key}={value:.2f}")
        else:
            evidence_items.append(f"{key}={value}")
    evidence_str = ", ".join(evidence_items[:3])  # Limit to 3 items

    return f"{trace.rule_id:<20} {trace.description:<30} ── {status} ({evidence_str})"


# =============================================================================
# HELPER FUNCTIONS FOR RULE TRACE CREATION (Phase 1)
# =============================================================================


def create_threshold_rule_trace(
    rule_id: str,
    description: str,
    metric_name: str,
    current_value: float,
    threshold: float,
    operator: Literal["<", "<=", ">", ">="],
    unit: str,
    evidence: Dict[str, Any],
    regime_target: str,
    category: str,
    priority: int,
) -> RuleTrace:
    """
    Create a RuleTrace for threshold-based rules using eval_condition().

    This is the preferred way to create threshold-based RuleTraces in Phase 1+.
    It ensures mathematical invariants are enforced and provides both
    eval_result (for new code) and threshold_info (for backward compatibility).

    Args:
        rule_id: Unique rule identifier (e.g., "r2_vol_high")
        description: Human-readable rule description
        metric_name: Name of the metric being compared
        current_value: Current observed value
        threshold: Threshold for comparison
        operator: Comparison operator ("<", "<=", ">", ">=")
        unit: Unit of measurement for display
        evidence: Dict of evidence values used in evaluation
        regime_target: Target regime ("R0", "R1", "R2", "R3")
        category: Rule category ("trend", "vol", "chop", "ext", "iv", "hysteresis")
        priority: Priority level (lower = higher priority)

    Returns:
        RuleTrace with eval_result and threshold_info both populated
    """
    # Evaluate condition using the unified function
    eval_result = eval_condition(
        metric_name=metric_name,
        current_value=current_value,
        threshold=threshold,
        operator=operator,
        unit=unit,
    )

    # Create backward-compatible ThresholdInfo
    threshold_info = ThresholdInfo.from_eval_result(eval_result)

    # Failed conditions list (for counterfactual generation)
    failed_conditions = [] if eval_result.passed else [threshold_info]

    return RuleTrace(
        rule_id=rule_id,
        description=description,
        passed=eval_result.passed,
        evidence=evidence,
        regime_target=regime_target,
        category=category,
        priority=priority,
        threshold_info=threshold_info,
        failed_conditions=failed_conditions,
        eval_result=eval_result,
    )


def create_categorical_rule_trace(
    rule_id: str,
    description: str,
    passed: bool,
    evidence: Dict[str, Any],
    regime_target: str,
    category: str,
    priority: int,
) -> RuleTrace:
    """
    Create a RuleTrace for categorical (non-threshold) rules.

    Use this for rules that compare enum states or boolean conditions
    rather than numeric thresholds. These rules don't have EvalResult
    since they're not threshold-based.

    Args:
        rule_id: Unique rule identifier
        description: Human-readable rule description
        passed: Whether the rule condition was satisfied
        evidence: Dict of evidence values used in evaluation
        regime_target: Target regime ("R0", "R1", "R2", "R3")
        category: Rule category
        priority: Priority level

    Returns:
        RuleTrace without eval_result (categorical rules)
    """
    return RuleTrace(
        rule_id=rule_id,
        description=description,
        passed=passed,
        evidence=evidence,
        regime_target=regime_target,
        category=category,
        priority=priority,
        threshold_info=None,  # No threshold for categorical
        failed_conditions=[],  # Categorical rules don't have threshold failures
        eval_result=None,  # No eval_result for categorical
    )
