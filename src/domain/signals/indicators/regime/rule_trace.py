"""
Rule Trace Models for Regime Classification Explainability.

Provides structured tracing of rule evaluations to enable:
- Auditability: Every regime label traces back to concrete inputs + rules
- Counterfactuals: Show users what would need to change to reach a different regime
- Debugging: Identify which conditions passed/failed and why
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ThresholdInfo:
    """
    Details for threshold-based comparisons in rule evaluation.

    Used both for documenting threshold crossings and for counterfactual generation
    (showing users how far they are from triggering a different regime).

    Attributes:
        metric_name: Name of the metric being compared (e.g., "atr_pctile_short")
        current_value: The actual value observed
        threshold: The threshold value for comparison
        operator: Comparison operator (">" or "<" or ">=" or "<=" or "==")
        gap: Distance to threshold (positive = how far from triggering)
             For categorical values, gap may be 0 or float('inf')
        unit: Unit of measurement ("%" or "ATR" or "" for dimensionless)
    """

    metric_name: str
    current_value: float
    threshold: float
    operator: str  # ">" or "<" or ">=" or "<=" or "=="
    gap: float  # Distance to threshold (positive = how far from triggering)
    unit: str = ""  # "%" or "ATR" or ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "operator": self.operator,
            "gap": self.gap,
            "unit": self.unit,
        }

    def format_comparison(self) -> str:
        """Format as human-readable comparison string."""
        unit_str = self.unit if self.unit else ""
        return (
            f"{self.metric_name}: {self.current_value:.2f}{unit_str} "
            f"{self.operator} {self.threshold:.2f}{unit_str}"
        )

    def format_gap(self) -> str:
        """Format the gap for display."""
        if self.gap == float("inf"):
            return "N/A (categorical)"
        unit_str = self.unit if self.unit else ""
        direction = "below" if self.operator in (">", ">=") else "above"
        return f"{abs(self.gap):.2f}{unit_str} {direction} threshold"


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
    failed_conditions: List[ThresholdInfo] = field(
        default_factory=list
    )  # Empty if passed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "passed": self.passed,
            "evidence": self.evidence,
            "regime_target": self.regime_target,
            "category": self.category,
            "priority": self.priority,
            "threshold_info": (
                self.threshold_info.to_dict() if self.threshold_info else None
            ),
            "failed_conditions": [c.to_dict() for c in self.failed_conditions],
        }


def generate_counterfactual(
    rules_fired: List[RuleTrace],
    target_regime: str,
) -> List[ThresholdInfo]:
    """
    Get the smallest-gap failed conditions to reach target regime.

    This enables showing users exactly what would need to change to reach
    a different regime classification.

    Args:
        rules_fired: All rules evaluated in the decision tree
        target_regime: The regime we want to know how to reach (e.g., "R0")

    Returns:
        Top 2 failed conditions with smallest gaps (closest to triggering)
    """
    # Filter rules that support target regime and didn't pass
    target_rules = [
        r for r in rules_fired if r.regime_target == target_regime and not r.passed
    ]

    # Collect all failed conditions
    all_failures: List[ThresholdInfo] = []
    for rule in target_rules:
        all_failures.extend(rule.failed_conditions)

    # Sort by gap (smallest first = closest to triggering)
    all_failures.sort(key=lambda x: abs(x.gap))
    return all_failures[:2]  # Top 2 smallest gaps


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
