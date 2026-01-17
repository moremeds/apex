"""
Constraint validator for filtering trial results.

Supports constraints on aggregated metrics with operators:
- >=, >, <=, <, ==, !=
- Range constraints (between)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from ..core import TrialAggregates, TrialResult


class Constraint(BaseModel):
    """Definition of a single constraint."""

    metric: str = Field(description="Metric to constrain (e.g., 'p10_sharpe')")
    operator: Literal[">=", ">", "<=", "<", "==", "!=", "between"] = Field(
        description="Comparison operator"
    )
    value: float = Field(description="Constraint value")
    value_max: Optional[float] = Field(default=None, description="Max value for 'between' operator")
    name: Optional[str] = Field(default=None, description="Optional constraint name")

    def check(self, actual_value: float) -> Tuple[bool, str]:
        """
        Check if value satisfies constraint.

        Returns:
            Tuple of (passed, violation_message)
        """
        name = self.name or f"{self.metric} {self.operator} {self.value}"

        ops = {
            ">=": lambda a, v: a >= v,
            ">": lambda a, v: a > v,
            "<=": lambda a, v: a <= v,
            "<": lambda a, v: a < v,
            "==": lambda a, v: a == v,
            "!=": lambda a, v: a != v,
        }

        if self.operator == "between":
            passed = self.value <= actual_value <= self.value_max
            if not passed:
                return False, f"{name}: {actual_value:.4f} not in [{self.value}, {self.value_max}]"
            return True, ""

        op_func = ops.get(self.operator)
        if op_func is None:
            return False, f"Unknown operator: {self.operator}"

        passed = op_func(actual_value, self.value)
        if not passed:
            return False, f"{name}: {actual_value:.4f} failed {self.operator} {self.value}"

        return True, ""


@dataclass
class ConstraintValidator:
    """
    Validates trial results against constraints.

    Constraints are defined in the experiment spec and can filter
    out trials that don't meet minimum requirements.

    Example:
        validator = ConstraintValidator([
            Constraint(metric="p10_sharpe", operator=">=", value=0.0),
            Constraint(metric="median_max_dd", operator="<=", value=0.25),
        ])

        valid_trials = [t for t in trials if validator.validate(t)]
    """

    constraints: List[Constraint]

    def validate(self, trial: TrialResult) -> bool:
        """
        Check if trial meets all constraints.

        Args:
            trial: Trial result to validate

        Returns:
            True if all constraints pass
        """
        passed, _ = self.validate_with_details(trial)
        return passed

    def validate_with_details(self, trial: TrialResult) -> Tuple[bool, List[str]]:
        """
        Validate trial and return violation details.

        Args:
            trial: Trial result to validate

        Returns:
            Tuple of (all_passed, list_of_violations)
        """
        violations = []
        agg = trial.aggregates

        for constraint in self.constraints:
            # Get metric value from aggregates
            value = self._get_metric_value(agg, constraint.metric)
            if value is None:
                violations.append(f"Unknown metric: {constraint.metric}")
                continue

            passed, message = constraint.check(value)
            if not passed:
                violations.append(message)

        return len(violations) == 0, violations

    def _get_metric_value(self, agg: TrialAggregates, metric: str) -> Optional[float]:
        """Get metric value from aggregates."""
        return getattr(agg, metric, None)

    @classmethod
    def from_config(cls, constraints: List[Dict[str, Any]]) -> "ConstraintValidator":
        """Create validator from config dictionaries."""
        return cls(constraints=[Constraint(**c) for c in constraints])
