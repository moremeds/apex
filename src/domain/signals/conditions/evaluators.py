"""
Condition Evaluators - Pluggable signal rule evaluation logic.

Each evaluator handles a specific ConditionType. Evaluators are:
- Stateless (no instance variables)
- Pure functions wrapped in classes for type safety
- Registered in EVALUATORS dict for dispatch

Performance: Direct dict lookup + method call, no reflection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

if TYPE_CHECKING:
    from ..models import ConditionType


class ConditionEvaluator(Protocol):
    """Protocol for condition evaluators."""

    def evaluate(
        self,
        config: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        """
        Evaluate if the condition is met.

        Args:
            config: Condition-specific configuration (thresholds, fields, etc.)
            prev_state: Previous indicator state (None if first evaluation)
            curr_state: Current indicator state

        Returns:
            True if condition is triggered
        """
        ...


class ThresholdCrossUpEvaluator:
    """
    Value crosses above threshold.

    Config:
        field: Field to check (default: "value")
        threshold: Threshold value
        detect_initial: If True, trigger when curr >= threshold on first eval
    """

    __slots__ = ()

    def evaluate(
        self,
        config: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        field = config.get("field", "value")
        threshold = config.get("threshold", 0)
        curr_val = curr_state.get(field, 0)

        if prev_state is None:
            if config.get("detect_initial", False):
                return curr_val >= threshold
            return False

        prev_val = prev_state.get(field, 0)
        return prev_val < threshold <= curr_val


class ThresholdCrossDownEvaluator:
    """
    Value crosses below threshold.

    Config:
        field: Field to check (default: "value")
        threshold: Threshold value
        detect_initial: If True, trigger when curr <= threshold on first eval
    """

    __slots__ = ()

    def evaluate(
        self,
        config: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        field = config.get("field", "value")
        threshold = config.get("threshold", 0)
        curr_val = curr_state.get(field, 0)

        if prev_state is None:
            if config.get("detect_initial", False):
                return curr_val <= threshold
            return False

        prev_val = prev_state.get(field, 0)
        return prev_val > threshold >= curr_val


class StateChangeEvaluator:
    """
    State transitions from one value to another.

    Config:
        field: Field to check (default: "zone")
        from: List of source states
        to: List of target states
    """

    __slots__ = ()

    def evaluate(
        self,
        config: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        if prev_state is None:
            return False

        field = config.get("field", "zone")
        from_states = config.get("from", [])
        to_states = config.get("to", [])
        prev_val = prev_state.get(field)
        curr_val = curr_state.get(field)

        return prev_val in from_states and curr_val in to_states


class CrossUpEvaluator:
    """
    Line A crosses above Line B.

    Config:
        line_a: Fast line field (default: "fast")
        line_b: Slow line field (default: "slow")
    """

    __slots__ = ()

    def evaluate(
        self,
        config: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        if prev_state is None:
            return False

        line_a = config.get("line_a", "fast")
        line_b = config.get("line_b", "slow")
        prev_a = prev_state.get(line_a, 0)
        prev_b = prev_state.get(line_b, 0)
        curr_a = curr_state.get(line_a, 0)
        curr_b = curr_state.get(line_b, 0)

        return prev_a <= prev_b and curr_a > curr_b


class CrossDownEvaluator:
    """
    Line A crosses below Line B.

    Config:
        line_a: Fast line field (default: "fast")
        line_b: Slow line field (default: "slow")
    """

    __slots__ = ()

    def evaluate(
        self,
        config: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        if prev_state is None:
            return False

        line_a = config.get("line_a", "fast")
        line_b = config.get("line_b", "slow")
        prev_a = prev_state.get(line_a, 0)
        prev_b = prev_state.get(line_b, 0)
        curr_a = curr_state.get(line_a, 0)
        curr_b = curr_state.get(line_b, 0)

        return prev_a >= prev_b and curr_a < curr_b


class RangeEntryEvaluator:
    """
    Value enters a range [lower, upper].

    Config:
        field: Field to check (default: "value")
        lower: Lower bound (default: 0)
        upper: Upper bound (default: 100)
    """

    __slots__ = ()

    def evaluate(
        self,
        config: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        if prev_state is None:
            return False

        field = config.get("field", "value")
        lower = config.get("lower", 0)
        upper = config.get("upper", 100)
        prev_val = prev_state.get(field, 0)
        curr_val = curr_state.get(field, 0)

        was_outside = prev_val < lower or prev_val > upper
        is_inside = lower <= curr_val <= upper

        return was_outside and is_inside


class RangeExitEvaluator:
    """
    Value exits a range [lower, upper].

    Config:
        field: Field to check (default: "value")
        lower: Lower bound (default: 0)
        upper: Upper bound (default: 100)
    """

    __slots__ = ()

    def evaluate(
        self,
        config: Dict[str, Any],
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        if prev_state is None:
            return False

        field = config.get("field", "value")
        lower = config.get("lower", 0)
        upper = config.get("upper", 100)
        prev_val = prev_state.get(field, 0)
        curr_val = curr_state.get(field, 0)

        was_inside = lower <= prev_val <= upper
        is_outside = curr_val < lower or curr_val > upper

        return was_inside and is_outside


# Singleton instances (stateless, so safe to share)
_threshold_cross_up = ThresholdCrossUpEvaluator()
_threshold_cross_down = ThresholdCrossDownEvaluator()
_state_change = StateChangeEvaluator()
_cross_up = CrossUpEvaluator()
_cross_down = CrossDownEvaluator()
_range_entry = RangeEntryEvaluator()
_range_exit = RangeExitEvaluator()


def _build_evaluators_dict():
    """Build evaluators dict with lazy import to avoid circular dependency."""
    from ..models import ConditionType

    return {
        ConditionType.THRESHOLD_CROSS_UP: _threshold_cross_up,
        ConditionType.THRESHOLD_CROSS_DOWN: _threshold_cross_down,
        ConditionType.STATE_CHANGE: _state_change,
        ConditionType.CROSS_UP: _cross_up,
        ConditionType.CROSS_DOWN: _cross_down,
        ConditionType.RANGE_ENTRY: _range_entry,
        ConditionType.RANGE_EXIT: _range_exit,
    }


# Lazy initialization to avoid circular import
_evaluators_cache: Optional[Dict["ConditionType", ConditionEvaluator]] = None


def get_evaluators() -> Dict["ConditionType", ConditionEvaluator]:
    """Get evaluators dict (lazy initialization)."""
    global _evaluators_cache
    if _evaluators_cache is None:
        _evaluators_cache = _build_evaluators_dict()
    return _evaluators_cache


class _EvaluatorsProxy:
    """Proxy for lazy EVALUATORS dict access."""

    def __getitem__(self, key: "ConditionType") -> ConditionEvaluator:
        return get_evaluators()[key]

    def get(self, key: "ConditionType", default=None) -> Optional[ConditionEvaluator]:
        return get_evaluators().get(key, default)

    def __contains__(self, key: "ConditionType") -> bool:
        return key in get_evaluators()


EVALUATORS = _EvaluatorsProxy()
