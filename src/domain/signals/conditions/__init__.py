"""
Pluggable Condition Evaluators for Signal Rules.

Replaces 7 hardcoded methods in SignalRule with pluggable evaluators:
- ThresholdCrossUpEvaluator
- ThresholdCrossDownEvaluator
- StateChangeEvaluator
- CrossUpEvaluator
- CrossDownEvaluator
- RangeEntryEvaluator
- RangeExitEvaluator

Usage:
    from src.domain.signals.conditions import EVALUATORS, ConditionType

    evaluator = EVALUATORS[ConditionType.THRESHOLD_CROSS_UP]
    triggered = evaluator.evaluate(config, prev_state, curr_state)
"""

from .evaluators import (
    EVALUATORS,
    ConditionEvaluator,
    CrossDownEvaluator,
    CrossUpEvaluator,
    RangeEntryEvaluator,
    RangeExitEvaluator,
    StateChangeEvaluator,
    ThresholdCrossDownEvaluator,
    ThresholdCrossUpEvaluator,
)

__all__ = [
    "ConditionEvaluator",
    "ThresholdCrossUpEvaluator",
    "ThresholdCrossDownEvaluator",
    "StateChangeEvaluator",
    "CrossUpEvaluator",
    "CrossDownEvaluator",
    "RangeEntryEvaluator",
    "RangeExitEvaluator",
    "EVALUATORS",
]
