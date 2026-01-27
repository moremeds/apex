"""
Score-Based Hysteresis State Machine.

Phase 5: Banded hysteresis using composite scores instead of bar counts.
Transitions require crossing the appropriate band, not just the midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class HysteresisBands:
    """
    Score bands for regime transitions.

    Enter bands: Score must exceed/drop below these to ENTER a regime.
    Exit bands: Score must exceed/drop below these to EXIT a regime.

    Gap between enter/exit creates hysteresis (prevents flip-flopping).
    """

    # R0 (Healthy) - need high score to enter, moderate to stay
    healthy_enter: float = 75.0  # Score must exceed this to become Healthy
    healthy_exit: float = 65.0  # Score must drop below this to leave Healthy

    # R2 (Risk-Off) - need low score to enter, slightly higher to exit
    risk_off_enter: float = 25.0  # Score must drop below this to become Risk-Off
    risk_off_exit: float = 35.0  # Score must exceed this to leave Risk-Off

    # R1 (Choppy) is everything in between
    # No explicit bands - it's the default state

    def validate(self) -> None:
        """Validate band configuration."""
        if self.healthy_exit >= self.healthy_enter:
            raise ValueError("healthy_exit must be < healthy_enter")
        if self.risk_off_exit <= self.risk_off_enter:
            raise ValueError("risk_off_exit must be > risk_off_enter")
        if self.risk_off_exit >= self.healthy_exit:
            raise ValueError("risk_off_exit must be < healthy_exit (R1 zone must exist)")


@dataclass
class TransitionEvent:
    """Record of a regime transition."""

    timestamp: pd.Timestamp
    from_regime: str
    to_regime: str
    score: float
    trigger: str  # Which band was crossed


@dataclass
class HysteresisState:
    """Current state of the hysteresis state machine."""

    current_regime: str = "R1"
    bars_in_regime: int = 0
    entry_score: float = 50.0
    min_score_in_regime: float = 100.0
    max_score_in_regime: float = 0.0
    transitions: List[TransitionEvent] = field(default_factory=list)


class ScoreHysteresisStateMachine:
    """
    Regime state machine with banded hysteresis.

    Key principle: Transitions require crossing explicit bands.
    - To enter R0: score must exceed healthy_enter
    - To exit R0: score must drop below healthy_exit
    - To enter R2: score must drop below risk_off_enter
    - To exit R2: score must exceed risk_off_exit

    This prevents oscillation around boundaries.
    """

    def __init__(self, bands: Optional[HysteresisBands] = None) -> None:
        self.bands = bands or HysteresisBands()
        self.bands.validate()
        self._state = HysteresisState()

    @property
    def current_regime(self) -> str:
        return self._state.current_regime

    @property
    def bars_in_regime(self) -> int:
        return self._state.bars_in_regime

    def reset(self, initial_regime: str = "R1") -> None:
        """Reset state machine."""
        self._state = HysteresisState(current_regime=initial_regime)

    def update(
        self,
        score: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Tuple[str, Optional[TransitionEvent]]:
        """
        Update regime based on score with hysteresis.

        Args:
            score: Composite score (0-100)
            timestamp: Optional timestamp for transition logging

        Returns:
            Tuple of (new_regime, transition_event or None)
        """
        current = self._state.current_regime
        new_regime = current
        transition = None
        trigger = ""

        # Update score tracking
        self._state.min_score_in_regime = min(self._state.min_score_in_regime, score)
        self._state.max_score_in_regime = max(self._state.max_score_in_regime, score)

        if current == "R0":
            # Currently Healthy - check for exit
            if score < self.bands.healthy_exit:
                if score <= self.bands.risk_off_enter:
                    new_regime = "R2"
                    trigger = f"score {score:.1f} < risk_off_enter {self.bands.risk_off_enter}"
                else:
                    new_regime = "R1"
                    trigger = f"score {score:.1f} < healthy_exit {self.bands.healthy_exit}"

        elif current == "R2":
            # Currently Risk-Off - check for exit
            if score > self.bands.risk_off_exit:
                if score >= self.bands.healthy_enter:
                    new_regime = "R0"
                    trigger = f"score {score:.1f} >= healthy_enter {self.bands.healthy_enter}"
                else:
                    new_regime = "R1"
                    trigger = f"score {score:.1f} > risk_off_exit {self.bands.risk_off_exit}"

        else:  # R1 (Choppy)
            # Check for entry into R0 or R2
            if score >= self.bands.healthy_enter:
                new_regime = "R0"
                trigger = f"score {score:.1f} >= healthy_enter {self.bands.healthy_enter}"
            elif score <= self.bands.risk_off_enter:
                new_regime = "R2"
                trigger = f"score {score:.1f} <= risk_off_enter {self.bands.risk_off_enter}"

        # Handle transition
        if new_regime != current:
            transition = TransitionEvent(
                timestamp=timestamp or pd.Timestamp.now(),
                from_regime=current,
                to_regime=new_regime,
                score=score,
                trigger=trigger,
            )
            self._state.transitions.append(transition)

            # Reset state for new regime
            self._state.current_regime = new_regime
            self._state.bars_in_regime = 1
            self._state.entry_score = score
            self._state.min_score_in_regime = score
            self._state.max_score_in_regime = score
        else:
            self._state.bars_in_regime += 1

        return new_regime, transition

    def classify_series(
        self,
        scores: pd.Series,
        initial_regime: str = "R1",
    ) -> pd.DataFrame:
        """
        Classify entire series with hysteresis.

        Args:
            scores: Series of composite scores
            initial_regime: Starting regime

        Returns:
            DataFrame with regime, bars_in_regime, transition columns
        """
        self.reset(initial_regime)

        regimes = []
        bars_in = []
        transitions = []

        for ts, score in scores.items():
            if np.isnan(score):
                regimes.append(self._state.current_regime)
                bars_in.append(self._state.bars_in_regime)
                transitions.append(False)
                continue

            regime, transition = self.update(score, ts)
            regimes.append(regime)
            bars_in.append(self._state.bars_in_regime)
            transitions.append(transition is not None)

        return pd.DataFrame(
            {
                "regime": regimes,
                "bars_in_regime": bars_in,
                "transition": transitions,
            },
            index=scores.index,
        )

    def get_transition_history(self) -> List[TransitionEvent]:
        """Get all transition events."""
        return self._state.transitions.copy()

    def get_transition_frequency(self) -> float:
        """Get transitions per bar (lower = more stable)."""
        total_bars = self._state.bars_in_regime + sum(
            t.score for t in self._state.transitions  # Approximate
        )
        if total_bars == 0:
            return 0.0
        return len(self._state.transitions) / max(total_bars, 1)
