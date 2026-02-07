"""
ExitManager - Deterministic exit priority ordering.

Provides a canonical exit evaluation with strict priority ordering.
Strategies always obey the highest-priority triggered exit.

Exit Priority:
    1. Catastrophic stop (gap / hard stop, e.g., 8% default)
    2. ATR trailing stop (peak - mult * ATR)
    3. Regime veto (R2 forced exit)
    4. Indicator deterioration (strategy-specific conditions)
    5. Time stop (max_hold_bars exceeded)

Usage:
    exit_mgr = ExitManager(
        hard_stop_pct=0.08,
        atr_trail_mult=3.0,
        max_hold_bars=40,
    )

    conditions = ExitConditions(
        current_price=185.0,
        entry_price=190.0,
        peak_price=195.0,
        current_atr=2.5,
        bars_held=35,
        regime_is_veto=False,
        indicator_exit=False,
    )

    result = exit_mgr.evaluate(conditions)
    if result is not None:
        print(f"EXIT: {result.reason} (priority {result.priority})")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


class ExitPriority(IntEnum):
    """Exit type priorities (lower number = higher priority)."""

    CATASTROPHIC_STOP = 1
    ATR_TRAILING_STOP = 2
    REGIME_VETO = 3
    INDICATOR_DETERIORATION = 4
    TIME_STOP = 5


@dataclass
class ExitSignal:
    """Result from exit evaluation."""

    triggered: bool
    priority: ExitPriority
    reason: str
    exit_price_hint: Optional[float] = None

    @property
    def name(self) -> str:
        return self.priority.name


@dataclass
class ExitConditions:
    """
    Inputs for exit evaluation.

    All fields that ExitManager needs to check the 5 exit types.
    Strategy fills these from its state and passes to evaluate().
    """

    current_price: float
    entry_price: float
    peak_price: float
    current_atr: float
    bars_held: int

    # Regime
    regime_is_veto: bool = False

    # Indicator deterioration (strategy-specific flag)
    indicator_exit: bool = False
    indicator_exit_reason: str = ""

    # Direction of position
    is_long: bool = True


class ExitManager:
    """
    Deterministic exit priority evaluator.

    Evaluates all 5 exit types in priority order and returns
    the single highest-priority triggered exit (or None).
    """

    def __init__(
        self,
        hard_stop_pct: float = 0.08,
        atr_trail_mult: float = 3.0,
        max_hold_bars: int = 40,
        regime_veto_regimes: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            hard_stop_pct: Maximum loss from entry before catastrophic exit
                           (0.08 = 8%). Applied to both longs and shorts.
            atr_trail_mult: ATR multiplier for trailing stop distance.
            max_hold_bars: Maximum bars to hold before time-based exit.
            regime_veto_regimes: Regime values that force exit (default ["R2"]).
        """
        self._hard_stop_pct = hard_stop_pct
        self._atr_trail_mult = atr_trail_mult
        self._max_hold_bars = max_hold_bars
        self._regime_veto_regimes = regime_veto_regimes or ["R2"]

    def evaluate(self, conditions: ExitConditions) -> Optional[ExitSignal]:
        """
        Evaluate exit conditions in priority order.

        Returns the highest-priority triggered exit, or None if no exit.
        """
        # Priority 1: Catastrophic / hard stop
        result = self._check_catastrophic_stop(conditions)
        if result and result.triggered:
            return result

        # Priority 2: ATR trailing stop
        result = self._check_atr_trailing_stop(conditions)
        if result and result.triggered:
            return result

        # Priority 3: Regime veto
        result = self._check_regime_veto(conditions)
        if result and result.triggered:
            return result

        # Priority 4: Indicator deterioration
        result = self._check_indicator_deterioration(conditions)
        if result and result.triggered:
            return result

        # Priority 5: Time stop
        result = self._check_time_stop(conditions)
        if result and result.triggered:
            return result

        return None

    def get_trailing_stop_level(
        self,
        peak_price: float,
        current_atr: float,
        is_long: bool = True,
    ) -> float:
        """Calculate current ATR trailing stop level."""
        distance = self._atr_trail_mult * current_atr
        if is_long:
            return peak_price - distance
        else:
            return peak_price + distance

    def get_hard_stop_level(self, entry_price: float, is_long: bool = True) -> float:
        """Calculate hard stop level."""
        if is_long:
            return entry_price * (1.0 - self._hard_stop_pct)
        else:
            return entry_price * (1.0 + self._hard_stop_pct)

    def _check_catastrophic_stop(self, c: ExitConditions) -> Optional[ExitSignal]:
        """Priority 1: Hard stop - max loss from entry."""
        stop_level = self.get_hard_stop_level(c.entry_price, c.is_long)

        if c.is_long and c.current_price <= stop_level:
            loss_pct = (c.entry_price - c.current_price) / c.entry_price
            return ExitSignal(
                triggered=True,
                priority=ExitPriority.CATASTROPHIC_STOP,
                reason=f"Hard stop hit: loss {loss_pct:.1%} >= {self._hard_stop_pct:.1%}",
                exit_price_hint=c.current_price,
            )
        elif not c.is_long and c.current_price >= stop_level:
            loss_pct = (c.current_price - c.entry_price) / c.entry_price
            return ExitSignal(
                triggered=True,
                priority=ExitPriority.CATASTROPHIC_STOP,
                reason=f"Hard stop hit (short): loss {loss_pct:.1%} >= {self._hard_stop_pct:.1%}",
                exit_price_hint=c.current_price,
            )
        return None

    def _check_atr_trailing_stop(self, c: ExitConditions) -> Optional[ExitSignal]:
        """Priority 2: ATR trailing stop from peak."""
        if c.current_atr <= 0:
            return None

        trail_level = self.get_trailing_stop_level(c.peak_price, c.current_atr, c.is_long)

        if c.is_long and c.current_price <= trail_level:
            return ExitSignal(
                triggered=True,
                priority=ExitPriority.ATR_TRAILING_STOP,
                reason=(
                    f"ATR trail stop: price {c.current_price:.2f} <= "
                    f"trail {trail_level:.2f} "
                    f"(peak {c.peak_price:.2f} - {self._atr_trail_mult}*ATR)"
                ),
                exit_price_hint=c.current_price,
            )
        elif not c.is_long and c.current_price >= trail_level:
            return ExitSignal(
                triggered=True,
                priority=ExitPriority.ATR_TRAILING_STOP,
                reason=(
                    f"ATR trail stop (short): price {c.current_price:.2f} >= "
                    f"trail {trail_level:.2f} "
                    f"(peak {c.peak_price:.2f} + {self._atr_trail_mult}*ATR)"
                ),
                exit_price_hint=c.current_price,
            )
        return None

    def _check_regime_veto(self, c: ExitConditions) -> Optional[ExitSignal]:
        """Priority 3: Regime veto forces exit."""
        if c.regime_is_veto:
            return ExitSignal(
                triggered=True,
                priority=ExitPriority.REGIME_VETO,
                reason="Regime veto: forced exit due to risk-off regime",
                exit_price_hint=c.current_price,
            )
        return None

    def _check_indicator_deterioration(self, c: ExitConditions) -> Optional[ExitSignal]:
        """Priority 4: Strategy-specific indicator deterioration."""
        if c.indicator_exit:
            reason = c.indicator_exit_reason or "Indicator deterioration"
            return ExitSignal(
                triggered=True,
                priority=ExitPriority.INDICATOR_DETERIORATION,
                reason=reason,
                exit_price_hint=c.current_price,
            )
        return None

    def _check_time_stop(self, c: ExitConditions) -> Optional[ExitSignal]:
        """Priority 5: Maximum holding period exceeded."""
        if c.bars_held >= self._max_hold_bars:
            return ExitSignal(
                triggered=True,
                priority=ExitPriority.TIME_STOP,
                reason=f"Time stop: held {c.bars_held} >= max {self._max_hold_bars} bars",
                exit_price_hint=c.current_price,
            )
        return None
