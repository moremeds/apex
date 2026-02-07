"""
Unit tests for ExitManager.

Covers: priority ordering determinism, all 5 exit types,
edge cases (zero ATR, flat peak), and long/short positions.
"""

from __future__ import annotations

import pytest

from src.domain.strategy.exit_manager import (
    ExitConditions,
    ExitManager,
    ExitPriority,
    ExitSignal,
)


def _default_manager(**overrides: object) -> ExitManager:
    defaults = {
        "hard_stop_pct": 0.08,
        "atr_trail_mult": 3.0,
        "max_hold_bars": 40,
        "regime_veto_regimes": ["R2"],
    }
    defaults.update(overrides)
    return ExitManager(**defaults)  # type: ignore[arg-type]


class TestExitPriority:
    def test_catastrophic_stop_is_highest_priority(self) -> None:
        assert ExitPriority.CATASTROPHIC_STOP < ExitPriority.ATR_TRAILING_STOP
        assert ExitPriority.CATASTROPHIC_STOP < ExitPriority.REGIME_VETO
        assert ExitPriority.CATASTROPHIC_STOP < ExitPriority.INDICATOR_DETERIORATION
        assert ExitPriority.CATASTROPHIC_STOP < ExitPriority.TIME_STOP

    def test_time_stop_is_lowest_priority(self) -> None:
        assert ExitPriority.TIME_STOP > ExitPriority.CATASTROPHIC_STOP
        assert ExitPriority.TIME_STOP > ExitPriority.ATR_TRAILING_STOP
        assert ExitPriority.TIME_STOP > ExitPriority.REGIME_VETO
        assert ExitPriority.TIME_STOP > ExitPriority.INDICATOR_DETERIORATION


class TestCatastrophicStop:
    def test_hard_stop_triggered_long(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.08)
        conditions = ExitConditions(
            current_price=92.0,  # Entry * (1 - 0.08) = 92
            entry_price=100.0,
            peak_price=105.0,
            current_atr=2.0,
            bars_held=5,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is not None
        assert result.priority == ExitPriority.CATASTROPHIC_STOP

    def test_hard_stop_not_triggered_when_above(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.08)
        conditions = ExitConditions(
            current_price=95.0,
            entry_price=100.0,
            peak_price=100.0,
            current_atr=2.0,
            bars_held=5,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        # No catastrophic stop, but might trigger ATR trail
        if result is not None:
            assert result.priority != ExitPriority.CATASTROPHIC_STOP

    def test_hard_stop_short_position(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.08)
        conditions = ExitConditions(
            current_price=108.5,  # Above entry * (1 + 0.08) = 108
            entry_price=100.0,
            peak_price=95.0,  # Peak = lowest for short
            current_atr=2.0,
            bars_held=5,
            is_long=False,
        )
        result = mgr.evaluate(conditions)
        assert result is not None
        assert result.priority == ExitPriority.CATASTROPHIC_STOP


class TestATRTrailingStop:
    def test_atr_trail_triggered_long(self) -> None:
        mgr = _default_manager(atr_trail_mult=3.0, hard_stop_pct=0.50)
        conditions = ExitConditions(
            current_price=94.0,  # Below peak(105) - 3*ATR(2) = 99
            entry_price=100.0,
            peak_price=105.0,
            current_atr=2.0,
            bars_held=10,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is not None
        assert result.priority == ExitPriority.ATR_TRAILING_STOP

    def test_atr_trail_not_triggered_above_level(self) -> None:
        mgr = _default_manager(atr_trail_mult=3.0, hard_stop_pct=0.50)
        conditions = ExitConditions(
            current_price=100.0,  # Above peak(105) - 3*ATR(2) = 99
            entry_price=100.0,
            peak_price=105.0,
            current_atr=2.0,
            bars_held=10,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is None

    def test_zero_atr_skips_trail_check(self) -> None:
        mgr = _default_manager(atr_trail_mult=3.0, hard_stop_pct=0.50, max_hold_bars=999)
        conditions = ExitConditions(
            current_price=100.0,
            entry_price=100.0,
            peak_price=105.0,
            current_atr=0.0,
            bars_held=10,
            is_long=True,
        )
        # With zero ATR, trail check is skipped (returns None)
        result = mgr.evaluate(conditions)
        assert result is None


class TestRegimeVeto:
    def test_regime_veto_triggers_exit(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.50, atr_trail_mult=20.0)
        conditions = ExitConditions(
            current_price=105.0,
            entry_price=100.0,
            peak_price=105.0,
            current_atr=1.0,
            bars_held=5,
            regime_is_veto=True,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is not None
        assert result.priority == ExitPriority.REGIME_VETO

    def test_no_regime_veto_when_flag_false(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.50, atr_trail_mult=20.0, max_hold_bars=999)
        conditions = ExitConditions(
            current_price=105.0,
            entry_price=100.0,
            peak_price=105.0,
            current_atr=1.0,
            bars_held=5,
            regime_is_veto=False,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is None


class TestIndicatorDeterioration:
    def test_indicator_exit_triggers(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.50, atr_trail_mult=20.0, max_hold_bars=999)
        conditions = ExitConditions(
            current_price=105.0,
            entry_price=100.0,
            peak_price=105.0,
            current_atr=1.0,
            bars_held=5,
            indicator_exit=True,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is not None
        assert result.priority == ExitPriority.INDICATOR_DETERIORATION


class TestTimeStop:
    def test_time_stop_at_max_bars(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.50, atr_trail_mult=20.0, max_hold_bars=40)
        conditions = ExitConditions(
            current_price=105.0,
            entry_price=100.0,
            peak_price=105.0,
            current_atr=1.0,
            bars_held=40,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is not None
        assert result.priority == ExitPriority.TIME_STOP

    def test_time_stop_not_triggered_before_max(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.50, atr_trail_mult=20.0, max_hold_bars=40)
        conditions = ExitConditions(
            current_price=105.0,
            entry_price=100.0,
            peak_price=105.0,
            current_atr=1.0,
            bars_held=39,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is None


class TestPriorityOrdering:
    def test_catastrophic_beats_atr_trail(self) -> None:
        """When both catastrophic and ATR stops trigger, catastrophic wins."""
        mgr = _default_manager(hard_stop_pct=0.08, atr_trail_mult=1.0)
        conditions = ExitConditions(
            current_price=90.0,  # Below both stops
            entry_price=100.0,
            peak_price=100.0,
            current_atr=5.0,
            bars_held=5,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is not None
        assert result.priority == ExitPriority.CATASTROPHIC_STOP

    def test_all_exits_triggered_returns_catastrophic(self) -> None:
        """When all 5 exits trigger, catastrophic wins."""
        mgr = _default_manager(hard_stop_pct=0.08, atr_trail_mult=1.0, max_hold_bars=5)
        conditions = ExitConditions(
            current_price=90.0,
            entry_price=100.0,
            peak_price=100.0,
            current_atr=5.0,
            bars_held=10,
            regime_is_veto=True,
            indicator_exit=True,
            is_long=True,
        )
        result = mgr.evaluate(conditions)
        assert result is not None
        assert result.priority == ExitPriority.CATASTROPHIC_STOP


class TestHelperMethods:
    def test_get_trailing_stop_level_long(self) -> None:
        mgr = _default_manager(atr_trail_mult=3.0)
        level = mgr.get_trailing_stop_level(peak_price=110.0, current_atr=2.0, is_long=True)
        assert level == pytest.approx(104.0)

    def test_get_trailing_stop_level_short(self) -> None:
        mgr = _default_manager(atr_trail_mult=3.0)
        level = mgr.get_trailing_stop_level(peak_price=90.0, current_atr=2.0, is_long=False)
        assert level == pytest.approx(96.0)

    def test_get_hard_stop_level_long(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.08)
        level = mgr.get_hard_stop_level(entry_price=100.0, is_long=True)
        assert level == pytest.approx(92.0)

    def test_get_hard_stop_level_short(self) -> None:
        mgr = _default_manager(hard_stop_pct=0.08)
        level = mgr.get_hard_stop_level(entry_price=100.0, is_long=False)
        assert level == pytest.approx(108.0)
