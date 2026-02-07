"""
Unit tests for StressValidator.

Covers: stress window extraction, MaxDD constraint,
baseline comparison, insufficient data handling,
and all 5 pre-defined stress windows.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.optimization.stress_validator import (
    STRESS_WINDOWS,
    StressValidator,
    StressWindow,
)


def _make_equity_curve(
    start_date: str = "2019-01-01",
    end_date: str = "2025-01-01",
    daily_return: float = 0.0003,
    crash_start: str | None = None,
    crash_end: str | None = None,
    crash_dd: float = -0.30,
) -> pd.Series:
    """Create synthetic equity curve with optional crash period.

    The crash_dd is the total drawdown during the crash window.
    The daily log returns are computed so the product yields the target DD.
    """
    dates = pd.bdate_range(start=start_date, end=end_date)
    equity = np.full(len(dates), 100.0)

    # Build equity with constant daily return
    for i in range(1, len(dates)):
        equity[i] = equity[i - 1] * (1 + daily_return)

    if crash_start and crash_end:
        crash_s = pd.Timestamp(crash_start)
        crash_e = pd.Timestamp(crash_end)
        crash_mask = (dates >= crash_s) & (dates <= crash_e)
        crash_indices = np.where(crash_mask)[0]
        if len(crash_indices) > 0:
            pre_crash_val = equity[crash_indices[0]]
            # Drop to crash_dd then stay flat during crash
            n_crash = len(crash_indices)
            half = n_crash // 2
            for j, idx in enumerate(crash_indices):
                if j <= half:
                    # Dropping phase
                    frac = j / max(half, 1)
                    equity[idx] = pre_crash_val * (1 + crash_dd * frac)
                else:
                    # Stay at bottom
                    equity[idx] = pre_crash_val * (1 + crash_dd)
            # After crash, continue from bottom
            bottom = equity[crash_indices[-1]]
            for i in range(crash_indices[-1] + 1, len(dates)):
                equity[i] = equity[i - 1] * (1 + daily_return)

    return pd.Series(equity, index=dates)


class TestStressWindows:
    def test_five_predefined_windows(self) -> None:
        assert len(STRESS_WINDOWS) == 5
        names = {w.name for w in STRESS_WINDOWS}
        assert "covid_crash" in names
        assert "bear_2022" in names
        assert "ai_meltup_2023" in names
        assert "regional_bank_2023" in names
        assert "aug_2024_unwind" in names

    def test_windows_are_chronologically_ordered(self) -> None:
        for i in range(len(STRESS_WINDOWS) - 1):
            assert STRESS_WINDOWS[i].start <= STRESS_WINDOWS[i + 1].start


class TestStressValidatorBasic:
    def test_gentle_strategy_passes_all(self) -> None:
        """Strategy with small drawdowns passes all windows."""
        equity = _make_equity_curve(daily_return=0.0005)
        validator = StressValidator(max_dd_multiplier=2.0)
        result = validator.validate(equity)
        assert result.all_passed is True
        assert len(result.failures) == 0

    def test_volatile_strategy_may_fail(self) -> None:
        """Strategy with extreme crash fails validation."""
        equity = _make_equity_curve(
            crash_start="2020-02-19",
            crash_end="2020-04-15",
            crash_dd=-0.80,  # 80% crash - much worse than threshold
        )
        # No baseline -> baseline_dd = 0.30, threshold = 2 * max(0.30, 0.05) = 0.60
        # Strategy DD ~0.80, abs(0.80) > 0.60 -> fail
        validator = StressValidator(max_dd_multiplier=2.0)
        result = validator.validate(equity)
        covid = result.window_results.get("covid_crash")
        assert covid is not None
        assert covid.passed is False
        assert result.all_passed is False


class TestStressValidatorBaseline:
    def test_baseline_comparison_with_default_threshold(self) -> None:
        """Without baseline, uses default 30% -> threshold = 2 * max(abs(0.30), 0.05) = 0.60.
        Note: _calc_max_drawdown returns negative values. The threshold uses
        abs(baseline_dd) to handle the sign correctly. When baseline is provided
        with e.g. -15% DD, threshold = 2 * max(0.15, 0.05) = 0.30.
        When no baseline, baseline_dd = 0.30 (positive constant).
        """
        # Strategy with tiny crash should pass with no baseline
        strategy = _make_equity_curve(
            crash_start="2020-02-19", crash_end="2020-04-15", crash_dd=-0.03
        )
        # No baseline -> baseline_dd = 0.30 -> threshold = 2 * max(abs(0.30), 0.05) = 0.60
        # Strategy DD ~0.03, abs(0.03) < 0.60 -> pass
        validator = StressValidator(max_dd_multiplier=2.0)
        result = validator.validate(strategy)
        covid = result.window_results.get("covid_crash")
        assert covid is not None
        assert covid.passed is True


class TestStressValidatorEdgeCases:
    def test_short_data_skips_window(self) -> None:
        """Data not covering a window results in skip (pass)."""
        # Only 2023 data - misses covid and bear_2022 windows
        equity = _make_equity_curve(start_date="2023-06-01", end_date="2024-12-31")
        validator = StressValidator()
        result = validator.validate(equity)
        # Windows with insufficient data are passed with "skipped" reason
        covid_result = result.window_results.get("covid_crash")
        assert covid_result is not None
        assert covid_result.passed is True
        assert "skipped" in covid_result.reason.lower() or covid_result.bars_in_window < 2

    def test_custom_windows(self) -> None:
        """Can pass custom stress windows."""
        custom = [
            StressWindow(
                name="test_crash",
                start=date(2024, 1, 15),
                end=date(2024, 2, 15),
                description="Test crash",
            )
        ]
        equity = _make_equity_curve(start_date="2023-01-01", end_date="2025-01-01")
        validator = StressValidator(windows=custom)
        result = validator.validate(equity)
        assert "test_crash" in result.window_results

    def test_num_passed_property(self) -> None:
        equity = _make_equity_curve(daily_return=0.0005)
        validator = StressValidator()
        result = validator.validate(equity)
        assert result.num_passed == result.num_total
        assert result.num_total == 5
