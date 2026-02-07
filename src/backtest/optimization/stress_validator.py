"""
Stress window validator for strategy backtests.

Validates that strategies survive named stress episodes with acceptable
drawdowns. Each stress window is a historical period of market stress.

Usage:
    validator = StressValidator()
    results = validator.validate(equity_curve, dates)
    for name, result in results.items():
        print(f"{name}: MaxDD={result.max_drawdown:.1%}, passed={result.passed}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StressWindow:
    """Named stress episode with start/end dates."""

    name: str
    start: date
    end: date
    description: str = ""


# Pre-defined stress windows (reviewer requirement)
STRESS_WINDOWS: List[StressWindow] = [
    StressWindow(
        name="covid_crash",
        start=date(2020, 2, 19),
        end=date(2020, 4, 30),
        description="COVID-19 market crash and initial recovery",
    ),
    StressWindow(
        name="bear_2022",
        start=date(2022, 1, 3),
        end=date(2022, 10, 13),
        description="2022 bear market (rate hikes + inflation)",
    ),
    StressWindow(
        name="ai_meltup_2023",
        start=date(2023, 1, 1),
        end=date(2023, 9, 30),
        description="2023 AI-driven rally (tests trend following)",
    ),
    StressWindow(
        name="regional_bank_2023",
        start=date(2023, 3, 8),
        end=date(2023, 3, 24),
        description="SVB/regional bank crisis",
    ),
    StressWindow(
        name="aug_2024_unwind",
        start=date(2024, 7, 10),
        end=date(2024, 8, 15),
        description="August 2024 carry trade unwind",
    ),
]


@dataclass
class StressWindowResult:
    """Result of validating a single stress window."""

    window_name: str
    max_drawdown: float
    total_return: float
    passed: bool
    reason: str = ""
    window_start: Optional[date] = None
    window_end: Optional[date] = None
    bars_in_window: int = 0


@dataclass
class StressValidationResult:
    """Aggregate result of all stress window validations."""

    window_results: Dict[str, StressWindowResult] = field(default_factory=dict)
    all_passed: bool = True
    failures: List[str] = field(default_factory=list)
    baseline_max_dd: float = 0.0

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.window_results.values() if r.passed)

    @property
    def num_total(self) -> int:
        return len(self.window_results)


class StressValidator:
    """
    Validates strategy equity curves against stress episodes.

    Constraint: strategy MaxDD in each stress window must be less than
    `max_dd_multiplier` times the baseline (buy-and-hold) MaxDD.
    """

    def __init__(
        self,
        windows: Optional[List[StressWindow]] = None,
        max_dd_multiplier: float = 2.0,
    ) -> None:
        """
        Args:
            windows: Stress windows to validate. Defaults to pre-defined set.
            max_dd_multiplier: Strategy MaxDD must be < multiplier * baseline MaxDD.
        """
        self._windows = windows or STRESS_WINDOWS
        self._max_dd_multiplier = max_dd_multiplier

    def validate(
        self,
        equity_curve: pd.Series,
        baseline_curve: Optional[pd.Series] = None,
    ) -> StressValidationResult:
        """
        Validate equity curve against all stress windows.

        Args:
            equity_curve: Strategy equity curve (datetime index, float values).
            baseline_curve: Buy-and-hold equity curve for comparison.
                           If None, uses absolute MaxDD threshold of 30%.

        Returns:
            StressValidationResult with per-window results.
        """
        result = StressValidationResult()

        for window in self._windows:
            window_result = self._validate_window(equity_curve, baseline_curve, window)
            result.window_results[window.name] = window_result

            if not window_result.passed:
                result.all_passed = False
                result.failures.append(f"{window.name}: {window_result.reason}")

        return result

    def _validate_window(
        self,
        equity_curve: pd.Series,
        baseline_curve: Optional[pd.Series],
        window: StressWindow,
    ) -> StressWindowResult:
        """Validate a single stress window."""
        # Extract window data
        start = pd.Timestamp(window.start)
        end = pd.Timestamp(window.end)

        # Filter equity curve to window
        mask = (equity_curve.index >= start) & (equity_curve.index <= end)
        window_equity = equity_curve[mask]

        if len(window_equity) < 2:
            return StressWindowResult(
                window_name=window.name,
                max_drawdown=0.0,
                total_return=0.0,
                passed=True,
                reason="Insufficient data in window (skipped)",
                window_start=window.start,
                window_end=window.end,
                bars_in_window=len(window_equity),
            )

        # Calculate strategy MaxDD in window
        strat_dd = self._calc_max_drawdown(window_equity)
        strat_return = window_equity.iloc[-1] / window_equity.iloc[0] - 1.0

        # Calculate baseline MaxDD in window
        if baseline_curve is not None:
            baseline_mask = (baseline_curve.index >= start) & (baseline_curve.index <= end)
            baseline_window = baseline_curve[baseline_mask]
            if len(baseline_window) >= 2:
                baseline_dd = self._calc_max_drawdown(baseline_window)
            else:
                baseline_dd = 0.30  # Default 30%
        else:
            baseline_dd = 0.30

        # Check constraint: strategy DD < multiplier * baseline DD
        # Note: _calc_max_drawdown returns negative values, use abs() for comparison
        dd_threshold = self._max_dd_multiplier * max(abs(baseline_dd), 0.05)
        passed = abs(strat_dd) < dd_threshold

        reason = ""
        if not passed:
            reason = (
                f"MaxDD {strat_dd:.1%} >= {self._max_dd_multiplier}x "
                f"baseline {baseline_dd:.1%} (threshold {dd_threshold:.1%})"
            )

        return StressWindowResult(
            window_name=window.name,
            max_drawdown=strat_dd,
            total_return=strat_return,
            passed=passed,
            reason=reason,
            window_start=window.start,
            window_end=window.end,
            bars_in_window=len(window_equity),
        )

    @staticmethod
    def _calc_max_drawdown(equity: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(equity) < 2:
            return 0.0
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return float(drawdown.min())
