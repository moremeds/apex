"""
Generic strategy Optuna objective with two-stage anti-overfit design.

Stage 1 — Non-negotiable gates (prune if ANY violated):
    - MaxDD cap (default -30%)
    - Minimum trades (default 20)
    - Positive Sharpe
    - Minimum exposure (default 30% of bars — prevents "win by being flat")

Stage 2 — Rank survivors (no pruning, just scoring):
    - Composite = Sharpe - turnover_penalty
    - Stress results stored as informational badges (user_attrs)

Usage:
    objective = StrategyObjective(
        strategy_name="pulse_dip",
        run_fn=lambda params: run_backtest("pulse_dip", params),
        max_params=7,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd
from optuna import Trial
from optuna.exceptions import TrialPruned

from .stress_validator import StressValidator

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a single backtest run."""

    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0  # Negative value (e.g., -0.15 = 15% drawdown)
    win_rate: float = 0.0
    trade_count: int = 0
    total_cost: float = 0.0
    equity_curve: Optional[pd.Series] = None
    baseline_curve: Optional[pd.Series] = None
    exposure_pct: float = 0.0  # Fraction of bars with open position (0.0-1.0)

    @property
    def calmar(self) -> float:
        if self.max_drawdown == 0:
            return 0.0
        return abs(self.total_return / self.max_drawdown)


class StrategyObjective:
    """
    Two-stage Optuna objective for strategy optimization.

    Stage 1 — Non-negotiable gates (prune immediately):
        - Max drawdown cap
        - Minimum trade count
        - Positive Sharpe
        - Minimum exposure percentage

    Stage 2 — Rank survivors:
        - Composite score = Sharpe - turnover_penalty
        - Stress results stored as badges (not gates)
    """

    def __init__(
        self,
        strategy_name: str,
        run_fn: Callable[[Dict[str, Any]], BacktestResult],
        max_params: int = 8,
        turnover_penalty_weight: float = 0.1,
        min_trades: int = 20,
        max_drawdown_cap: float = -0.30,
        min_exposure_pct: float = 0.30,
        stress_validator: Optional[StressValidator] = None,
    ) -> None:
        """
        Args:
            strategy_name: Name of the strategy being optimized.
            run_fn: Function that runs backtest with params and returns result.
            max_params: Maximum number of variable parameters.
            turnover_penalty_weight: Weight for turnover cost penalty.
            min_trades: Minimum required trades (prune if fewer).
            max_drawdown_cap: MaxDD threshold (negative, e.g., -0.30 = -30%).
            min_exposure_pct: Min fraction of bars with position (prune if lower).
            stress_validator: Optional stress window validator (results stored as badges).
        """
        self._strategy_name = strategy_name
        self._run_fn = run_fn
        self._max_params = max_params
        self._turnover_penalty_weight = turnover_penalty_weight
        self._min_trades = min_trades
        self._max_drawdown_cap = max_drawdown_cap
        self._min_exposure_pct = min_exposure_pct
        self._stress_validator = stress_validator or StressValidator()

    def __call__(self, trial: Trial) -> float:
        """
        Optuna objective function.

        Returns composite score (higher = better).
        """
        # Get parameters from trial
        params = self._suggest_params(trial)

        # Check parameter budget
        variable_params = sum(1 for p in trial.params.values() if not isinstance(p, (bool,)))
        if variable_params > self._max_params:
            raise TrialPruned(f"Parameter budget exceeded: {variable_params} > {self._max_params}")

        # Run backtest
        try:
            result = self._run_fn(params)
        except Exception as e:
            logger.error(f"Backtest failed for trial {trial.number}: {e}")
            raise TrialPruned(f"Backtest error: {e}") from e

        # ====================================================================
        # STAGE 1: Non-negotiable gates (prune if ANY violated)
        # These are absolute requirements, not tunable.
        # ====================================================================

        if result.max_drawdown < self._max_drawdown_cap:
            raise TrialPruned(
                f"MaxDD {result.max_drawdown:.1%} exceeds cap {self._max_drawdown_cap:.1%}"
            )

        if result.trade_count < self._min_trades:
            raise TrialPruned(f"Too few trades: {result.trade_count} < {self._min_trades}")

        if result.sharpe < 0.0:
            raise TrialPruned(f"Negative Sharpe: {result.sharpe:.3f}")

        if result.exposure_pct < self._min_exposure_pct:
            raise TrialPruned(
                f"Insufficient exposure: {result.exposure_pct:.1%} < {self._min_exposure_pct:.1%}"
            )

        # ====================================================================
        # STAGE 2: Rank survivors (no more pruning, just scoring)
        # ====================================================================

        # Composite score with turnover penalty
        turnover_cost = result.total_cost * self._turnover_penalty_weight
        composite_score = result.sharpe - turnover_cost

        # Stress validation: informational badge, NOT a gate
        if result.equity_curve is not None:
            try:
                stress_result = self._stress_validator.validate(
                    result.equity_curve, result.baseline_curve
                )
                trial.set_user_attr("stress_passed", stress_result.all_passed)
                trial.set_user_attr("stress_failures", stress_result.failures)
            except Exception as e:
                logger.debug(f"Stress validation skipped for trial {trial.number}: {e}")
                trial.set_user_attr("stress_passed", None)

        # Store metrics as user attributes (for dashboard display)
        trial.set_user_attr("total_return", result.total_return)
        trial.set_user_attr("max_drawdown", result.max_drawdown)
        trial.set_user_attr("trade_count", result.trade_count)
        trial.set_user_attr("win_rate", result.win_rate)
        trial.set_user_attr("calmar", result.calmar)
        trial.set_user_attr("exposure_pct", result.exposure_pct)

        logger.info(
            f"Trial {trial.number}: Sharpe={result.sharpe:.3f}, "
            f"Return={result.total_return:.1%}, "
            f"MaxDD={result.max_drawdown:.1%}, "
            f"Trades={result.trade_count}, "
            f"Exposure={result.exposure_pct:.0%}, "
            f"Score={composite_score:.3f}"
        )

        return composite_score

    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest parameters based on strategy name.

        Each strategy defines its own parameter space.
        SqueezePlay: bb_period, bb_std, kc_multiplier are frozen (use YAML defaults).
        """
        if self._strategy_name == "pulse_dip":
            return {
                "rsi_period": trial.suggest_int("rsi_period", 10, 21),
                "rsi_entry_threshold": trial.suggest_float("rsi_entry_threshold", 25, 45, step=5),
                "atr_stop_mult": trial.suggest_float("atr_stop_mult", 2.0, 4.0, step=0.5),
                "adx_entry_min": trial.suggest_float("adx_entry_min", 10, 25, step=5),
                "trend_strength_moderate": trial.suggest_float(
                    "trend_strength_moderate", 0.10, 0.30, step=0.05
                ),
                "exit_bearish_bars": trial.suggest_int("exit_bearish_bars", 2, 5),
                "hard_stop_pct": trial.suggest_float("hard_stop_pct", 0.05, 0.12, step=0.01),
            }
        elif self._strategy_name == "squeeze_play":
            # bb_period, bb_std, kc_multiplier are FROZEN — use YAML defaults
            # Only optimize release/entry timing and risk params
            return {
                "release_persist_bars": trial.suggest_int("release_persist_bars", 1, 4),
                "close_outside_bars": trial.suggest_int("close_outside_bars", 1, 3),
                "adx_min": trial.suggest_float("adx_min", 15, 30, step=5),
                "atr_stop_mult": trial.suggest_float("atr_stop_mult", 1.5, 3.5, step=0.5),
            }
        elif self._strategy_name == "trend_pulse":
            return {
                "atr_stop_mult": trial.suggest_float("atr_stop_mult", 2.5, 5.0, step=0.5),
                "exit_bearish_bars": trial.suggest_int("exit_bearish_bars", 2, 5),
                "adx_entry_min": trial.suggest_float("adx_entry_min", 10, 25, step=5),
                "cooldown_bars": trial.suggest_int("cooldown_bars", 3, 8),
                "min_confidence": trial.suggest_float("min_confidence", 0.3, 0.7, step=0.1),
                "trend_strength_moderate": trial.suggest_float(
                    "trend_strength_moderate", 0.10, 0.30, step=0.05
                ),
            }
        else:
            # Default: return empty params (strategy uses YAML defaults)
            return {}
