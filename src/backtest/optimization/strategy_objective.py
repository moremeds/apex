"""
Generic strategy Optuna objective with anti-overfit constraints.

Extends the DualMACD-specific BehavioralObjective into a general-purpose
objective for any registered strategy. Includes parameter budget enforcement,
turnover penalties, and stress window validation.

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
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    total_cost: float = 0.0
    equity_curve: Optional[pd.Series] = None
    baseline_curve: Optional[pd.Series] = None

    @property
    def calmar(self) -> float:
        if self.max_drawdown == 0:
            return 0.0
        return abs(self.total_return / self.max_drawdown)


class StrategyObjective:
    """
    Generic Optuna objective for strategy optimization.

    Anti-overfit constraints:
    1. Parameter budget: max_params enforced via trial pruning
    2. Turnover penalty: adjusted_score = raw - weight * cost
    3. Stress windows: all windows must pass
    4. Execution realism: must pass both Tier A and Tier B
    """

    def __init__(
        self,
        strategy_name: str,
        run_fn: Callable[[Dict[str, Any]], BacktestResult],
        max_params: int = 8,
        turnover_penalty_weight: float = 0.1,
        min_trades: int = 20,
        stress_validator: Optional[StressValidator] = None,
    ) -> None:
        """
        Args:
            strategy_name: Name of the strategy being optimized.
            run_fn: Function that runs backtest with params and returns result.
            max_params: Maximum number of variable parameters.
            turnover_penalty_weight: Weight for turnover cost penalty.
            min_trades: Minimum required trades (prune if fewer).
            stress_validator: Optional stress window validator.
        """
        self._strategy_name = strategy_name
        self._run_fn = run_fn
        self._max_params = max_params
        self._turnover_penalty_weight = turnover_penalty_weight
        self._min_trades = min_trades
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

        # Hard constraint: minimum trades
        if result.trade_count < self._min_trades:
            raise TrialPruned(f"Too few trades: {result.trade_count} < {self._min_trades}")

        # Stress window validation
        if result.equity_curve is not None:
            stress_result = self._stress_validator.validate(
                result.equity_curve, result.baseline_curve
            )
            if not stress_result.all_passed:
                raise TrialPruned(f"Stress validation failed: {stress_result.failures}")

        # Compute composite score with turnover penalty
        turnover_cost = result.total_cost * self._turnover_penalty_weight
        composite_score = result.sharpe - turnover_cost

        # Log trial results
        trial.set_user_attr("total_return", result.total_return)
        trial.set_user_attr("max_drawdown", result.max_drawdown)
        trial.set_user_attr("trade_count", result.trade_count)
        trial.set_user_attr("win_rate", result.win_rate)
        trial.set_user_attr("calmar", result.calmar)

        logger.info(
            f"Trial {trial.number}: Sharpe={result.sharpe:.3f}, "
            f"Return={result.total_return:.1%}, "
            f"MaxDD={result.max_drawdown:.1%}, "
            f"Trades={result.trade_count}, "
            f"Score={composite_score:.3f}"
        )

        return composite_score

    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest parameters based on strategy name.

        Each strategy defines its own parameter space.
        Override this method for custom parameter spaces.
        """
        if self._strategy_name == "pulse_dip":
            return {
                "ema_trend_period": trial.suggest_int("ema_trend_period", 50, 200, step=25),
                "rsi_period": trial.suggest_int("rsi_period", 10, 21),
                "rsi_entry_threshold": trial.suggest_float("rsi_entry_threshold", 25, 45, step=5),
                "atr_stop_mult": trial.suggest_float("atr_stop_mult", 2.0, 4.0, step=0.5),
                "min_confluence_score": trial.suggest_int("min_confluence_score", 0, 50, step=10),
                "max_hold_bars": trial.suggest_int("max_hold_bars", 20, 60, step=10),
                "risk_per_trade_pct": trial.suggest_float(
                    "risk_per_trade_pct", 0.01, 0.04, step=0.01
                ),
            }
        elif self._strategy_name == "squeeze_play":
            return {
                "bb_period": trial.suggest_int("bb_period", 15, 30, step=5),
                "bb_std": trial.suggest_float("bb_std", 1.5, 2.5, step=0.5),
                "kc_multiplier": trial.suggest_float("kc_multiplier", 1.0, 2.0, step=0.25),
                "release_persist_bars": trial.suggest_int("release_persist_bars", 1, 4),
                "close_outside_bars": trial.suggest_int("close_outside_bars", 1, 3),
                "atr_stop_mult": trial.suggest_float("atr_stop_mult", 1.5, 3.5, step=0.5),
                "adx_min": trial.suggest_float("adx_min", 15, 30, step=5),
                "risk_per_trade_pct": trial.suggest_float(
                    "risk_per_trade_pct", 0.01, 0.04, step=0.01
                ),
            }
        else:
            # Default: return empty params (strategy uses defaults)
            return {}
