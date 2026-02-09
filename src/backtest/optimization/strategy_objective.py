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

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Type

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


# Frozen params: structural parameters that should NOT be optimized.
# Optimizing these overfits to noise (e.g., BB geometry for SqueezePlay).
FROZEN_PARAMS: Dict[str, Set[str]] = {
    "squeeze_play": {"bb_period", "bb_std", "kc_multiplier"},
    "trend_pulse": {"risk_per_trade_pct", "cooldown_bars"},
    "pulse_dip": set(),
    "regime_flex": set(),
    "sector_pulse": set(),
}

# Correlated param groups: params that move together count as 1 for budget.
# Currently none needed — reserved for future multi-horizon params.
CORRELATED_GROUPS: Dict[str, List[List[str]]] = {}

# Number of params actually optimized by Optuna (from _build_param_space).
# This is the true overfitting budget — raw YAML param count includes
# structural params (warmup_bars, indicator periods) that are never optimized.
OPTUNA_PARAM_COUNTS: Dict[str, int] = {
    "pulse_dip": 7,  # rsi_period, rsi_entry_threshold, atr_stop_mult, hard_stop_pct, min_confluence_score, max_hold_bars, risk_per_trade_pct
    "squeeze_play": 4,  # release_persist_bars, close_outside_bars, adx_min, atr_stop_mult
    "trend_pulse": 7,  # zig_threshold_pct, trend_strength_moderate, min_confidence, hard_stop_pct, atr_stop_mult, exit_bearish_bars, adx_entry_min
    "regime_flex": 4,  # r0_gross_pct, r1_gross_pct, r3_gross_pct, ramp_bars
    "sector_pulse": 5,  # top_n_sectors, confidence_threshold, drift_threshold_pct, max_turnover_pct, risk_per_sector_pct
}


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
        strategy_class: Optional[Type] = None,
    ) -> None:
        """
        Args:
            strategy_name: Name of the strategy being optimized.
            run_fn: Function that runs backtest with params and returns result.
            max_params: Maximum effective parameter budget.
            turnover_penalty_weight: Weight for turnover cost penalty.
            min_trades: Minimum required trades (prune if fewer).
            max_drawdown_cap: MaxDD threshold (negative, e.g., -0.30 = -30%).
            min_exposure_pct: Min fraction of bars with position (prune if lower).
            stress_validator: Optional stress window validator (results stored as badges).
            strategy_class: Optional Strategy class for constructor param validation.
        """
        self._strategy_name = strategy_name
        self._run_fn = run_fn
        self._max_params = max_params
        self._turnover_penalty_weight = turnover_penalty_weight
        self._min_trades = min_trades
        self._max_drawdown_cap = max_drawdown_cap
        self._min_exposure_pct = min_exposure_pct
        self._stress_validator = stress_validator or StressValidator()
        self._strategy_class = strategy_class

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

        Each strategy defines its own parameter space. Frozen params use YAML
        defaults and are excluded from optimization. After building the param
        dict, validates that all suggested names exist in the strategy constructor
        (prevents silent-drop bug where runner filters unknown params).
        """
        params = self._build_param_space(trial)
        self._validate_params_against_constructor(params)
        self._validate_param_budget()
        return params

    def _build_param_space(self, trial: Trial) -> Dict[str, Any]:
        """Build the Optuna param space for the current strategy."""
        if self._strategy_name == "pulse_dip":
            # PulseDip constructor: ema_trend_period, rsi_period,
            # rsi_entry_threshold, atr_stop_mult, hard_stop_pct,
            # min_confluence_score, max_hold_bars, risk_per_trade_pct
            return {
                "rsi_period": trial.suggest_int("rsi_period", 10, 21),
                "rsi_entry_threshold": trial.suggest_float("rsi_entry_threshold", 25, 45, step=5),
                "atr_stop_mult": trial.suggest_float("atr_stop_mult", 2.0, 4.0, step=0.5),
                "hard_stop_pct": trial.suggest_float("hard_stop_pct", 0.05, 0.12, step=0.01),
                "min_confluence_score": trial.suggest_int("min_confluence_score", 10, 30),
                "max_hold_bars": trial.suggest_int("max_hold_bars", 20, 60),
                "risk_per_trade_pct": trial.suggest_float(
                    "risk_per_trade_pct", 0.01, 0.05, step=0.01
                ),
            }
        elif self._strategy_name == "squeeze_play":
            # bb_period, bb_std, kc_multiplier are FROZEN — use YAML defaults
            return {
                "release_persist_bars": trial.suggest_int("release_persist_bars", 1, 4),
                "close_outside_bars": trial.suggest_int("close_outside_bars", 1, 3),
                "adx_min": trial.suggest_float("adx_min", 15, 30, step=5),
                "atr_stop_mult": trial.suggest_float("atr_stop_mult", 1.5, 3.5, step=0.5),
            }
        elif self._strategy_name == "trend_pulse":
            # cooldown_bars and risk_per_trade_pct are FROZEN (structural)
            return {
                "zig_threshold_pct": trial.suggest_float("zig_threshold_pct", 1.5, 5.0, step=0.5),
                "trend_strength_moderate": trial.suggest_float(
                    "trend_strength_moderate", 0.10, 0.30, step=0.05
                ),
                "min_confidence": trial.suggest_float("min_confidence", 0.2, 0.6, step=0.05),
                "hard_stop_pct": trial.suggest_float("hard_stop_pct", 0.10, 0.20, step=0.02),
                "atr_stop_mult": trial.suggest_float("atr_stop_mult", 3.0, 6.0, step=0.5),
                "exit_bearish_bars": trial.suggest_int("exit_bearish_bars", 2, 5),
                "adx_entry_min": trial.suggest_float("adx_entry_min", 10, 25, step=5),
            }
        elif self._strategy_name == "regime_flex":
            # All params meaningful, none frozen
            return {
                "r0_gross_pct": trial.suggest_float("r0_gross_pct", 0.7, 1.0, step=0.1),
                "r1_gross_pct": trial.suggest_float("r1_gross_pct", 0.2, 0.7, step=0.1),
                "r3_gross_pct": trial.suggest_float("r3_gross_pct", 0.1, 0.5, step=0.1),
                "ramp_bars": trial.suggest_int("ramp_bars", 3, 20),
            }
        elif self._strategy_name == "sector_pulse":
            # rebalance_day and slippage_bps are FROZEN (structural/execution)
            return {
                "top_n_sectors": trial.suggest_int("top_n_sectors", 2, 5),
                "confidence_threshold": trial.suggest_float(
                    "confidence_threshold", 0.005, 0.10, step=0.005
                ),
                "drift_threshold_pct": trial.suggest_float(
                    "drift_threshold_pct", 0.03, 0.15, step=0.01
                ),
                "max_turnover_pct": trial.suggest_float("max_turnover_pct", 0.15, 0.50, step=0.05),
                "risk_per_sector_pct": trial.suggest_float(
                    "risk_per_sector_pct", 0.05, 0.20, step=0.05
                ),
            }
        else:
            return {}

    def _validate_params_against_constructor(self, params: Dict[str, Any]) -> None:
        """
        Validate that all suggested param names exist in the strategy constructor.

        Prevents the silent-drop bug where strategy_compare_runner filters
        unknown params via inspect.signature(), causing Optuna to "optimize"
        params that are silently discarded at runtime.
        """
        if not self._strategy_class or not params:
            return

        sig = inspect.signature(self._strategy_class.__init__)
        accepted = set(sig.parameters.keys()) - {"self", "strategy_id", "symbols", "context"}
        invalid = set(params.keys()) - accepted
        if invalid:
            raise ValueError(
                f"Optuna suggests params not in {self._strategy_name} constructor: "
                f"{sorted(invalid)}. Accepted: {sorted(accepted)}"
            )

    def _validate_param_budget(self) -> None:
        """Log warning if effective param count exceeds budget."""
        frozen = FROZEN_PARAMS.get(self._strategy_name, set())
        correlated = CORRELATED_GROUPS.get(self._strategy_name, [])
        correlated_savings = sum(len(g) - 1 for g in correlated)

        try:
            from src.domain.strategy.param_loader import get_strategy_params

            yaml_params = get_strategy_params(self._strategy_name)
            total = len(yaml_params)
        except KeyError:
            return  # No YAML config — skip budget check

        effective = total - len(frozen) - correlated_savings
        if effective > self._max_params:
            logger.warning(
                f"{self._strategy_name}: {effective} effective params "
                f"(total={total}, frozen={len(frozen)}, correlated={correlated_savings}) "
                f"exceeds budget of {self._max_params}"
            )
