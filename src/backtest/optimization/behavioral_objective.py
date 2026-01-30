"""
Optuna objective function for DualMACD behavioral gate optimization.

Optimizes behavioral quality (blocks bad trades, preserves freedom),
NOT alpha or PnL. Hard constraints prune bad trials early.
"""

from __future__ import annotations

import logging
from typing import Callable

from optuna import Trial
from optuna.exceptions import TrialPruned

from src.backtest.analysis.dual_macd.behavioral_metrics import BehavioralMetricsCalculator
from src.backtest.analysis.dual_macd.behavioral_models import BehavioralMetrics

logger = logging.getLogger(__name__)


class BehavioralObjective:
    """
    Optuna objective for DualMACD gate parameter optimization.

    The objective rewards:
    1. Blocking trades that would have lost money (high loss ratio)
    2. Negative average PnL on blocked trades
    3. Better Sharpe on allowed trades (capped at 1.25x baseline, G3)
    4. Reduced max drawdown

    Hard constraints (pruned if violated):
    - blocked_trade_loss_ratio >= 0.6
    - allowed_trade_count >= 0.7 * baseline_trade_count
    - max_dd_gated <= 0.85 * max_dd_baseline
    """

    def __init__(
        self,
        run_fn: Callable[[int, int], BehavioralMetrics],
    ) -> None:
        """
        Args:
            run_fn: Function(slope_lookback, hist_norm_window) -> BehavioralMetrics
                    Runs gated vs baseline and returns metrics.
        """
        self._run_fn = run_fn
        self._calculator = BehavioralMetricsCalculator()

    def __call__(self, trial: Trial) -> float:
        """Optuna objective function."""
        slope_lookback = trial.suggest_categorical("slope_lookback", [2, 3, 5])
        hist_norm_window = trial.suggest_categorical("hist_norm_window", [126, 252, 504])

        metrics = self._run_fn(slope_lookback, hist_norm_window)

        logger.info(
            f"Trial sl={slope_lookback} hnw={hist_norm_window}: "
            f"loss_ratio={metrics.blocked_trade_loss_ratio:.2f}, "
            f"allowed={metrics.allowed_trade_count}/{metrics.baseline_trade_count}, "
            f"blocked_avg_pnl={metrics.blocked_trade_avg_pnl:+.4f}, "
            f"dd_gated={metrics.max_dd_gated:.3f} dd_base={metrics.max_dd_baseline:.3f}"
        )

        # Hard constraints → prune
        if metrics.blocked_trade_loss_ratio < 0.6:
            raise TrialPruned(
                f"blocked_trade_loss_ratio={metrics.blocked_trade_loss_ratio:.2f} < 0.6"
            )

        if metrics.baseline_trade_count > 0:
            if metrics.allowed_trade_count < 0.7 * metrics.baseline_trade_count:
                raise TrialPruned(
                    f"allowed_trade_count={metrics.allowed_trade_count} "
                    f"< 0.7 * {metrics.baseline_trade_count}"
                )

        if metrics.max_dd_baseline > 0:
            if metrics.max_dd_gated > 0.85 * metrics.max_dd_baseline:
                raise TrialPruned(
                    f"max_dd_gated={metrics.max_dd_gated:.3f} "
                    f"> 0.85 * {metrics.max_dd_baseline:.3f}"
                )

        # G3: Anti-conservative ceiling on Sharpe ratio term
        sharpe_ratio = min(
            metrics.allowed_trade_sharpe / max(metrics.baseline_sharpe, 0.01),
            1.25,
        )

        # Composite score: higher = better gate
        score = (
            1.0 * (-metrics.blocked_trade_avg_pnl)
            + 0.8 * metrics.blocked_trade_loss_ratio
            + 0.6 * sharpe_ratio
            - 0.5 * max(0, metrics.max_dd_gated - 0.85 * metrics.max_dd_baseline)
        )

        trial.set_user_attr("blocked_loss_ratio", metrics.blocked_trade_loss_ratio)
        trial.set_user_attr("blocked_avg_pnl", metrics.blocked_trade_avg_pnl)
        trial.set_user_attr("allowed_sharpe", metrics.allowed_trade_sharpe)
        trial.set_user_attr("baseline_sharpe", metrics.baseline_sharpe)
        trial.set_user_attr("allowed_count", metrics.allowed_trade_count)
        trial.set_user_attr("baseline_count", metrics.baseline_trade_count)
        trial.set_user_attr("max_dd_gated", metrics.max_dd_gated)
        trial.set_user_attr("max_dd_baseline", metrics.max_dd_baseline)

        return score
