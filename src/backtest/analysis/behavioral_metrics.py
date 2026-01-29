"""
BehavioralMetricsCalculator — computes gate quality metrics.

Inputs: List[TradeDecision] (post-warmup, G4) + gated/baseline RunResults.
Enforces anti-conservative ceiling (G3).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .behavioral_models import BehavioralMetrics, TradeDecision

logger = logging.getLogger(__name__)


class BehavioralMetricsCalculator:
    """
    Computes behavioral quality metrics for the DualMACD gate.

    All decisions must be post-warmup (G4 enforced by caller).
    """

    # G3: Anti-conservative ceiling
    SHARPE_CAP_MULTIPLIER = 1.25

    def calculate(
        self,
        decisions: List[TradeDecision],
        gated_sharpe: float = 0.0,
        baseline_sharpe: float = 0.0,
        gated_max_dd: float = 0.0,
        baseline_max_dd: float = 0.0,
    ) -> BehavioralMetrics:
        """
        Calculate behavioral metrics from trade decisions.

        Args:
            decisions: Post-warmup trade decisions (G4)
            gated_sharpe: Sharpe ratio of the gated portfolio
            baseline_sharpe: Sharpe ratio of the baseline (ungated) portfolio
            gated_max_dd: Max drawdown of the gated portfolio
            baseline_max_dd: Max drawdown of the baseline portfolio

        Returns:
            BehavioralMetrics with all fields populated
        """
        blocked = [d for d in decisions if not d.allowed]
        allowed = [d for d in decisions if d.allowed]

        # Blocked trade loss ratio: fraction that would have lost money
        blocked_with_pnl = [d for d in blocked if d.virtual_pnl_pct is not None]
        if blocked_with_pnl:
            losses = sum(1 for d in blocked_with_pnl if (d.virtual_pnl_pct or 0) < 0)
            blocked_trade_loss_ratio = losses / len(blocked_with_pnl)
            blocked_trade_avg_pnl = float(
                np.mean([d.virtual_pnl_pct for d in blocked_with_pnl])  # type: ignore[misc]
            )
        else:
            blocked_trade_loss_ratio = 0.0
            blocked_trade_avg_pnl = 0.0

        # G3: Cap allowed trade Sharpe to prevent over-restriction reward
        capped_sharpe = gated_sharpe
        if baseline_sharpe > 0:
            cap = baseline_sharpe * self.SHARPE_CAP_MULTIPLIER
            capped_sharpe = min(gated_sharpe, cap)

        size_down_count = sum(1 for d in decisions if d.action == "SIZE_DOWN")
        bypass_count = sum(1 for d in decisions if d.action == "BYPASS")

        return BehavioralMetrics(
            blocked_trade_loss_ratio=blocked_trade_loss_ratio,
            blocked_trade_avg_pnl=blocked_trade_avg_pnl,
            allowed_trade_sharpe=capped_sharpe,
            baseline_sharpe=baseline_sharpe,
            max_dd_gated=gated_max_dd,
            max_dd_baseline=baseline_max_dd,
            allowed_trade_count=len(allowed),
            baseline_trade_count=len(decisions),
            size_down_count=size_down_count,
            bypass_count=bypass_count,
        )

    def passes_hard_constraints(self, metrics: BehavioralMetrics) -> bool:
        """Check if metrics pass all hard constraints for Optuna."""
        # Blocked trades must mostly be losers
        if metrics.blocked_trade_loss_ratio < 0.6:
            return False

        # Must preserve trade freedom
        if metrics.allowed_trade_ratio < 0.7:
            return False

        # Must improve max drawdown
        if metrics.max_dd_baseline > 0 and metrics.max_dd_ratio > 0.85:
            return False

        return True
