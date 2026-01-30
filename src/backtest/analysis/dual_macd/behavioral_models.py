"""
Data models for DualMACD behavioral gate validation.

TradeDecision captures every entry signal with gate state and counterfactual outcome.
BehavioralMetrics aggregates gate quality into a single scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass(frozen=True)
class GatePolicy:
    """Per-sector gate behavior when DualMACD signals adverse trend."""

    action_on_block: str = "BLOCK"  # BLOCK / SIZE_DOWN / BYPASS
    size_factor: float = 0.5  # position multiplier when SIZE_DOWN


@dataclass(frozen=True)
class TradeDecision:
    """
    A single trade entry decision with DualMACD gate state.

    Every bar that generates a base strategy entry signal produces one
    TradeDecision, regardless of whether the gate allows or blocks it.

    Counterfactual fields (virtual_exit_price, virtual_pnl_pct) are
    resolved post-exit using the BASE strategy exit signal only (G1).
    """

    timestamp: date
    symbol: str
    regime: str  # R0/R1/R2/R3
    strategy_direction: str  # LONG / SHORT

    intended_size: float

    # DualMACD snapshot at decision time
    trend_state: str  # BULLISH / BEARISH / IMPROVING / DETERIORATING
    tactical_signal: str  # DIP_BUY / RALLY_SELL / NONE
    momentum_balance: str  # FAST_DOMINANT / SLOW_DOMINANT / BALANCED
    confidence: float  # 0.0-1.0, curvature-based
    gate_strength: float  # mirrors confidence when tactical_signal != NONE, else 0.0

    # Gate decision
    allowed: bool
    action: str  # ENTER / BLOCK / SIZE_DOWN
    block_reason: Optional[str] = None
    size_factor: float = 1.0  # 1.0=full, 0.5=half (SIZE_DOWN), 0.0=blocked

    # Counterfactual PnL (resolved post-exit via base strategy exits, G1)
    virtual_entry_price: float = 0.0
    virtual_exit_price: Optional[float] = None
    virtual_pnl_pct: Optional[float] = None


@dataclass
class BehavioralMetrics:
    """
    Aggregated behavioral quality metrics for the DualMACD gate.

    These measure whether the gate blocks bad trades and preserves
    trade freedom — NOT whether it generates alpha.
    """

    # Primary: fraction of blocked trades that would have lost money
    blocked_trade_loss_ratio: float = 0.0

    # Average virtual PnL of blocked trades (negative = gate is working)
    blocked_trade_avg_pnl: float = 0.0

    # Sharpe of allowed trades vs baseline (all trades)
    allowed_trade_sharpe: float = 0.0
    baseline_sharpe: float = 0.0

    # Max drawdown comparison
    max_dd_gated: float = 0.0
    max_dd_baseline: float = 0.0

    # Trade count comparison
    allowed_trade_count: int = 0
    baseline_trade_count: int = 0

    # Policy action counts
    size_down_count: int = 0
    bypass_count: int = 0

    # Per-action average virtual PnL (resolved counterfactuals)
    size_down_avg_pnl: float = 0.0
    bypass_avg_pnl: float = 0.0

    # Derived ratios
    @property
    def allowed_trade_ratio(self) -> float:
        """Fraction of baseline trades that the gate allows."""
        if self.baseline_trade_count == 0:
            return 0.0
        return self.allowed_trade_count / self.baseline_trade_count

    @property
    def max_dd_ratio(self) -> float:
        """Gated max_dd / baseline max_dd (< 1.0 is better)."""
        if self.max_dd_baseline == 0.0:
            return 0.0
        return self.max_dd_gated / self.max_dd_baseline

    @property
    def blocked_trade_count(self) -> int:
        return self.baseline_trade_count - self.allowed_trade_count


@dataclass
class BehavioralRunConfig:
    """Configuration for a behavioral gate validation run."""

    symbol: str
    start_date: date
    end_date: date

    # DualMACD gate params (Optuna targets)
    slope_lookback: int = 3
    hist_norm_window: int = 252

    # Base strategy
    base_strategy: str = "ma_cross"
    base_params: dict = field(default_factory=dict)

    # Gate settings
    regime: str = "R0"  # default regime label
    direction: str = "LONG"  # default direction
