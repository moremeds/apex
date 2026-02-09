"""
RegimeGate - Per-strategy regime policy enforcement.

Wraps the existing ActionResolver with strategy-level configuration
for regime-based trade gating. Each strategy defines which regimes
allow trading, minimum dwell times, switch cooldowns, and per-regime
trade limits.

Usage:
    gate = RegimeGate(policy=RegimePolicy(
        allowed_regimes=["R0", "R3"],
        min_dwell_bars=5,
        switch_cooldown_bars=10,
    ))

    result = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=12)
    if result.allowed:
        # Proceed with trade, using result.size_factor for position sizing
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..signals.indicators.regime.models import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class RegimePolicy:
    """
    Per-strategy regime policy configuration.

    Can be loaded from config/strategy/regime_policy.yaml via from_dict()
    or constructed directly in strategy __init__.
    """

    allowed_regimes: List[str] = field(default_factory=lambda: ["R0", "R1", "R3"])
    min_dwell_bars: int = 5
    switch_cooldown_bars: int = 10
    max_trades_per_regime: Dict[str, int] = field(
        default_factory=lambda: {"R0": 50, "R1": 20, "R3": 10}
    )
    forced_degross_regimes: List[str] = field(default_factory=lambda: ["R2"])
    degross_target_pct: float = 0.0
    size_factors: Dict[str, float] = field(
        default_factory=lambda: {"R0": 1.0, "R1": 0.6, "R3": 0.3}
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimePolicy":
        """Create from dictionary (e.g., YAML config)."""
        return cls(
            allowed_regimes=data.get("allowed_regimes", ["R0", "R1", "R3"]),
            min_dwell_bars=data.get("min_dwell_bars", 5),
            switch_cooldown_bars=data.get("switch_cooldown_bars", 10),
            max_trades_per_regime=data.get("max_trades_per_regime", {"R0": 50, "R1": 20, "R3": 10}),
            forced_degross_regimes=data.get("forced_degross_regimes", ["R2"]),
            degross_target_pct=data.get("degross_target_pct", 0.0),
            size_factors=data.get("size_factors", {"R0": 1.0, "R1": 0.6, "R3": 0.3}),
        )


@dataclass
class RegimeGateResult:
    """Result of regime gate evaluation."""

    allowed: bool
    size_factor: float = 1.0
    reason: str = ""
    forced_exit: bool = False

    @classmethod
    def allow(cls, size_factor: float = 1.0, reason: str = "") -> "RegimeGateResult":
        return cls(allowed=True, size_factor=size_factor, reason=reason)

    @classmethod
    def deny(cls, reason: str) -> "RegimeGateResult":
        return cls(allowed=False, size_factor=0.0, reason=reason)

    @classmethod
    def force_exit(cls, reason: str) -> "RegimeGateResult":
        return cls(allowed=False, size_factor=0.0, reason=reason, forced_exit=True)


class RegimeGate:
    """
    Per-strategy regime policy enforcement.

    Tracks regime transitions per symbol and evaluates whether
    trading is allowed based on the strategy's regime policy.
    """

    def __init__(self, policy: RegimePolicy) -> None:
        self._policy = policy
        # Per-symbol tracking
        self._current_regime: Dict[str, MarketRegime] = {}
        self._bars_in_regime: Dict[str, int] = {}
        self._bars_since_switch: Dict[str, int] = {}
        self._trade_counts: Dict[str, Dict[str, int]] = {}
        # Global bar counter per symbol
        self._bar_count: Dict[str, int] = {}

    @property
    def policy(self) -> RegimePolicy:
        return self._policy

    def evaluate(
        self,
        symbol: str,
        current_regime: MarketRegime,
        bar_count: int,
    ) -> RegimeGateResult:
        """
        Evaluate whether trading is allowed for a symbol given its regime.

        Args:
            symbol: Ticker symbol.
            current_regime: Current regime classification.
            bar_count: Current bar index (for tracking).

        Returns:
            RegimeGateResult with allowed/denied status and size factor.
        """
        regime_str = current_regime.value
        self._bar_count[symbol] = bar_count

        # Track regime transitions
        prev_regime = self._current_regime.get(symbol)
        if prev_regime != current_regime:
            self.on_regime_change(symbol, prev_regime, current_regime, bar_count)

        # Increment bars in current regime
        self._bars_in_regime[symbol] = self._bars_in_regime.get(symbol, 0) + 1
        self._bars_since_switch[symbol] = self._bars_since_switch.get(symbol, 0) + 1

        # Check forced de-gross regimes (highest priority)
        if regime_str in self._policy.forced_degross_regimes:
            return RegimeGateResult.force_exit(
                f"Regime {regime_str} forces de-gross to " f"{self._policy.degross_target_pct:.0%}"
            )

        # Check if regime is allowed
        if regime_str not in self._policy.allowed_regimes:
            return RegimeGateResult.deny(
                f"Regime {regime_str} not in allowed list " f"{self._policy.allowed_regimes}"
            )

        # Check minimum dwell time
        bars_in = self._bars_in_regime.get(symbol, 0)
        if bars_in < self._policy.min_dwell_bars:
            return RegimeGateResult.deny(
                f"Regime dwell {bars_in}/{self._policy.min_dwell_bars} bars "
                f"not met for {regime_str}"
            )

        # Check switch cooldown
        bars_since = self._bars_since_switch.get(symbol, 0)
        if bars_since < self._policy.switch_cooldown_bars:
            return RegimeGateResult.deny(
                f"Switch cooldown {bars_since}/" f"{self._policy.switch_cooldown_bars} bars not met"
            )

        # Check per-regime trade count limit
        sym_trades = self._trade_counts.get(symbol, {})
        max_trades = self._policy.max_trades_per_regime.get(regime_str, 999)
        current_trades = sym_trades.get(regime_str, 0)
        if current_trades >= max_trades:
            return RegimeGateResult.deny(
                f"Max trades {current_trades}/{max_trades} " f"reached for regime {regime_str}"
            )

        # All checks pass - get size factor
        size_factor = self._policy.size_factors.get(regime_str, 1.0)
        return RegimeGateResult.allow(
            size_factor=size_factor,
            reason=f"Regime {regime_str} allowed, size_factor={size_factor:.2f}",
        )

    def on_regime_change(
        self,
        symbol: str,
        old_regime: Optional[MarketRegime],
        new_regime: MarketRegime,
        bar_idx: int,
    ) -> None:
        """
        Track regime transition for a symbol.

        Resets dwell counter and switch cooldown.
        """
        self._current_regime[symbol] = new_regime
        self._bars_in_regime[symbol] = 0
        self._bars_since_switch[symbol] = 0

        if old_regime is not None:
            logger.info(
                f"RegimeGate: {symbol} regime change "
                f"{old_regime.value} -> {new_regime.value} at bar {bar_idx}"
            )

    def record_trade(self, symbol: str, regime: MarketRegime) -> None:
        """Record a trade for per-regime trade counting."""
        regime_str = regime.value
        if symbol not in self._trade_counts:
            self._trade_counts[symbol] = {}
        self._trade_counts[symbol][regime_str] = self._trade_counts[symbol].get(regime_str, 0) + 1

    def reset_trade_counts(self, symbol: Optional[str] = None) -> None:
        """Reset trade counts, optionally for a specific symbol."""
        if symbol:
            self._trade_counts.pop(symbol, None)
        else:
            self._trade_counts.clear()

    def get_bars_in_regime(self, symbol: str) -> int:
        """Get number of bars the symbol has been in current regime."""
        return self._bars_in_regime.get(symbol, 0)

    def get_current_regime(self, symbol: str) -> Optional[MarketRegime]:
        """Get the last known regime for a symbol."""
        return self._current_regime.get(symbol)
