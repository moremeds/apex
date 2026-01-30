"""
DualMACD Behavioral Gate Signal Generator.

Wraps any base SignalGenerator and applies DualMACD state as a behavioral
gate — blocking entries in adverse trend states while preserving trade
freedom in favorable conditions.

Key guardrails:
- G1: Counterfactual exits use base strategy exits only
- G2: Symmetric gate logic (different rules for LONG vs SHORT)
- G4: Warmup period excluded from all metrics
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.backtest.analysis.dual_macd.behavioral_models import GatePolicy, TradeDecision
from src.backtest.analysis.dual_macd.trade_decision_logger import TradeDecisionLogger
from src.domain.signals.indicators.momentum.dual_macd import (
    DualMACDConfig,
    DualMACDIndicator,
    TacticalSignal,
    TrendState,
)

logger = logging.getLogger(__name__)


class DualMACDGateSignalGenerator:
    """
    Wraps a base SignalGenerator with DualMACD behavioral gating.

    The gate blocks entries in adverse trend states:
    - LONG: blocks DETERIORATING/BEARISH (unless DIP_BUY tactical signal fires)
    - SHORT: blocks IMPROVING/BULLISH (unless RALLY_SELL tactical signal fires)

    This is a behavioral constraint, not an alpha source.
    """

    def __init__(
        self,
        base_generator: Any,  # SignalGenerator protocol
        slope_lookback: int = 3,
        hist_norm_window: int = 252,
        direction: str = "LONG",
        regime: str = "R0",
        gate_policies: Optional[Dict[str, GatePolicy]] = None,
        symbol_to_sector: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_generator = base_generator
        self.direction = direction.upper()
        self.regime = regime
        self._gate_policies = gate_policies or {}
        self._symbol_to_sector = symbol_to_sector or {}

        # DualMACD config with tunable params
        self._macd_config = DualMACDConfig(
            slope_lookback=slope_lookback,
            hist_norm_window=hist_norm_window,
        )
        self._macd_indicator = DualMACDIndicator()
        self._macd_params = {
            "slow_fast": self._macd_config.slow_fast,
            "slow_slow": self._macd_config.slow_slow,
            "slow_signal": self._macd_config.slow_signal,
            "fast_fast": self._macd_config.fast_fast,
            "fast_slow": self._macd_config.fast_slow,
            "fast_signal": self._macd_config.fast_signal,
            "slope_lookback": slope_lookback,
            "hist_norm_window": hist_norm_window,
            "histogram_multiplier": self._macd_config.histogram_multiplier,
            "eps": self._macd_config.eps,
        }

    @property
    def warmup_bars(self) -> int:
        """Max of base warmup and DualMACD warmup."""
        base_warmup = getattr(self.base_generator, "warmup_bars", 0)
        return max(base_warmup, self._macd_indicator.warmup_periods)

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate gated entry/exit signals.

        Returns entries filtered by DualMACD gate, exits unchanged from base.
        """
        base_entries, base_exits = self.base_generator.generate(data, params, secondary_data)

        gate_mask = self._compute_gate_mask(data, params.get("symbol", ""))
        gated_entries = base_entries & gate_mask

        return gated_entries, base_exits

    def generate_with_decisions(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series, TradeDecisionLogger]:
        """
        Generate gated signals with full decision logging.

        Returns:
            (gated_entries, base_exits, decision_logger)
        """
        base_entries, base_exits = self.base_generator.generate(data, params, secondary_data)

        # Compute DualMACD state for all bars
        macd_result = self._macd_indicator.calculate(data[["close"]], self._macd_params)
        states = self._compute_states(macd_result, data)

        # Build gate mask and log decisions
        decision_logger = TradeDecisionLogger()
        gate_mask = pd.Series(True, index=data.index)

        warmup = self.warmup_bars
        close_prices = data["close"]

        for i, idx in enumerate(data.index):
            if not base_entries.iloc[i]:
                continue

            if i < warmup:
                # During warmup, allow all entries but don't log
                continue

            state = states[i]
            ts = idx.date() if hasattr(idx, "date") else idx
            sym = params.get("symbol", "UNKNOWN")

            allowed, action, block_reason, size_factor = self._evaluate_gate(state, sym)

            gate_strength = state["confidence"] if state["tactical_signal"] != "NONE" else 0.0

            decision = TradeDecision(
                timestamp=ts,
                symbol=sym,
                regime=self.regime,
                strategy_direction=self.direction,
                intended_size=1.0,
                trend_state=state["trend_state"],
                tactical_signal=state["tactical_signal"],
                momentum_balance=state["momentum_balance"],
                confidence=state["confidence"],
                gate_strength=gate_strength,
                allowed=allowed,
                action=action,
                block_reason=block_reason,
                size_factor=size_factor,
                virtual_entry_price=float(close_prices.iloc[i]),
            )
            decision_logger.log_decision(decision)

            if not allowed:
                gate_mask.iloc[i] = False

        gated_entries = base_entries & gate_mask

        # Resolve counterfactuals using base exits (G1)
        decision_logger.resolve_all_counterfactuals(base_exits, close_prices)

        return gated_entries, base_exits, decision_logger

    def _compute_gate_mask(self, data: pd.DataFrame, symbol: str = "") -> pd.Series:
        """Compute boolean gate mask (True = allow entry)."""
        macd_result = self._macd_indicator.calculate(data[["close"]], self._macd_params)
        states = self._compute_states(macd_result, data)

        mask = pd.Series(True, index=data.index)
        for i in range(len(data)):
            if i < self.warmup_bars:
                continue
            state = states[i]
            allowed, _, _, _ = self._evaluate_gate(state, symbol)
            if not allowed:
                mask.iloc[i] = False

        return mask

    def _get_policy_for_symbol(self, symbol: str) -> GatePolicy:
        """Resolve gate policy: symbol-level first, then sector, then default."""
        if not self._gate_policies:
            return GatePolicy(action_on_block="BLOCK")

        sym = symbol.upper()

        # 1. Direct symbol lookup (from cluster policies)
        if sym in self._gate_policies:
            return self._gate_policies[sym]

        # 2. Sector lookup
        sector = self._symbol_to_sector.get(sym, "")
        if sector in self._gate_policies:
            return self._gate_policies[sector]

        return self._gate_policies.get("default", GatePolicy(action_on_block="BLOCK"))

    def _evaluate_gate(
        self, state: Dict[str, Any], symbol: str = ""
    ) -> Tuple[bool, str, Optional[str], float]:
        """
        Evaluate gate for a single bar. Symmetric logic (G2).

        Returns:
            (allowed, action, block_reason, size_factor)
        """
        trend = state["trend_state"]
        tactical = state["tactical_signal"]

        # Check if trend is adverse
        adverse = False
        if self.direction == "LONG":
            if trend in (TrendState.DETERIORATING.value, TrendState.BEARISH.value):
                if tactical == TacticalSignal.DIP_BUY.value:
                    return True, "ENTER", None, 1.0
                adverse = True
        else:  # SHORT
            if trend in (TrendState.IMPROVING.value, TrendState.BULLISH.value):
                if tactical == TacticalSignal.RALLY_SELL.value:
                    return True, "ENTER", None, 1.0
                adverse = True

        if not adverse:
            return True, "ENTER", None, 1.0

        # Adverse trend — apply policy
        policy = self._get_policy_for_symbol(symbol)
        reason = f"{self.direction} blocked: trend={trend}"

        if policy.action_on_block == "BYPASS":
            return True, "BYPASS", None, 1.0
        elif policy.action_on_block == "SIZE_DOWN":
            return True, "SIZE_DOWN", reason, policy.size_factor
        else:  # BLOCK
            return False, "BLOCK", reason, 0.0

    def _compute_states(
        self, macd_result: pd.DataFrame, data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Compute DualMACD state for every bar."""
        states: List[Dict[str, Any]] = []
        for i in range(len(macd_result)):
            current = macd_result.iloc[i]
            previous = macd_result.iloc[i - 1] if i > 0 else None
            state = self._macd_indicator._get_state(current, previous, self._macd_params)
            states.append(state)
        return states

    def get_warmup_end_date(self, data: pd.DataFrame) -> date:
        """Return the date after which DualMACD warmup is complete (G4)."""
        if len(data) <= self.warmup_bars:
            idx = data.index[-1]
        else:
            idx = data.index[self.warmup_bars]
        return date.fromisoformat(str(idx.date() if hasattr(idx, "date") else idx))
