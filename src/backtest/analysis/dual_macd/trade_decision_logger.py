"""
In-memory trade decision log with JSONL export and counterfactual resolution.

Logs every gate decision (allow/block) and resolves counterfactual PnL
using base strategy exits only (G1: counterfactual exit isolation).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .behavioral_models import TradeDecision

logger = logging.getLogger(__name__)


class TradeDecisionLogger:
    """
    Accumulates TradeDecision objects and resolves counterfactual exits.

    Usage:
        log = TradeDecisionLogger()
        log.log_decision(decision)
        ...
        log.resolve_counterfactual(ts, symbol, exit_price)
        df = log.to_dataframe()
    """

    def __init__(self) -> None:
        self._decisions: List[TradeDecision] = []
        # Index for fast lookup: (symbol, timestamp) -> list index
        self._pending_exits: Dict[Tuple[str, date], int] = {}

    def log_decision(self, decision: TradeDecision) -> None:
        """Append a trade decision to the log."""
        idx = len(self._decisions)
        self._decisions.append(decision)

        # Track trades that need counterfactual exit resolution:
        # - Blocked trades (allowed=False): full counterfactual PnL
        # - SIZE_DOWN trades: PnL scaled by size_factor
        # - BYPASS trades: full PnL (gate has no information advantage)
        needs_resolution = (not decision.allowed) or decision.action in (
            "SIZE_DOWN",
            "BYPASS",
        )
        if needs_resolution and decision.virtual_exit_price is None:
            key = (decision.symbol, decision.timestamp)
            self._pending_exits[key] = idx

    def resolve_counterfactual(
        self,
        timestamp: date,
        symbol: str,
        exit_price: float,
    ) -> None:
        """
        Resolve counterfactual exit for a blocked trade.

        Uses the BASE strategy exit signal (G1: counterfactual exit isolation).
        The exit bar must match what the baseline strategy would have used.

        Args:
            timestamp: Original entry timestamp of the blocked trade
            symbol: Symbol
            exit_price: Price at the base strategy's exit bar
        """
        key = (symbol, timestamp)
        idx = self._pending_exits.pop(key, None)
        if idx is None:
            return

        old = self._decisions[idx]
        entry_price = old.virtual_entry_price
        if entry_price <= 0:
            return

        if old.strategy_direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        # SIZE_DOWN: scale PnL by size_factor to reflect reduced position
        if old.action == "SIZE_DOWN":
            pnl_pct = pnl_pct * old.size_factor

        # Replace with resolved version (frozen dataclass → reconstruct)
        self._decisions[idx] = TradeDecision(
            timestamp=old.timestamp,
            symbol=old.symbol,
            regime=old.regime,
            strategy_direction=old.strategy_direction,
            intended_size=old.intended_size,
            trend_state=old.trend_state,
            tactical_signal=old.tactical_signal,
            momentum_balance=old.momentum_balance,
            confidence=old.confidence,
            gate_strength=old.gate_strength,
            allowed=old.allowed,
            action=old.action,
            block_reason=old.block_reason,
            virtual_entry_price=old.virtual_entry_price,
            virtual_exit_price=exit_price,
            virtual_pnl_pct=pnl_pct,
        )

    def resolve_all_counterfactuals(
        self,
        base_exits: pd.Series,
        close_prices: pd.Series,
    ) -> None:
        """
        Batch-resolve all pending counterfactual exits using base exit signals.

        Args:
            base_exits: Boolean Series (True at exit bars), indexed by date
            close_prices: Close price Series, same index as base_exits
        """
        exit_dates = base_exits.index[base_exits].tolist()

        for key, idx in list(self._pending_exits.items()):
            symbol, entry_date = key

            # Find next base exit after entry
            exit_price = None
            for exit_date in exit_dates:
                d = exit_date.date() if hasattr(exit_date, "date") else exit_date
                if d > entry_date:
                    exit_price = float(close_prices.loc[exit_date])
                    break

            if exit_price is not None:
                self.resolve_counterfactual(entry_date, symbol, exit_price)

    @property
    def decisions(self) -> List[TradeDecision]:
        return list(self._decisions)

    def get_post_warmup(self, warmup_end: date) -> List[TradeDecision]:
        """Return only decisions after warmup period (G4)."""
        return [d for d in self._decisions if d.timestamp > warmup_end]

    def get_blocked(self, warmup_end: Optional[date] = None) -> List[TradeDecision]:
        """Return blocked decisions, optionally filtered by warmup."""
        decisions = self.get_post_warmup(warmup_end) if warmup_end else self._decisions
        return [d for d in decisions if not d.allowed]

    def get_allowed(self, warmup_end: Optional[date] = None) -> List[TradeDecision]:
        """Return allowed decisions, optionally filtered by warmup."""
        decisions = self.get_post_warmup(warmup_end) if warmup_end else self._decisions
        return [d for d in decisions if d.allowed]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all decisions to a DataFrame."""
        if not self._decisions:
            return pd.DataFrame()

        records = []
        for d in self._decisions:
            records.append(
                {
                    "timestamp": d.timestamp,
                    "symbol": d.symbol,
                    "regime": d.regime,
                    "strategy_direction": d.strategy_direction,
                    "intended_size": d.intended_size,
                    "trend_state": d.trend_state,
                    "tactical_signal": d.tactical_signal,
                    "momentum_balance": d.momentum_balance,
                    "confidence": d.confidence,
                    "gate_strength": d.gate_strength,
                    "allowed": d.allowed,
                    "action": d.action,
                    "block_reason": d.block_reason,
                    "virtual_entry_price": d.virtual_entry_price,
                    "virtual_exit_price": d.virtual_exit_price,
                    "virtual_pnl_pct": d.virtual_pnl_pct,
                    "size_factor": d.size_factor,
                }
            )
        return pd.DataFrame(records)

    def to_jsonl(self, path: Path) -> None:
        """Export decisions to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for d in self._decisions:
                record = {
                    "timestamp": str(d.timestamp),
                    "symbol": d.symbol,
                    "regime": d.regime,
                    "strategy_direction": d.strategy_direction,
                    "intended_size": d.intended_size,
                    "trend_state": d.trend_state,
                    "tactical_signal": d.tactical_signal,
                    "momentum_balance": d.momentum_balance,
                    "confidence": d.confidence,
                    "gate_strength": d.gate_strength,
                    "allowed": d.allowed,
                    "action": d.action,
                    "block_reason": d.block_reason,
                    "virtual_entry_price": d.virtual_entry_price,
                    "virtual_exit_price": d.virtual_exit_price,
                    "virtual_pnl_pct": d.virtual_pnl_pct,
                    "size_factor": d.size_factor,
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Exported {len(self._decisions)} decisions to {path}")

    @classmethod
    def from_jsonl(cls, path: Path) -> "TradeDecisionLogger":
        """Load decisions from JSONL file."""
        log = cls()
        with open(path) as f:
            for line in f:
                record = json.loads(line.strip())
                record["timestamp"] = date.fromisoformat(record["timestamp"])
                decision = TradeDecision(**record)
                log._decisions.append(decision)
        return log
