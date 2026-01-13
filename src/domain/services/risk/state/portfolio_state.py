"""
Portfolio State - Thread-safe aggregate state management.

Uses immutable PositionState objects with replace-not-mutate pattern
for thread safety between the hot path (tick processing) and cold path
(snapshot building).

Thread Safety:
- PositionState values are immutable (frozen dataclasses)
- Dict mutations (self._positions[symbol] = new_state) are atomic in CPython
- Lock only protects dict structure, not values
- Shallow dict copy for snapshots is safe because values are immutable

Usage:
    state = PortfolioState()

    # Hot path: apply delta (updates single position)
    state.apply_delta(delta)

    # Cold path: get snapshot (thread-safe copy)
    snapshot = state.to_snapshot()
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from .position_state import PositionState, PositionDelta
from src.models.risk_snapshot import RiskSnapshot
from src.utils.timezone import now_utc

if TYPE_CHECKING:
    from src.models.position import Position
    from src.models.position_risk import PositionRisk


@dataclass
class PortfolioAggregates:
    """Cached portfolio-level aggregates for O(1) delta updates."""

    total_unrealized_pnl: float = 0.0
    total_daily_pnl: float = 0.0
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0
    total_gross_notional: float = 0.0
    total_net_notional: float = 0.0
    gamma_notional_near_term: float = 0.0
    vega_notional_near_term: float = 0.0
    positions_with_missing_md: int = 0
    missing_greeks_count: int = 0


class PortfolioState:
    """
    Thread-safe portfolio state using immutable position states.

    Design:
    - Dict values are immutable PositionState objects
    - Mutations replace dict values atomically (CPython GIL)
    - Lock protects dict structure (add/remove keys)
    - Aggregates updated incrementally via deltas
    """

    def __init__(self) -> None:
        # Dict values are immutable PositionState objects
        self._positions: Dict[str, PositionState] = {}
        self._lock = threading.Lock()

        # Cached aggregates for O(1) delta updates
        self._aggregates = PortfolioAggregates()

        # By-underlying tracking
        self._delta_by_underlying: Dict[str, float] = defaultdict(float)
        self._notional_by_underlying: Dict[str, float] = defaultdict(float)

    def apply_delta(self, delta: PositionDelta) -> None:
        """
        O(1) update: replace position state with new immutable object.

        Thread-safe: dict value replacement is atomic in CPython.

        Args:
            delta: The incremental update to apply.
        """
        with self._lock:
            old = self._positions.get(delta.symbol)
            if old is None:
                return  # Unknown position, ignore

            # Create new immutable state (thread-safe replace)
            new_state = old.with_update(delta)
            self._positions[delta.symbol] = new_state

            # Update aggregates by delta (O(1))
            self._aggregates.total_unrealized_pnl += delta.pnl_change
            self._aggregates.total_daily_pnl += delta.daily_pnl_change
            self._aggregates.portfolio_delta += delta.delta_change
            self._aggregates.portfolio_gamma += delta.gamma_change
            self._aggregates.portfolio_vega += delta.vega_change
            self._aggregates.portfolio_theta += delta.theta_change

            # Update notional (need to track both gross and net)
            old_notional = old.notional
            new_notional = new_state.notional
            self._aggregates.total_net_notional += (new_notional - old_notional)
            self._aggregates.total_gross_notional += (abs(new_notional) - abs(old_notional))

            # Update by-underlying tracking
            underlying = delta.underlying
            self._delta_by_underlying[underlying] += delta.delta_change
            self._notional_by_underlying[underlying] += delta.notional_change

            # Update reliability/greeks counts when state changes
            if old.is_reliable != new_state.is_reliable:
                if new_state.is_reliable:
                    self._aggregates.positions_with_missing_md -= 1
                else:
                    self._aggregates.positions_with_missing_md += 1

            if old.has_greeks != new_state.has_greeks:
                if new_state.has_greeks:
                    self._aggregates.missing_greeks_count -= 1
                else:
                    self._aggregates.missing_greeks_count += 1

    def get_position(self, symbol: str) -> Optional[PositionState]:
        """Get position state by symbol (thread-safe read)."""
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if position exists."""
        return symbol in self._positions

    def to_snapshot(self) -> RiskSnapshot:
        """
        Thread-safe snapshot creation via shallow dict copy.

        Safe because:
        - Values are immutable PositionState objects
        - Shallow copy creates new dict with same immutable refs
        - Aggregate VALUES (not reference) copied atomically under lock
        """
        with self._lock:
            # Shallow copy is safe because values are immutable
            positions_copy = self._positions.copy()
            delta_by_underlying = dict(self._delta_by_underlying)
            notional_by_underlying = dict(self._notional_by_underlying)

            # CRITICAL: Copy aggregate VALUES under lock, not reference
            # (fixes race condition where aggregates change mid-snapshot)
            total_unrealized_pnl = self._aggregates.total_unrealized_pnl
            total_daily_pnl = self._aggregates.total_daily_pnl
            portfolio_delta = self._aggregates.portfolio_delta
            portfolio_gamma = self._aggregates.portfolio_gamma
            portfolio_vega = self._aggregates.portfolio_vega
            portfolio_theta = self._aggregates.portfolio_theta
            total_gross_notional = self._aggregates.total_gross_notional
            total_net_notional = self._aggregates.total_net_notional
            gamma_notional_near_term = self._aggregates.gamma_notional_near_term
            vega_notional_near_term = self._aggregates.vega_notional_near_term
            positions_with_missing_md = self._aggregates.positions_with_missing_md
            missing_greeks_count = self._aggregates.missing_greeks_count
            position_count = len(positions_copy)

            # Calculate concentration
            max_symbol = ""
            max_notional = 0.0
            concentration_pct = 0.0
            if notional_by_underlying and total_gross_notional > 0:
                max_item = max(
                    notional_by_underlying.items(),
                    key=lambda x: abs(x[1]),
                )
                max_symbol = max_item[0]
                max_notional = abs(max_item[1])
                concentration_pct = max_notional / total_gross_notional

        # Build snapshot outside lock (all values are now local copies)
        # Note: position_risks and expiry_buckets populated by cold path
        return RiskSnapshot(
            timestamp=now_utc(),
            total_unrealized_pnl=total_unrealized_pnl,
            total_daily_pnl=total_daily_pnl,
            portfolio_delta=portfolio_delta,
            portfolio_gamma=portfolio_gamma,
            portfolio_vega=portfolio_vega,
            portfolio_theta=portfolio_theta,
            total_gross_notional=total_gross_notional,
            total_net_notional=total_net_notional,
            max_underlying_notional=max_notional,
            max_underlying_symbol=max_symbol,
            concentration_pct=concentration_pct,
            delta_by_underlying=delta_by_underlying,
            notional_by_underlying=notional_by_underlying,
            gamma_notional_near_term=gamma_notional_near_term,
            vega_notional_near_term=vega_notional_near_term,
            positions_with_missing_md=positions_with_missing_md,
            missing_greeks_count=missing_greeks_count,
            total_positions=position_count,
        )

    def add_position(self, state: PositionState) -> None:
        """
        Add a new position to the portfolio.

        Used during initial load or position changes.
        """
        with self._lock:
            self._positions[state.symbol] = state

            # Update aggregates
            self._aggregates.total_unrealized_pnl += state.unrealized_pnl
            self._aggregates.total_daily_pnl += state.daily_pnl
            self._aggregates.portfolio_delta += state.delta
            self._aggregates.portfolio_gamma += state.gamma
            self._aggregates.portfolio_vega += state.vega
            self._aggregates.portfolio_theta += state.theta
            self._aggregates.total_net_notional += state.notional
            self._aggregates.total_gross_notional += abs(state.notional)

            if not state.is_reliable:
                self._aggregates.positions_with_missing_md += 1
            if not state.has_greeks:
                self._aggregates.missing_greeks_count += 1

            # Update by-underlying tracking
            self._delta_by_underlying[state.underlying] += state.delta
            self._notional_by_underlying[state.underlying] += state.notional

    def remove_position(self, symbol: str) -> Optional[PositionState]:
        """
        Remove a position from the portfolio.

        Returns the removed state, or None if not found.
        """
        with self._lock:
            state = self._positions.pop(symbol, None)
            if state is None:
                return None

            # Update aggregates (subtract)
            self._aggregates.total_unrealized_pnl -= state.unrealized_pnl
            self._aggregates.total_daily_pnl -= state.daily_pnl
            self._aggregates.portfolio_delta -= state.delta
            self._aggregates.portfolio_gamma -= state.gamma
            self._aggregates.portfolio_vega -= state.vega
            self._aggregates.portfolio_theta -= state.theta
            self._aggregates.total_net_notional -= state.notional
            self._aggregates.total_gross_notional -= abs(state.notional)

            if not state.is_reliable:
                self._aggregates.positions_with_missing_md -= 1
            if not state.has_greeks:
                self._aggregates.missing_greeks_count -= 1

            # Update by-underlying tracking
            self._delta_by_underlying[state.underlying] -= state.delta
            self._notional_by_underlying[state.underlying] -= state.notional

            return state

    def clear(self) -> None:
        """
        Clear all state (for reconnect re-sync).

        Published with FULL_RESYNC event to trigger complete rebuild.
        """
        with self._lock:
            self._positions.clear()
            self._aggregates = PortfolioAggregates()
            self._delta_by_underlying.clear()
            self._notional_by_underlying.clear()

    @property
    def position_count(self) -> int:
        """Number of positions in portfolio."""
        return len(self._positions)

    @property
    def symbols(self) -> List[str]:
        """List of position symbols."""
        return list(self._positions.keys())

    def get_position_states(self) -> Dict[str, PositionState]:
        """
        Get thread-safe copy of all position states.

        Used by RiskFacade to build position_risks list.
        """
        with self._lock:
            return self._positions.copy()
