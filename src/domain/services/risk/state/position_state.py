"""
Position State - Immutable per-position calculated state.

Uses frozen dataclasses for thread safety in the streaming architecture.
Updates are performed by creating new instances via with_update().

Thread Safety:
- PositionState is frozen (immutable)
- PositionDelta is frozen (immutable)
- Updates create new objects, never mutate existing
- Dict operations (replace value) are atomic in CPython

Usage:
    # Create initial state
    state = PositionState.from_position(position, mark_price=155.0, ...)

    # Update with delta (returns NEW state)
    new_state = state.with_update(delta)

    # Old state is unchanged (immutable)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.models.position import Position
from src.utils.timezone import now_utc


@dataclass(frozen=True, slots=True)
class PositionDelta:
    """
    Incremental position update message.

    Published on POSITION_DELTA events for streaming TUI updates.
    Immutable for thread safety.

    Attributes:
        symbol: Position symbol.
        underlying: Underlying asset symbol.
        timestamp: When this delta was calculated.
        new_mark_price: Updated mark price.
        pnl_change: Change in unrealized P&L.
        daily_pnl_change: Change in daily P&L.
        delta_change: Change in position delta.
        gamma_change: Change in position gamma.
        vega_change: Change in position vega.
        theta_change: Change in position theta.
        notional_change: Change in notional exposure.
        is_reliable: True if based on good quality data.
    """

    symbol: str
    underlying: str
    timestamp: datetime
    new_mark_price: float
    pnl_change: float
    daily_pnl_change: float
    delta_change: float
    gamma_change: float
    vega_change: float
    theta_change: float
    notional_change: float
    delta_dollars_change: float
    underlying_price: float
    is_reliable: bool
    has_greeks: bool

    def to_event(self) -> "PositionDeltaEvent":
        """
        Convert to PositionDeltaEvent for event bus publishing.

        Returns:
            PositionDeltaEvent with all delta fields.
        """
        from src.domain.events.domain_events import PositionDeltaEvent

        return PositionDeltaEvent(
            timestamp=self.timestamp,
            symbol=self.symbol,
            underlying=self.underlying,
            new_mark_price=self.new_mark_price,
            pnl_change=self.pnl_change,
            daily_pnl_change=self.daily_pnl_change,
            delta_change=self.delta_change,
            gamma_change=self.gamma_change,
            vega_change=self.vega_change,
            theta_change=self.theta_change,
            notional_change=self.notional_change,
            delta_dollars_change=self.delta_dollars_change,
            underlying_price=self.underlying_price,
            is_reliable=self.is_reliable,
            has_greeks=self.has_greeks,
        )


@dataclass(frozen=True, slots=True)
class PositionState:
    """
    Immutable per-position calculated state.

    Thread-safe due to frozen=True. Updates are performed by creating
    new instances via with_update().

    Attributes:
        symbol: Position symbol.
        underlying: Underlying asset symbol.
        quantity: Position size.
        multiplier: Contract multiplier.
        avg_cost: Average cost basis.
        mark_price: Current mark price.
        yesterday_close: Previous day's close.
        session_open: Today's session open.
        unrealized_pnl: Total unrealized P&L.
        daily_pnl: P&L since yesterday's close.
        delta: Position delta exposure.
        gamma: Position gamma.
        vega: Position vega.
        theta: Position theta.
        notional: Position notional value.
        is_reliable: True if based on good quality data.
        has_greeks: True if real (not synthetic) Greeks available.
        last_update: Timestamp of last update.
    """

    # Identity
    symbol: str
    underlying: str

    # Position attributes (rarely change)
    quantity: float
    multiplier: int
    avg_cost: float

    # Market data (updates frequently)
    mark_price: float
    yesterday_close: float
    session_open: float

    # Calculated metrics
    unrealized_pnl: float
    daily_pnl: float
    delta: float
    gamma: float
    vega: float
    theta: float
    notional: float
    delta_dollars: float
    underlying_price: float

    # Data quality
    is_reliable: bool
    has_greeks: bool

    # Timestamp
    last_update: datetime

    @classmethod
    def from_position(
        cls,
        position: Position,
        mark_price: float,
        yesterday_close: float,
        session_open: float,
        unrealized_pnl: float,
        daily_pnl: float,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        notional: float,
        delta_dollars: float,
        underlying_price: float,
        is_reliable: bool,
        has_greeks: bool,
        timestamp: Optional[datetime] = None,
    ) -> PositionState:
        """Create PositionState from Position and calculated values."""
        return cls(
            symbol=position.symbol,
            underlying=position.underlying,
            quantity=position.quantity,
            multiplier=position.multiplier,
            avg_cost=position.avg_price,
            mark_price=mark_price,
            yesterday_close=yesterday_close,
            session_open=session_open,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=daily_pnl,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            notional=notional,
            delta_dollars=delta_dollars,
            underlying_price=underlying_price,
            is_reliable=is_reliable,
            has_greeks=has_greeks,
            last_update=timestamp or now_utc(),
        )

    def with_update(self, delta: PositionDelta) -> PositionState:
        """
        Return new PositionState with delta applied.

        Immutable update pattern: creates new object, original unchanged.

        Args:
            delta: The incremental update to apply.

        Returns:
            New PositionState with updated values.
        """
        return PositionState(
            symbol=self.symbol,
            underlying=self.underlying,
            quantity=self.quantity,
            multiplier=self.multiplier,
            avg_cost=self.avg_cost,
            mark_price=delta.new_mark_price,
            yesterday_close=self.yesterday_close,
            session_open=self.session_open,
            unrealized_pnl=self.unrealized_pnl + delta.pnl_change,
            daily_pnl=self.daily_pnl + delta.daily_pnl_change,
            delta=self.delta + delta.delta_change,
            gamma=self.gamma + delta.gamma_change,
            vega=self.vega + delta.vega_change,
            theta=self.theta + delta.theta_change,
            notional=self.notional + delta.notional_change,
            delta_dollars=self.delta_dollars + delta.delta_dollars_change,
            underlying_price=delta.underlying_price,
            is_reliable=delta.is_reliable,
            has_greeks=delta.has_greeks,
            last_update=delta.timestamp,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "quantity": self.quantity,
            "multiplier": self.multiplier,
            "avg_cost": self.avg_cost,
            "mark_price": self.mark_price,
            "yesterday_close": self.yesterday_close,
            "session_open": self.session_open,
            "unrealized_pnl": self.unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "notional": self.notional,
            "delta_dollars": self.delta_dollars,
            "underlying_price": self.underlying_price,
            "is_reliable": self.is_reliable,
            "has_greeks": self.has_greeks,
            "last_update": self.last_update.isoformat(),
        }
