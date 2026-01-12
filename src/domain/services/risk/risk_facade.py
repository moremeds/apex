"""
Risk Facade - External interface to the streaming risk system.

Provides a simple API for:
- Processing market data ticks (hot path)
- Getting portfolio snapshots (cold path)
- Handling position changes (initialization/sync)

Thread Safety:
    - Delegates to PortfolioState which handles thread safety internally
    - Tick processing is stateless (uses immutable inputs)
    - Position mapping updated atomically via dict replacement

Usage:
    facade = RiskFacade(portfolio_state)
    facade.load_positions(positions, initial_ticks)

    # Hot path (called per tick)
    delta = facade.on_tick(tick)
    if delta:
        event_bus.publish(POSITION_DELTA, delta.to_event())

    # Cold path (periodic)
    snapshot = facade.get_snapshot()
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, TYPE_CHECKING

from .streaming.tick_processor import TickProcessor, create_initial_state
from .state.portfolio_state import PortfolioState
from .state.position_state import PositionState, PositionDelta

if TYPE_CHECKING:
    from src.domain.events.domain_events import MarketDataTickEvent
    from src.models.position import Position
    from src.models.risk_snapshot import RiskSnapshot


class RiskFacade:
    """
    Simple external interface to the streaming risk system.

    Coordinates between:
    - TickProcessor: Converts ticks to deltas (stateless)
    - PortfolioState: Maintains portfolio state (thread-safe)
    - Position lookup: Maps symbols to Position objects

    Design:
    - Single entry point for tick processing
    - Maintains symbol->Position mapping for O(1) lookup
    - Delegates all state management to PortfolioState
    """

    def __init__(self, portfolio_state: Optional[PortfolioState] = None) -> None:
        """
        Initialize RiskFacade.

        Args:
            portfolio_state: Portfolio state manager. Created if not provided.
        """
        self._state = portfolio_state or PortfolioState()
        self._processor = TickProcessor()

        # Symbol -> Position mapping for tick processing
        # Uses replace-not-mutate pattern (atomic dict replacement)
        self._positions: Dict[str, Position] = {}
        self._positions_lock = threading.Lock()

    def on_tick(self, tick: MarketDataTickEvent) -> Optional[PositionDelta]:
        """
        Process a market data tick and return delta if valid.

        This is the HOT PATH - called for every tick.

        If position has no state yet, initializes from this tick (first tick seeding).

        Args:
            tick: Market data tick event.

        Returns:
            PositionDelta if tick produces a valid update, None otherwise.
        """
        # Look up position (O(1))
        position = self._positions.get(tick.symbol)
        if position is None:
            return None  # No position for this symbol

        # Get current calculated state
        current_state = self._state.get_position(tick.symbol)
        if current_state is None:
            # First tick for this position - seed initial state
            initial_state = create_initial_state(position, tick, strict_quality=False)
            if initial_state is None:
                return None  # Tick not valid for initialization
            self._state.add_position(initial_state)
            # Return None for first tick (no delta, just initialization)
            return None

        # Process tick (stateless calculation)
        delta = self._processor.process_tick(tick, position, current_state)
        if delta is None:
            return None  # Tick filtered (bad quality, wide spread, etc.)

        # Apply delta to portfolio state
        self._state.apply_delta(delta)

        return delta

    def get_snapshot(self) -> RiskSnapshot:
        """
        Get current portfolio snapshot.

        This is the COLD PATH - called periodically for risk signals.

        Returns:
            RiskSnapshot with current portfolio metrics.
        """
        return self._state.to_snapshot()

    def load_positions(
        self,
        positions: List[Position],
        initial_ticks: Optional[Dict[str, MarketDataTickEvent]] = None,
    ) -> int:
        """
        Load positions into the facade.

        Called on startup and after position changes. Creates initial
        PositionState for each position using provided ticks.

        Args:
            positions: List of Position objects.
            initial_ticks: Optional dict of symbol -> tick for initialization.
                If not provided, positions will be added but won't have
                calculated state until first tick arrives.

        Returns:
            Number of positions successfully initialized with state.
        """
        initial_ticks = initial_ticks or {}
        initialized = 0

        # Build new position mapping
        new_positions = {p.symbol: p for p in positions}

        # Update position mapping atomically
        with self._positions_lock:
            self._positions = new_positions

        # Clear existing state and rebuild
        self._state.clear()

        # Initialize state for each position with available ticks
        for position in positions:
            tick = initial_ticks.get(position.symbol)
            if tick is not None:
                # Try to create initial state from tick
                state = create_initial_state(position, tick, strict_quality=False)
                if state is not None:
                    self._state.add_position(state)
                    initialized += 1

        return initialized

    def add_position(
        self,
        position: Position,
        tick: Optional[MarketDataTickEvent] = None,
    ) -> bool:
        """
        Add a single position.

        Used for incremental position updates.

        Args:
            position: Position to add.
            tick: Optional initial tick for state creation.

        Returns:
            True if position was added with calculated state.
        """
        # Update position mapping
        with self._positions_lock:
            new_positions = self._positions.copy()
            new_positions[position.symbol] = position
            self._positions = new_positions

        if tick is not None:
            state = create_initial_state(position, tick, strict_quality=False)
            if state is not None:
                self._state.add_position(state)
                return True

        return False

    def remove_position(self, symbol: str) -> Optional[PositionState]:
        """
        Remove a position.

        Args:
            symbol: Symbol of position to remove.

        Returns:
            Removed PositionState if it existed.
        """
        # Update position mapping
        with self._positions_lock:
            new_positions = self._positions.copy()
            new_positions.pop(symbol, None)
            self._positions = new_positions

        return self._state.remove_position(symbol)

    def clear(self) -> None:
        """
        Clear all positions and state.

        Called on reconnect/resync.
        """
        with self._positions_lock:
            self._positions = {}
        self._state.clear()

    def has_position(self, symbol: str) -> bool:
        """Check if a position exists."""
        return symbol in self._positions

    @property
    def position_count(self) -> int:
        """Number of positions with calculated state."""
        return self._state.position_count

    @property
    def symbols(self) -> List[str]:
        """List of position symbols."""
        return list(self._positions.keys())

    def get_position_state(self, symbol: str) -> Optional[PositionState]:
        """Get calculated state for a position."""
        return self._state.get_position(symbol)
