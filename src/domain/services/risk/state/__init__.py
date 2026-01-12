"""
State management for streaming risk calculations.

Provides thread-safe, immutable state management for the hot path:
- PositionState: Immutable per-position state (frozen dataclass)
- PositionDelta: Incremental update message
- PortfolioState: Thread-safe portfolio aggregate state

Design principles:
- All PositionState objects are immutable (frozen=True)
- Updates create new PositionState, never mutate existing
- PortfolioState uses replace-not-mutate for thread safety
- Shallow dict copy for snapshots is safe because values are immutable
"""

from .position_state import PositionState, PositionDelta
from .portfolio_state import PortfolioState

__all__ = [
    "PositionState",
    "PositionDelta",
    "PortfolioState",
]
