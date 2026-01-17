"""
Streaming risk calculations for real-time TUI updates.

This module provides the hot path for tick-to-display latency optimization.
Instead of waiting for batch snapshots, ticks are processed immediately
and deltas are published for direct TUI consumption.

Components:
- TickProcessor: Processes market data ticks into position deltas
- DeltaPublisher: Bridges RiskFacade to event bus, publishes deltas

Design Principles:
- TUI never waits for snapshot building
- Deltas are incremental (O(1) to apply)
- Bad ticks are filtered before delta emission
"""

from .delta_publisher import DeltaPublisher
from .tick_processor import TickProcessor, create_initial_state

__all__ = [
    "TickProcessor",
    "DeltaPublisher",
    "create_initial_state",
]
