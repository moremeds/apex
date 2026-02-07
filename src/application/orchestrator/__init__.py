"""
Orchestrator module - Main application workflow control.

Split into focused components:
- Orchestrator: Main coordinator (lifecycle, event wiring)
- DataCoordinator: Position/market data fetching and reconciliation
- SnapshotCoordinator: Risk snapshot building and dispatching
- SignalCoordinator: Signal pipeline wiring (ticks -> bars -> indicators -> signals)
- ShutdownCoordinator: Graceful shutdown with timeouts
"""

from .data_coordinator import DataCoordinator
from .orchestrator import Orchestrator
from .shutdown import ShutdownCoordinator, create_shutdown_handler
from .signal_coordinator import SignalCoordinator
from .snapshot_coordinator import SnapshotCoordinator

__all__ = [
    "Orchestrator",
    "DataCoordinator",
    "SnapshotCoordinator",
    "SignalCoordinator",
    "ShutdownCoordinator",
    "create_shutdown_handler",
]
