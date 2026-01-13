"""
Orchestrator module - Main application workflow control.

Split into focused components:
- Orchestrator: Main coordinator (lifecycle, event wiring)
- DataCoordinator: Position/market data fetching and reconciliation
- SnapshotCoordinator: Risk snapshot building and dispatching
- ShutdownCoordinator: Graceful shutdown with timeouts
"""

from .orchestrator import Orchestrator
from .data_coordinator import DataCoordinator
from .snapshot_coordinator import SnapshotCoordinator
from .shutdown import ShutdownCoordinator, create_shutdown_handler

__all__ = [
    "Orchestrator",
    "DataCoordinator",
    "SnapshotCoordinator",
    "ShutdownCoordinator",
    "create_shutdown_handler",
]
