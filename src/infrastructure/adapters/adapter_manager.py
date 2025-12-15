"""
Adapter Manager for lifecycle management of all adapters.

Manages:
- Live adapters (QuoteProvider, PositionProvider, AccountProvider)
- Historical adapters (BarProvider)
- Execution adapters (ExecutionProvider)

Provides:
- Unified startup/shutdown
- Health monitoring
- Per-adapter Prometheus metrics
- Hot restart capability
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
import asyncio

from ...utils.logging_setup import get_logger
from ...domain.interfaces.quote_provider import QuoteProvider
from ...domain.interfaces.bar_provider import BarProvider
from ...domain.interfaces.execution_provider import ExecutionProvider
from ...domain.interfaces.position_provider import PositionProvider
from ...domain.interfaces.account_provider import AccountProvider

if TYPE_CHECKING:
    from ...infrastructure.monitoring import HealthMonitor, HealthStatus
    from ..observability import AdapterMetrics


logger = get_logger(__name__)


@dataclass
class AdapterStatus:
    """Status of a single adapter."""
    name: str
    adapter_type: str  # "live", "historical", "execution"
    broker: str  # "ib", "futu"
    connected: bool = False
    last_error: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    connect_time: Optional[datetime] = None
    reconnect_count: int = 0


class AdapterManager:
    """
    Manages lifecycle of all broker adapters.

    Organizes adapters by type:
    - live_adapters: Real-time data (quotes, positions, accounts)
    - historical_adapters: Historical bar data
    - execution_adapters: Order submission

    Each adapter can be started, stopped, and restarted independently.
    """

    def __init__(
        self,
        health_monitor: Optional["HealthMonitor"] = None,
        adapter_metrics: Optional["AdapterMetrics"] = None,
    ):
        """
        Initialize adapter manager.

        Args:
            health_monitor: Optional health monitor for status reporting.
            adapter_metrics: Optional AdapterMetrics for Prometheus metrics.
        """
        self._live_adapters: Dict[str, Any] = {}  # QuoteProvider/PositionProvider
        self._historical_adapters: Dict[str, BarProvider] = {}
        self._execution_adapters: Dict[str, ExecutionProvider] = {}

        self._status: Dict[str, AdapterStatus] = {}
        self._health_monitor = health_monitor
        self._adapter_metrics = adapter_metrics

        # Connection state
        self._started = False

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register_live_adapter(self, name: str, adapter: Any, broker: str = "unknown") -> None:
        """
        Register a live adapter (quotes, positions, account).

        Args:
            name: Unique name (e.g., "ib_live", "futu_live").
            adapter: Adapter implementing QuoteProvider/PositionProvider.
            broker: Broker identifier (e.g., "ib", "futu").
        """
        self._live_adapters[name] = adapter
        self._status[name] = AdapterStatus(
            name=name,
            adapter_type="live",
            broker=broker,
        )
        logger.info(f"Registered live adapter: {name} ({broker})")

    def register_historical_adapter(
        self,
        name: str,
        adapter: BarProvider,
        broker: str = "unknown"
    ) -> None:
        """
        Register a historical adapter (bar data).

        Args:
            name: Unique name (e.g., "ib_historical").
            adapter: Adapter implementing BarProvider.
            broker: Broker identifier.
        """
        self._historical_adapters[name] = adapter
        self._status[name] = AdapterStatus(
            name=name,
            adapter_type="historical",
            broker=broker,
        )
        logger.info(f"Registered historical adapter: {name} ({broker})")

    def register_execution_adapter(
        self,
        name: str,
        adapter: ExecutionProvider,
        broker: str = "unknown"
    ) -> None:
        """
        Register an execution adapter (order submission).

        Args:
            name: Unique name (e.g., "ib_execution").
            adapter: Adapter implementing ExecutionProvider.
            broker: Broker identifier.
        """
        self._execution_adapters[name] = adapter
        self._status[name] = AdapterStatus(
            name=name,
            adapter_type="execution",
            broker=broker,
        )
        logger.info(f"Registered execution adapter: {name} ({broker})")

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    async def start_all(self) -> None:
        """Start all registered adapters."""
        if self._started:
            logger.warning("AdapterManager already started")
            return

        logger.info("Starting all adapters...")

        # Start all adapter types in parallel
        all_adapters = [
            *self._live_adapters.items(),
            *self._historical_adapters.items(),
            *self._execution_adapters.items(),
        ]

        tasks = [self._start_adapter(name, adapter) for name, adapter in all_adapters]
        await asyncio.gather(*tasks, return_exceptions=True)

        self._started = True

        connected = sum(1 for s in self._status.values() if s.connected)
        logger.info(f"AdapterManager started: {connected}/{len(all_adapters)} adapters connected")

    async def stop_all(self) -> None:
        """Stop all registered adapters."""
        if not self._started:
            return

        logger.info("Stopping all adapters...")

        all_adapters = [
            *self._live_adapters.items(),
            *self._historical_adapters.items(),
            *self._execution_adapters.items(),
        ]

        tasks = [self._stop_adapter(name, adapter) for name, adapter in all_adapters]
        await asyncio.gather(*tasks, return_exceptions=True)

        self._started = False
        logger.info("AdapterManager stopped")

    async def restart_adapter(self, name: str) -> bool:
        """
        Restart a specific adapter.

        Args:
            name: Adapter name to restart.

        Returns:
            True if restart successful, False otherwise.
        """
        adapter = self._get_adapter_by_name(name)
        if not adapter:
            logger.error(f"Adapter not found: {name}")
            return False

        logger.info(f"Restarting adapter: {name}")

        # Stop
        await self._stop_adapter(name, adapter)

        # Wait briefly
        await asyncio.sleep(1.0)

        # Start
        success = await self._start_adapter(name, adapter)

        if success:
            status = self._status.get(name)
            if status:
                status.reconnect_count += 1
                # Record reconnect metric
                self._record_reconnect_metric(status)
            logger.info(f"Adapter {name} restarted successfully")
        else:
            logger.error(f"Failed to restart adapter {name}")

        return success

    async def _start_adapter(self, name: str, adapter: Any) -> bool:
        """Start a single adapter."""
        try:
            if hasattr(adapter, 'connect'):
                await adapter.connect()

            status = self._status.get(name)
            if status:
                status.connected = True
                status.connect_time = datetime.now()
                status.last_error = None
                status.last_updated = datetime.now()

                # Record metrics
                self._record_connection_metric(status, connected=True)

            self._update_health(name, "HEALTHY", "Connected")
            logger.info(f"✓ Started {name}")
            return True

        except Exception as e:
            status = self._status.get(name)
            if status:
                status.connected = False
                status.last_error = str(e)
                status.last_updated = datetime.now()

                # Record metrics
                self._record_connection_metric(status, connected=False)
                self._record_error_metric(status, "connection")

            self._update_health(name, "UNHEALTHY", f"Connection failed: {str(e)[:50]}")
            logger.error(f"✗ Failed to start {name}: {e}")
            return False

    async def _stop_adapter(self, name: str, adapter: Any) -> None:
        """Stop a single adapter."""
        try:
            if hasattr(adapter, 'disconnect'):
                await adapter.disconnect()

            status = self._status.get(name)
            if status:
                status.connected = False
                status.last_updated = datetime.now()

                # Record metrics
                self._record_connection_metric(status, connected=False)

            logger.info(f"Stopped {name}")

        except Exception as e:
            logger.error(f"Error stopping {name}: {e}")

    # -------------------------------------------------------------------------
    # Adapter Access
    # -------------------------------------------------------------------------

    def get_live_adapter(self, name: str) -> Optional[Any]:
        """Get a live adapter by name."""
        return self._live_adapters.get(name)

    def get_historical_adapter(self, name: str) -> Optional[BarProvider]:
        """Get a historical adapter by name."""
        return self._historical_adapters.get(name)

    def get_execution_adapter(self, name: str) -> Optional[ExecutionProvider]:
        """Get an execution adapter by name."""
        return self._execution_adapters.get(name)

    def get_all_live_adapters(self) -> Dict[str, Any]:
        """Get all live adapters."""
        return self._live_adapters.copy()

    def get_all_historical_adapters(self) -> Dict[str, BarProvider]:
        """Get all historical adapters."""
        return self._historical_adapters.copy()

    def get_all_execution_adapters(self) -> Dict[str, ExecutionProvider]:
        """Get all execution adapters."""
        return self._execution_adapters.copy()

    def _get_adapter_by_name(self, name: str) -> Optional[Any]:
        """Get any adapter by name."""
        return (
            self._live_adapters.get(name)
            or self._historical_adapters.get(name)
            or self._execution_adapters.get(name)
        )

    # -------------------------------------------------------------------------
    # Status & Health
    # -------------------------------------------------------------------------

    def get_status(self, name: str) -> Optional[AdapterStatus]:
        """Get status for a specific adapter."""
        return self._status.get(name)

    def get_all_status(self) -> Dict[str, AdapterStatus]:
        """Get status for all adapters."""
        return self._status.copy()

    def get_connected_adapters(self) -> List[str]:
        """Get names of connected adapters."""
        return [name for name, status in self._status.items() if status.connected]

    def is_any_connected(self) -> bool:
        """Check if any adapter is connected."""
        return any(s.connected for s in self._status.values())

    async def check_health(self) -> Dict[str, AdapterStatus]:
        """
        Check health of all adapters and update status.

        Returns:
            Dict mapping adapter name to status.
        """
        for name, status in self._status.items():
            adapter = self._get_adapter_by_name(name)
            if adapter and hasattr(adapter, 'is_connected'):
                try:
                    is_connected = adapter.is_connected()
                    status.connected = is_connected
                    status.last_updated = datetime.now()

                    if is_connected:
                        self._update_health(name, "HEALTHY", "Connected")
                    else:
                        self._update_health(name, "UNHEALTHY", "Disconnected")

                except Exception as e:
                    status.connected = False
                    status.last_error = str(e)
                    self._update_health(name, "UNHEALTHY", f"Health check failed: {str(e)[:50]}")

        return self._status.copy()

    def _update_health(self, name: str, status_str: str, message: str) -> None:
        """Update health monitor for an adapter."""
        if self._health_monitor is None:
            return

        try:
            from ..monitoring import HealthStatus

            status_map = {
                "HEALTHY": HealthStatus.HEALTHY,
                "DEGRADED": HealthStatus.DEGRADED,
                "UNHEALTHY": HealthStatus.UNHEALTHY,
            }
            health_status = status_map.get(status_str, HealthStatus.UNKNOWN)

            adapter_status = self._status.get(name)
            metadata = {}
            if adapter_status:
                metadata = {
                    "adapter_type": adapter_status.adapter_type,
                    "broker": adapter_status.broker,
                    "connected": adapter_status.connected,
                    "reconnect_count": adapter_status.reconnect_count,
                }

            self._health_monitor.update_component_health(
                component_name=f"adapter_{name}",
                status=health_status,
                message=message,
                metadata=metadata,
            )

        except Exception as e:
            logger.debug(f"Failed to update health for {name}: {e}")

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """Set health monitor after initialization."""
        self._health_monitor = health_monitor

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for all adapters.

        Returns:
            Dict with metrics suitable for Prometheus export.
        """
        metrics = {
            "total_adapters": len(self._status),
            "connected_adapters": sum(1 for s in self._status.values() if s.connected),
            "adapters_by_type": {
                "live": len(self._live_adapters),
                "historical": len(self._historical_adapters),
                "execution": len(self._execution_adapters),
            },
            "adapters_by_broker": {},
            "adapter_details": {},
        }

        # Count by broker
        broker_counts: Dict[str, int] = {}
        for status in self._status.values():
            broker_counts[status.broker] = broker_counts.get(status.broker, 0) + 1
        metrics["adapters_by_broker"] = broker_counts

        # Detailed status per adapter
        for name, status in self._status.items():
            metrics["adapter_details"][name] = {
                "type": status.adapter_type,
                "broker": status.broker,
                "connected": status.connected,
                "reconnect_count": status.reconnect_count,
                "last_error": status.last_error,
            }

        return metrics

    def set_adapter_metrics(self, adapter_metrics: "AdapterMetrics") -> None:
        """Set adapter metrics after initialization."""
        self._adapter_metrics = adapter_metrics

    def _record_connection_metric(self, status: AdapterStatus, connected: bool) -> None:
        """Record connection status metric for an adapter."""
        if self._adapter_metrics is None:
            return

        try:
            self._adapter_metrics.record_connection_status(
                adapter_name=status.name,
                broker=status.broker,
                connected=connected,
                adapter_type=status.adapter_type,
            )
        except Exception as e:
            logger.debug(f"Failed to record connection metric for {status.name}: {e}")

    def _record_reconnect_metric(self, status: AdapterStatus) -> None:
        """Record reconnect metric for an adapter."""
        if self._adapter_metrics is None:
            return

        try:
            self._adapter_metrics.record_reconnect(
                adapter_name=status.name,
                broker=status.broker,
                adapter_type=status.adapter_type,
            )
        except Exception as e:
            logger.debug(f"Failed to record reconnect metric for {status.name}: {e}")

    def _record_error_metric(self, status: AdapterStatus, error_type: str) -> None:
        """Record error metric for an adapter."""
        if self._adapter_metrics is None:
            return

        try:
            self._adapter_metrics.record_error(
                adapter_name=status.name,
                broker=status.broker,
                error_type=error_type,
            )
        except Exception as e:
            logger.debug(f"Failed to record error metric for {status.name}: {e}")
