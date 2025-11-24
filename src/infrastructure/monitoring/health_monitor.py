"""
Health Monitor - Component health tracking.

Tracks health status of all system components (IB connection, data freshness, etc).
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    component_name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime = None
    metadata: Dict = None

    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class HealthMonitor:
    """
    System health monitoring service.

    Tracks health of:
    - IB connection
    - File loader
    - Market data freshness
    - Position reconciliation status
    - Snapshot staleness
    """

    def __init__(self):
        """Initialize health monitor."""
        self._component_health: Dict[str, ComponentHealth] = {}

    def update_component_health(
        self,
        component_name: str,
        status: HealthStatus,
        message: str = "",
        metadata: Dict = None,
    ) -> None:
        """
        Update health status for a component.

        Args:
            component_name: Name of component (e.g., "ib_adapter", "file_loader").
            status: Health status.
            message: Optional status message.
            metadata: Optional metadata dict.
        """
        self._component_health[component_name] = ComponentHealth(
            component_name=component_name,
            status=status,
            message=message,
            last_check=datetime.now(),
            metadata=metadata or {},
        )
        logger.debug(f"{component_name} health updated: {status.value} - {message}")
        logger.debug(f"Total health components in monitor: {len(self._component_health)}")

    def get_component_health(self, component_name: str) -> ComponentHealth | None:
        """Get health status for a component."""
        return self._component_health.get(component_name)

    def get_all_health(self) -> List[ComponentHealth]:
        """Get health status for all components."""
        return list(self._component_health.values())

    def is_system_healthy(self) -> bool:
        """
        Check if entire system is healthy.

        Returns:
            True if all components are HEALTHY, False otherwise.
        """
        if not self._component_health:
            return False

        return all(
            h.status == HealthStatus.HEALTHY
            for h in self._component_health.values()
        )

    def get_unhealthy_components(self) -> List[ComponentHealth]:
        """Get list of unhealthy or degraded components."""
        return [
            h for h in self._component_health.values()
            if h.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        ]

    def summary(self) -> Dict[str, int]:
        """
        Get summary of component health counts.

        Returns:
            Dict with counts per status.
        """
        summary = {
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "unknown": 0,
        }

        for health in self._component_health.values():
            if health.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif health.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
            elif health.status == HealthStatus.UNHEALTHY:
                summary["unhealthy"] += 1
            else:
                summary["unknown"] += 1

        return summary
