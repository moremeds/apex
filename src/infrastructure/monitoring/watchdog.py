"""
Watchdog - Connection and snapshot freshness monitoring.

Monitors:
- Connection status (auto-reconnect on failure)
- Snapshot staleness (alert if no update)
- Missing market data ratio
"""

from __future__ import annotations
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from ...domain.events import PriorityEventBus
from ...domain.interfaces.broker_adapter import BrokerAdapter
from ...domain.interfaces.market_data_provider import MarketDataProvider
from ...domain.interfaces.event_bus import EventBus, EventType
from ...utils.timezone import age_seconds
from .health_monitor import HealthMonitor, HealthStatus
from ...utils.logging_setup import get_logger


logger = get_logger(__name__)


class Watchdog:
    """
    System watchdog for connection and data freshness monitoring.

    Responsibilities:
    - Monitor IB connection and trigger auto-reconnect
    - Monitor snapshot freshness (alert if stale)
    - Monitor missing market data ratio
    """

    def __init__(
        self,
        health_monitor: HealthMonitor,
        event_bus: PriorityEventBus,
        config: Dict[str, Any],
    ):
        """
        Initialize watchdog.

        Args:
            health_monitor: HealthMonitor instance.
            event_bus: EventBus for publishing alerts.
            config: Watchdog configuration dict.
        """
        self.health_monitor = health_monitor
        self.event_bus = event_bus
        self.config = config

        self.snapshot_stale_sec = config.get("snapshot_stale_sec", 10)
        self.max_missing_md_ratio = config.get("max_missing_md_ratio", 0.2)
        self.reconnect_backoff = config.get("reconnect_backoff_sec", {})

        self._last_snapshot_time: Optional[datetime] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Initialize health components
        self._initialize_health_components()

    def _initialize_health_components(self) -> None:
        """Initialize health components with default states."""
        logger.info("Watchdog: Initializing health components...")
        self.health_monitor.update_component_health(
            "market_data_coverage",
            HealthStatus.UNKNOWN,
            "Waiting for data...",
            {"missing_count": 0, "total": 0},
        )
        logger.info("Watchdog: Registered market_data_coverage")

        self.health_monitor.update_component_health(
            "snapshot_freshness",
            HealthStatus.UNKNOWN,
            "Waiting for first snapshot...",
        )
        logger.info("Watchdog: Registered snapshot_freshness")

        # Log total components after initialization
        all_health = self.health_monitor.get_all_health()
        logger.info(f"Watchdog: Total health components after init: {len(all_health)}")

    async def start(self) -> None:
        """Start watchdog monitoring loop."""
        if self._running:
            logger.warning("Watchdog already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Watchdog started")

    async def stop(self) -> None:
        """Stop watchdog monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Watchdog stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_snapshot_freshness()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                await asyncio.sleep(5)

    async def _check_snapshot_freshness(self) -> None:
        """Check if snapshot is stale."""
        if self._last_snapshot_time is None:
            return

        age = age_seconds(self._last_snapshot_time)
        if age > self.snapshot_stale_sec:
            logger.warning(f"Snapshot stale: {age:.1f}s")
            self.health_monitor.update_component_health(
                "snapshot_freshness",
                HealthStatus.DEGRADED,
                f"Snapshot not updated for {age:.1f}s",
            )
            self.event_bus.publish(
                EventType.MARKET_DATA_STALE,
                {"age_seconds": age, "threshold": self.snapshot_stale_sec},
            )
        else:
            self.health_monitor.update_component_health(
                "snapshot_freshness",
                HealthStatus.HEALTHY,
                f"Snapshot fresh ({age:.1f}s old)",
            )

    def update_snapshot_time(self, timestamp: datetime) -> None:
        """
        Update last snapshot timestamp.

        Call this after each successful snapshot generation.

        Args:
            timestamp: Snapshot timestamp.
        """
        self._last_snapshot_time = timestamp

    def check_missing_market_data(
        self, positions_count: int, missing_md_count: int
    ) -> None:
        """
        Check if missing market data ratio exceeds threshold.

        Args:
            positions_count: Total position count.
            missing_md_count: Count of positions with missing market data.
        """
        if positions_count == 0:
            self.health_monitor.update_component_health(
                "market_data_coverage",
                HealthStatus.HEALTHY,
                "No positions",
                {"missing_count": 0, "total": 0},
            )
            return

        missing_ratio = missing_md_count / positions_count
        if missing_ratio > self.max_missing_md_ratio:
            logger.warning(
                f"Missing market data ratio high: {missing_ratio:.2%} "
                f"({missing_md_count}/{positions_count})"
            )
            self.health_monitor.update_component_health(
                "market_data_coverage",
                HealthStatus.DEGRADED,
                f"Missing MD: {missing_ratio:.1%}",
                {"missing_count": missing_md_count, "total": positions_count},
            )
        else:
            # Healthy case - include counts for visibility
            self.health_monitor.update_component_health(
                "market_data_coverage",
                HealthStatus.HEALTHY,
                f"MD coverage: {(1-missing_ratio):.1%}",
                {"missing_count": missing_md_count, "total": positions_count},
            )

    async def reconnect_provider(
        self, provider: BrokerAdapter | MarketDataProvider, provider_name: str
    ) -> bool:
        """
        Attempt to reconnect a provider with exponential backoff.

        Args:
            provider: Provider to reconnect.
            provider_name: Provider name for logging.

        Returns:
            True if reconnected successfully, False otherwise.
        """
        initial_delay = self.reconnect_backoff.get("initial", 1)
        max_delay = self.reconnect_backoff.get("max", 60)
        factor = self.reconnect_backoff.get("factor", 2)

        delay = initial_delay
        attempt = 0

        while self._running:
            attempt += 1
            logger.info(f"Reconnecting {provider_name} (attempt {attempt}, delay {delay}s)")

            try:
                await provider.connect()
                if provider.is_connected():
                    logger.info(f"{provider_name} reconnected successfully")
                    self.event_bus.publish(
                        EventType.CONNECTION_RESTORED,
                        {"provider": provider_name, "attempts": attempt},
                    )
                    return True
            except Exception as e:
                logger.error(f"Reconnect failed: {e}")

            await asyncio.sleep(delay)
            delay = min(delay * factor, max_delay)

        return False
