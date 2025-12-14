"""
BrokerManager - Unified interface for multiple broker adapters.

Manages connections to multiple brokers (IBKR, Futu, etc.) and provides
aggregated positions and account information.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, field
import asyncio

from ...utils.logging_setup import get_logger
from ...domain.interfaces.broker_adapter import BrokerAdapter
from ...models.position import Position, PositionSource
from ...models.account import AccountInfo
from ...models.order import Order, Trade

if TYPE_CHECKING:
    from ...infrastructure.monitoring import HealthMonitor, HealthStatus


logger = get_logger(__name__)


@dataclass
class BrokerStatus:
    """Status of a broker connection."""
    name: str
    connected: bool = False
    last_error: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    position_count: int = 0


class BrokerManager(BrokerAdapter):
    """
    Manages multiple broker adapters and aggregates their data.

    Provides:
    - Unified connection management
    - Aggregated positions across all brokers
    - Aggregated account information
    - Aggregated orders and trades
    - Per-broker status tracking
    - Health monitoring integration
    """

    def __init__(self, health_monitor: Optional["HealthMonitor"] = None):
        """
        Initialize broker manager with empty adapter list.

        Args:
            health_monitor: Optional HealthMonitor for reporting broker health.
        """
        self._adapters: Dict[str, BrokerAdapter] = {}
        self._status: Dict[str, BrokerStatus] = {}
        self._connected = False
        self._health_monitor = health_monitor

        # Position subscription callback
        self._on_position_update: Optional[Callable[[str, List[Position]], None]] = None
        self._position_subscription_active: bool = False

    def register_adapter(self, name: str, adapter: BrokerAdapter) -> None:
        """
        Register a broker adapter.

        Args:
            name: Unique name for the broker (e.g., "ibkr", "futu").
            adapter: The adapter implementing BrokerAdapter.
        """
        self._adapters[name] = adapter
        self._status[name] = BrokerStatus(name=name)
        logger.info(f"Registered broker adapter: {name}")

    def get_adapter(self, name: str) -> Optional[BrokerAdapter]:
        """Get a specific adapter by name."""
        return self._adapters.get(name)

    def get_status(self, name: str) -> Optional[BrokerStatus]:
        """Get status for a specific broker."""
        return self._status.get(name)

    def get_all_status(self) -> Dict[str, BrokerStatus]:
        """Get status for all brokers."""
        return self._status.copy()

    async def connect(self) -> None:
        """
        Connect to all registered broker adapters.

        Attempts to connect to each adapter independently.
        Failures are logged but don't prevent other adapters from connecting.
        """
        if not self._adapters:
            logger.warning("No broker adapters registered")
            return

        connect_tasks = []
        for name, adapter in self._adapters.items():
            connect_tasks.append(self._connect_adapter(name, adapter))

        await asyncio.gather(*connect_tasks)

        # Set overall connected status if at least one adapter connected
        self._connected = any(s.connected for s in self._status.values())

        connected_count = sum(1 for s in self._status.values() if s.connected)
        logger.info(f"BrokerManager connected: {connected_count}/{len(self._adapters)} adapters")

    async def _connect_adapter(self, name: str, adapter: BrokerAdapter) -> None:
        """Connect a single adapter with error handling."""
        try:
            await adapter.connect()
            self._status[name].connected = True
            self._status[name].last_error = None
            self._status[name].last_updated = datetime.now()
            logger.info(f"✓ Connected to {name}")
            self._update_health(name, "HEALTHY", "Connected")
        except Exception as e:
            self._status[name].connected = False
            self._status[name].last_error = str(e)
            self._status[name].last_updated = datetime.now()
            logger.error(f"✗ Failed to connect to {name}: {e}")
            self._update_health(name, "UNHEALTHY", f"Connection failed: {str(e)[:50]}")

    async def disconnect(self) -> None:
        """Disconnect from all broker adapters."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.disconnect()
                self._status[name].connected = False
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")

        self._connected = False
        logger.info("BrokerManager disconnected from all adapters")

    def is_connected(self) -> bool:
        """Check if at least one broker is connected."""
        return self._connected

    async def fetch_positions(self) -> List[Position]:
        """
        Fetch positions from all connected brokers.

        Returns:
            Aggregated list of positions from all brokers.
            Positions are tagged with their source for reconciliation.
        """
        all_positions: List[Position] = []

        fetch_tasks = []
        for name, adapter in self._adapters.items():
            if self._status[name].connected:
                fetch_tasks.append(self._fetch_positions_from_adapter(name, adapter))

        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue  # Already logged in _fetch_positions_from_adapter
            all_positions.extend(result)

        logger.info(f"Fetched {len(all_positions)} total positions from all brokers")
        return all_positions

    async def _fetch_positions_from_adapter(
        self, name: str, adapter: BrokerAdapter
    ) -> List[Position]:
        """Fetch positions from a single adapter with error handling."""
        try:
            positions = await adapter.fetch_positions()
            self._status[name].position_count = len(positions)
            self._status[name].last_updated = datetime.now()
            self._status[name].last_error = None
            logger.debug(f"Fetched {len(positions)} positions from {name}")
            self._update_health(name, "HEALTHY", f"Fetched {len(positions)} positions")
            return positions
        except Exception as e:
            self._status[name].last_error = str(e)
            self._status[name].last_updated = datetime.now()
            logger.error(f"Failed to fetch positions from {name}: {e}")
            self._update_health(name, "UNHEALTHY", f"Position fetch failed: {str(e)[:50]}")
            return []

    async def fetch_account_info(self) -> AccountInfo:
        """
        Fetch and aggregate account information from all connected brokers.

        Returns:
            Aggregated AccountInfo with combined balances across all brokers.
        """
        aggregated = AccountInfo(
            net_liquidation=0.0,
            total_cash=0.0,
            buying_power=0.0,
            margin_used=0.0,
            margin_available=0.0,
            maintenance_margin=0.0,
            init_margin_req=0.0,
            excess_liquidity=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(),
            account_id="AGGREGATED",
        )

        account_count = 0
        for name, adapter in self._adapters.items():
            if not self._status[name].connected:
                continue

            try:
                account = await adapter.fetch_account_info()
                aggregated = self._merge_account_info(aggregated, account)
                account_count += 1
                logger.debug(f"Fetched account info from {name}: NetLiq=${account.net_liquidation:,.2f}")
            except NotImplementedError:
                logger.debug(f"{name} adapter does not support account info")
            except Exception as e:
                logger.error(f"Failed to fetch account info from {name}: {e}")

        if account_count > 0:
            logger.info(
                f"Aggregated account info from {account_count} brokers: "
                f"NetLiq=${aggregated.net_liquidation:,.2f}, "
                f"BuyingPower=${aggregated.buying_power:,.2f}"
            )
        else:
            logger.warning("No account info fetched from any broker")

        return aggregated

    def _merge_account_info(self, base: AccountInfo, other: AccountInfo) -> AccountInfo:
        """
        Merge account info by summing relevant fields.

        Args:
            base: Base account info to merge into.
            other: Account info to merge from.

        Returns:
            New AccountInfo with combined values.
        """
        return AccountInfo(
            net_liquidation=base.net_liquidation + other.net_liquidation,
            total_cash=base.total_cash + other.total_cash,
            buying_power=base.buying_power + other.buying_power,
            margin_used=base.margin_used + other.margin_used,
            margin_available=base.margin_available + other.margin_available,
            maintenance_margin=base.maintenance_margin + other.maintenance_margin,
            init_margin_req=base.init_margin_req + other.init_margin_req,
            excess_liquidity=base.excess_liquidity + other.excess_liquidity,
            realized_pnl=base.realized_pnl + other.realized_pnl,
            unrealized_pnl=base.unrealized_pnl + other.unrealized_pnl,
            timestamp=datetime.now(),
            account_id="AGGREGATED",
        )

    async def fetch_positions_by_broker(self) -> Dict[str, List[Position]]:
        """
        Fetch positions grouped by broker.

        Returns:
            Dict mapping broker name to list of positions.
        """
        positions_by_broker: Dict[str, List[Position]] = {}

        for name, adapter in self._adapters.items():
            if not self._status[name].connected:
                positions_by_broker[name] = []
                continue

            try:
                positions = await adapter.fetch_positions()
                positions_by_broker[name] = positions
                self._status[name].position_count = len(positions)
            except Exception as e:
                logger.error(f"Failed to fetch positions from {name}: {e}")
                positions_by_broker[name] = []

        return positions_by_broker

    async def fetch_account_info_by_broker(self) -> Dict[str, AccountInfo]:
        """
        Fetch account info grouped by broker.

        Returns:
            Dict mapping broker name to AccountInfo.
        """
        accounts_by_broker: Dict[str, AccountInfo] = {}

        for name, adapter in self._adapters.items():
            if not self._status[name].connected:
                continue

            try:
                account = await adapter.fetch_account_info()
                accounts_by_broker[name] = account
                self._update_health(name, "HEALTHY", f"Account: ${account.net_liquidation:,.0f}")
            except NotImplementedError:
                logger.debug(f"{name} adapter does not support account info")
            except Exception as e:
                logger.error(f"Failed to fetch account info from {name}: {e}")
                self._update_health(name, "DEGRADED", f"Account fetch failed: {str(e)[:50]}")

        return accounts_by_broker

    def _update_health(
        self, broker_name: str, status: str, message: str, metadata: Optional[Dict] = None
    ) -> None:
        """
        Update health status for a broker in the health monitor.

        Args:
            broker_name: Name of the broker (e.g., "ibkr", "futu").
            status: Health status string ("HEALTHY", "DEGRADED", "UNHEALTHY").
            message: Status message.
            metadata: Optional additional metadata.
        """
        if self._health_monitor is None:
            return

        from ..monitoring import HealthStatus

        # Map string to enum
        status_map = {
            "HEALTHY": HealthStatus.HEALTHY,
            "DEGRADED": HealthStatus.DEGRADED,
            "UNHEALTHY": HealthStatus.UNHEALTHY,
            "UNKNOWN": HealthStatus.UNKNOWN,
        }
        health_status = status_map.get(status, HealthStatus.UNKNOWN)

        # Use broker-specific component name (e.g., "futu_adapter", "ibkr_adapter")
        component_name = f"{broker_name}_adapter"

        # Include broker status in metadata
        broker_status = self._status.get(broker_name)
        full_metadata = metadata or {}
        if broker_status:
            full_metadata.update({
                "connected": broker_status.connected,
                "position_count": broker_status.position_count,
                "last_error": broker_status.last_error,
            })

        self._health_monitor.update_component_health(
            component_name=component_name,
            status=health_status,
            message=message,
            metadata=full_metadata,
        )

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor after initialization.

        Args:
            health_monitor: HealthMonitor instance to use.
        """
        self._health_monitor = health_monitor

    async def check_all_health(self) -> Dict[str, BrokerStatus]:
        """
        Check health of all brokers and update health monitor.

        Returns:
            Dict mapping broker name to current BrokerStatus.
        """
        for name, adapter in self._adapters.items():
            try:
                # Check if adapter reports connected
                is_connected = adapter.is_connected()

                # Try to reconnect if disconnected
                if not is_connected and hasattr(adapter, '_ensure_connected'):
                    try:
                        await adapter._ensure_connected()
                        is_connected = adapter.is_connected()
                    except Exception as e:
                        logger.debug(f"Reconnect attempt for {name} failed: {e}")

                self._status[name].connected = is_connected

                if is_connected:
                    self._update_health(name, "HEALTHY", "Connected and operational")
                else:
                    self._update_health(name, "UNHEALTHY", "Not connected")
                    self._status[name].last_error = "Connection lost"

            except Exception as e:
                self._status[name].connected = False
                self._status[name].last_error = str(e)
                self._update_health(name, "UNHEALTHY", f"Health check failed: {str(e)[:50]}")

        return self._status.copy()

    async def fetch_orders(
        self,
        include_open: bool = True,
        include_completed: bool = True,
        days_back: int = 30,
    ) -> List[Order]:
        """
        Fetch orders from all connected brokers.

        Args:
            include_open: Include open/pending orders.
            include_completed: Include filled/cancelled/expired orders.
            days_back: Number of days to look back for completed orders.

        Returns:
            Aggregated list of orders from all brokers.
        """
        all_orders: List[Order] = []

        for name, adapter in self._adapters.items():
            if not self._status[name].connected:
                continue

            try:
                orders = await adapter.fetch_orders(
                    include_open=include_open,
                    include_completed=include_completed,
                    days_back=days_back,
                )
                all_orders.extend(orders)
                logger.debug(f"Fetched {len(orders)} orders from {name}")
            except NotImplementedError:
                logger.debug(f"{name} adapter does not support orders")
            except Exception as e:
                logger.error(f"Failed to fetch orders from {name}: {e}")

        logger.info(f"Fetched {len(all_orders)} total orders from all brokers")
        return all_orders

    async def fetch_trades(self, days_back: int = 30) -> List[Trade]:
        """
        Fetch trades from all connected brokers.

        Args:
            days_back: Number of days to look back.

        Returns:
            Aggregated list of trades from all brokers.
        """
        all_trades: List[Trade] = []

        for name, adapter in self._adapters.items():
            if not self._status[name].connected:
                continue

            try:
                trades = await adapter.fetch_trades(days_back=days_back)
                all_trades.extend(trades)
                logger.debug(f"Fetched {len(trades)} trades from {name}")
            except NotImplementedError:
                logger.debug(f"{name} adapter does not support trades")
            except Exception as e:
                logger.error(f"Failed to fetch trades from {name}: {e}")

        logger.info(f"Fetched {len(all_trades)} total trades from all brokers")
        return all_trades

    # -------------------------------------------------------------------------
    # Position Subscription (Event-Driven)
    # -------------------------------------------------------------------------

    def set_position_callback(self, callback: Callable[[str, List[Position]], None]) -> None:
        """
        Set callback for position updates from any broker.

        The callback receives (broker_name, positions_list) whenever positions change.

        Args:
            callback: Function to call with (broker_name, positions).
        """
        self._on_position_update = callback

    async def subscribe_positions(self) -> None:
        """
        Subscribe to position updates from all connected brokers.

        After calling this, position changes (fills, closes) will trigger
        the callback set via set_position_callback().
        """
        if self._position_subscription_active:
            logger.debug("Position subscription already active")
            return

        for name, adapter in self._adapters.items():
            if not self._status[name].connected:
                continue

            # Check if adapter supports position subscription
            if hasattr(adapter, 'set_position_callback') and hasattr(adapter, 'subscribe_positions'):
                try:
                    # Set callback that routes to our aggregator
                    adapter.set_position_callback(
                        lambda positions, broker=name: self._on_broker_position_update(broker, positions)
                    )
                    await adapter.subscribe_positions()
                    logger.info(f"Subscribed to position updates from {name}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to positions from {name}: {e}")
            else:
                logger.debug(f"{name} adapter does not support position subscription")

        self._position_subscription_active = True
        logger.info("BrokerManager position subscription active")

    def unsubscribe_positions(self) -> None:
        """Unsubscribe from position updates on all brokers."""
        for name, adapter in self._adapters.items():
            if hasattr(adapter, 'unsubscribe_positions'):
                try:
                    adapter.unsubscribe_positions()
                except Exception as e:
                    logger.error(f"Error unsubscribing from {name} positions: {e}")

        self._position_subscription_active = False
        logger.info("BrokerManager position subscription stopped")

    def _on_broker_position_update(self, broker_name: str, positions: List[Position]) -> None:
        """
        Handle position update from a specific broker.

        Forwards to the registered callback with broker name.
        """
        self._status[broker_name].position_count = len(positions)
        self._status[broker_name].last_updated = datetime.now()

        logger.debug(f"Position update from {broker_name}: {len(positions)} positions")

        if self._on_position_update:
            self._on_position_update(broker_name, positions)
