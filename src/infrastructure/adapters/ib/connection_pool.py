"""
IB Connection Pool for managing multiple IB connections on the same event loop.

This elegantly solves the "Future attached to a different loop" error by:
- Using multiple IB() instances on the SAME event loop (no threading)
- Each instance has a different client ID for different purposes

TWS supports up to 32 simultaneous client connections.

Usage:
    pool = IbConnectionPool(config)
    await pool.connect()

    # Use monitoring connection for positions/quotes
    positions = await pool.monitoring.reqPositionsAsync()

    # Use historical connection for bar data (won't block monitoring)
    bars = await pool.historical.reqHistoricalDataAsync(...)

    await pool.disconnect()

See: https://github.com/ib-api-reloaded/ib_async/issues/186
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from config.models import IbClientIdsConfig

from ....utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ib_async import IB


logger = get_logger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for IB connection pool."""

    host: str = "127.0.0.1"
    port: int = 7497
    client_ids: Optional[IbClientIdsConfig] = None
    connect_timeout: int = 10

    def __post_init__(self) -> None:
        if self.client_ids is None:
            self.client_ids = IbClientIdsConfig()


class IbConnectionPool:
    """
    Pool of IB connections for different purposes on the same event loop.

    Manages:
    - monitoring: For positions, quotes, market data (client_id from config)
    - historical: For bar data requests, ATR (client_id from config)
    - execution: For order submission/management (A4: isolated from data ops)

    All connections run on the SAME event loop - no threading, no loop conflicts.
    """

    def __init__(self, config: ConnectionPoolConfig):
        """
        Initialize connection pool.

        Args:
            config: Pool configuration with host, port, client_ids.
        """
        self._config = config
        self._monitoring: Optional[IB] = None
        self._historical: Optional[IB] = None
        self._execution: Optional[IB] = None  # A4: Dedicated execution connection
        self._connected = False

    @property
    def monitoring(self) -> Optional["IB"]:
        """Get monitoring IB connection (positions, quotes, orders)."""
        return self._monitoring

    @property
    def historical(self) -> Optional["IB"]:
        """Get historical IB connection (bar data, ATR)."""
        return self._historical

    @property
    def execution(self) -> Optional["IB"]:
        """
        Get execution IB connection (order submission/management).

        A4: Dedicated connection for execution ensures order operations
        are never blocked by data operations (streaming, historical fetches).
        """
        return self._execution

    def is_connected(self) -> bool:
        """Check if all pool connections are alive."""
        if not self._connected:
            return False

        mon_ok = self._monitoring is not None and self._monitoring.isConnected()
        hist_ok = self._historical is not None and self._historical.isConnected()
        exec_ok = self._execution is not None and self._execution.isConnected()
        return mon_ok and hist_ok and exec_ok

    def is_monitoring_connected(self) -> bool:
        """Check if monitoring connection is alive."""
        return self._monitoring is not None and self._monitoring.isConnected()

    def is_historical_connected(self) -> bool:
        """Check if historical connection is alive."""
        return self._historical is not None and self._historical.isConnected()

    def is_execution_connected(self) -> bool:
        """Check if execution connection is alive (A4)."""
        return self._execution is not None and self._execution.isConnected()

    async def connect(self) -> None:
        """
        Connect all IB instances on the SAME event loop.

        This is the elegant solution - no threading, no loop caching issues.
        Each IB() gets a different client ID but shares the event loop.
        """
        from ib_async import IB

        host = self._config.host
        port = self._config.port
        timeout = self._config.connect_timeout
        client_ids = self._config.client_ids

        try:
            # Monitoring connection (positions, quotes, orders)
            self._monitoring = IB()
            await self._monitoring.connectAsync(
                host,
                port,
                clientId=client_ids.monitoring,
                timeout=timeout,
            )
            logger.info(f"IB pool: monitoring connected " f"(client_id={client_ids.monitoring})")

            # Historical connection (bar data, ATR)
            # Use first ID from historical pool
            hist_client_id = client_ids.historical_pool[0] if client_ids.historical_pool else 3
            self._historical = IB()
            await self._historical.connectAsync(
                host,
                port,
                clientId=hist_client_id,
                timeout=timeout,
            )
            logger.info(f"IB pool: historical connected " f"(client_id={hist_client_id})")

            # A4: Execution connection (order submission/management)
            # Isolated from data operations to ensure orders are never blocked
            self._execution = IB()
            await self._execution.connectAsync(
                host,
                port,
                clientId=client_ids.execution,
                timeout=timeout,
            )
            logger.info(f"IB pool: execution connected " f"(client_id={client_ids.execution})")

            self._connected = True
            logger.info(f"IB connection pool ready at {host}:{port} (3 connections)")

        except Exception as e:
            logger.error(f"Failed to connect IB pool: {e}")
            await self.disconnect()
            raise ConnectionError(f"IB pool connection failed: {e}")

    async def connect_monitoring_only(self) -> None:
        """Connect only the monitoring connection (for backward compatibility)."""
        from ib_async import IB

        host = self._config.host
        port = self._config.port
        timeout = self._config.connect_timeout
        client_ids = self._config.client_ids

        try:
            self._monitoring = IB()
            await self._monitoring.connectAsync(
                host,
                port,
                clientId=client_ids.monitoring,
                timeout=timeout,
            )
            self._connected = True
            logger.info(f"IB pool: monitoring connected " f"(client_id={client_ids.monitoring})")
        except Exception as e:
            logger.error(f"Failed to connect monitoring: {e}")
            raise ConnectionError(f"IB monitoring connection failed: {e}")

    async def connect_historical(self) -> None:
        """Connect the historical connection (if not already connected)."""
        if self._historical is not None and self._historical.isConnected():
            return

        from ib_async import IB

        host = self._config.host
        port = self._config.port
        timeout = self._config.connect_timeout
        client_ids = self._config.client_ids

        hist_client_id = client_ids.historical_pool[0] if client_ids.historical_pool else 3

        try:
            self._historical = IB()
            await self._historical.connectAsync(
                host,
                port,
                clientId=hist_client_id,
                timeout=timeout,
            )
            logger.info(f"IB pool: historical connected " f"(client_id={hist_client_id})")
        except Exception as e:
            logger.error(f"Failed to connect historical: {e}")
            raise ConnectionError(f"IB historical connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect all IB instances."""
        # A4: Disconnect execution first (active orders should be handled)
        if self._execution:
            try:
                self._execution.disconnect()
                logger.info("IB pool: execution disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting execution: {e}")
            self._execution = None

        if self._historical:
            try:
                self._historical.disconnect()
                logger.info("IB pool: historical disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting historical: {e}")
            self._historical = None

        if self._monitoring:
            try:
                self._monitoring.disconnect()
                logger.info("IB pool: monitoring disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting monitoring: {e}")
            self._monitoring = None

        self._connected = False
        logger.info("IB connection pool disconnected")

    async def connect_execution(self) -> None:
        """
        Connect the execution connection (A4: if not already connected).

        Dedicated connection for order operations, isolated from data operations.
        """
        if self._execution is not None and self._execution.isConnected():
            return

        from ib_async import IB

        host = self._config.host
        port = self._config.port
        timeout = self._config.connect_timeout
        client_ids = self._config.client_ids

        try:
            self._execution = IB()
            await self._execution.connectAsync(
                host,
                port,
                clientId=client_ids.execution,
                timeout=timeout,
            )
            logger.info(f"IB pool: execution connected " f"(client_id={client_ids.execution})")
        except Exception as e:
            logger.error(f"Failed to connect execution: {e}")
            raise ConnectionError(f"IB execution connection failed: {e}")

    async def ensure_connected(self) -> None:
        """Ensure all connections are alive, reconnect if needed."""
        if not self.is_connected():
            logger.info("IB pool connection lost, reconnecting...")
            await self.connect()

    async def ensure_historical_connected(self) -> None:
        """Ensure historical connection is alive."""
        if not self.is_historical_connected():
            await self.connect_historical()

    async def ensure_execution_connected(self) -> None:
        """Ensure execution connection is alive (A4)."""
        if not self.is_execution_connected():
            await self.connect_execution()

    def get_status(self) -> dict:
        """Get pool connection status."""
        return {
            "connected": self._connected,
            "monitoring": {
                "connected": self.is_monitoring_connected(),
                "client_id": self._config.client_ids.monitoring,
            },
            "historical": {
                "connected": self.is_historical_connected(),
                "client_id": (
                    self._config.client_ids.historical_pool[0]
                    if self._config.client_ids.historical_pool
                    else None
                ),
            },
            "execution": {  # A4: Execution connection status
                "connected": self.is_execution_connected(),
                "client_id": self._config.client_ids.execution,
            },
            "host": self._config.host,
            "port": self._config.port,
        }
