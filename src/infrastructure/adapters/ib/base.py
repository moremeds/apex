"""
Base IB adapter with shared connection logic.

All IB adapters inherit from this base class which provides:
- Connection management (connect, disconnect, reconnect)
- Client ID management
- Event bus integration
- Common utilities
"""

from __future__ import annotations

from abc import ABC
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Optional, Union

from ....domain.interfaces.event_bus import EventBus
from ....utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ib_async import IB

logger = get_logger(__name__)


class IbBaseAdapter(ABC):
    """
    Base class for all IB adapters.

    Provides shared connection management and utilities.
    Subclasses implement specific functionality (live, historical, execution).
    """

    # Adapter type - override in subclasses
    ADAPTER_TYPE = "base"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        reconnect_backoff_initial: int = 1,
        reconnect_backoff_max: int = 60,
        reconnect_backoff_factor: float = 2.0,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize base IB adapter.

        Args:
            host: IB TWS/Gateway host.
            port: IB TWS/Gateway port (7497 for TWS, 4001 for Gateway).
            client_id: Reserved IB client ID for this adapter.
            reconnect_backoff_initial: Initial reconnect delay (seconds).
            reconnect_backoff_max: Max reconnect delay (seconds).
            reconnect_backoff_factor: Backoff multiplier.
            event_bus: Optional event bus for publishing events.
        """
        self.host = host
        self.port = port
        self._client_id = client_id
        self.reconnect_backoff_initial = reconnect_backoff_initial
        self.reconnect_backoff_max = reconnect_backoff_max
        self.reconnect_backoff_factor = reconnect_backoff_factor

        self.ib: Optional[IB] = None  # ib_async.IB instance (lazy init)
        self._connected = False
        self._event_bus = event_bus

        # Connection stats
        self._connect_time: Optional[datetime] = None
        self._reconnect_count: int = 0
        self._last_error: Optional[str] = None

    @property
    def client_id(self) -> int:
        """
        Get the actual client ID used for connection.

        Override in subclasses if a dynamic ID is required.
        """
        return self._client_id

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to Interactive Brokers TWS/Gateway.

        Uses the client ID specific to this adapter type.
        """
        try:
            from ib_async import IB

            actual_client_id = self.client_id
            self.ib = IB()
            await self.ib.connectAsync(self.host, self.port, clientId=actual_client_id, timeout=5)
            self._connected = True
            self._connect_time = datetime.now()
            self._last_error = None

            logger.info(
                f"Connected to IB at {self.host}:{self.port} "
                f"(client_id={actual_client_id}, type={self.ADAPTER_TYPE})"
            )

            # Hook for subclasses to perform post-connect setup
            await self._on_connected()

        except ImportError:
            self._last_error = "ib_async library not installed"
            logger.error("ib_async library not installed. Install with: pip install ib_async")
            raise ConnectionError("ib_async library not installed")
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to connect to IB at {self.host}:{self.port}: {e}")
            raise ConnectionError(f"IB connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self.ib:
            # Hook for subclasses to perform pre-disconnect cleanup
            await self._on_disconnecting()

            self.ib.disconnect()
            self._connected = False

            logger.info(f"Disconnected from IB (type={self.ADAPTER_TYPE})")

    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connected and self.ib is not None and self.ib.isConnected()

    async def ensure_connected(self) -> None:
        """Ensure connection is alive, reconnect if needed."""
        if not self.is_connected():
            logger.info(f"IB {self.ADAPTER_TYPE} connection lost, reconnecting...")
            self._reconnect_count += 1
            await self.connect()

    # -------------------------------------------------------------------------
    # Hooks for Subclasses
    # -------------------------------------------------------------------------

    async def _on_connected(self) -> None:
        """
        Hook called after successful connection.

        Override in subclasses to perform adapter-specific setup.
        """

    async def _on_disconnecting(self) -> None:
        """
        Hook called before disconnecting.

        Override in subclasses to perform adapter-specific cleanup.
        """

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def format_expiry_for_ib(self, expiry: Union[date, str, None]) -> Optional[str]:
        """
        Convert expiry to YYYYMMDD string format required by IBKR.

        Args:
            expiry: Expiry in various formats (date, str).

        Returns:
            YYYYMMDD string or None if invalid.
        """
        if expiry is None or expiry == "":
            return None

        if isinstance(expiry, date):
            return expiry.strftime("%Y%m%d")

        if isinstance(expiry, str):
            # Already in YYYYMMDD format
            if len(expiry) == 8 and expiry.isdigit():
                return expiry

            # Handle YYYY-MM-DD format
            if "-" in expiry:
                try:
                    dt = datetime.strptime(expiry, "%Y-%m-%d")
                    return dt.strftime("%Y%m%d")
                except ValueError:
                    logger.error(f"Invalid date format: {expiry}")
                    return None

        logger.error(f"Unexpected expiry type: {type(expiry)}, value: {expiry}")
        return None

    def get_connection_info(self) -> dict:
        """Get connection information for monitoring."""
        return {
            "adapter_type": self.ADAPTER_TYPE,
            "host": self.host,
            "port": self.port,
            "client_id": self._client_id,
            "connected": self.is_connected(),
            "connect_time": self._connect_time.isoformat() if self._connect_time else None,
            "reconnect_count": self._reconnect_count,
            "last_error": self._last_error,
        }

    def publish_event(self, event_type: Any, payload: dict) -> None:
        """
        Publish event to event bus if available.

        Args:
            event_type: EventType enum value.
            payload: Event payload dict.
        """
        if self._event_bus:
            payload["source"] = f"IB_{self.ADAPTER_TYPE.upper()}"
            payload["timestamp"] = datetime.now()
            self._event_bus.publish(event_type, payload)
