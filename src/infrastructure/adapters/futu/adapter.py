"""
Futu OpenD adapter with auto-reconnect and push subscription.

Implements BrokerAdapter interface for Futu OpenD gateway.

Architecture:
    - adapter.py: Connection management, lifecycle, main interface
    - position_fetcher.py: Position queries with caching
    - account_fetcher.py: Account info queries with caching
    - order_fetcher.py: Order and trade queries
    - trade_handler.py: Real-time push notifications
    - converters.py: Data type conversions
"""

from __future__ import annotations
from typing import List, Optional, Callable, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import functools

from ....utils.logging_setup import get_logger
import asyncio

from ....domain.interfaces.broker_adapter import BrokerAdapter
from ....domain.interfaces.event_bus import EventBus, EventType
from ....models.position import Position
from ....models.account import AccountInfo
from ....models.order import Order, Trade

from .trade_handler import create_trade_handler
from .position_fetcher import PositionFetcher
from .account_fetcher import AccountFetcher
from .order_fetcher import OrderFetcher


logger = get_logger(__name__)


class FutuAdapter(BrokerAdapter):
    """
    Futu OpenD adapter with auto-reconnect.

    Implements BrokerAdapter using futu-api SDK.
    Requires Futu OpenD gateway to be running locally or on a server.

    Note: All Futu SDK calls are synchronous (blocking). This adapter wraps
    them with asyncio.run_in_executor() to prevent blocking the event loop.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        security_firm: str = "FUTUSECURITIES",
        trd_env: str = "REAL",
        filter_trdmarket: str = "US",
        reconnect_backoff_initial: int = 1,
        reconnect_backoff_max: int = 60,
        reconnect_backoff_factor: float = 2.0,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize Futu adapter.

        Args:
            host: Futu OpenD host.
            port: Futu OpenD port (default 11111).
            security_firm: Security firm (FUTUSECURITIES, FUTUINC, etc.).
            trd_env: Trading environment (REAL or SIMULATE).
            filter_trdmarket: Market filter (US, HK, CN, etc.).
            reconnect_backoff_initial: Initial reconnect delay (seconds).
            reconnect_backoff_max: Max reconnect delay (seconds).
            reconnect_backoff_factor: Backoff multiplier.
            event_bus: Optional event bus for publishing events.
        """
        self.host = host
        self.port = port
        self.security_firm = security_firm
        self.trd_env = trd_env
        self.filter_trdmarket = filter_trdmarket
        self.reconnect_backoff_initial = reconnect_backoff_initial
        self.reconnect_backoff_max = reconnect_backoff_max
        self.reconnect_backoff_factor = reconnect_backoff_factor

        self._trd_ctx = None  # OpenSecTradeContext instance (lazy init)
        self._connected = False
        self._acc_id: Optional[int] = None  # Selected account ID

        # Event bus for publishing events
        self._event_bus = event_bus

        # Thread pool for running blocking Futu SDK calls
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="futu_")

        # Push subscription for real-time trade updates
        self._trade_handler = None
        self._push_enabled = False
        self._cache_lock = threading.Lock()

        # Callback for position updates (subscription-based)
        self._on_position_update: Optional[Callable[[List[Position]], None]] = None

        # Event loop reference (captured during connect for cross-thread ops)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize fetchers (after all attributes are set)
        self._position_fetcher = PositionFetcher(self, cache_ttl_sec=30)
        self._account_fetcher = AccountFetcher(self, cache_ttl_sec=10)
        self._order_fetcher = OrderFetcher(self)

    # -------------------------------------------------------------------------
    # Async Wrapper for Blocking Calls
    # -------------------------------------------------------------------------

    async def _run_blocking(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a blocking Futu SDK call in a thread pool executor.

        This prevents blocking the asyncio event loop.

        Args:
            func: The blocking function to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the blocking function.
        """
        loop = self._loop or asyncio.get_running_loop()
        partial_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self._executor, partial_func)

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Futu OpenD gateway."""
        self._loop = asyncio.get_running_loop()

        try:
            from futu import (
                OpenSecTradeContext,
                TrdMarket,
                SecurityFirm,
                TrdEnv,
                RET_OK,
            )

            trd_market = getattr(TrdMarket, self.filter_trdmarket, TrdMarket.US)
            sec_firm = getattr(SecurityFirm, self.security_firm, SecurityFirm.FUTUSECURITIES)

            self._trd_ctx = OpenSecTradeContext(
                filter_trdmarket=trd_market,
                host=self.host,
                port=self.port,
                security_firm=sec_firm,
            )

            ret, data = await self._run_blocking(self._trd_ctx.get_acc_list)
            if ret != RET_OK:
                raise ConnectionError(f"Failed to get account list: {data}")

            if data.empty:
                raise ConnectionError("No trading accounts found")

            trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)
            matching_accounts = data[data["trd_env"] == trd_env_enum]

            if matching_accounts.empty:
                self._acc_id = int(data["acc_id"].iloc[0])
                logger.warning(
                    f"No {self.trd_env} account found, using first account: {self._acc_id}"
                )
            else:
                self._acc_id = int(matching_accounts["acc_id"].iloc[0])

            self._connected = True
            logger.info(
                f"Connected to Futu OpenD at {self.host}:{self.port}, "
                f"account={self._acc_id}, market={self.filter_trdmarket}"
            )

            self._setup_push_subscription()

        except ImportError:
            logger.error("futu-api library not installed. Install with: pip install futu-api")
            raise ConnectionError("futu-api library not installed")
        except Exception as e:
            logger.error(f"Failed to connect to Futu OpenD at {self.host}:{self.port}: {e}")
            raise ConnectionError(f"Futu connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Futu OpenD."""
        if self._trd_ctx:
            await self._run_blocking(self._trd_ctx.close)
            self._trd_ctx = None
            self._connected = False
            self._acc_id = None
            logger.info("Disconnected from Futu OpenD")

        self._executor.shutdown(wait=True, cancel_futures=True)

    def is_connected(self) -> bool:
        """Check if connected to Futu OpenD."""
        return self._connected and self._acc_id is not None

    async def _ensure_connected(self) -> None:
        """Ensure connection is alive, reconnect if needed."""
        if not self.is_connected():
            logger.info("Futu connection lost, attempting to reconnect...")
            self._connected = False
            if self._trd_ctx:
                try:
                    await self._run_blocking(self._trd_ctx.close)
                except Exception:
                    pass
                self._trd_ctx = None
            await self.connect()

    # -------------------------------------------------------------------------
    # Push Subscription
    # -------------------------------------------------------------------------

    def _setup_push_subscription(self) -> None:
        """Set up push subscription for real-time trade notifications."""
        if not self._trd_ctx:
            logger.warning("Cannot set up push subscription: no trading context")
            return

        try:
            self._trade_handler = create_trade_handler(self._on_trade_received)

            if self._trade_handler:
                self._trd_ctx.set_handler(self._trade_handler)
                self._push_enabled = True
                logger.info("Futu trade push subscription enabled")
            else:
                logger.warning("Failed to create trade handler")

        except Exception as e:
            logger.error(f"Failed to set up Futu push subscription: {e}")
            self._push_enabled = False

    def _on_trade_received(self, trade_data: dict) -> None:
        """Callback invoked when a trade is received via push."""
        try:
            code = trade_data.get('code', 'unknown')
            qty = trade_data.get('qty', 0)
            trd_side = trade_data.get('trd_side', 'unknown')

            logger.info(
                f"Trade notification: {code} qty={qty} side={trd_side} - "
                "refreshing positions"
            )

            if self._event_bus:
                self._event_bus.publish(EventType.POSITION_UPDATED, {
                    "symbol": code,
                    "trade": trade_data,
                    "source": "FUTU",
                    "timestamp": datetime.now(),
                })

            # Invalidate position cache
            self._position_fetcher.invalidate_cache()

            # Trigger position callback if set
            if self._on_position_update:
                self._trigger_position_refresh()

        except Exception as e:
            logger.error(f"Error handling trade notification: {e}")

    def _trigger_position_refresh(self) -> None:
        """Trigger async position refresh from sync callback context."""
        if self._loop is None:
            logger.debug("No event loop captured, skipping position refresh")
            return
        try:
            asyncio.run_coroutine_threadsafe(self._refresh_and_notify_positions(), self._loop)
        except RuntimeError:
            logger.debug("Could not schedule position refresh")

    async def _refresh_and_notify_positions(self) -> None:
        """Fetch fresh positions and notify callback."""
        try:
            positions = await self.fetch_positions()
            if self._on_position_update:
                self._on_position_update(positions)
        except Exception as e:
            logger.error(f"Failed to refresh positions after trade: {e}")

    def set_position_callback(self, callback: Callable[[List[Position]], None]) -> None:
        """
        Set callback for position updates.

        The callback receives the full list of positions whenever a trade occurs.

        Args:
            callback: Function to call with updated positions list.
        """
        self._on_position_update = callback

    async def subscribe_positions(self) -> None:
        """
        Subscribe to position updates via Futu trade push.

        For Futu, position updates are triggered by trade notifications.
        The push subscription is set up during connect(), so this method
        just verifies the subscription is active.
        """
        if not self._connected:
            logger.warning("Cannot subscribe to positions: not connected")
            return

        if self._push_enabled:
            logger.info("Futu position subscription active (via trade push)")
        else:
            logger.warning("Futu push subscription not enabled - positions will poll only")

    def unsubscribe_positions(self) -> None:
        """
        Unsubscribe from position updates.

        For Futu, this clears the callback but doesn't disable push
        (which is also used for other notifications).
        """
        self._on_position_update = None
        logger.info("Futu position callback cleared")

    # -------------------------------------------------------------------------
    # Data Fetching (delegated to specialized fetchers)
    # -------------------------------------------------------------------------

    async def fetch_positions(self) -> List[Position]:
        """Fetch positions from Futu OpenD."""
        return await self._position_fetcher.fetch()

    async def fetch_account_info(self) -> AccountInfo:
        """Fetch account information from Futu OpenD."""
        return await self._account_fetcher.fetch()

    async def fetch_orders(
        self,
        include_open: bool = True,
        include_completed: bool = True,
        days_back: int = 30,
    ) -> List[Order]:
        """Fetch order history from Futu OpenD."""
        return await self._order_fetcher.fetch_orders(
            include_open=include_open,
            include_completed=include_completed,
            days_back=days_back,
        )

    async def fetch_trades(self, days_back: int = 30) -> List[Trade]:
        """Fetch trade/execution history from Futu OpenD."""
        return await self._order_fetcher.fetch_trades(days_back=days_back)
