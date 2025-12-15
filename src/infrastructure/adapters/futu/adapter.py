"""
Futu OpenD adapter with auto-reconnect and push subscription.

Implements BrokerAdapter interface for Futu OpenD gateway.

Phase 2 Fix: All blocking Futu SDK calls are now wrapped with
asyncio.run_in_executor() to prevent blocking the event loop.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
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
from .converters import (
    convert_position,
    convert_order,
    convert_trade,
    convert_trade_with_fee,
    build_trade_from_order,
)


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

        # Cache for account info (Futu rate limit: 10 calls per 30 seconds)
        self._account_cache: Optional[AccountInfo] = None
        self._account_cache_time: Optional[datetime] = None
        self._account_cache_ttl_sec: int = 10

        # Cache for positions (Futu rate limit: 10 calls per 30 seconds)
        self._position_cache: Optional[List[Position]] = None
        self._position_cache_time: Optional[datetime] = None
        self._position_cache_ttl_sec: int = 30
        self._position_cooldown_until: Optional[datetime] = None

        # Push subscription for real-time trade updates
        self._trade_handler = None
        self._push_enabled = False
        self._cache_lock = threading.Lock()

        # Callback for position updates (subscription-based)
        self._on_position_update: Optional[Callable[[List[Position]], None]] = None

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
        loop = asyncio.get_event_loop()
        # Use functools.partial to bind arguments
        partial_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self._executor, partial_func)

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Futu OpenD gateway."""
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

            # Context creation is synchronous but fast
            self._trd_ctx = OpenSecTradeContext(
                filter_trdmarket=trd_market,
                host=self.host,
                port=self.port,
                security_firm=sec_firm,
            )

            # Run blocking get_acc_list in executor
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
            # Close is fast, but run in executor to be safe
            await self._run_blocking(self._trd_ctx.close)
            self._trd_ctx = None
            self._connected = False
            self._acc_id = None
            logger.info("Disconnected from Futu OpenD")

        # Shutdown the executor
        self._executor.shutdown(wait=False)

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

            # Invalidate cache and fetch fresh positions
            with self._cache_lock:
                self._position_cache = None
                self._position_cache_time = None
                self._position_cooldown_until = None

            # Trigger position callback if set (async-safe via event loop)
            if self._on_position_update:
                self._trigger_position_refresh()

        except Exception as e:
            logger.error(f"Error handling trade notification: {e}")

    def _trigger_position_refresh(self) -> None:
        """Trigger async position refresh from sync callback context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the async refresh on the running loop
                asyncio.run_coroutine_threadsafe(self._refresh_and_notify_positions(), loop)
            else:
                # No running loop - skip (positions will refresh on next poll)
                logger.debug("No running event loop, skipping position refresh")
        except RuntimeError:
            logger.debug("Could not get event loop for position refresh")

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
    # Position Fetching
    # -------------------------------------------------------------------------

    async def fetch_positions(self) -> List[Position]:
        """Fetch positions from Futu OpenD (non-blocking)."""
        now = datetime.now()

        with self._cache_lock:
            if self._position_cooldown_until and now < self._position_cooldown_until:
                logger.warning(
                    "Futu position fetch skipped due to rate-limit cooldown "
                    f"(retry after {self._position_cooldown_until.isoformat(timespec='seconds')})"
                )
                if self._position_cache is not None:
                    return self._position_cache
                return []

            if (
                self._position_cache is not None
                and self._position_cache_time is not None
                and (now - self._position_cache_time).total_seconds() < self._position_cache_ttl_sec
            ):
                logger.debug("Using cached Futu positions")
                return self._position_cache

        await self._ensure_connected()

        from futu import RET_OK, TrdEnv

        positions = []
        try:
            trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)

            # Run blocking position_list_query in executor
            ret, data = await self._run_blocking(
                self._trd_ctx.position_list_query,
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                refresh_cache=False,
            )

            if ret != RET_OK:
                if "disconnect" in str(data).lower() or "connection" in str(data).lower():
                    logger.warning("Futu connection issue detected, reconnecting...")
                    self._connected = False
                    await self._ensure_connected()
                    ret, data = await self._run_blocking(
                        self._trd_ctx.position_list_query,
                        trd_env=trd_env_enum,
                        acc_id=self._acc_id,
                        refresh_cache=False,
                    )
                    if ret != RET_OK:
                        raise Exception(f"Position query failed after reconnect: {data}")
                else:
                    if "frequent" not in str(data).lower():
                        logger.error(f"Failed to fetch positions from Futu: {data}")
                    raise Exception(f"Position query failed: {data}")

            if data.empty:
                logger.debug("No positions found in Futu account")
                return []

            for _, row in data.iterrows():
                position = convert_position(row, self._acc_id)
                if position:
                    positions.append(position)

            logger.debug(f"Fetched {len(positions)} positions from Futu")
            self._connected = True

            with self._cache_lock:
                self._position_cache = positions
                self._position_cache_time = datetime.now()
                self._position_cooldown_until = None

        except Exception as e:
            if "disconnect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Failed to fetch positions from Futu: {e}")
                self._connected = False

            if "frequent" in str(e).lower():
                cooldown_seconds = 30
                with self._cache_lock:
                    self._position_cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
                logger.warning(f"Futu rate limit hit; backing off for {cooldown_seconds}s")
                with self._cache_lock:
                    if self._position_cache is not None:
                        return self._position_cache
                logger.warning("Futu rate limited and no cached positions available")
            else:
                logger.error(f"Failed to fetch positions from Futu: {e}")
            raise

        return positions

    # -------------------------------------------------------------------------
    # Account Info Fetching
    # -------------------------------------------------------------------------

    async def fetch_account_info(self) -> AccountInfo:
        """Fetch account information from Futu OpenD (non-blocking)."""
        now = datetime.now()
        if (
            self._account_cache is not None
            and self._account_cache_time is not None
            and (now - self._account_cache_time).total_seconds() < self._account_cache_ttl_sec
        ):
            logger.debug("Using cached Futu account info")
            return self._account_cache

        await self._ensure_connected()

        from futu import RET_OK, TrdEnv, Currency

        try:
            trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)

            # Run blocking accinfo_query in executor
            ret, data = await self._run_blocking(
                self._trd_ctx.accinfo_query,
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                refresh_cache=False,
                currency=Currency.USD,
            )

            if ret != RET_OK:
                if "disconnect" in str(data).lower() or "connection" in str(data).lower():
                    logger.warning("Futu connection issue detected, reconnecting...")
                    self._connected = False
                    await self._ensure_connected()
                    ret, data = await self._run_blocking(
                        self._trd_ctx.accinfo_query,
                        trd_env=trd_env_enum,
                        acc_id=self._acc_id,
                        refresh_cache=False,
                        currency=Currency.USD,
                    )
                    if ret != RET_OK:
                        raise Exception(f"Account info query failed after reconnect: {data}")
                else:
                    if "frequent" not in str(data).lower():
                        logger.error(f"Failed to fetch account info from Futu: {data}")
                    raise Exception(f"Account info query failed: {data}")

            if data.empty:
                raise Exception("No account info returned")

            row = data.iloc[0]

            def safe_float(key: str, default: float = 0.0) -> float:
                try:
                    value = row.get(key)
                    if value is None or (isinstance(value, float) and value != value):
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default

            net_liquidation = safe_float("total_assets")
            total_cash = safe_float("cash")
            buying_power = safe_float("power")
            maintenance_margin = safe_float("maintenance_margin", 0.0)
            init_margin_req = safe_float("initial_margin", 0.0)
            margin_used = init_margin_req
            margin_available = safe_float("available_funds", buying_power)
            excess_liquidity = safe_float("risk_level", 0.0)
            realized_pnl = safe_float("realized_pl", 0.0)
            unrealized_pnl = safe_float("unrealized_pl", 0.0)

            logger.debug(
                f"Fetched Futu account info: TotalAssets=${net_liquidation:,.2f}, "
                f"BuyingPower=${buying_power:,.2f}, Cash=${total_cash:,.2f}"
            )
            self._connected = True

            account_info = AccountInfo(
                net_liquidation=net_liquidation,
                total_cash=total_cash,
                buying_power=buying_power,
                margin_used=margin_used,
                margin_available=margin_available,
                maintenance_margin=maintenance_margin,
                init_margin_req=init_margin_req,
                excess_liquidity=excess_liquidity,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                timestamp=datetime.now(),
                account_id=str(self._acc_id) if self._acc_id else None,
            )

            self._account_cache = account_info
            self._account_cache_time = datetime.now()

            if self._event_bus:
                self._event_bus.publish(EventType.ACCOUNT_UPDATED, {
                    "account": account_info,
                    "source": "FUTU",
                    "timestamp": datetime.now(),
                })

            return account_info

        except Exception as e:
            if "disconnect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Failed to fetch account info from Futu: {e}")
                self._connected = False

            if "frequent" in str(e).lower():
                if self._account_cache is not None:
                    logger.debug("Rate limited - returning cached account info")
                    return self._account_cache
                logger.warning("Futu rate limited and no cached account info available")
            else:
                logger.error(f"Failed to fetch account info from Futu: {e}")
            raise

    # -------------------------------------------------------------------------
    # Order Fetching
    # -------------------------------------------------------------------------

    async def fetch_orders(
        self,
        include_open: bool = True,
        include_completed: bool = True,
        days_back: int = 30,
    ) -> List[Order]:
        """Fetch order history from Futu OpenD (non-blocking)."""
        await self._ensure_connected()

        from futu import RET_OK, TrdEnv

        orders = []
        trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)

        try:
            if include_open:
                # Run blocking order_list_query in executor
                ret, data = await self._run_blocking(
                    self._trd_ctx.order_list_query,
                    trd_env=trd_env_enum,
                    acc_id=self._acc_id,
                    refresh_cache=False,
                )
                if ret == RET_OK and not data.empty:
                    for _, row in data.iterrows():
                        order = convert_order(row, self._acc_id)
                        if order:
                            orders.append(order)
                    logger.debug(f"Fetched {len(data)} orders from Futu")

            if include_completed:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)

                # Run blocking history_order_list_query in executor
                ret, data = await self._run_blocking(
                    self._trd_ctx.history_order_list_query,
                    trd_env=trd_env_enum,
                    acc_id=self._acc_id,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )
                if ret == RET_OK and not data.empty:
                    for _, row in data.iterrows():
                        order = convert_order(row, self._acc_id)
                        if order:
                            if not any(o.order_id == order.order_id for o in orders):
                                orders.append(order)
                    logger.debug("Fetched historical orders from Futu")

            logger.info(f"Fetched {len(orders)} total orders from Futu")

        except Exception as e:
            logger.error(f"Failed to fetch orders from Futu: {e}")
            raise

        return orders

    # -------------------------------------------------------------------------
    # Trade Fetching
    # -------------------------------------------------------------------------

    async def fetch_trades(self, days_back: int = 30) -> List[Trade]:
        """
        Fetch trade/execution history from Futu OpenD (non-blocking).

        Uses a two-step approach:
        1. Fetch filled orders via history_order_list_query with fees from order_fee_query
        2. Validate against deal_list_query to ensure no trades are missing
        """
        await self._ensure_connected()

        try:
            orders_with_fees = await self._fetch_filled_orders_with_fees(days_back)
            trades_from_orders = self._build_trades_from_orders(orders_with_fees)
            deals = await self._fetch_deals(days_back)
            trades = self._validate_and_merge_trades(trades_from_orders, deals)

            logger.info(f"Fetched {len(trades)} total trades from Futu (last {days_back} days)")
            return trades

        except Exception as e:
            logger.error(f"Failed to fetch trades from Futu: {e}")
            raise

    async def _fetch_filled_orders_with_fees(self, days_back: int) -> List[Dict]:
        """Fetch filled orders and their associated fees (non-blocking)."""
        from futu import RET_OK, TrdEnv, OrderStatus as FutuOrderStatus

        trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        orders_with_fees = []

        # Run blocking history_order_list_query in executor
        ret, data = await self._run_blocking(
            self._trd_ctx.history_order_list_query,
            trd_env=trd_env_enum,
            acc_id=self._acc_id,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            status_filter_list=[FutuOrderStatus.FILLED_ALL],
        )

        if ret != RET_OK:
            logger.warning(f"Failed to fetch filled orders: {data}")
            return []

        if data.empty:
            logger.debug("No filled orders found")
            return []

        logger.debug(f"Fetched {len(data)} filled orders from Futu history")

        order_ids = data['order_id'].astype(str).tolist()

        fees_by_order: Dict[str, float] = {}
        for i in range(0, len(order_ids), 400):
            batch_ids = order_ids[i:i + 400]
            # Run blocking order_fee_query in executor
            ret_fee, fee_data = await self._run_blocking(
                self._trd_ctx.order_fee_query,
                order_id_list=batch_ids,
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
            )
            if ret_fee == RET_OK and not fee_data.empty:
                for _, fee_row in fee_data.iterrows():
                    oid = str(fee_row.get('order_id', ''))
                    fee_amount = float(fee_row.get('fee_amount', 0) or 0)
                    fees_by_order[oid] = fee_amount
                logger.debug(f"Fetched fees for {len(fee_data)} orders")
            else:
                logger.warning(f"Failed to fetch order fees: {fee_data}")

        for _, row in data.iterrows():
            order_id = str(row.get('order_id', ''))
            order_dict = row.to_dict()
            order_dict['fee_amount'] = fees_by_order.get(order_id, 0.0)
            orders_with_fees.append(order_dict)

        return orders_with_fees

    async def _fetch_deals(self, days_back: int) -> List[Dict]:
        """Fetch deals (executions) from Futu (non-blocking)."""
        from futu import RET_OK, TrdEnv

        trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        deals = []
        seen_deal_ids = set()

        # Run blocking deal_list_query in executor
        ret, data = await self._run_blocking(
            self._trd_ctx.deal_list_query,
            trd_env=trd_env_enum,
            acc_id=self._acc_id,
            refresh_cache=False,
        )
        if ret == RET_OK and not data.empty:
            for _, row in data.iterrows():
                deal_id = str(row.get('deal_id', ''))
                if deal_id and deal_id not in seen_deal_ids:
                    deals.append(row.to_dict())
                    seen_deal_ids.add(deal_id)
            logger.debug(f"Fetched {len(data)} deals (today) from Futu")

        # Run blocking history_deal_list_query in executor
        ret, data = await self._run_blocking(
            self._trd_ctx.history_deal_list_query,
            trd_env=trd_env_enum,
            acc_id=self._acc_id,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
        if ret == RET_OK and not data.empty:
            for _, row in data.iterrows():
                deal_id = str(row.get('deal_id', ''))
                if deal_id and deal_id not in seen_deal_ids:
                    deals.append(row.to_dict())
                    seen_deal_ids.add(deal_id)
            logger.debug(f"Fetched {len(data)} historical deals from Futu")

        return deals

    def _build_trades_from_orders(self, orders_with_fees: List[Dict]) -> Dict[str, Trade]:
        """Build Trade objects from filled orders with fees."""
        trades_by_order_id: Dict[str, Trade] = {}

        for order in orders_with_fees:
            trade = build_trade_from_order(order, self._acc_id)
            if trade:
                trades_by_order_id[trade.order_id] = trade

        return trades_by_order_id

    def _validate_and_merge_trades(
        self,
        trades_from_orders: Dict[str, Trade],
        deals: List[Dict],
    ) -> List[Trade]:
        """Validate trades from orders against deals and merge."""
        final_trades: List[Trade] = []
        seen_deal_ids: set = set()
        orders_with_deals: set = set()

        deals_by_order: Dict[str, List[Dict]] = {}
        for deal in deals:
            order_id = str(deal.get('order_id', ''))
            if order_id:
                if order_id not in deals_by_order:
                    deals_by_order[order_id] = []
                deals_by_order[order_id].append(deal)

        for order_id, order_deals in deals_by_order.items():
            order_trade = trades_from_orders.get(order_id)

            total_fee = order_trade.commission if order_trade else 0.0
            fee_per_deal = total_fee / len(order_deals) if order_deals else 0.0

            for deal in order_deals:
                deal_id = str(deal.get('deal_id', ''))
                if deal_id in seen_deal_ids:
                    continue
                seen_deal_ids.add(deal_id)
                orders_with_deals.add(order_id)

                trade = convert_trade_with_fee(deal, fee_per_deal, self._acc_id)
                if trade:
                    final_trades.append(trade)

        for order_id, order_trade in trades_from_orders.items():
            if order_id not in orders_with_deals:
                logger.warning(
                    f"Filled order {order_id} has no corresponding deals - "
                    f"using order data: {order_trade.symbol} qty={order_trade.quantity}"
                )
                final_trades.append(order_trade)

        for deal in deals:
            order_id = str(deal.get('order_id', ''))
            deal_id = str(deal.get('deal_id', ''))
            if order_id and order_id not in trades_from_orders and deal_id not in seen_deal_ids:
                logger.warning(
                    f"Deal {deal_id} has no corresponding filled order - "
                    f"order_id={order_id}, adding without fee"
                )
                trade = convert_trade(deal, self._acc_id)
                if trade:
                    final_trades.append(trade)
                    seen_deal_ids.add(deal_id)

        return final_trades
