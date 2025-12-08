"""
Futu OpenD adapter with auto-reconnect and push subscription.

Implements BrokerAdapter interface for Futu OpenD gateway.
"""

from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import threading
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


logger = logging.getLogger(__name__)


class FutuAdapter(BrokerAdapter):
    """
    Futu OpenD adapter with auto-reconnect.

    Implements BrokerAdapter using futu-api SDK.
    Requires Futu OpenD gateway to be running locally or on a server.
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

            self._trd_ctx = OpenSecTradeContext(
                filter_trdmarket=trd_market,
                host=self.host,
                port=self.port,
                security_firm=sec_firm,
            )

            ret, data = self._trd_ctx.get_acc_list()
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
            self._trd_ctx.close()
            self._trd_ctx = None
            self._connected = False
            self._acc_id = None
            logger.info("Disconnected from Futu OpenD")

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
                    self._trd_ctx.close()
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
                "invalidating position cache"
            )

            if self._event_bus:
                self._event_bus.publish(EventType.POSITION_UPDATED, {
                    "symbol": code,
                    "trade": trade_data,
                    "source": "FUTU",
                    "timestamp": datetime.now(),
                })

            with self._cache_lock:
                self._position_cache = None
                self._position_cache_time = None
                self._position_cooldown_until = None

        except Exception as e:
            logger.error(f"Error handling trade notification: {e}")

    # -------------------------------------------------------------------------
    # Position Fetching
    # -------------------------------------------------------------------------

    async def fetch_positions(self) -> List[Position]:
        """Fetch positions from Futu OpenD."""
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

            ret, data = self._trd_ctx.position_list_query(
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                refresh_cache=False,
            )

            if ret != RET_OK:
                if "disconnect" in str(data).lower() or "connection" in str(data).lower():
                    logger.warning("Futu connection issue detected, reconnecting...")
                    self._connected = False
                    await self._ensure_connected()
                    ret, data = self._trd_ctx.position_list_query(
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
        """Fetch account information from Futu OpenD."""
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

            ret, data = self._trd_ctx.accinfo_query(
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
                    ret, data = self._trd_ctx.accinfo_query(
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
        """Fetch order history from Futu OpenD."""
        await self._ensure_connected()

        from futu import RET_OK, TrdEnv

        orders = []
        trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)

        try:
            if include_open:
                ret, data = self._trd_ctx.order_list_query(
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

                ret, data = self._trd_ctx.history_order_list_query(
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
        Fetch trade/execution history from Futu OpenD.

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
        """Fetch filled orders and their associated fees."""
        from futu import RET_OK, TrdEnv, OrderStatus as FutuOrderStatus

        trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        orders_with_fees = []

        ret, data = self._trd_ctx.history_order_list_query(
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
            # Rate limit: 10 req/30s - add 3s delay between batches
            if i > 0:
                await asyncio.sleep(3)

            batch_ids = order_ids[i:i + 400]
            ret_fee, fee_data = self._trd_ctx.order_fee_query(
                order_id_list=batch_ids,
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
            )
            if ret_fee == RET_OK and fee_data is not None and not fee_data.empty:
                for _, fee_row in fee_data.iterrows():
                    oid = str(fee_row.get('order_id', ''))
                    fee_amount = float(fee_row.get('fee_amount', 0) or 0)
                    fees_by_order[oid] = fee_amount
                logger.debug(f"Fetched fees for {len(fee_data)} orders (batch {i // 400 + 1})")
            else:
                logger.warning(f"Failed to fetch order fees: {fee_data}")

        for _, row in data.iterrows():
            order_id = str(row.get('order_id', ''))
            order_dict = row.to_dict()
            order_dict['fee_amount'] = fees_by_order.get(order_id, 0.0)
            orders_with_fees.append(order_dict)

        return orders_with_fees

    async def _fetch_deals(self, days_back: int) -> List[Dict]:
        """Fetch deals (executions) from Futu."""
        from futu import RET_OK, TrdEnv

        trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        deals = []
        seen_deal_ids = set()

        ret, data = self._trd_ctx.deal_list_query(
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

        ret, data = self._trd_ctx.history_deal_list_query(
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

    async def fetch_order_fees(self, order_ids: List[str]) -> List[Dict]:
        """
        Fetch fees for specific order IDs.

        Args:
            order_ids: List of order IDs to fetch fees for

        Returns:
            List of fee records as dicts with order_id, fee_amount, fee_list keys
        """
        await self._ensure_connected()
        from futu import RET_OK, TrdEnv

        trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)
        fees = []

        ret_fee, fee_data = self._trd_ctx.order_fee_query(
            order_id_list=order_ids,
            trd_env=trd_env_enum,
            acc_id=self._acc_id,
        )

        if ret_fee == RET_OK and fee_data is not None and not fee_data.empty:
            for _, row in fee_data.iterrows():
                fees.append(row.to_dict())
            logger.debug(f"Fetched fees for {len(fees)} orders")
        else:
            logger.warning(f"Failed to fetch order fees: {fee_data}")

        return fees

    async def fetch_orders_raw(
        self,
        days_back: int = 30,
        include_open: bool = True,
        include_completed: bool = True,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Fetch orders returning raw API dictionaries (not converted to Order objects).

        Args:
            days_back: Number of days to look back (used if start_date/end_date not provided)
            include_open: Include open orders
            include_completed: Include completed/filled orders
            start_date: Explicit start date (overrides days_back)
            end_date: Explicit end date (overrides days_back)

        Returns:
            List of raw order dicts from Futu API

        Raises:
            Exception: If API calls fail
        """
        await self._ensure_connected()
        from futu import RET_OK, TrdEnv, OrderStatus as FutuOrderStatus

        trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)
        raw_orders = []
        seen_order_ids = set()
        errors = []

        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days_back)

        # Fetch open orders
        if include_open:
            ret, data = self._trd_ctx.order_list_query(
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                refresh_cache=False,
            )
            if ret != RET_OK:
                error_msg = f"Failed to fetch open orders: {data}"
                logger.warning(error_msg)
                errors.append(error_msg)
            elif data is not None and not data.empty:
                for _, row in data.iterrows():
                    order_id = str(row.get('order_id', ''))
                    if order_id and order_id not in seen_order_ids:
                        raw_orders.append(row.to_dict())
                        seen_order_ids.add(order_id)
                logger.info(f"Fetched {len(data)} open orders from Futu")
            else:
                logger.debug("No open orders found in Futu")

        # Fetch historical orders
        if include_completed:
            ret, data = self._trd_ctx.history_order_list_query(
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
            if ret != RET_OK:
                error_msg = f"Failed to fetch historical orders ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}): {data}"
                logger.warning(error_msg)
                errors.append(error_msg)
            elif data is not None and not data.empty:
                for _, row in data.iterrows():
                    order_id = str(row.get('order_id', ''))
                    if order_id and order_id not in seen_order_ids:
                        raw_orders.append(row.to_dict())
                        seen_order_ids.add(order_id)
                logger.debug(f"Fetched {len(data)} historical orders from Futu ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
            else:
                logger.debug(f"No historical orders found in Futu ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

        # If all calls failed, raise an exception
        if errors and len(raw_orders) == 0:
            raise Exception(f"Futu order fetch failed: {'; '.join(errors)}")

        return raw_orders

    async def fetch_deals_raw(
        self,
        days_back: int = 30,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Fetch deals returning raw API dictionaries (not converted to Trade objects).

        Args:
            days_back: Number of days to look back (used if start_date/end_date not provided)
            start_date: Explicit start date (overrides days_back)
            end_date: Explicit end date (overrides days_back)

        Returns:
            List of raw deal dicts from Futu API

        Raises:
            Exception: If API calls fail
        """
        await self._ensure_connected()
        from futu import RET_OK, TrdEnv

        trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)
        raw_deals = []
        seen_deal_ids = set()
        errors = []

        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days_back)

        # Check if date range includes today
        today = datetime.now().date()
        include_today = end_date.date() >= today

        # Today's deals (only if end_date includes today)
        if include_today:
            ret, data = self._trd_ctx.deal_list_query(
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                refresh_cache=False,
            )
            if ret != RET_OK:
                error_msg = f"Failed to fetch today's deals: {data}"
                logger.warning(error_msg)
                errors.append(error_msg)
            elif data is not None and not data.empty:
                for _, row in data.iterrows():
                    deal_id = str(row.get('deal_id', ''))
                    if deal_id and deal_id not in seen_deal_ids:
                        raw_deals.append(row.to_dict())
                        seen_deal_ids.add(deal_id)
                logger.debug(f"Fetched {len(data)} deals (today) from Futu")
            else:
                logger.debug("No deals today in Futu")

        # Historical deals
        ret, data = self._trd_ctx.history_deal_list_query(
            trd_env=trd_env_enum,
            acc_id=self._acc_id,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
        if ret != RET_OK:
            error_msg = f"Failed to fetch historical deals ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}): {data}"
            logger.warning(error_msg)
            errors.append(error_msg)
        elif data is not None and not data.empty:
            for _, row in data.iterrows():
                deal_id = str(row.get('deal_id', ''))
                if deal_id and deal_id not in seen_deal_ids:
                    raw_deals.append(row.to_dict())
                    seen_deal_ids.add(deal_id)
            logger.debug(f"Fetched {len(data)} historical deals from Futu ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        else:
            logger.debug(f"No historical deals found in Futu ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

        # If all calls failed, raise an exception
        if errors and len(raw_deals) == 0:
            raise Exception(f"Futu deal fetch failed: {'; '.join(errors)}")

        return raw_deals
