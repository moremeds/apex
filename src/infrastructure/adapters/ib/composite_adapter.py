"""
OPT-016: Composite IB adapter wrapping split adapters.

This adapter replaces the legacy monolithic IbAdapter by delegating to
specialized adapters (IbLiveAdapter, IbHistoricalAdapter, IbExecutionAdapter)
connected via IbConnectionPool.

Benefits:
- Each adapter can be tested independently
- Connection pooling for concurrent operations (3 separate IB connections)
- Clear separation of concerns
- Single source of truth for each capability
- Easier maintenance and debugging

Usage:
    pool_config = ConnectionPoolConfig(
        host="127.0.0.1",
        port=7497,
        client_ids=config.ibkr.client_ids,
    )
    adapter = IbCompositeAdapter(pool_config, event_bus)
    await adapter.connect()

    positions = await adapter.fetch_positions()
    market_data = await adapter.fetch_market_data(positions)
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, cast

from ....domain.interfaces.broker_adapter import BrokerAdapter
from ....domain.interfaces.market_data_provider import MarketDataProvider
from ....models.account import AccountInfo
from ....models.market_data import MarketData
from ....models.order import Order, OrderSide, OrderSource, OrderStatus, OrderType, Trade
from ....models.position import AssetType, Position, PositionSource
from ....utils.logging_setup import get_logger
from .connection_pool import ConnectionPoolConfig, IbConnectionPool
from .execution_adapter import IbExecutionAdapter
from .historical_adapter import IbHistoricalAdapter
from .live_adapter import IbLiveAdapter

if TYPE_CHECKING:
    from ....domain.events import PriorityEventBus
    from ....domain.events.domain_events import BarData, OrderUpdate, QuoteTick


logger = get_logger(__name__)


class IbCompositeAdapter(BrokerAdapter, MarketDataProvider):
    """
    Composite IB adapter that delegates to specialized adapters.

    Replaces the legacy monolithic IbAdapter with a clean composition
    of single-responsibility adapters connected via IbConnectionPool.

    Implements:
    - BrokerAdapter: Positions, account, orders, trades
    - MarketDataProvider: Market data fetching and streaming

    Connection Architecture:
    - Uses IbConnectionPool for 3 separate IB connections
    - Monitoring connection: positions, quotes, account
    - Historical connection: bar data, backfill
    - Execution connection: orders, fills (isolated for reliability)
    """

    def __init__(
        self,
        pool_config: ConnectionPoolConfig,
        event_bus: Optional["PriorityEventBus"] = None,
    ):
        """
        Initialize composite adapter.

        Args:
            pool_config: Configuration for IB connection pool.
            event_bus: Optional event bus for publishing events.
        """
        self._pool_config = pool_config
        self._pool: Optional[IbConnectionPool] = None
        self._event_bus = event_bus

        # Specialized adapters (initialized on connect)
        self._live_adapter: Optional[IbLiveAdapter] = None
        self._historical_adapter: Optional[IbHistoricalAdapter] = None
        self._execution_adapter: Optional[IbExecutionAdapter] = None

        # Market data cache (LRU with bounded size)
        self._market_data_cache: OrderedDict[str, MarketData] = OrderedDict()
        self._market_data_cache_max_size: int = 1000

        # Streaming callback
        self._streaming_callback: Optional[Callable[[str, MarketData], None]] = None

        self._connected = False

    # -------------------------------------------------------------------------
    # Connection Management (BrokerAdapter + MarketDataProvider)
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect all adapters via connection pool.

        Creates 3 IB connections for:
        - Monitoring (positions, quotes, account)
        - Historical (bar data)
        - Execution (orders)
        """
        try:
            # Create and connect pool
            self._pool = IbConnectionPool(self._pool_config)
            await self._pool.connect()

            # Get client_ids (ConnectionPoolConfig.__post_init__ guarantees non-None)
            client_ids = self._pool_config.client_ids
            if client_ids is None:
                raise ValueError("client_ids is required in pool config")

            # Initialize live adapter with monitoring connection
            self._live_adapter = IbLiveAdapter(
                host=self._pool_config.host,
                port=self._pool_config.port,
                client_id=client_ids.monitoring,
                event_bus=self._event_bus,
            )
            # Inject the already-connected IB instance
            self._live_adapter.ib = self._pool.monitoring
            self._live_adapter._connected = True
            await self._live_adapter._on_connected()

            # Initialize historical adapter
            hist_client_id = client_ids.historical_pool[0] if client_ids.historical_pool else 3
            self._historical_adapter = IbHistoricalAdapter(
                host=self._pool_config.host,
                port=self._pool_config.port,
                client_id=hist_client_id,
            )
            self._historical_adapter.ib = self._pool.historical
            self._historical_adapter._connected = True
            await self._historical_adapter._on_connected()

            # Initialize execution adapter
            self._execution_adapter = IbExecutionAdapter(
                host=self._pool_config.host,
                port=self._pool_config.port,
                client_id=client_ids.execution,
                event_bus=self._event_bus,
            )
            self._execution_adapter.ib = self._pool.execution
            self._execution_adapter._connected = True
            await self._execution_adapter._on_connected()

            self._connected = True
            logger.info(
                f"IbCompositeAdapter connected via pool at "
                f"{self._pool_config.host}:{self._pool_config.port}"
            )

        except Exception as e:
            logger.error(f"Failed to connect IbCompositeAdapter: {e}")
            await self.disconnect()
            raise ConnectionError(f"IbCompositeAdapter connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect all adapters and pool."""
        # Clean up adapters first
        if self._live_adapter:
            try:
                await self._live_adapter._on_disconnecting()
            except Exception as e:
                logger.warning(f"Error cleaning up live adapter: {e}")
            self._live_adapter = None

        if self._historical_adapter:
            try:
                await self._historical_adapter._on_disconnecting()
            except Exception as e:
                logger.warning(f"Error cleaning up historical adapter: {e}")
            self._historical_adapter = None

        if self._execution_adapter:
            try:
                await self._execution_adapter._on_disconnecting()
            except Exception as e:
                logger.warning(f"Error cleaning up execution adapter: {e}")
            self._execution_adapter = None

        # Disconnect pool
        if self._pool:
            await self._pool.disconnect()
            self._pool = None

        self._connected = False
        logger.info("IbCompositeAdapter disconnected")

    def is_connected(self) -> bool:
        """Check if the composite adapter is connected."""
        return self._connected and self._pool is not None and self._pool.is_connected()

    # -------------------------------------------------------------------------
    # BrokerAdapter: Position Operations
    # -------------------------------------------------------------------------

    async def fetch_positions(self) -> List[Position]:
        """
        Fetch current positions from IB.

        Delegates to IbLiveAdapter which handles position caching.
        """
        if not self._live_adapter:
            logger.warning("Live adapter not connected, returning empty positions")
            return []

        try:
            # IbLiveAdapter.fetch_positions returns List[PositionSnapshot]
            # We need to convert to Position model
            snapshots = await self._live_adapter.fetch_positions()

            # Convert PositionSnapshot to Position
            positions = []
            for snap in snapshots:
                # Convert string enums to proper Enum types
                asset_type = AssetType(snap.asset_type) if snap.asset_type else AssetType.STOCK
                source = PositionSource(snap.source) if snap.source else PositionSource.IB
                # Validate and cast right to Literal["C", "P"] | None
                right: Literal["C", "P"] | None = None
                if snap.right in ("C", "P"):
                    right = cast(Literal["C", "P"], snap.right)
                pos = Position(
                    symbol=snap.symbol,
                    quantity=snap.quantity,
                    avg_price=snap.avg_price,  # Fixed: was avg_cost
                    asset_type=asset_type,
                    underlying=snap.underlying,
                    expiry=snap.expiry,
                    strike=snap.strike,
                    right=right,
                    multiplier=snap.multiplier or 100,
                    source=source,
                )
                positions.append(pos)

            return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    # -------------------------------------------------------------------------
    # BrokerAdapter: Account Operations
    # -------------------------------------------------------------------------

    async def fetch_account_info(self) -> AccountInfo:
        """
        Fetch account information from IB.

        Delegates to IbLiveAdapter which handles account caching.
        """
        if not self._live_adapter:
            raise ConnectionError("Live adapter not connected")

        try:
            # IbLiveAdapter.fetch_account returns AccountSnapshot
            snapshot = await self._live_adapter.fetch_account()

            # Convert AccountSnapshot to AccountInfo
            return AccountInfo(
                account_id=snapshot.account_id,
                net_liquidation=snapshot.net_liquidation,
                total_cash=snapshot.total_cash,  # Fixed: was cash_balance
                buying_power=snapshot.buying_power,
                margin_used=snapshot.margin_used,
                margin_available=snapshot.margin_available,
                maintenance_margin=snapshot.maintenance_margin,
                init_margin_req=snapshot.init_margin_req,
                excess_liquidity=snapshot.excess_liquidity,
                realized_pnl=snapshot.realized_pnl,
                unrealized_pnl=snapshot.unrealized_pnl,
                timestamp=snapshot.timestamp,
            )

        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            raise

    # -------------------------------------------------------------------------
    # BrokerAdapter: Order Operations
    # -------------------------------------------------------------------------

    async def fetch_orders(
        self,
        include_open: bool = True,
        include_completed: bool = True,
        days_back: int = 30,
    ) -> List[Order]:
        """
        Fetch order history from IB.

        Delegates to IbExecutionAdapter.
        """
        if not self._execution_adapter:
            logger.warning("Execution adapter not connected, returning empty orders")
            return []

        orders = []

        try:
            if include_open:
                open_orders = await self._execution_adapter.get_open_orders()
                for order_update in open_orders:
                    orders.append(self._order_update_to_order(order_update))

            if include_completed:
                history = await self._execution_adapter.get_order_history(days_back=days_back)
                for order_update in history:
                    orders.append(self._order_update_to_order(order_update))

            return orders

        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            return []

    async def fetch_trades(self, days_back: int = 30) -> List[Trade]:
        """
        Fetch trade (execution) history from IB.

        Delegates to IbExecutionAdapter.
        """
        if not self._execution_adapter:
            logger.warning("Execution adapter not connected, returning empty trades")
            return []

        try:
            fills = await self._execution_adapter.get_fills(days_back=days_back)

            trades = []
            for fill in fills:
                # Convert side string to OrderSide enum
                side = OrderSide.BUY if fill.side == "BUY" else OrderSide.SELL
                trades.append(
                    Trade(
                        trade_id=fill.exec_id,  # Use exec_id as trade_id
                        order_id=fill.order_id,
                        source=OrderSource.IB,
                        account_id=fill.account_id,
                        symbol=fill.symbol,
                        underlying=fill.underlying,
                        asset_type=fill.asset_type,
                        side=side,
                        quantity=fill.quantity,
                        price=fill.price,
                        commission=fill.commission,
                        trade_time=fill.timestamp,
                    )
                )

            return trades

        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []

    # -------------------------------------------------------------------------
    # MarketDataProvider: Market Data Operations
    # -------------------------------------------------------------------------

    async def fetch_market_data(self, positions: List[Position]) -> List[MarketData]:
        """
        Fetch market data (prices + Greeks) for positions.

        Delegates to IbLiveAdapter.fetch_market_data.
        """
        if not self._live_adapter:
            logger.warning("Live adapter not connected, returning empty market data")
            return []

        try:
            market_data = await self._live_adapter.fetch_market_data(positions)

            # Update cache
            for md in market_data:
                self._update_cache(md.symbol, md)

            return market_data

        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return []

    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Fetch quotes for symbols (market indicators like VIX, SPY).

        Delegates to IbLiveAdapter.fetch_market_indicators.
        """
        if not self._live_adapter:
            logger.warning("Live adapter not connected, returning empty quotes")
            return {}

        try:
            quotes = await self._live_adapter.fetch_market_indicators(symbols)

            # Update cache
            for symbol, md in quotes.items():
                self._update_cache(symbol, md)

            return quotes

        except Exception as e:
            logger.error(f"Failed to fetch quotes: {e}")
            return {}

    # -------------------------------------------------------------------------
    # MarketDataProvider: Streaming
    # -------------------------------------------------------------------------

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data updates."""
        if self._live_adapter:
            await self._live_adapter.subscribe_quotes(symbols)

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data updates."""
        if self._live_adapter:
            await self._live_adapter.unsubscribe_quotes(symbols)

    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest cached market data for a symbol."""
        if self._live_adapter:
            return self._live_adapter.get_latest(symbol)
        return self._market_data_cache.get(symbol)

    def set_streaming_callback(self, callback: Optional[Callable[[str, MarketData], None]]) -> None:
        """Set callback for streaming market data updates."""
        self._streaming_callback = callback
        if self._live_adapter:
            self._live_adapter.set_quote_callback(
                self._wrap_streaming_callback(callback) if callback else None
            )

    def enable_streaming(self) -> None:
        """Enable streaming mode."""
        if self._live_adapter:
            self._live_adapter.enable_streaming()

    def disable_streaming(self) -> None:
        """Disable streaming mode."""
        if self._live_adapter:
            self._live_adapter.disable_streaming()

    def supports_streaming(self) -> bool:
        """Check if streaming is supported."""
        return True

    def supports_greeks(self) -> bool:
        """Check if Greeks are supported."""
        return True

    # -------------------------------------------------------------------------
    # Historical Data (additional functionality)
    # -------------------------------------------------------------------------

    async def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List["BarData"]:
        """
        Fetch historical bar data.

        Delegates to IbHistoricalAdapter.
        """
        if not self._historical_adapter:
            logger.warning("Historical adapter not connected")
            return []

        try:
            return await self._historical_adapter.fetch_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Failed to fetch historical bars: {e}")
            return []

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _update_cache(self, symbol: str, md: MarketData) -> None:
        """Update market data cache with LRU eviction."""
        if symbol in self._market_data_cache:
            self._market_data_cache.move_to_end(symbol)
        self._market_data_cache[symbol] = md
        while len(self._market_data_cache) > self._market_data_cache_max_size:
            self._market_data_cache.popitem(last=False)

    def _wrap_streaming_callback(
        self, callback: Callable[[str, MarketData], None]
    ) -> Callable[["QuoteTick"], None]:
        """Wrap streaming callback to convert QuoteTick to symbol+MarketData."""

        def wrapper(quote_tick: "QuoteTick") -> None:
            # QuoteTick has symbol, bid, ask, last, timestamp
            md = MarketData(
                symbol=quote_tick.symbol,
                bid=quote_tick.bid,
                ask=quote_tick.ask,
                last=quote_tick.last,
                mid=quote_tick.mid,
                timestamp=quote_tick.timestamp,
            )
            callback(quote_tick.symbol, md)

        return wrapper

    def _order_update_to_order(self, order_update: "OrderUpdate") -> Order:
        """Convert OrderUpdate to Order model."""
        # Convert string enum values to proper enum types
        side = OrderSide.BUY if order_update.side == "BUY" else OrderSide.SELL
        order_type = OrderType(order_update.order_type)
        status = OrderStatus(order_update.status)

        return Order(
            order_id=order_update.order_id,
            source=OrderSource.IB,
            account_id=order_update.account_id,
            symbol=order_update.symbol,
            underlying=order_update.underlying,
            asset_type=order_update.asset_type,
            side=side,
            order_type=order_type,
            quantity=order_update.quantity,
            limit_price=order_update.limit_price,
            stop_price=order_update.stop_price,
            status=status,
            filled_quantity=order_update.filled_quantity,
            avg_fill_price=order_update.avg_fill_price,
            updated_time=order_update.timestamp,
        )

    # -------------------------------------------------------------------------
    # Status and Monitoring
    # -------------------------------------------------------------------------

    def get_connection_info(self) -> dict:
        """Get connection status information."""
        return {
            "adapter_type": "composite",
            "connected": self.is_connected(),
            "pool_status": self._pool.get_status() if self._pool else None,
            "live_connected": self._live_adapter.is_connected() if self._live_adapter else False,
            "historical_connected": (
                self._historical_adapter.is_connected() if self._historical_adapter else False
            ),
            "execution_connected": (
                self._execution_adapter.is_connected() if self._execution_adapter else False
            ),
        }
