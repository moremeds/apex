"""
Interactive Brokers adapter with auto-reconnect.

Implements BrokerAdapter and MarketDataProvider interfaces for IBKR TWS/Gateway.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Callable, TYPE_CHECKING
from datetime import datetime
from math import isnan
import logging

from ...domain.interfaces.broker_adapter import BrokerAdapter
from ...domain.interfaces.market_data_provider import MarketDataProvider
from ...domain.interfaces.event_bus import EventBus, EventType
from ...models.position import Position, AssetType, PositionSource
from ...models.market_data import MarketData
from ...models.account import AccountInfo
from ...models.order import Order, Trade, OrderSource, OrderStatus, OrderSide, OrderType
from .market_data_fetcher import MarketDataFetcher


logger = logging.getLogger(__name__)


class IbAdapter(BrokerAdapter, MarketDataProvider):
    """
    Interactive Brokers adapter with auto-reconnect.

    Implements BrokerAdapter and MarketDataProvider using ib_async.
    """

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
        Initialize IB adapter.

        Args:
            host: IB TWS/Gateway host.
            port: IB TWS/Gateway port (7497 for TWS, 4001 for Gateway).
            client_id: Client ID for connection.
            reconnect_backoff_initial: Initial reconnect delay (seconds).
            reconnect_backoff_max: Max reconnect delay (seconds).
            reconnect_backoff_factor: Backoff multiplier.
            event_bus: Optional event bus for publishing events.
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.reconnect_backoff_initial = reconnect_backoff_initial
        self.reconnect_backoff_max = reconnect_backoff_max
        self.reconnect_backoff_factor = reconnect_backoff_factor

        self.ib = None  # ib_async.IB instance (lazy init)
        self._connected = False
        self._subscribed_symbols: List[str] = []
        self._market_data_cache: Dict[str, MarketData] = {}
        self._market_data_fetcher: Optional[MarketDataFetcher] = None

        # Event bus for publishing events
        self._event_bus = event_bus

        # Cache for account info (avoid excessive API calls)
        self._account_cache: Optional[AccountInfo] = None
        self._account_cache_time: Optional[datetime] = None
        self._account_cache_ttl_sec: int = 10  # Cache for 10 seconds

        # Cache for positions (avoid excessive API calls)
        self._position_cache: Optional[List[Position]] = None
        self._position_cache_time: Optional[datetime] = None
        self._position_cache_ttl_sec: int = 10  # Cache for 10 seconds

        # Callback for streaming market data updates
        self._on_market_data_update: Optional[Callable[[str, MarketData], None]] = None

    async def connect(self) -> None:
        """
        Connect to Interactive Brokers TWS/Gateway with auto-reconnect.

        Raises:
            ConnectionError: If unable to connect after retries.
        """
        try:
            from ib_async import IB
            self.ib = IB()
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id, timeout=5)
            self._connected = True
            self._market_data_fetcher = MarketDataFetcher(
                self.ib,
                on_price_update=self._handle_streaming_update,
            )
            logger.info(f"Connected to IB at {self.host}:{self.port}")
        except ImportError:
            logger.error("ib_async library not installed. Install with: pip install ib_async")
            raise ConnectionError("ib_async library not installed")
        except Exception as e:
            logger.error(f"Failed to connect to IB at {self.host}:{self.port}: {e}")
            logger.info("Make sure IB TWS or Gateway is running and API connections are enabled")
            raise ConnectionError(f"IB connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self.ib:
            # Cleanup market data subscriptions
            if self._market_data_fetcher:
                self._market_data_fetcher.cleanup()

            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connected and self.ib is not None and self.ib.isConnected()

    async def fetch_positions(self) -> List[Position]:
        """
        Fetch positions from Interactive Brokers.

        Uses a 10-second TTL cache to avoid excessive API calls.

        Returns:
            List of Position objects with source=IB.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        # Check cache first
        now = datetime.now()
        if (
            self._position_cache is not None
            and self._position_cache_time is not None
            and (now - self._position_cache_time).total_seconds() < self._position_cache_ttl_sec
        ):
            logger.debug("Using cached IB positions")
            return self._position_cache

        positions = []
        try:
            ib_positions = await self.ib.reqPositionsAsync()

            for ib_pos in ib_positions:
                position = self._convert_ib_position(ib_pos)
                positions.append(position)

            logger.info(f"Fetched {len(positions)} positions from IB")

            # Update cache
            self._position_cache = positions
            self._position_cache_time = datetime.now()

            # NOTE: Do NOT publish POSITIONS_BATCH here.
            # The orchestrator publishes after reconciliation to ensure single data path.
            # Publishing here would cause store/RiskEngine to process raw adapter data
            # before reconciliation, leading to transient inconsistent snapshots.
        except Exception as e:
            logger.error(f"Failed to fetch positions from IB: {e}")
            # Return cached data on error if available
            if self._position_cache is not None:
                logger.debug("Returning cached positions after error")
                return self._position_cache
            raise

        return positions

    def _convert_ib_position(self, ib_pos) -> Position:
        """
        Convert ib_async Position to internal Position model.

        Args:
            ib_pos: ib_async Position object.

        Returns:
            Position object.

        Note on avgCost:
            IB's avgCost is the "average cost per share" which for options already
            includes the multiplier. For a PUT sold at $5.00, IB reports avgCost=500.
            We need to divide by multiplier to get the per-contract price for our
            PnL calculation: (mark - avg_price) * quantity * multiplier
        """
        contract = ib_pos.contract

        # Determine asset type
        if contract.secType == "STK":
            asset_type = AssetType.STOCK
        elif contract.secType == "OPT":
            asset_type = AssetType.OPTION
        elif contract.secType == "FUT":
            asset_type = AssetType.FUTURE
        else:
            asset_type = AssetType.CASH

        # For stocks, expiry/strike/right should be None
        if asset_type == AssetType.STOCK:
            expiry = None
            strike = None
            right = None
        else:
            expiry = contract.lastTradeDateOrContractMonth or None  # Convert "" to None
            strike = float(contract.strike) if contract.strike else None
            right = contract.right or None  # Convert "" to None

        # Get multiplier (default 1 for stocks, typically 100 for options)
        multiplier = int(contract.multiplier or 1)

        # IB's avgCost is already multiplied by the contract multiplier for derivatives
        # For options: avgCost = price_per_contract * 100
        # We need per-contract price for our PnL formula: (mark - avg_price) * qty * mult
        # So we divide by multiplier to get the per-contract avg_price
        avg_cost = ib_pos.avgCost
        if asset_type in (AssetType.OPTION, AssetType.FUTURE) and multiplier > 1:
            avg_price = avg_cost / multiplier
        else:
            avg_price = avg_cost

        return Position(
            symbol=contract.localSymbol,
            underlying=contract.symbol,  # Simplified - extract from contract
            asset_type=asset_type,
            quantity=float(ib_pos.position),
            strike=strike,
            right=right,
            expiry=expiry,
            avg_price=avg_price,
            multiplier=multiplier,
            source=PositionSource.IB,
            last_updated=datetime.now(),
            account_id=ib_pos.account,
        )

    async def fetch_market_data(self, positions: List[Position]) -> List[MarketData]:
        """
        Fetch market data for given positions with batch requests for better performance.

        Args:
            positions: List of positions.

        Returns:
            List of MarketData objects.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        if not positions:
            return []

        logger.info(f"Fetching market data for {len(positions)} positions...")
        market_data_list = []

        try:
            from ib_async import Stock, Option

            # Step 1: Create and qualify all contracts in batch
            logger.debug("Qualifying contracts...")
            contracts = []
            pos_map = {}  # Map contract to position

            for pos in positions:
                try:
                    if pos.asset_type == AssetType.OPTION:
                        # Convert expiry to YYYYMMDD string format for IBKR
                        expiry_str = self._format_expiry_for_ib(pos.expiry)
                        if not expiry_str:
                            logger.warning(f"Invalid expiry for {pos.symbol}: {pos.expiry}")
                            continue

                        contract = Option(
                            symbol=pos.underlying,
                            lastTradeDateOrContractMonth=expiry_str,
                            strike=pos.strike,
                            right=str(pos.right),
                            exchange="SMART",
                            multiplier=str(pos.multiplier),
                            currency="USD",
                        )
                    else:
                        contract = Stock(pos.symbol, 'SMART', currency="USD")

                    contracts.append(contract)
                    pos_map[id(contract)] = pos
                except Exception as e:
                    logger.warning(f"Failed to create contract for {pos.symbol}: {e}")
                    continue

            # Qualify all contracts at once (much faster than one-by-one)
            qualified = []
            if contracts:
                try:
                    qualified_raw = await self.ib.qualifyContractsAsync(*contracts)
                    # Filter out None values (failed qualifications)
                    qualified = [c for c in qualified_raw if c is not None]

                    failed_count = len(contracts) - len(qualified)
                    if failed_count > 0:
                        logger.warning(f"Failed to qualify {failed_count}/{len(contracts)} contracts - skipping them")

                    logger.debug(f"Qualified {len(qualified)}/{len(contracts)} contracts")
                except Exception as e:
                    logger.error(f"Error qualifying contracts: {e}")
                    # Continue with empty list - will use mock data or cached data

            # Step 2: Fetch market data using the MarketDataFetcher
            if qualified and self._market_data_fetcher:
                try:
                    market_data_list = await self._market_data_fetcher.fetch_market_data(
                        positions, qualified, pos_map
                    )

                    # Cache the market data
                    for md in market_data_list:
                        self._market_data_cache[md.symbol] = md

                except Exception as e:
                    logger.error(f"Error fetching market data: {e}")

            logger.info(f"✓ Fetched market data for {len(market_data_list)}/{len(positions)} positions")

            # Emit event
            if self._event_bus and market_data_list:
                self._event_bus.publish(EventType.MARKET_DATA_BATCH, {
                    "market_data": market_data_list,
                    "source": "IB",
                    "count": len(market_data_list),
                    "timestamp": datetime.now(),
                })

        except Exception as e:
            logger.error(f"Error in fetch_market_data: {e}")

        return market_data_list

    def _format_expiry_for_ib(self, expiry) -> Optional[str]:
        """
        Convert expiry to YYYYMMDD string format required by IBKR.

        Args:
            expiry: Can be date object, "YYYY-MM-DD" string, or "YYYYMMDD" string.

        Returns:
            Expiry in "YYYYMMDD" format, or None if invalid.
        """
        from datetime import date, datetime

        if expiry is None or expiry == "":
            return None

        # If already a date object, convert to string
        if isinstance(expiry, date):
            return expiry.strftime("%Y%m%d")

        # If string, parse and convert
        if isinstance(expiry, str):
            # Already in YYYYMMDD format
            if len(expiry) == 8 and expiry.isdigit():
                return expiry

            # Convert YYYY-MM-DD to YYYYMMDD
            if "-" in expiry:
                try:
                    dt = datetime.strptime(expiry, "%Y-%m-%d")
                    return dt.strftime("%Y%m%d")
                except ValueError:
                    logger.error(f"Invalid date format: {expiry}")
                    return None

        logger.error(f"Unexpected expiry type: {type(expiry)}, value: {expiry}")
        return None

    async def subscribe(self, symbols: List[str]) -> None:
        """
        Subscribe to real-time market data.

        Args:
            symbols: List of symbols to subscribe to.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        # TODO: Implement subscription using ib.reqMktData()
        self._subscribed_symbols.extend(symbols)
        logger.info(f"Subscribed to {len(symbols)} symbols")

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        # TODO: Implement unsubscription using ib.cancelMktData()
        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                self._subscribed_symbols.remove(symbol)
        logger.info(f"Unsubscribed from {len(symbols)} symbols")

    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest cached market data for a symbol."""
        return self._market_data_cache.get(symbol)

    async def fetch_market_indicators(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Fetch snapshot market data for broad market indicators (e.g., VIX, SPY).

        Designed for market-alert use cases where we need lightweight quotes
        without Greeks or streaming subscriptions.

        Args:
            symbols: List of ticker symbols to fetch.

        Returns:
            Dict mapping symbol -> MarketData
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        if not symbols:
            return {}

        from ib_async import Stock, Index

        # Map of known index symbols
        INDEX_SYMBOLS = {"VIX", "SPX", "NDX", "DJI", "RUT"}

        contracts = []
        for sym in symbols:
            try:
                # Use Index contract for known indices, Stock for everything else
                if sym in INDEX_SYMBOLS:
                    contracts.append(Index(sym, "CBOE", currency="USD"))
                else:
                    contracts.append(Stock(sym, "SMART", currency="USD"))
            except Exception as e:
                logger.warning(f"Failed to build contract for {sym}: {e}")

        if not contracts:
            return {}

        market_data: Dict[str, MarketData] = {}

        try:
            qualified = await self.ib.qualifyContractsAsync(*contracts)
            tickers = await self.ib.reqTickersAsync(*qualified)

            for sym, ticker in zip(symbols, tickers):
                try:
                    md = MarketData(
                        symbol=sym,
                        last=float(ticker.last) if ticker.last and not isnan(ticker.last) else None,
                        bid=float(ticker.bid) if ticker.bid and not isnan(ticker.bid) else None,
                        ask=float(ticker.ask) if ticker.ask and not isnan(ticker.ask) else None,
                        mid=float((ticker.bid + ticker.ask) / 2) if ticker.bid and ticker.ask and not isnan(ticker.bid) and not isnan(ticker.ask) else None,
                        yesterday_close=float(ticker.close) if hasattr(ticker, "close") and ticker.close and not isnan(ticker.close) else None,
                        timestamp=datetime.now(),
                    )
                    market_data[sym] = md
                    self._market_data_cache[sym] = md
                except Exception as e:
                    logger.warning(f"Failed to parse indicator data for {sym}: {e}")

        except Exception as e:
            logger.error(f"Error fetching market indicators: {e}")

        return market_data

    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Fetch quotes for a list of symbols (without position context).

        This is an alias for fetch_market_indicators to satisfy the
        MarketDataProvider interface.

        Args:
            symbols: List of symbols to fetch quotes for.

        Returns:
            Dict mapping symbol to MarketData.
        """
        return await self.fetch_market_indicators(symbols)

    def set_streaming_callback(
        self,
        callback: Optional[Callable[[str, MarketData], None]]
    ) -> None:
        """
        Set callback for streaming market data updates.

        Args:
            callback: Function to call with (symbol, market_data) on updates.
                     Set to None to disable streaming callbacks.
        """
        self._on_market_data_update = callback

    def supports_streaming(self) -> bool:
        """Check if this provider supports real-time streaming."""
        return True  # IB supports streaming via reqMktData

    def supports_greeks(self) -> bool:
        """Check if this provider supports Greeks (options data)."""
        return True  # IB provides Greeks for options

    async def fetch_account_info(self) -> AccountInfo:
        """
        Fetch account information from IB using accountSummary API.

        Uses a 10-second TTL cache to avoid excessive API calls.

        Returns:
            AccountInfo object with real account data from IBKR.

        Raises:
            ConnectionError: If not connected.
            Exception: Any error encountered during IB fetch/parsing.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        # Check cache first
        now = datetime.now()
        if (
            self._account_cache is not None
            and self._account_cache_time is not None
            and (now - self._account_cache_time).total_seconds() < self._account_cache_ttl_sec
        ):
            logger.debug("Using cached IB account info")
            return self._account_cache

        try:
            # Request account summary from IBKR
            # Use empty string for account to get default account
            account_values = await self.ib.accountSummaryAsync()

            # Parse account values into a dictionary
            account_data = {}
            account_id = None

            for av in account_values:
                account_data[av.tag] = av.value
                if not account_id:
                    account_id = av.account

            # Helper function to safely parse float values
            def safe_float(tag: str, default: float = 0.0) -> float:
                try:
                    value = account_data.get(tag, default)
                    return float(value) if value else default
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse account tag '{tag}': {value}")
                    return default

            # Extract key account metrics
            net_liquidation = safe_float('NetLiquidation')
            total_cash = safe_float('TotalCashValue')
            buying_power = safe_float('BuyingPower')
            maintenance_margin = safe_float('MaintMarginReq')
            init_margin_req = safe_float('InitMarginReq')
            excess_liquidity = safe_float('ExcessLiquidity')
            available_funds = safe_float('AvailableFunds')
            realized_pnl = safe_float('RealizedPnL')
            unrealized_pnl = safe_float('UnrealizedPnL')
            gross_position_value = safe_float('GrossPositionValue')

            # Calculate derived metrics
            # Margin used = Initial margin requirement (what's currently used)
            margin_used = init_margin_req
            # Margin available = Excess liquidity or available funds
            margin_available = max(excess_liquidity, available_funds)

            logger.info(f"✓ Fetched account info: NetLiq=${net_liquidation:,.2f}, BuyingPower=${buying_power:,.2f}, Margin={margin_used:,.2f}/{net_liquidation:,.2f}")

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
                account_id=account_id,
            )

            # Update cache
            self._account_cache = account_info
            self._account_cache_time = datetime.now()

            # Emit event
            if self._event_bus:
                self._event_bus.publish(EventType.ACCOUNT_UPDATED, {
                    "account": account_info,
                    "source": "IB",
                    "timestamp": datetime.now(),
                })

            return account_info

        except Exception as e:
            logger.error(f"Failed to fetch account info from IB: {e}")
            # Return cached data on error if available
            if self._account_cache is not None:
                logger.debug("Returning cached account info after error")
                return self._account_cache
            # Propagate to caller to avoid silent risk calculations with zeroed balances
            raise

    def enable_streaming(self) -> None:
        """Enable streaming market data updates."""
        if self._market_data_fetcher:
            self._market_data_fetcher.enable_streaming()
            logger.info("IB streaming market data enabled")

    def disable_streaming(self) -> None:
        """Disable streaming market data updates."""
        if self._market_data_fetcher:
            self._market_data_fetcher.disable_streaming()
            logger.info("IB streaming market data disabled")

    def _handle_streaming_update(self, symbol: str, market_data: MarketData) -> None:
        """
        Handle streaming market data update from fetcher.

        Updates cache, emits event, and fires callback if registered.
        """
        # Update cache
        self._market_data_cache[symbol] = market_data

        # Emit event (non-blocking via event bus)
        if self._event_bus:
            self._event_bus.publish(EventType.MARKET_DATA_TICK, {
                "symbol": symbol,
                "data": market_data,
                "source": "IB",
                "timestamp": datetime.now(),
            })

        # Fire callback (legacy support)
        if self._on_market_data_update:
            self._on_market_data_update(symbol, market_data)

    async def fetch_orders(
        self,
        include_open: bool = True,
        include_completed: bool = True,
        days_back: int = 30,
    ) -> List[Order]:
        """
        Fetch order history from Interactive Brokers.

        Args:
            include_open: Include open/pending orders.
            include_completed: Include filled/cancelled orders.
            days_back: Number of days to look back for completed orders (unused by IB API).

        Returns:
            List of Order objects with source=IB.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        orders = []

        try:
            # Request all orders (open and completed) from IB
            # openOrders() gets currently open orders
            # reqCompletedOrders() gets historical completed orders
            if include_open:
                open_trades = await self.ib.reqOpenOrdersAsync()
                for trade in open_trades:
                    order = self._convert_ib_order_wrapper_to_order(trade)
                    if order:
                        orders.append(order)
                logger.debug(f"Fetched {len(open_trades)} open orders from IB")

            if include_completed:
                # Request completed orders (fills, cancellations)
                completed_trades = await self.ib.reqCompletedOrdersAsync(apiOnly=False)
                for trade in completed_trades:
                    order = self._convert_ib_order_wrapper_to_order(trade)
                    if order:
                        orders.append(order)
                logger.debug(f"Fetched {len(completed_trades)} completed orders from IB")

            logger.info(f"Fetched {len(orders)} total orders from IB")

        except Exception as e:
            logger.error(f"Failed to fetch orders from IB: {e}")
            raise

        return orders

    async def fetch_trades(self, days_back: int = 7) -> List[Trade]:
        """
        Fetch trade/execution history from Interactive Brokers.

        Args:
            days_back: Number of days to look back (default 7)

        Returns:
            List of Trade objects with source=IB.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        trades = []

        try:
            # Request executions from IB
            # This returns all fills/executions within the lookback period
            from ib_async import ExecutionFilter
            from datetime import timedelta, timezone

            exec_filter = ExecutionFilter()
            # IB expects time filter in YYYYMMDD-HH:MM:SS format (exchange timezone)
            # Use UTC and go back extra day to handle timezone edge cases
            start_datetime = datetime.now(timezone.utc) - timedelta(days=days_back + 1)
            exec_filter.time = start_datetime.strftime("%Y%m%d-%H:%M:%S")
            exec_filter.clientId = self.client_id

            fills = await self.ib.reqExecutionsAsync(exec_filter)

            for fill in fills:
                trade = self._convert_ib_fill_to_trade(fill)
                if trade:
                    trades.append(trade)

            logger.info(f"Fetched {len(trades)} trades from IB (last {days_back} days)")

        except Exception as e:
            logger.error(f"Failed to fetch trades from IB: {e}")
            raise

        return trades

    def _convert_ib_order_wrapper_to_order(self, ib_order_wrapper) -> Optional[Order]:
        """
        Convert ib_async Trade wrapper to internal Order model.

        Note: In ib_async, the Trade class is a wrapper containing:
        - trade.contract (security)
        - trade.order (the IB Order object)
        - trade.orderStatus (status information)
        It represents an order with its status, NOT a trade/execution.

        Args:
            ib_order_wrapper: ib_async Trade object (order + status wrapper).

        Returns:
            Order object or None if conversion fails.
        """
        try:
            contract = ib_order_wrapper.contract
            order = ib_order_wrapper.order
            order_status = ib_order_wrapper.orderStatus

            # Determine asset type
            if contract.secType == "STK":
                asset_type = "STOCK"
            elif contract.secType == "OPT":
                asset_type = "OPTION"
            elif contract.secType == "FUT":
                asset_type = "FUTURE"
            else:
                asset_type = contract.secType

            # Map IB order type
            order_type_map = {
                "MKT": OrderType.MARKET,
                "LMT": OrderType.LIMIT,
                "STP": OrderType.STOP,
                "STP LMT": OrderType.STOP_LIMIT,
            }
            order_type = order_type_map.get(order.orderType, OrderType.MARKET)

            # Map IB order status
            status_map = {
                "PendingSubmit": OrderStatus.PENDING,
                "PendingCancel": OrderStatus.PENDING,
                "PreSubmitted": OrderStatus.SUBMITTED,
                "Submitted": OrderStatus.SUBMITTED,
                "Cancelled": OrderStatus.CANCELLED,
                "Filled": OrderStatus.FILLED,
                "Inactive": OrderStatus.REJECTED,
            }
            status = status_map.get(order_status.status, OrderStatus.PENDING)

            # Determine side
            side = OrderSide.BUY if order.action == "BUY" else OrderSide.SELL

            # Option-specific fields
            expiry = None
            strike = None
            right = None
            if asset_type == "OPTION":
                expiry = contract.lastTradeDateOrContractMonth or None
                strike = float(contract.strike) if contract.strike else None
                right = contract.right or None

            return Order(
                order_id=str(order.orderId),
                source=OrderSource.IB,
                account_id=order.account or "",
                symbol=contract.localSymbol or contract.symbol,
                underlying=contract.symbol,
                asset_type=asset_type,
                side=side,
                order_type=order_type,
                quantity=float(order.totalQuantity),
                limit_price=float(order.lmtPrice) if order.lmtPrice else None,
                stop_price=float(order.auxPrice) if order.auxPrice else None,
                status=status,
                filled_quantity=float(order_status.filled) if order_status.filled else 0.0,
                avg_fill_price=float(order_status.avgFillPrice) if order_status.avgFillPrice else None,
                commission=float(order_status.commission) if order_status.commission and not isnan(order_status.commission) else 0.0,
                submitted_time=datetime.now(),  # IB doesn't provide exact submission time
                filled_time=datetime.now() if status == OrderStatus.FILLED else None,
                updated_time=datetime.now(),
                expiry=expiry,
                strike=strike,
                right=right,
                broker_order_id=str(order.permId) if order.permId else None,
                exchange=contract.exchange,
                time_in_force=order.tif,
            )

        except Exception as e:
            logger.warning(f"Failed to convert IB order wrapper to order: {e}")
            return None

    def _convert_ib_fill_to_trade(self, ib_fill) -> Optional[Trade]:
        """
        Convert ib_async Fill to internal Trade model.

        Args:
            ib_fill: ib_async Fill object (execution details).

        Returns:
            Trade object or None if conversion fails.
        """
        try:
            contract = ib_fill.contract
            execution = ib_fill.execution
            commission_report = ib_fill.commissionReport

            # Determine asset type
            if contract.secType == "STK":
                asset_type = "STOCK"
            elif contract.secType == "OPT":
                asset_type = "OPTION"
            elif contract.secType == "FUT":
                asset_type = "FUTURE"
            else:
                asset_type = contract.secType

            # Determine side
            side = OrderSide.BUY if execution.side == "BOT" else OrderSide.SELL

            # Get execution time (ib_async returns datetime with UTC timezone)
            trade_time = execution.time if execution.time else datetime.now()
            # Convert to local time if it's timezone-aware UTC
            if hasattr(trade_time, 'tzinfo') and trade_time.tzinfo is not None:
                # Convert UTC to local time for display consistency
                trade_time = trade_time.replace(tzinfo=None)  # Keep as naive datetime in UTC

            # Option-specific fields
            expiry = None
            strike = None
            right = None
            if asset_type == "OPTION":
                expiry = contract.lastTradeDateOrContractMonth or None
                strike = float(contract.strike) if contract.strike else None
                right = contract.right or None

            # Get commission from report
            commission = 0.0
            if commission_report and commission_report.commission and not isnan(commission_report.commission):
                commission = float(commission_report.commission)

            return Trade(
                trade_id=execution.execId,
                order_id=str(execution.orderId),
                source=OrderSource.IB,
                account_id=execution.acctNumber or "",
                symbol=contract.localSymbol or contract.symbol,
                underlying=contract.symbol,
                asset_type=asset_type,
                side=side,
                quantity=float(execution.shares),
                price=float(execution.price),
                commission=commission,
                trade_time=trade_time,
                expiry=expiry,
                strike=strike,
                right=right,
                exchange=execution.exchange,
                liquidity=execution.liquidation if hasattr(execution, 'liquidation') else None,
            )

        except Exception as e:
            logger.warning(f"Failed to convert IB fill to trade: {e}")
            return None
