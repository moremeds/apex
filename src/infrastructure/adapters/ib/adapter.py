"""
Interactive Brokers adapter with auto-reconnect.

Implements BrokerAdapter and MarketDataProvider interfaces for IBKR TWS/Gateway.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Callable
from datetime import datetime, date
from math import isnan
import logging

from ....domain.interfaces.broker_adapter import BrokerAdapter
from ....domain.interfaces.market_data_provider import MarketDataProvider
from ....domain.interfaces.event_bus import EventBus, EventType
from ....models.position import Position, AssetType
from ....models.market_data import MarketData
from ....models.account import AccountInfo
from ....models.order import Order, Trade

from ..market_data_fetcher import MarketDataFetcher
from .converters import convert_position, convert_order, convert_fill


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

        # Cache for account info
        self._account_cache: Optional[AccountInfo] = None
        self._account_cache_time: Optional[datetime] = None
        self._account_cache_ttl_sec: int = 10

        # Cache for positions
        self._position_cache: Optional[List[Position]] = None
        self._position_cache_time: Optional[datetime] = None
        self._position_cache_ttl_sec: int = 10

        # Callback for streaming market data updates
        self._on_market_data_update: Optional[Callable[[str, MarketData], None]] = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Interactive Brokers TWS/Gateway with auto-reconnect."""
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
            raise ConnectionError(f"IB connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if self.ib:
            if self._market_data_fetcher:
                self._market_data_fetcher.cleanup()

            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connected and self.ib is not None and self.ib.isConnected()

    # -------------------------------------------------------------------------
    # Position Fetching
    # -------------------------------------------------------------------------

    async def fetch_positions(self) -> List[Position]:
        """Fetch positions from Interactive Brokers."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

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
                position = convert_position(ib_pos)
                positions.append(position)

            logger.info(f"Fetched {len(positions)} positions from IB")

            self._position_cache = positions
            self._position_cache_time = datetime.now()

        except Exception as e:
            logger.error(f"Failed to fetch positions from IB: {e}")
            if self._position_cache is not None:
                logger.debug("Returning cached positions after error")
                return self._position_cache
            raise

        return positions

    # -------------------------------------------------------------------------
    # Account Info Fetching
    # -------------------------------------------------------------------------

    async def fetch_account_info(self) -> AccountInfo:
        """Fetch account information from IB using accountSummary API."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        now = datetime.now()
        if (
            self._account_cache is not None
            and self._account_cache_time is not None
            and (now - self._account_cache_time).total_seconds() < self._account_cache_ttl_sec
        ):
            logger.debug("Using cached IB account info")
            return self._account_cache

        try:
            account_values = await self.ib.accountSummaryAsync()

            account_data = {}
            account_id = None

            for av in account_values:
                account_data[av.tag] = av.value
                if not account_id:
                    account_id = av.account

            def safe_float(tag: str, default: float = 0.0) -> float:
                try:
                    value = account_data.get(tag, default)
                    return float(value) if value else default
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse account tag '{tag}': {value}")
                    return default

            net_liquidation = safe_float('NetLiquidation')
            total_cash = safe_float('TotalCashValue')
            buying_power = safe_float('BuyingPower')
            maintenance_margin = safe_float('MaintMarginReq')
            init_margin_req = safe_float('InitMarginReq')
            excess_liquidity = safe_float('ExcessLiquidity')
            available_funds = safe_float('AvailableFunds')
            realized_pnl = safe_float('RealizedPnL')
            unrealized_pnl = safe_float('UnrealizedPnL')

            margin_used = init_margin_req
            margin_available = max(excess_liquidity, available_funds)

            logger.info(f"Fetched account info: NetLiq=${net_liquidation:,.2f}, BuyingPower=${buying_power:,.2f}")

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

            self._account_cache = account_info
            self._account_cache_time = datetime.now()

            if self._event_bus:
                self._event_bus.publish(EventType.ACCOUNT_UPDATED, {
                    "account": account_info,
                    "source": "IB",
                    "timestamp": datetime.now(),
                })

            return account_info

        except Exception as e:
            logger.error(f"Failed to fetch account info from IB: {e}")
            if self._account_cache is not None:
                logger.debug("Returning cached account info after error")
                return self._account_cache
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
        """Fetch order history from Interactive Brokers."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        orders = []

        try:
            if include_open:
                open_trades = await self.ib.reqOpenOrdersAsync()
                for trade in open_trades:
                    order = convert_order(trade)
                    if order:
                        orders.append(order)
                logger.debug(f"Fetched {len(open_trades)} open orders from IB")

            if include_completed:
                completed_trades = await self.ib.reqCompletedOrdersAsync(apiOnly=False)
                for trade in completed_trades:
                    order = convert_order(trade)
                    if order:
                        orders.append(order)
                logger.debug(f"Fetched {len(completed_trades)} completed orders from IB")

            logger.info(f"Fetched {len(orders)} total orders from IB")

        except Exception as e:
            logger.error(f"Failed to fetch orders from IB: {e}")
            raise

        return orders

    # -------------------------------------------------------------------------
    # Trade Fetching
    # -------------------------------------------------------------------------

    async def fetch_trades(self, days_back: int = 7) -> List[Trade]:
        """Fetch trade/execution history from Interactive Brokers."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        trades = []

        try:
            from ib_async import ExecutionFilter
            from datetime import timedelta, timezone

            exec_filter = ExecutionFilter()
            start_datetime = datetime.now(timezone.utc) - timedelta(days=days_back + 1)
            exec_filter.time = start_datetime.strftime("%Y%m%d-%H:%M:%S")
            exec_filter.clientId = self.client_id

            fills = await self.ib.reqExecutionsAsync(exec_filter)

            for fill in fills:
                trade = convert_fill(fill)
                if trade:
                    trades.append(trade)

            logger.info(f"Fetched {len(trades)} trades from IB (last {days_back} days)")

        except Exception as e:
            logger.error(f"Failed to fetch trades from IB: {e}")
            raise

        return trades

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    async def fetch_market_data(self, positions: List[Position]) -> List[MarketData]:
        """Fetch market data for given positions."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        if not positions:
            return []

        logger.info(f"Fetching market data for {len(positions)} positions...")
        market_data_list = []

        try:
            from ib_async import Stock, Option

            contracts = []
            positions_for_contracts = []  # Parallel list: positions_for_contracts[i] is position for contracts[i]

            for pos in positions:
                try:
                    if pos.asset_type == AssetType.OPTION:
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
                    positions_for_contracts.append(pos)
                except Exception as e:
                    logger.warning(f"Failed to create contract for {pos.symbol}: {e}")
                    continue

            qualified = []
            pos_map = {}  # Mapping from qualified contract id to position
            if contracts:
                try:
                    qualified_raw = await self.ib.qualifyContractsAsync(*contracts)

                    # Build mapping from qualified contracts to positions
                    # qualified_raw maintains same order as input contracts
                    for i, qualified_contract in enumerate(qualified_raw):
                        if qualified_contract is not None:
                            qualified.append(qualified_contract)
                            pos_map[id(qualified_contract)] = positions_for_contracts[i]

                    failed_count = len(contracts) - len(qualified)
                    if failed_count > 0:
                        logger.warning(f"Failed to qualify {failed_count}/{len(contracts)} contracts")

                    logger.debug(f"Qualified {len(qualified)}/{len(contracts)} contracts")
                except Exception as e:
                    logger.error(f"Error qualifying contracts: {e}")

            if qualified and self._market_data_fetcher:
                try:
                    market_data_list = await self._market_data_fetcher.fetch_market_data(
                        positions, qualified, pos_map
                    )

                    for md in market_data_list:
                        self._market_data_cache[md.symbol] = md

                except Exception as e:
                    logger.error(f"Error fetching market data: {e}")

            logger.info(f"Fetched market data for {len(market_data_list)}/{len(positions)} positions")

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
        """Convert expiry to YYYYMMDD string format required by IBKR."""
        if expiry is None or expiry == "":
            return None

        if isinstance(expiry, date):
            return expiry.strftime("%Y%m%d")

        if isinstance(expiry, str):
            if len(expiry) == 8 and expiry.isdigit():
                return expiry

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
        """Subscribe to real-time market data."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        self._subscribed_symbols.extend(symbols)
        logger.info(f"Subscribed to {len(symbols)} symbols")

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                self._subscribed_symbols.remove(symbol)
        logger.info(f"Unsubscribed from {len(symbols)} symbols")

    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest cached market data for a symbol."""
        return self._market_data_cache.get(symbol)

    async def fetch_market_indicators(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Fetch snapshot market data for broad market indicators."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        if not symbols:
            return {}

        from ib_async import Stock, Index

        INDEX_SYMBOLS = {"VIX", "SPX", "NDX", "DJI", "RUT"}

        contracts = []
        for sym in symbols:
            try:
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
        """Fetch quotes for a list of symbols."""
        return await self.fetch_market_indicators(symbols)

    def set_streaming_callback(
        self,
        callback: Optional[Callable[[str, MarketData], None]]
    ) -> None:
        """Set callback for streaming market data updates."""
        self._on_market_data_update = callback

    def supports_streaming(self) -> bool:
        """Check if this provider supports real-time streaming."""
        return True

    def supports_greeks(self) -> bool:
        """Check if this provider supports Greeks (options data)."""
        return True

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
        """Handle streaming market data update from fetcher."""
        self._market_data_cache[symbol] = market_data

        if self._event_bus:
            self._event_bus.publish(EventType.MARKET_DATA_TICK, {
                "symbol": symbol,
                "data": market_data,
                "source": "IB",
                "timestamp": datetime.now(),
            })

        if self._on_market_data_update:
            self._on_market_data_update(symbol, market_data)

