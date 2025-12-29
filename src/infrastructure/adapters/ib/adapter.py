"""
Interactive Brokers adapter with auto-reconnect.

Implements BrokerAdapter and MarketDataProvider interfaces for IBKR TWS/Gateway.
"""

from __future__ import annotations
import asyncio
from collections import OrderedDict
from threading import Lock
from typing import List, Dict, Optional, Callable
from datetime import datetime, date
from math import isnan

from ....domain.events import PriorityEventBus
from ....utils.logging_setup import get_logger
from ....domain.interfaces.broker_adapter import BrokerAdapter
from ....domain.interfaces.market_data_provider import MarketDataProvider
from ....domain.interfaces.event_bus import EventBus, EventType
from ....models.position import Position, AssetType
from ....models.market_data import MarketData
from ....models.account import AccountInfo
from ....models.order import Order, Trade

from ..market_data_fetcher import MarketDataFetcher
from .converters import convert_position, convert_order, convert_fill


logger = get_logger(__name__)


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
        event_bus: Optional[PriorityEventBus] = None,
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
            event_bus: Optional priority event bus for publishing events.
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
        # C11: Use OrderedDict for LRU eviction with bounded size
        self._market_data_cache: OrderedDict[str, MarketData] = OrderedDict()
        self._market_data_cache_max_size: int = 1000  # C11: Prevent unbounded growth
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

        # Position subscription callback
        self._on_position_update: Optional[Callable[[str, List[Position]], None]] = None
        self._position_subscription_active: bool = False

        # Guard to prevent concurrent market data fetches
        self._market_data_fetch_in_progress: bool = False

        # Cache for qualified contracts - keyed by position symbol
        # Avoids re-qualifying same contracts every fetch cycle
        self._qualified_contract_cache: Dict[str, object] = {}
        self._contract_cache_lock = Lock()  # Protects cache from concurrent access

        # Contract qualification retry settings
        self._qualify_max_retries: int = 3
        self._qualify_retry_delay: float = 1.0  # seconds
        self._qualify_retry_backoff: float = 2.0

        # IB maintenance window detection (ET timezone)
        # IB typically has maintenance 23:45-00:45 ET
        self._maintenance_start_hour: int = 23
        self._maintenance_start_minute: int = 45
        self._maintenance_end_hour: int = 0
        self._maintenance_end_minute: int = 45

    # -------------------------------------------------------------------------
    # Cache Helpers
    # -------------------------------------------------------------------------

    def _update_market_data_cache(self, symbol: str, md: MarketData) -> None:
        """
        C11: Update market data cache with LRU eviction.

        Maintains bounded cache size to prevent memory leaks during long sessions.
        """
        # Move existing entry to end (LRU) or add new
        if symbol in self._market_data_cache:
            self._market_data_cache.move_to_end(symbol)
        self._market_data_cache[symbol] = md

        # Evict oldest entries if over limit
        while len(self._market_data_cache) > self._market_data_cache_max_size:
            self._market_data_cache.popitem(last=False)

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
            # M4: Set event loop for non-blocking callback dispatch
            try:
                loop = asyncio.get_running_loop()
                self._market_data_fetcher.set_event_loop(loop)
            except RuntimeError:
                pass  # No running loop, will use sync dispatch
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

    def _is_ib_maintenance_window(self) -> bool:
        """
        Check if current time is within IB maintenance window.

        IB typically has scheduled maintenance from 23:45 to 00:45 ET.
        During this window, contract qualification often fails.

        Returns:
            True if within maintenance window, False otherwise.
        """
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            now_et = datetime.now(et)
            hour, minute = now_et.hour, now_et.minute

            # Check if in maintenance window (23:45 - 00:45 ET)
            if hour == self._maintenance_start_hour and minute >= self._maintenance_start_minute:
                return True
            if hour == self._maintenance_end_hour and minute <= self._maintenance_end_minute:
                return True
            return False
        except (ImportError, KeyError) as e:
            # M16: Specific exceptions - ImportError if zoneinfo missing, KeyError if tz data missing
            logger.debug(f"Timezone detection failed, assuming not in maintenance: {e}")
            return False

    async def _qualify_contracts_with_retry(
        self,
        contracts: List,
        positions: List
    ) -> List:
        """
        Qualify contracts with retry logic and graceful degradation.

        Handles IB maintenance window and transient failures gracefully.

        Args:
            contracts: List of IB contracts to qualify.
            positions: Corresponding positions (for logging and cache).

        Returns:
            List of qualified contracts (may be partial on failure).
        """
        import asyncio

        if not contracts:
            return []

        # Check if in maintenance window - warn and use shorter timeout
        in_maintenance = self._is_ib_maintenance_window()
        if in_maintenance:
            logger.warning("IB maintenance window detected - contract qualification may be degraded")

        qualified = []
        retry_delay = self._qualify_retry_delay
        last_error = None

        for attempt in range(self._qualify_max_retries):
            try:
                # Use shorter timeout during maintenance window (C2: removed global)
                qualify_timeout = 15.0 if in_maintenance else 30.0

                qualified_raw = await asyncio.wait_for(
                    self.ib.qualifyContractsAsync(*contracts),
                    timeout=qualify_timeout
                )

                # Filter out None results and cache successful qualifications
                failed_contracts = []
                for i, qc in enumerate(qualified_raw):
                    if qc is not None:
                        qualified.append(qc)
                        if i < len(positions):
                            with self._contract_cache_lock:
                                self._qualified_contract_cache[positions[i].symbol] = qc
                    else:
                        # Track failed contracts for detailed logging
                        if i < len(positions):
                            failed_contracts.append((positions[i].symbol, contracts[i]))
                        else:
                            failed_contracts.append((f"contract_{i}", contracts[i]))

                # Log details of failed contracts (may be ambiguous or invalid)
                if failed_contracts:
                    for symbol, contract in failed_contracts:
                        logger.warning(
                            f"Contract qualification failed for {symbol}: {contract} "
                            f"(may be ambiguous - check IB for multiple matches)"
                        )

                if qualified:
                    if attempt > 0:
                        logger.info(f"Contract qualification succeeded on retry {attempt + 1}")
                    return qualified

            except asyncio.TimeoutError:
                last_error = f"Timeout after {qualify_timeout}s"
                logger.warning(f"Contract qualification timeout (attempt {attempt + 1}/{self._qualify_max_retries})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Contract qualification failed (attempt {attempt + 1}/{self._qualify_max_retries}): {e}")

            # Don't retry if in maintenance window (it will likely fail anyway)
            if in_maintenance:
                logger.info("Skipping retries during IB maintenance window")
                break

            # Exponential backoff before retry
            if attempt < self._qualify_max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= self._qualify_retry_backoff

        # All retries exhausted - log final error
        failed_count = len(contracts) - len(qualified)
        if failed_count > 0:
            msg = f"Failed to qualify {failed_count}/{len(contracts)} contracts"
            if in_maintenance:
                msg += " (IB maintenance window)"
            else:
                msg += f" after {self._qualify_max_retries} attempts: {last_error}"
            logger.error(msg)

        return qualified

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
                # C2: removed global, use local variable
                try:
                    raw_value = account_data.get(tag, default)
                    return float(raw_value) if raw_value else default
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse account tag '{tag}': {account_data.get(tag)}")
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
        import asyncio

        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        if not positions:
            return []

        # Guard against concurrent fetches - return cached data if fetch in progress
        if self._market_data_fetch_in_progress:
            logger.debug("Market data fetch already in progress, returning cached data")
            return list(self._market_data_cache.values())

        self._market_data_fetch_in_progress = True
        market_data_list = []

        try:
            from ib_async import Stock, Option

            # Separate positions into cached (already qualified) and new (need qualification)
            cached_qualified = []
            cached_positions = []
            new_contracts = []
            new_positions = []

            for pos in positions:
                # Check if we have a cached qualified contract
                with self._contract_cache_lock:
                    cached_contract = self._qualified_contract_cache.get(pos.symbol)
                if cached_contract is not None:
                    cached_qualified.append(cached_contract)
                    cached_positions.append(pos)
                    continue

                # Need to build and qualify this contract
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

                    new_contracts.append(contract)
                    new_positions.append(pos)
                except Exception as e:
                    logger.warning(f"Failed to create contract for {pos.symbol}: {e}")
                    continue

            # Build pos_map for market data fetcher - map contract id to position
            pos_map = {}

            # Add cached contracts to pos_map
            for qc, pos in zip(cached_qualified, cached_positions):
                pos_map[id(qc)] = pos

            # Qualify only NEW contracts (not in cache) - with retry logic
            newly_qualified = []
            if new_contracts:
                logger.info(f"Qualifying {len(new_contracts)} new contracts...")

                # Use retry-capable qualification (handles IB maintenance window)
                newly_qualified = await self._qualify_contracts_with_retry(
                    new_contracts, new_positions
                )

                # M9: Add successfully qualified contracts to pos_map
                # Copy cache snapshot under lock, then match outside to reduce contention
                with self._contract_cache_lock:
                    cache_snapshot = dict(self._qualified_contract_cache)

                # Match outside the lock (O(n) instead of O(n*m))
                symbol_to_pos = {pos.symbol: pos for pos in new_positions}
                for qc in newly_qualified:
                    for symbol, cached in cache_snapshot.items():
                        if cached == qc and symbol in symbol_to_pos:
                            pos_map[id(qc)] = symbol_to_pos[symbol]
                            break

                logger.info(f"Qualified {len(newly_qualified)}/{len(new_contracts)} new contracts")

            # Combine cached and newly qualified contracts
            qualified = cached_qualified + newly_qualified

            if cached_qualified and not newly_qualified:
                logger.debug(f"Using {len(cached_qualified)} cached contracts (no new contracts to qualify)")
            elif cached_qualified:
                logger.info(f"Fetching market data: {len(cached_qualified)} cached + {len(newly_qualified)} new = {len(qualified)} total")
            else:
                logger.info(f"Fetching market data for {len(qualified)} contracts")

            if qualified and self._market_data_fetcher:
                try:
                    # OPT-004: Reduced from 60s - market data typically arrives in 5-10s
                    market_data_list = await asyncio.wait_for(
                        self._market_data_fetcher.fetch_market_data(
                            positions, qualified, pos_map
                        ),
                        timeout=15.0
                    )

                    for md in market_data_list:
                        self._update_market_data_cache(md.symbol, md)

                except asyncio.TimeoutError:
                    logger.error(f"Timeout fetching market data after 15s")
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
        finally:
            self._market_data_fetch_in_progress = False

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
                    self._update_market_data_cache(sym, md)
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
        """
        Handle streaming market data update from fetcher.

        Note: EventBus publishing moved to MarketDataManager (Phase 4 single streaming path).
        IbAdapter only updates cache and forwards to callback.
        """
        self._update_market_data_cache(symbol, market_data)

        # Forward to callback (MarketDataManager handles EventBus publish)
        if self._on_market_data_update:
            self._on_market_data_update(symbol, market_data)

    # -------------------------------------------------------------------------
    # Position Subscription (Phase 6)
    # -------------------------------------------------------------------------

    def set_position_callback(
        self,
        callback: Callable[[List[Position]], None]
    ) -> None:
        """
        Set callback for position updates.

        Args:
            callback: Function called with positions list on position change.
        """
        self._on_position_update = callback

    async def subscribe_positions(self) -> None:
        """
        Subscribe to position updates via IB positionEvent.

        When positions change, the callback set via set_position_callback() is invoked.
        """
        if not self.ib or not self._connected:
            logger.warning("Cannot subscribe to positions: not connected")
            return

        if self._position_subscription_active:
            logger.debug("Position subscription already active")
            return

        # Subscribe to IB position events
        self.ib.positionEvent += self._on_ib_position_event
        self._position_subscription_active = True
        logger.info("IB position subscription enabled")

    def unsubscribe_positions(self) -> None:
        """Unsubscribe from position updates."""
        if not self._position_subscription_active:
            return

        if self.ib:
            self.ib.positionEvent -= self._on_ib_position_event

        self._position_subscription_active = False
        logger.info("IB position subscription disabled")

    def _on_ib_position_event(self, position) -> None:
        """
        Handle IB position event.

        Called by ib_async when a position changes.
        Converts and forwards to the callback.
        """
        if not self._on_position_update:
            return

        try:
            # Convert single IB position to our model
            converted = convert_position(position)
            if converted:
                # Invalidate position cache since something changed
                self._position_cache = None
                self._position_cache_time = None

                # Forward to callback (BrokerManager wraps to add broker name)
                # Note: IB sends individual position updates, callback expects list
                self._on_position_update([converted])
                logger.debug(f"Position update: {converted.symbol}")

        except Exception as e:
            logger.warning(f"Error processing IB position event: {e}")
