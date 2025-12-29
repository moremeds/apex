"""
IB Live Adapter for real-time data streaming.

Handles:
- Live quote streaming (implements QuoteProvider)
- Position data (implements PositionProvider)
- Account data (implements AccountProvider)

Uses reserved monitoring client ID.
"""

from __future__ import annotations
from collections import OrderedDict
from threading import Lock
from typing import List, Dict, Optional, Callable
from datetime import datetime
from math import isnan
import asyncio

from ....utils.logging_setup import get_logger
from ....domain.interfaces.event_bus import EventType
from ....domain.interfaces.quote_provider import QuoteProvider
from ....domain.interfaces.position_provider import PositionProvider
from ....domain.interfaces.account_provider import AccountProvider
from ....domain.events.domain_events import QuoteTick, PositionSnapshot, AccountSnapshot
from ....models.position import Position, AssetType
from ....models.market_data import MarketData
from ....models.account import AccountInfo

from ..market_data_fetcher import MarketDataFetcher
from .base import IbBaseAdapter
from .converters import convert_position


logger = get_logger(__name__)


class IbLiveAdapter(IbBaseAdapter, QuoteProvider, PositionProvider, AccountProvider):
    """
    IB adapter for live/streaming data.

    Implements:
    - QuoteProvider: Real-time quote streaming
    - PositionProvider: Position data and updates
    - AccountProvider: Account balances and margin

    Uses reserved monitoring client ID.
    """

    ADAPTER_TYPE = "live"

    def __init__(self, *args, **kwargs):
        """Initialize live adapter."""
        super().__init__(*args, **kwargs)

        # Market data - C11: Use OrderedDict for LRU eviction
        self._market_data_fetcher: Optional[MarketDataFetcher] = None
        self._market_data_cache: OrderedDict[str, MarketData] = OrderedDict()
        self._market_data_cache_max_size: int = 1000
        self._subscribed_symbols: List[str] = []
        self._quote_callback: Optional[Callable[[QuoteTick], None]] = None

        # Position cache
        self._position_cache: Optional[List[Position]] = None
        self._position_cache_time: Optional[datetime] = None
        self._position_cache_ttl_sec: int = 10
        self._position_callback: Optional[Callable[[List[PositionSnapshot]], None]] = None
        self._position_subscription_active: bool = False

        # Lock to prevent concurrent market data fetches
        self._market_data_fetch_lock: asyncio.Lock = asyncio.Lock()

        # Cache for qualified contracts - keyed by position symbol
        self._qualified_contract_cache: Dict[str, object] = {}
        self._contract_cache_lock = Lock()  # Protects cache from concurrent access

        # Account cache
        self._account_cache: Optional[AccountInfo] = None
        self._account_cache_time: Optional[datetime] = None
        self._account_cache_ttl_sec: int = 10
        self._account_callback: Optional[Callable[[AccountSnapshot], None]] = None

    # -------------------------------------------------------------------------
    # Cache Helpers
    # -------------------------------------------------------------------------

    def _update_market_data_cache(self, symbol: str, md: MarketData) -> None:
        """
        C11: Update market data cache with LRU eviction.
        """
        if symbol in self._market_data_cache:
            self._market_data_cache.move_to_end(symbol)
        self._market_data_cache[symbol] = md
        while len(self._market_data_cache) > self._market_data_cache_max_size:
            self._market_data_cache.popitem(last=False)

    # -------------------------------------------------------------------------
    # Connection Hooks
    # -------------------------------------------------------------------------

    async def _on_connected(self) -> None:
        """Set up market data fetcher after connection."""
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
        logger.debug("IbLiveAdapter: Market data fetcher initialized")

    async def _on_disconnecting(self) -> None:
        """Clean up before disconnect."""
        if self._market_data_fetcher:
            self._market_data_fetcher.cleanup()
            self._market_data_fetcher = None

        if self._position_subscription_active:
            self.unsubscribe_positions()

    # -------------------------------------------------------------------------
    # QuoteProvider Implementation
    # -------------------------------------------------------------------------

    async def subscribe_quotes(self, symbols: List[str]) -> None:
        """Subscribe to real-time quotes."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        self._subscribed_symbols.extend(symbols)
        logger.info(f"Subscribed to {len(symbols)} symbols")

    async def unsubscribe_quotes(self, symbols: List[str]) -> None:
        """Unsubscribe from quotes."""
        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                self._subscribed_symbols.remove(symbol)
        logger.info(f"Unsubscribed from {len(symbols)} symbols")

    def set_quote_callback(self, callback: Optional[Callable[[QuoteTick], None]]) -> None:
        """Set callback for incoming quotes."""
        self._quote_callback = callback

    def get_latest_quote(self, symbol: str) -> Optional[QuoteTick]:
        """Get latest cached quote."""
        md = self._market_data_cache.get(symbol)
        if md:
            return self._market_data_to_quote_tick(md)
        return None

    def get_all_quotes(self) -> Dict[str, QuoteTick]:
        """Get all cached quotes."""
        return {
            symbol: self._market_data_to_quote_tick(md)
            for symbol, md in self._market_data_cache.items()
        }

    def get_subscribed_symbols(self) -> List[str]:
        """Get subscribed symbols."""
        return self._subscribed_symbols.copy()

    async def fetch_snapshot(self, symbols: List[str]) -> Dict[str, QuoteTick]:
        """Fetch one-time quote snapshot."""
        market_data = await self.fetch_market_indicators(symbols)
        return {
            symbol: self._market_data_to_quote_tick(md)
            for symbol, md in market_data.items()
        }

    def _market_data_to_quote_tick(self, md: MarketData) -> QuoteTick:
        """Convert MarketData to QuoteTick domain event."""
        return QuoteTick(
            symbol=md.symbol,
            bid=md.bid,
            ask=md.ask,
            last=md.last,
            iv=md.iv,
            delta=md.delta,
            gamma=md.gamma,
            vega=md.vega,
            theta=md.theta,
            underlying_price=md.underlying_price,
            source="IB",
            timestamp=md.timestamp or datetime.now(),
        )

    def _handle_streaming_update(self, symbol: str, market_data: MarketData) -> None:
        """Handle streaming market data update."""
        self._update_market_data_cache(symbol, market_data)

        if self._quote_callback:
            quote_tick = self._market_data_to_quote_tick(market_data)
            self._quote_callback(quote_tick)

    # -------------------------------------------------------------------------
    # PositionProvider Implementation
    # -------------------------------------------------------------------------

    async def fetch_positions(self) -> List[PositionSnapshot]:
        """Fetch positions from IB."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        now = datetime.now()
        if (
            self._position_cache is not None
            and self._position_cache_time is not None
            and (now - self._position_cache_time).total_seconds() < self._position_cache_ttl_sec
        ):
            logger.debug("Using cached IB positions")
            return [self._position_to_snapshot(p) for p in self._position_cache]

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
                return [self._position_to_snapshot(p) for p in self._position_cache]
            raise

        return [self._position_to_snapshot(p) for p in positions]

    async def fetch_position(self, symbol: str) -> Optional[PositionSnapshot]:
        """Fetch specific position."""
        positions = await self.fetch_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def fetch_positions_by_underlying(self, underlying: str) -> List[PositionSnapshot]:
        """Fetch positions for underlying."""
        positions = await self.fetch_positions()
        return [p for p in positions if p.underlying == underlying]

    async def subscribe_positions(self) -> None:
        """Subscribe to position updates."""
        if not self.ib or not self._connected:
            logger.warning("Cannot subscribe to positions: not connected")
            return

        if self._position_subscription_active:
            return

        self.ib.positionEvent += self._on_ib_position_event
        self._position_subscription_active = True
        logger.info("IB position subscription enabled")

    def unsubscribe_positions(self) -> None:
        """Unsubscribe from positions."""
        if not self._position_subscription_active:
            return

        if self.ib:
            self.ib.positionEvent -= self._on_ib_position_event
        self._position_subscription_active = False
        logger.info("IB position subscription disabled")

    def set_position_callback(
        self,
        callback: Optional[Callable[[List[PositionSnapshot]], None]]
    ) -> None:
        """Set position update callback."""
        self._position_callback = callback

    def get_cached_positions(self) -> List[PositionSnapshot]:
        """Get cached positions."""
        if self._position_cache:
            return [self._position_to_snapshot(p) for p in self._position_cache]
        return []

    def get_position_count(self) -> int:
        """Get position count."""
        return len(self._position_cache) if self._position_cache else 0

    def get_positions_by_asset_type(self, asset_type: str) -> List[PositionSnapshot]:
        """Get positions by asset type."""
        if not self._position_cache:
            return []
        return [
            self._position_to_snapshot(p)
            for p in self._position_cache
            if p.asset_type.value == asset_type
        ]

    def _on_ib_position_event(self, position) -> None:
        """Handle IB position event."""
        if not self._position_callback:
            return

        try:
            converted = convert_position(position)
            if converted:
                self._position_cache = None
                self._position_cache_time = None
                snapshot = self._position_to_snapshot(converted)
                self._position_callback([snapshot])

        except Exception as e:
            logger.warning(f"Error processing IB position event: {e}")

    def _position_to_snapshot(self, pos: Position) -> PositionSnapshot:
        """Convert Position to PositionSnapshot domain event."""
        return PositionSnapshot(
            symbol=pos.symbol,
            underlying=pos.underlying,
            asset_type=pos.asset_type.value,
            quantity=pos.quantity,
            avg_price=pos.avg_price,
            multiplier=pos.multiplier,
            expiry=pos.expiry,
            strike=pos.strike,
            right=pos.right,
            days_to_expiry=pos.days_to_expiry(),
            source="IB",
            timestamp=pos.last_updated,
        )

    # -------------------------------------------------------------------------
    # AccountProvider Implementation
    # -------------------------------------------------------------------------

    async def fetch_account(self) -> AccountSnapshot:
        """Fetch account info from IB."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        now = datetime.now()
        if (
            self._account_cache is not None
            and self._account_cache_time is not None
            and (now - self._account_cache_time).total_seconds() < self._account_cache_ttl_sec
        ):
            return self._account_to_snapshot(self._account_cache)

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
                    return default

            account_info = AccountInfo(
                net_liquidation=safe_float('NetLiquidation'),
                total_cash=safe_float('TotalCashValue'),
                buying_power=safe_float('BuyingPower'),
                margin_used=safe_float('InitMarginReq'),
                margin_available=max(safe_float('ExcessLiquidity'), safe_float('AvailableFunds')),
                maintenance_margin=safe_float('MaintMarginReq'),
                init_margin_req=safe_float('InitMarginReq'),
                excess_liquidity=safe_float('ExcessLiquidity'),
                realized_pnl=safe_float('RealizedPnL'),
                unrealized_pnl=safe_float('UnrealizedPnL'),
                timestamp=datetime.now(),
                account_id=account_id,
            )

            self._account_cache = account_info
            self._account_cache_time = datetime.now()

            self.publish_event(EventType.ACCOUNT_UPDATED, {"account": account_info})

            return self._account_to_snapshot(account_info)

        except Exception as e:
            logger.error(f"Failed to fetch account info from IB: {e}")
            if self._account_cache:
                return self._account_to_snapshot(self._account_cache)
            raise

    async def fetch_accounts(self) -> Dict[str, AccountSnapshot]:
        """Fetch all accounts."""
        account = await self.fetch_account()
        return {account.account_id: account}

    async def subscribe_account(self) -> None:
        """Subscribe to account updates (not implemented for IB)."""
        logger.debug("IB account subscription not implemented - use polling")

    def unsubscribe_account(self) -> None:
        """Unsubscribe from account updates."""
        pass

    def set_account_callback(
        self,
        callback: Optional[Callable[[AccountSnapshot], None]]
    ) -> None:
        """Set account update callback."""
        self._account_callback = callback

    def get_cached_account(self) -> Optional[AccountSnapshot]:
        """Get cached account."""
        if self._account_cache:
            return self._account_to_snapshot(self._account_cache)
        return None

    def get_account_id(self) -> Optional[str]:
        """Get account ID."""
        return self._account_cache.account_id if self._account_cache else None

    def get_buying_power(self) -> float:
        """Get buying power."""
        return self._account_cache.buying_power if self._account_cache else 0.0

    def get_margin_utilization(self) -> float:
        """Get margin utilization."""
        if self._account_cache and self._account_cache.net_liquidation > 0:
            return (self._account_cache.margin_used / self._account_cache.net_liquidation) * 100
        return 0.0

    def _account_to_snapshot(self, acc: AccountInfo) -> AccountSnapshot:
        """Convert AccountInfo to AccountSnapshot domain event."""
        return AccountSnapshot(
            account_id=acc.account_id or "",
            net_liquidation=acc.net_liquidation,
            total_cash=acc.total_cash,
            buying_power=acc.buying_power,
            margin_used=acc.margin_used,
            margin_available=acc.margin_available,
            maintenance_margin=acc.maintenance_margin,
            init_margin_req=acc.init_margin_req,
            excess_liquidity=acc.excess_liquidity,
            realized_pnl=acc.realized_pnl,
            unrealized_pnl=acc.unrealized_pnl,
            source="IB",
            timestamp=acc.timestamp,
        )

    # -------------------------------------------------------------------------
    # Market Data Methods (Legacy compatibility)
    # -------------------------------------------------------------------------

    async def fetch_market_data(self, positions: List[Position]) -> List[MarketData]:
        """Fetch market data for positions (legacy method)."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        if not positions or not self._market_data_fetcher:
            return []

        # Use lock to prevent concurrent fetches (try_lock returns immediately if locked)
        if self._market_data_fetch_lock.locked():
            logger.debug("Market data fetch already in progress, returning cached data")
            return list(self._market_data_cache.values())

        async with self._market_data_fetch_lock:
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
                        expiry_str = self.format_expiry_for_ib(pos.expiry)
                        if not expiry_str:
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

            # Build pos_map for market data fetcher
            pos_map = {}

            # Add cached contracts to pos_map
            for qc, pos in zip(cached_qualified, cached_positions):
                pos_map[id(qc)] = pos

            # Qualify only NEW contracts (not in cache)
            newly_qualified = []
            if new_contracts:
                try:
                    logger.info(f"Qualifying {len(new_contracts)} new contracts...")
                    qualified_raw = await asyncio.wait_for(
                        self.ib.qualifyContractsAsync(*new_contracts),
                        timeout=30.0
                    )

                    failed_contracts = []
                    for i, qualified_contract in enumerate(qualified_raw):
                        if qualified_contract is not None:
                            newly_qualified.append(qualified_contract)
                            # Cache the qualified contract
                            symbol = new_positions[i].symbol
                            with self._contract_cache_lock:
                                self._qualified_contract_cache[symbol] = qualified_contract
                            # Add to pos_map
                            pos_map[id(qualified_contract)] = new_positions[i]
                        else:
                            # Track which contracts failed for detailed logging
                            pos = new_positions[i]
                            contract = new_contracts[i]
                            failed_contracts.append((pos.symbol, contract))

                    if failed_contracts:
                        # Log details of each failed contract (may be ambiguous or invalid)
                        for symbol, contract in failed_contracts:
                            logger.warning(
                                f"Contract qualification failed for {symbol}: {contract} "
                                f"(may be ambiguous - check IB for multiple matches)"
                            )
                        logger.warning(f"Failed to qualify {len(failed_contracts)}/{len(new_contracts)} contracts total")

                    logger.info(f"Qualified {len(newly_qualified)}/{len(new_contracts)} new contracts")
                except asyncio.TimeoutError:
                    logger.error(f"Timeout qualifying {len(new_contracts)} contracts after 30s")
                    # Continue with cached contracts if available
                except Exception as e:
                    logger.error(f"Error qualifying contracts: {e}")

            # Combine cached and newly qualified contracts
            qualified = cached_qualified + newly_qualified

            if not qualified:
                return []

            if cached_qualified:
                logger.debug(f"Using {len(cached_qualified)} cached + {len(newly_qualified)} new = {len(qualified)} total contracts")

            try:
                # Add timeout to market data fetch
                # OPT-004: Reduced from 60s - market data typically arrives in 5-10s
                market_data_list = await asyncio.wait_for(
                    self._market_data_fetcher.fetch_market_data(
                        positions, qualified, pos_map
                    ),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching market data after 15s")
                return []

            for md in market_data_list:
                self._update_market_data_cache(md.symbol, md)

            return market_data_list

    async def fetch_market_indicators(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Fetch market indicators."""
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
            qualified_raw = await self.ib.qualifyContractsAsync(*contracts)
            # Filter None contracts and track which symbols succeeded
            qualified_with_syms = [
                (sym, c) for sym, c in zip(symbols, qualified_raw) if c is not None
            ]
            if not qualified_with_syms:
                logger.warning("No contracts qualified for market indicators")
                return market_data

            valid_symbols, qualified = zip(*qualified_with_syms)
            tickers = await self.ib.reqTickersAsync(*qualified)

            for sym, ticker in zip(valid_symbols, tickers):
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
                    logger.warning(f"Failed to parse data for {sym}: {e}")

        except Exception as e:
            logger.error(f"Error fetching market indicators: {e}")

        return market_data

    def enable_streaming(self) -> None:
        """Enable streaming."""
        if self._market_data_fetcher:
            self._market_data_fetcher.enable_streaming()

    def disable_streaming(self) -> None:
        """Disable streaming."""
        if self._market_data_fetcher:
            self._market_data_fetcher.disable_streaming()

    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data (legacy)."""
        return self._market_data_cache.get(symbol)
