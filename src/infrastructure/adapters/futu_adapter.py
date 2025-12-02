"""
Futu OpenD adapter with auto-reconnect and push subscription.

Implements PositionProvider interface for Futu OpenD gateway.
Uses the futu-api SDK to connect to Futu OpenD and fetch positions/account info.
Supports real-time position updates via trade deal push notifications.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
import logging
import re
import threading

from ...domain.interfaces.position_provider import PositionProvider
from ...domain.interfaces.event_bus import EventBus, EventType
from ...models.position import Position, AssetType, PositionSource
from ...models.account import AccountInfo


logger = logging.getLogger(__name__)


def create_trade_deal_handler(on_deal_callback: Callable[[dict], None]):
    """
    Factory function to create a Futu trade deal handler.

    Creates a handler class that inherits from Futu's TradeDealHandlerBase
    to receive real-time trade deal notifications.

    Args:
        on_deal_callback: Callback function to invoke when a deal is received.

    Returns:
        Handler instance or None if futu library not available.
    """
    try:
        from futu import TradeDealHandlerBase

        class FutuTradeDealHandler(TradeDealHandlerBase):
            """
            Handler for Futu trade deal push notifications.

            Inherits from Futu SDK's TradeDealHandlerBase to receive
            real-time notifications when trades are executed.
            """

            def __init__(self, callback: Callable[[dict], None]):
                super().__init__()
                self._callback = callback
                self._lock = threading.Lock()

            def on_recv_rsp(self, rsp_str):
                """
                Called by Futu SDK when a trade deal notification is received.

                Args:
                    rsp_str: Response string/data from Futu SDK containing deal info.
                """
                try:
                    import pandas as pd

                    # rsp_str is typically a tuple (ret_code, data) from Futu SDK
                    # where data is a DataFrame with deal information
                    if isinstance(rsp_str, tuple) and len(rsp_str) >= 2:
                        ret_code, data = rsp_str[0], rsp_str[1]
                        if ret_code != 0:
                            logger.warning(f"Futu trade deal notification error: {data}")
                            return

                        # data is a pandas DataFrame with deal info
                        if isinstance(data, pd.DataFrame) and not data.empty:
                            for _, row in data.iterrows():
                                deal_data = {
                                    'deal_id': row.get('deal_id', None),
                                    'order_id': row.get('order_id', None),
                                    'code': row.get('code', None),
                                    'stock_name': row.get('stock_name', None),
                                    'qty': float(row.get('qty', 0)),
                                    'price': float(row.get('price', 0)),
                                    'trd_side': row.get('trd_side', None),
                                    'create_time': row.get('create_time', None),
                                }
                                logger.info(
                                    f"Futu trade deal received: {deal_data['code']} "
                                    f"qty={deal_data['qty']} price={deal_data['price']} "
                                    f"side={deal_data['trd_side']}"
                                )

                                with self._lock:
                                    if self._callback:
                                        self._callback(deal_data)

                except Exception as e:
                    logger.error(f"Error processing Futu trade deal notification: {e}")

        return FutuTradeDealHandler(on_deal_callback)

    except ImportError:
        logger.warning("futu-api library not installed, trade deal handler unavailable")
        return None


class FutuAdapter(PositionProvider):
    """
    Futu OpenD adapter with auto-reconnect.

    Implements PositionProvider using futu-api SDK.
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
        # With both positions and account queries, need at least 6 seconds between each type
        # Using 10 seconds to have safety margin
        self._account_cache: Optional[AccountInfo] = None
        self._account_cache_time: Optional[datetime] = None
        self._account_cache_ttl_sec: int = 10  # Cache for 10 seconds to avoid rate limits

        # Cache for positions (Futu rate limit: 10 calls per 30 seconds)
        self._position_cache: Optional[List[Position]] = None
        self._position_cache_time: Optional[datetime] = None
        self._position_cache_ttl_sec: int = 30  # Cache for 30 seconds (Futu limit: 10 calls/30s)
        # Cooldown after hitting rate limits to avoid hammering OpenD
        self._position_cooldown_until: Optional[datetime] = None

        # Push subscription for real-time trade updates
        self._trade_deal_handler = None
        self._push_enabled = False
        self._cache_lock = threading.Lock()  # Thread safety for cache updates from push

    async def connect(self) -> None:
        """
        Connect to Futu OpenD gateway.

        Raises:
            ConnectionError: If unable to connect.
        """
        try:
            from futu import (
                OpenSecTradeContext,
                TrdMarket,
                SecurityFirm,
                TrdEnv,
                RET_OK,
            )

            # Map string config to Futu enums
            trd_market = getattr(TrdMarket, self.filter_trdmarket, TrdMarket.US)
            sec_firm = getattr(SecurityFirm, self.security_firm, SecurityFirm.FUTUSECURITIES)

            self._trd_ctx = OpenSecTradeContext(
                filter_trdmarket=trd_market,
                host=self.host,
                port=self.port,
                security_firm=sec_firm,
            )

            # Get account list to verify connection and select account
            ret, data = self._trd_ctx.get_acc_list()
            if ret != RET_OK:
                raise ConnectionError(f"Failed to get account list: {data}")

            if data.empty:
                raise ConnectionError("No trading accounts found")

            # Select the appropriate account based on trd_env
            trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)
            matching_accounts = data[data["trd_env"] == trd_env_enum]

            if matching_accounts.empty:
                # Fall back to first account
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

            # Set up push subscription for real-time trade deal updates
            self._setup_push_subscription()

        except ImportError:
            logger.error("futu-api library not installed. Install with: pip install futu-api")
            raise ConnectionError("futu-api library not installed")
        except Exception as e:
            logger.error(f"Failed to connect to Futu OpenD at {self.host}:{self.port}: {e}")
            logger.info("Make sure Futu OpenD is running and accessible")
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
        """
        Check if connected to Futu OpenD.

        Note: Futu uses a request-response pattern where connections may
        close between calls. We track logical connection state rather than
        physical connection state.
        """
        # We track logical connection - if we've successfully connected once
        # and haven't had a fatal error, we're "connected"
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
        """
        Set up push subscription for real-time trade deal notifications.

        This allows immediate cache invalidation when trades execute,
        ensuring positions are refreshed on the next fetch.
        """
        if not self._trd_ctx:
            logger.warning("Cannot set up push subscription: no trade context")
            return

        try:
            # Create the trade deal handler with callback
            self._trade_deal_handler = create_trade_deal_handler(self._on_trade_deal)

            if self._trade_deal_handler:
                # Register the handler with the trade context
                self._trd_ctx.set_handler(self._trade_deal_handler)
                self._push_enabled = True
                logger.info("Futu trade deal push subscription enabled")
            else:
                logger.warning("Failed to create trade deal handler")

        except Exception as e:
            logger.error(f"Failed to set up Futu push subscription: {e}")
            self._push_enabled = False

    def _on_trade_deal(self, deal_data: dict) -> None:
        """
        Callback invoked when a trade deal is received via push.

        Invalidates the position cache, emits event, and forces a fresh fetch.
        This is called from Futu's internal thread, so we use the cache lock.

        Args:
            deal_data: Dictionary containing trade deal information.
        """
        try:
            code = deal_data.get('code', 'unknown')
            qty = deal_data.get('qty', 0)
            trd_side = deal_data.get('trd_side', 'unknown')

            logger.info(
                f"Trade deal notification: {code} qty={qty} side={trd_side} - "
                "invalidating position cache"
            )

            # Emit event for trade deal (position update signal)
            if self._event_bus:
                self._event_bus.publish(EventType.POSITION_UPDATED, {
                    "symbol": code,
                    "deal": deal_data,
                    "source": "FUTU",
                    "timestamp": datetime.now(),
                })

            # Invalidate position cache to force refresh on next fetch
            with self._cache_lock:
                self._position_cache = None
                self._position_cache_time = None
                # Clear any cooldown since we need fresh data after a trade
                self._position_cooldown_until = None

            logger.debug("Position cache invalidated due to trade deal")

        except Exception as e:
            logger.error(f"Error handling trade deal notification: {e}")

    async def fetch_positions(self) -> List[Position]:
        """
        Fetch positions from Futu OpenD.

        Returns:
            List of Position objects with source=FUTU.

        Raises:
            ConnectionError: If not connected.
        """
        now = datetime.now()

        # Check cache with lock (cache may be invalidated by push handler)
        with self._cache_lock:
            # Respect cooldown if we recently hit the OpenD rate limit
            if self._position_cooldown_until and now < self._position_cooldown_until:
                logger.warning(
                    "Futu position fetch skipped due to rate-limit cooldown "
                    f"(retry after {self._position_cooldown_until.isoformat(timespec='seconds')})"
                )
                if (
                    self._position_cache is not None
                    and self._position_cache_time is not None
                    and (now - self._position_cache_time).total_seconds() < self._position_cache_ttl_sec
                ):
                    logger.debug("Returning cached Futu positions during cooldown")
                    return self._position_cache
                return []

            # Check cache first (Futu rate limit: 10 calls per 30 seconds)
            if (
                self._position_cache is not None
                and self._position_cache_time is not None
                and (now - self._position_cache_time).total_seconds() < self._position_cache_ttl_sec
            ):
                logger.debug("Using cached Futu positions")
                return self._position_cache

        # Auto-reconnect if needed
        await self._ensure_connected()

        from futu import RET_OK, TrdEnv

        positions = []
        try:
            trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)

            # Use refresh_cache=False to use Futu's internal cache
            # Futu rate limit is strict (10 calls/30s), so we rely on their cache
            # Our app-level cache (10s TTL) controls how often we call the API
            ret, data = self._trd_ctx.position_list_query(
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                refresh_cache=False,
            )

            if ret != RET_OK:
                # Check if it's a connection error and try to reconnect
                if "disconnect" in str(data).lower() or "connection" in str(data).lower():
                    logger.warning("Futu connection issue detected, reconnecting...")
                    self._connected = False
                    await self._ensure_connected()
                    # Retry once
                    ret, data = self._trd_ctx.position_list_query(
                        trd_env=trd_env_enum,
                        acc_id=self._acc_id,
                        refresh_cache=False,
                    )
                    if ret != RET_OK:
                        raise Exception(f"Position query failed after reconnect: {data}")
                else:
                    # Don't log rate limit errors at error level - they're handled below
                    if "frequent" not in str(data).lower():
                        logger.error(f"Failed to fetch positions from Futu: {data}")
                    raise Exception(f"Position query failed: {data}")

            if data.empty:
                logger.debug("No positions found in Futu account")
                return []

            for _, row in data.iterrows():
                position = self._convert_futu_position(row)
                if position:
                    positions.append(position)

            logger.debug(f"Fetched {len(positions)} positions from Futu")
            # Mark as connected since operation succeeded
            self._connected = True

            # Update cache with lock (cache may be accessed by push handler)
            with self._cache_lock:
                self._position_cache = positions
                self._position_cache_time = datetime.now()
                # Clear any prior cooldown after a successful fetch
                self._position_cooldown_until = None

            # NOTE: Do NOT publish POSITIONS_BATCH here.
            # The orchestrator publishes after reconciliation to ensure single data path.
            # Publishing here would cause store/RiskEngine to process raw adapter data
            # before reconciliation, leading to transient inconsistent snapshots.

        except Exception as e:
            # Only mark disconnected on connection-related errors
            if "disconnect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Failed to fetch positions from Futu: {e}")
                self._connected = False

            # If rate limited and we have cached data, return it instead of failing
            if "frequent" in str(e).lower():
                # Back off for the full 30-second Futu window to avoid hammering
                cooldown_seconds = 30
                with self._cache_lock:
                    self._position_cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
                logger.warning(
                    f"Futu rate limit hit; backing off for {cooldown_seconds}s "
                    f"(until {self._position_cooldown_until.isoformat(timespec='seconds')})"
                )
                with self._cache_lock:
                    if self._position_cache is not None:
                        logger.debug("Rate limited - returning cached positions")
                        return self._position_cache
                # No cache available, log at warning level
                logger.warning("Futu rate limited and no cached positions available")
            else:
                logger.error(f"Failed to fetch positions from Futu: {e}")
            raise

        return positions

    def _convert_futu_position(self, row) -> Optional[Position]:
        """
        Convert Futu position row to internal Position model.

        Args:
            row: pandas DataFrame row from position_list_query.

        Returns:
            Position object or None if conversion fails.
        """
        try:
            code = row.get("code", "")
            stock_name = row.get("stock_name", "")
            qty = float(row.get("qty", 0))

            if qty == 0:
                return None

            # Parse the Futu code format (e.g., "US.AAPL", "US.AAPL240119C190000")
            asset_type, symbol, underlying, expiry, strike, right = self._parse_futu_code(code)

            # Get cost and market values
            avg_price = float(row.get("cost_price", 0) or row.get("average_cost", 0) or 0)

            return Position(
                symbol=symbol,
                underlying=underlying,
                asset_type=asset_type,
                quantity=qty,
                avg_price=avg_price,
                multiplier=100 if asset_type == AssetType.OPTION else 1,
                expiry=expiry,
                strike=strike,
                right=right,
                source=PositionSource.FUTU,
                last_updated=datetime.now(),
                account_id=str(self._acc_id) if self._acc_id else None,
            )

        except Exception as e:
            logger.warning(f"Failed to convert Futu position: {e}, row={row.to_dict()}")
            return None

    def _parse_futu_code(self, code: str) -> tuple:
        """
        Parse Futu security code to extract asset details.

        Futu code formats:
        - Stock: "US.AAPL", "HK.00700"
        - Option: "US.AAPL240119C190000" (underlying + YYMMDD + C/P + strike*1000)

        Args:
            code: Futu security code string.

        Returns:
            Tuple of (asset_type, symbol, underlying, expiry, strike, right)
        """
        # Remove market prefix (e.g., "US.", "HK.")
        if "." in code:
            market, ticker = code.split(".", 1)
        else:
            ticker = code

        # Check if it's an option (has date and C/P in the format)
        # Option format: SYMBOL + YYMMDD + C/P + STRIKE*1000
        # Example: AAPL240119C190000 (AAPL Jan 19, 2024 Call $190)
        option_pattern = r"^([A-Z]+)(\d{6})([CP])(\d+)$"
        match = re.match(option_pattern, ticker)

        if match:
            underlying = match.group(1)
            date_str = match.group(2)  # YYMMDD
            right = match.group(3)  # C or P
            strike_raw = match.group(4)

            # Convert YYMMDD to YYYYMMDD
            year = int(date_str[:2])
            year_full = 2000 + year if year < 50 else 1900 + year
            expiry = f"{year_full}{date_str[2:]}"

            # Strike is stored as strike * 1000
            strike = float(strike_raw) / 1000.0

            return (
                AssetType.OPTION,
                ticker,  # Full option symbol
                underlying,
                expiry,
                strike,
                right,
            )
        else:
            # It's a stock
            return (
                AssetType.STOCK,
                ticker,
                ticker,
                None,
                None,
                None,
            )

    async def fetch_account_info(self) -> AccountInfo:
        """
        Fetch account information from Futu OpenD.

        Returns:
            AccountInfo object with account data.

        Raises:
            ConnectionError: If not connected.
            Exception: Any error encountered during fetch/parsing.
        """
        # Check cache first (Futu rate limit: 10 calls per 30 seconds)
        now = datetime.now()
        if (
            self._account_cache is not None
            and self._account_cache_time is not None
            and (now - self._account_cache_time).total_seconds() < self._account_cache_ttl_sec
        ):
            logger.debug("Using cached Futu account info")
            return self._account_cache

        # Auto-reconnect if needed
        await self._ensure_connected()

        from futu import RET_OK, TrdEnv, Currency

        try:
            trd_env_enum = getattr(TrdEnv, self.trd_env, TrdEnv.REAL)

            # Use refresh_cache=False to use Futu's internal cache
            # This avoids hitting Futu's rate limit (10 calls per 30 seconds)
            # Our application-level cache (10s TTL) controls refresh frequency
            ret, data = self._trd_ctx.accinfo_query(
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                refresh_cache=False,
                currency=Currency.USD,  # Request in USD for consistency
            )

            if ret != RET_OK:
                # Check if it's a connection error and try to reconnect
                if "disconnect" in str(data).lower() or "connection" in str(data).lower():
                    logger.warning("Futu connection issue detected, reconnecting...")
                    self._connected = False
                    await self._ensure_connected()
                    # Retry once
                    ret, data = self._trd_ctx.accinfo_query(
                        trd_env=trd_env_enum,
                        acc_id=self._acc_id,
                        refresh_cache=False,
                        currency=Currency.USD,
                    )
                    if ret != RET_OK:
                        raise Exception(f"Account info query failed after reconnect: {data}")
                else:
                    # Don't log rate limit errors at error level - they're handled below
                    if "frequent" not in str(data).lower():
                        logger.error(f"Failed to fetch account info from Futu: {data}")
                    raise Exception(f"Account info query failed: {data}")

            if data.empty:
                raise Exception("No account info returned")

            row = data.iloc[0]

            # Helper function to safely parse float values
            def safe_float(key: str, default: float = 0.0) -> float:
                try:
                    value = row.get(key)
                    if value is None or (isinstance(value, float) and value != value):  # NaN check
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default

            # Extract key account metrics
            # Futu field names from documentation
            net_liquidation = safe_float("total_assets")
            total_cash = safe_float("cash")
            buying_power = safe_float("power")

            # Margin-related fields
            maintenance_margin = safe_float("maintenance_margin", 0.0)
            init_margin_req = safe_float("initial_margin", 0.0)

            # Calculate margin used and available
            margin_used = init_margin_req
            margin_available = safe_float("available_funds", buying_power)

            # Excess liquidity / risk level
            excess_liquidity = safe_float("risk_level", 0.0)  # May need adjustment

            # P&L fields
            realized_pnl = safe_float("realized_pl", 0.0)
            unrealized_pnl = safe_float("unrealized_pl", 0.0)

            logger.debug(
                f"Fetched Futu account info: TotalAssets=${net_liquidation:,.2f}, "
                f"BuyingPower=${buying_power:,.2f}, Cash=${total_cash:,.2f}"
            )
            # Mark as connected since operation succeeded
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

            # Update cache
            self._account_cache = account_info
            self._account_cache_time = datetime.now()

            # Emit event
            if self._event_bus:
                self._event_bus.publish(EventType.ACCOUNT_UPDATED, {
                    "account": account_info,
                    "source": "FUTU",
                    "timestamp": datetime.now(),
                })

            return account_info

        except Exception as e:
            # Only mark disconnected on connection-related errors
            if "disconnect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Failed to fetch account info from Futu: {e}")
                self._connected = False

            # If rate limited and we have cached data, return it instead of failing
            if "frequent" in str(e).lower():
                if self._account_cache is not None:
                    logger.debug("Rate limited - returning cached account info")
                    return self._account_cache
                # No cache available, log at warning level
                logger.warning(f"Futu rate limited and no cached account info available")
            else:
                logger.error(f"Failed to fetch account info from Futu: {e}")
            raise
