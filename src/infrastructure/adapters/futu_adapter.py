"""
Futu OpenD adapter with auto-reconnect.

Implements PositionProvider interface for Futu OpenD gateway.
Uses the futu-api SDK to connect to Futu OpenD and fetch positions/account info.
"""

from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime
import logging
import re

from ...domain.interfaces.position_provider import PositionProvider
from ...models.position import Position, AssetType, PositionSource
from ...models.account import AccountInfo


logger = logging.getLogger(__name__)


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

        # Cache for account info (Futu rate limit: 10 calls per 30 seconds)
        self._account_cache: Optional[AccountInfo] = None
        self._account_cache_time: Optional[datetime] = None
        self._account_cache_ttl_sec: int = 5  # Cache for 5 seconds to avoid rate limits

        # Cache for positions (Futu rate limit: 10 calls per 30 seconds)
        self._position_cache: Optional[List[Position]] = None
        self._position_cache_time: Optional[datetime] = None
        self._position_cache_ttl_sec: int = 5  # Cache for 5 seconds to avoid rate limits

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

    async def fetch_positions(self) -> List[Position]:
        """
        Fetch positions from Futu OpenD.

        Returns:
            List of Position objects with source=FUTU.

        Raises:
            ConnectionError: If not connected.
        """
        # Check cache first (Futu rate limit: 10 calls per 30 seconds)
        now = datetime.now()
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

            ret, data = self._trd_ctx.position_list_query(
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                refresh_cache=True,
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
                        refresh_cache=True,
                    )
                    if ret != RET_OK:
                        raise Exception(f"Position query failed after reconnect: {data}")
                else:
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

            # Update cache
            self._position_cache = positions
            self._position_cache_time = datetime.now()

        except Exception as e:
            logger.error(f"Failed to fetch positions from Futu: {e}")
            # Only mark disconnected on connection-related errors
            if "disconnect" in str(e).lower() or "connection" in str(e).lower():
                self._connected = False

            # If rate limited and we have cached data, return it instead of failing
            if "frequent" in str(e).lower() and self._position_cache is not None:
                logger.warning("Rate limited - returning cached positions")
                return self._position_cache
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

            ret, data = self._trd_ctx.accinfo_query(
                trd_env=trd_env_enum,
                acc_id=self._acc_id,
                refresh_cache=True,
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
                        refresh_cache=True,
                        currency=Currency.USD,
                    )
                    if ret != RET_OK:
                        raise Exception(f"Account info query failed after reconnect: {data}")
                else:
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

            return account_info

        except Exception as e:
            logger.error(f"Failed to fetch account info from Futu: {e}")
            # Only mark disconnected on connection-related errors
            if "disconnect" in str(e).lower() or "connection" in str(e).lower():
                self._connected = False

            # If rate limited and we have cached data, return it instead of failing
            if "frequent" in str(e).lower() and self._account_cache is not None:
                logger.warning("Rate limited - returning cached account info")
                return self._account_cache
            raise
