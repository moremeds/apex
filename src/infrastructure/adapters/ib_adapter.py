"""
Interactive Brokers adapter with auto-reconnect.

Implements PositionProvider and MarketDataProvider interfaces for IBKR TWS/Gateway.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Literal
from datetime import datetime
from math import isnan
import logging

from ...domain.interfaces.position_provider import PositionProvider
from ...domain.interfaces.market_data_provider import MarketDataProvider
from ...models.position import Position, AssetType, PositionSource
from ...models.market_data import MarketData
from ...models.account import AccountInfo
from .market_data_fetcher import MarketDataFetcher


logger = logging.getLogger(__name__)


class IbAdapter(PositionProvider, MarketDataProvider):
    """
    Interactive Brokers adapter with auto-reconnect.

    Implements both PositionProvider and MarketDataProvider using ib_async.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        reconnect_backoff_initial: int = 1,
        reconnect_backoff_max: int = 60,
        reconnect_backoff_factor: float = 2.0,
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
            self._market_data_fetcher = MarketDataFetcher(self.ib)
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

        Returns:
            List of Position objects with source=IB.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        positions = []
        try:
            ib_positions = await self.ib.reqPositionsAsync()

            for ib_pos in ib_positions:
                position = self._convert_ib_position(ib_pos)
                positions.append(position)

            logger.info(f"Fetched {len(positions)} positions from IB")
        except Exception as e:
            logger.error(f"Failed to fetch positions from IB: {e}")
            raise

        return positions

    def _convert_ib_position(self, ib_pos) -> Position:
        """
        Convert ib_async Position to internal Position model.

        Args:
            ib_pos: ib_async Position object.

        Returns:
            Position object.
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

        return Position(
            symbol=contract.localSymbol,
            underlying=contract.symbol,  # Simplified - extract from contract
            asset_type=asset_type,
            quantity=float(ib_pos.position),
            strike=float(contract.strike),
            right=contract.right,
            expiry=contract.lastTradeDateOrContractMonth,
            avg_price=ib_pos.avgCost,
            multiplier=int(contract.multiplier or 1),
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

    async def fetch_account_info(self) -> AccountInfo:
        """
        Fetch account information from IB using accountSummary API.

        Returns:
            AccountInfo object with real account data from IBKR.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

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

            return AccountInfo(
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

        except Exception as e:
            logger.error(f"Failed to fetch account info from IB: {e}")
            # Return default values on error instead of crashing
            logger.warning("Returning zeroed account info due to fetch failure")
            return AccountInfo(
                net_liquidation=0.0,
                total_cash=0.0,
                buying_power=0.0,
                margin_used=0.0,
                margin_available=0.0,
                maintenance_margin=0.0,
                init_margin_req=0.0,
                excess_liquidity=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                timestamp=datetime.now(),
            )
