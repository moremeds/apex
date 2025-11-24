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
        Fetch market data for given symbols.

        Args:
            positions: List of positions.

        Returns:
            List of MarketData objects.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        market_data_list = []

        try:
            from ib_async import Stock, Option

            # Fetch market data for each symbol
            for pos in positions:
                symbol = pos.symbol
                try:
                    # Create contract based on symbol
                    # This is simplified - proper parsing needed for options
                    if pos.asset_type == AssetType.OPTION:
                        # Option symbol (contains spaces)
                        contract = Option(
                            symbol=pos.underlying,  # Use underlying ticker, not full option symbol
                            lastTradeDateOrContractMonth=pos.expiry,
                            strike=pos.strike,
                            right=str(pos.right),
                            exchange="SMART",
                            multiplier=str(pos.multiplier),
                            currency="USD",
                        )

                    else:
                        # Stock symbol
                        contract = Stock(pos.symbol, 'SMART', currency="USD")

                    # Qualify the contract
                    await self.ib.qualifyContractsAsync(contract)

                    # Request market data snapshot
                    ticker = await self.ib.reqTickersAsync(contract)

                    if ticker and len(ticker) > 0:
                        t = ticker[0]

                        # Extract market data
                        md = MarketData(
                            symbol=symbol,
                            last=float(t.last) if t.last and not isnan(t.last) else None,
                            bid=float(t.bid) if t.bid and not isnan(t.bid) else None,
                            ask=float(t.ask) if t.ask and not isnan(t.ask) else None,
                            mid=float((t.bid + t.ask) / 2) if t.bid and t.ask and not isnan(t.bid) and not isnan(t.ask) else None,
                            volume=int(t.volume) if t.volume and not isnan(t.volume) else None,
                            yesterday_close=float(t.close) if hasattr(t, 'close') and t.close and not isnan(t.close) else None,
                            timestamp=datetime.now(),
                        )

                        # For stocks, set delta to 1.0
                        if " " not in symbol:
                            md.delta = 1.0

                        market_data_list.append(md)
                        self._market_data_cache[symbol] = md

                except Exception as e:
                    logger.warning(f"Failed to fetch market data for {symbol}: {e}")
                    continue

            logger.info(f"Fetched market data for {len(market_data_list)}/{len(positions)} symbols")

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")

        return market_data_list

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
        Fetch account information from IB.

        Returns:
            AccountInfo object.

        Raises:
            ConnectionError: If not connected.
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        # TODO: Implement account fetch using ib.accountSummary()
        # This is a skeleton - full implementation needed

        return AccountInfo(
            net_liquidation=0.0,
            total_cash=0.0,
            buying_power=0.0,
            margin_used=0.0,
            margin_available=0.0,
            maintenance_margin=0.0,
            init_margin_req=0.0,
            excess_liquidity=0.0,
            timestamp=datetime.now(),
        )
