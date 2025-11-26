"""
Market data fetcher with fallback mechanisms.

Handles fetching market data from IBKR with separate strategies for stocks and options.
Provides fallback to snapshot data if streaming data is unavailable.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from math import isnan
import logging
import asyncio

from ...models.position import Position, AssetType
from ...models.market_data import MarketData, GreeksSource


logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetches market data from IBKR with intelligent fallback mechanisms.

    Supports:
    - Streaming data with Greeks for options (via reqMktData)
    - Snapshot data fallback (via reqTickers)
    - Separate handling for stocks vs options
    """

    def __init__(self, ib, data_timeout: float = 3.0, poll_interval: float = 0.1):
        """
        Initialize market data fetcher.

        Args:
            ib: ib_async.IB instance
            data_timeout: Maximum time to wait for data population (seconds)
            poll_interval: Interval between data availability checks (seconds)
        """
        self.ib = ib
        self._active_tickers: Dict[str, object] = {}
        self.data_timeout = data_timeout
        self.poll_interval = poll_interval

    async def fetch_market_data(
        self,
        positions: List[Position],
        qualified_contracts: List,
        pos_map: Dict[int, Position]
    ) -> List[MarketData]:
        """
        Fetch market data for positions with fallback logic.

        Args:
            positions: List of positions
            qualified_contracts: List of qualified IB contracts
            pos_map: Mapping from contract ID to position

        Returns:
            List of MarketData objects
        """
        if not qualified_contracts:
            return []

        # Separate stocks and options
        stock_contracts = []
        option_contracts = []

        for contract in qualified_contracts:
            contract_id = id(contract)
            pos = pos_map.get(contract_id)
            if pos:
                if pos.asset_type == AssetType.STOCK:
                    stock_contracts.append((contract, pos))
                elif pos.asset_type == AssetType.OPTION:
                    option_contracts.append((contract, pos))

        logger.info(f"Fetching market data: {len(stock_contracts)} stocks, {len(option_contracts)} options")

        market_data_list = []

        # Fetch stock data using snapshot method (faster, no Greeks needed)
        if stock_contracts:
            stock_md = await self._fetch_stock_snapshot([c for c, _ in stock_contracts], stock_contracts)
            market_data_list.extend(stock_md)

        # Fetch option data with streaming (for Greeks) and fallback to snapshot
        if option_contracts:
            option_md = await self._fetch_option_streaming_with_fallback(
                [c for c, _ in option_contracts],
                option_contracts
            )
            market_data_list.extend(option_md)

        return market_data_list

    async def _fetch_stock_snapshot(
        self,
        contracts: List,
        contract_pos_pairs: List[Tuple]
    ) -> List[MarketData]:
        """
        Fetch stock market data using snapshot method.

        Args:
            contracts: List of stock contracts
            contract_pos_pairs: List of (contract, position) tuples

        Returns:
            List of MarketData objects
        """
        logger.debug(f"Fetching snapshot data for {len(contracts)} stocks...")
        market_data_list = []

        try:
            # Use reqTickers for fast snapshot data
            tickers = await self.ib.reqTickersAsync(*contracts)

            for i, ticker in enumerate(tickers):
                try:
                    _, pos = contract_pos_pairs[i]
                    md = self._extract_market_data(ticker, pos)

                    # Set delta to 1.0 for stocks
                    md.delta = 1.0

                    market_data_list.append(md)
                    logger.debug(f"✓ Stock data for {pos.symbol}: bid={md.bid}, ask={md.ask}")

                except Exception as e:
                    logger.warning(f"Failed to parse stock data: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching stock snapshots: {e}")

        return market_data_list

    async def _fetch_option_streaming_with_fallback(
        self,
        contracts: List,
        contract_pos_pairs: List[Tuple]
    ) -> List[MarketData]:
        """
        Fetch option market data with streaming (for Greeks) and fallback to snapshot.

        Tries:
        1. Streaming with Greeks (reqMktData with generic tick 106)
        2. Fallback to snapshot if streaming fails (reqTickers)

        Args:
            contracts: List of option contracts
            contract_pos_pairs: List of (contract, position) tuples

        Returns:
            List of MarketData objects
        """
        logger.debug(f"Fetching streaming data with Greeks for {len(contracts)} options...")

        # Try streaming first
        market_data_list = await self._fetch_option_streaming(contracts, contract_pos_pairs)

        # Check for missing data and fallback to snapshot
        missing_indices = []
        for i, md in enumerate(market_data_list):
            if md is None or (not md.bid and not md.ask and not md.last):
                missing_indices.append(i)

        if missing_indices:
            logger.warning(f"{len(missing_indices)} options have no streaming data, falling back to snapshot...")
            fallback_md = await self._fetch_option_snapshot(
                [contracts[i] for i in missing_indices],
                [contract_pos_pairs[i] for i in missing_indices]
            )

            # Replace missing data with fallback
            fallback_idx = 0
            for i in missing_indices:
                if fallback_idx < len(fallback_md):
                    market_data_list[i] = fallback_md[fallback_idx]
                    fallback_idx += 1

        # Filter out None values
        return [md for md in market_data_list if md is not None]

    async def _wait_for_data_population(
        self,
        tickers: List,
        timeout: float
    ) -> int:
        """
        Wait for ticker data to populate using polling with timeout.

        Polls tickers at regular intervals to check if data has arrived.
        Exits early if all tickers are populated or timeout is reached.

        Args:
            tickers: List of ticker objects to monitor
            timeout: Maximum time to wait in seconds

        Returns:
            Number of tickers that successfully populated with data
        """
        start_time = asyncio.get_event_loop().time()
        last_populated_count = 0

        while True:
            # Check how many tickers have data
            populated_count = sum(
                1 for t in tickers
                if (t.bid and not isnan(t.bid)) or
                   (t.ask and not isnan(t.ask)) or
                   (t.last and not isnan(t.last))
            )

            # Log progress if count changed
            if populated_count > last_populated_count:
                logger.debug(f"Data population progress: {populated_count}/{len(tickers)} tickers")
                last_populated_count = populated_count

            # Exit early if all data populated
            if populated_count == len(tickers):
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.debug(f"All data populated in {elapsed:.2f}s")
                return populated_count

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Data population timeout ({timeout}s): {populated_count}/{len(tickers)} populated")
                return populated_count

            # Wait before next poll
            await asyncio.sleep(self.poll_interval)

    async def _fetch_option_streaming(
        self,
        contracts: List,
        contract_pos_pairs: List[Tuple]
    ) -> List[Optional[MarketData]]:
        """
        Fetch option market data using streaming (reqMktData with Greeks).

        Args:
            contracts: List of option contracts
            contract_pos_pairs: List of (contract, position) tuples

        Returns:
            List of MarketData objects (may contain None for failed requests)
        """
        market_data_list = []

        try:
            requested_symbols = {pos.symbol for _, pos in contract_pos_pairs}

            # Cancel subscriptions that are no longer needed
            for symbol, old_ticker in list(self._active_tickers.items()):
                if symbol not in requested_symbols:
                    try:
                        self.ib.cancelMktData(old_ticker.contract)
                    except Exception as e:
                        logger.debug(f"Error cancelling old subscription for {symbol}: {e}")
                    self._active_tickers.pop(symbol, None)

            # Request streaming data with Greeks (generic tick 106)
            tickers = []
            for i, contract in enumerate(contracts):
                _, pos = contract_pos_pairs[i]

                # Generic tick type 106 enables option computations (Greeks + IV)
                # This requests: delta, gamma, vega, theta, implied volatility, and live prices
                existing = self._active_tickers.get(pos.symbol)
                if existing:
                    tickers.append(existing)
                else:
                    ticker = self.ib.reqMktData(contract, '106', False, False)
                    tickers.append(ticker)
                    self._active_tickers[pos.symbol] = ticker
                    logger.debug(f"Requested streaming data with Greeks for option {i+1}/{len(contracts)}")

            # Wait for data to populate using robust polling mechanism
            logger.debug(f"Polling for option data population (timeout={self.data_timeout}s)...")
            populated_count = await self._wait_for_data_population(tickers, timeout=self.data_timeout)

            # Log results
            empty_count = len(tickers) - populated_count
            if populated_count == len(tickers):
                logger.debug(f"✓ All {len(tickers)} option tickers populated successfully")
            elif populated_count > 0:
                logger.warning(f"⚠ Partial data: {populated_count}/{len(tickers)} tickers populated, {empty_count} empty")
            else:
                logger.warning(f"✗ No data populated for {len(tickers)} option tickers")

            # Extract data from tickers
            for i, ticker in enumerate(tickers):
                try:
                    _, pos = contract_pos_pairs[i]
                    md = self._extract_market_data(ticker, pos)

                    # Try to extract Greeks for options
                    self._extract_greeks(ticker, md, pos)

                    market_data_list.append(md)

                except Exception as e:
                    logger.warning(f"Failed to parse streaming option data: {e}")
                    market_data_list.append(None)  # Will trigger fallback

        except Exception as e:
            logger.error(f"Error in option streaming: {e}")
            # Return list of None to trigger full fallback
            market_data_list = [None] * len(contracts)

        return market_data_list

    async def _fetch_option_snapshot(
        self,
        contracts: List,
        contract_pos_pairs: List[Tuple]
    ) -> List[MarketData]:
        """
        Fetch option market data using snapshot method (no Greeks).

        Args:
            contracts: List of option contracts
            contract_pos_pairs: List of (contract, position) tuples

        Returns:
            List of MarketData objects
        """
        logger.debug(f"Fetching snapshot data for {len(contracts)} options (no Greeks)...")
        market_data_list = []

        try:
            # Use reqTickers for snapshot data
            tickers = await self.ib.reqTickersAsync(*contracts)

            for i, ticker in enumerate(tickers):
                try:
                    _, pos = contract_pos_pairs[i]
                    md = self._extract_market_data(ticker, pos)

                    # Note: No Greeks available in snapshot mode
                    logger.debug(f"✓ Snapshot data for {pos.symbol}: bid={md.bid}, ask={md.ask} (no Greeks)")

                    market_data_list.append(md)

                except Exception as e:
                    logger.warning(f"Failed to parse snapshot option data: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching option snapshots: {e}")

        return market_data_list

    def _extract_market_data(self, ticker, pos: Position) -> MarketData:
        """
        Extract market data from ticker object.

        Args:
            ticker: IB ticker object
            pos: Position

        Returns:
            MarketData object
        """
        return MarketData(
            symbol=pos.symbol,
            last=float(ticker.last) if ticker.last and not isnan(ticker.last) else None,
            bid=float(ticker.bid) if ticker.bid and not isnan(ticker.bid) else None,
            ask=float(ticker.ask) if ticker.ask and not isnan(ticker.ask) else None,
            mid=float((ticker.bid + ticker.ask) / 2) if ticker.bid and ticker.ask and not isnan(ticker.bid) and not isnan(ticker.ask) else None,
            volume=int(ticker.volume) if ticker.volume and not isnan(ticker.volume) else None,
            yesterday_close=float(ticker.close) if hasattr(ticker, 'close') and ticker.close and not isnan(ticker.close) else None,
            timestamp=datetime.now(),
        )

    def _extract_greeks(self, ticker, md: MarketData, pos: Position) -> None:
        """
        Extract Greeks and IV from ticker and populate MarketData object.

        Args:
            ticker: IB ticker object
            md: MarketData object to populate
            pos: Position
        """
        if pos.asset_type != AssetType.OPTION:
            return

        # Extract IV first (available directly on ticker)
        if hasattr(ticker, 'impliedVolatility') and ticker.impliedVolatility and not isnan(ticker.impliedVolatility):
            md.iv = float(ticker.impliedVolatility)
            logger.debug(f"✓ IV for {pos.symbol}: {md.iv:.3f} ({md.iv*100:.1f}%)")
        else:
            logger.debug(f"No IV available for {pos.symbol}")

        # Extract Greeks from modelGreeks
        if hasattr(ticker, 'modelGreeks') and ticker.modelGreeks:
            greeks = ticker.modelGreeks
            md.delta = float(greeks.delta) if greeks.delta and not isnan(greeks.delta) else None
            md.gamma = float(greeks.gamma) if greeks.gamma and not isnan(greeks.gamma) else None
            md.vega = float(greeks.vega) if greeks.vega and not isnan(greeks.vega) else None
            md.theta = float(greeks.theta) if greeks.theta and not isnan(greeks.theta) else None
            md.greeks_source = GreeksSource.IBKR

            # Extract underlying price (critical for delta dollars calculation)
            if hasattr(greeks, 'undPrice') and greeks.undPrice and not isnan(greeks.undPrice):
                md.underlying_price = float(greeks.undPrice)
                logger.debug(f"✓ Underlying price for {pos.underlying}: ${md.underlying_price:.2f}")

            if md.delta:
                logger.debug(f"✓ Greeks for {pos.symbol}: Δ={md.delta:.3f}, γ={md.gamma:.4f}, ν={md.vega:.2f}, θ={md.theta:.2f}")
            else:
                logger.debug(f"Greeks for {pos.symbol}: values are None")
        else:
            logger.debug(f"No modelGreeks available for {pos.symbol}")

    def cleanup(self) -> None:
        """Cancel all active market data subscriptions."""
        for symbol, ticker in self._active_tickers.items():
            try:
                self.ib.cancelMktData(ticker.contract)
                logger.debug(f"Cancelled market data for {symbol}")
            except Exception as e:
                logger.warning(f"Error cancelling market data for {symbol}: {e}")

        self._active_tickers.clear()
