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

from ...models.position import Position, AssetType
from ...models.market_data import MarketData


logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetches market data from IBKR with intelligent fallback mechanisms.

    Supports:
    - Streaming data with Greeks for options (via reqMktData)
    - Snapshot data fallback (via reqTickers)
    - Separate handling for stocks vs options
    """

    def __init__(self, ib):
        """
        Initialize market data fetcher.

        Args:
            ib: ib_async.IB instance
        """
        self.ib = ib
        self._active_tickers: Dict[str, object] = {}

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
            # Cancel previous subscriptions
            for symbol, old_ticker in list(self._active_tickers.items()):
                try:
                    self.ib.cancelMktData(old_ticker.contract)
                except Exception as e:
                    logger.debug(f"Error cancelling old subscription for {symbol}: {e}")
            self._active_tickers.clear()

            # Request streaming data with Greeks (generic tick 106)
            tickers = []
            for i, contract in enumerate(contracts):
                # Generic tick type 106 enables option computations (Greeks)
                # Empty string '' means no generic ticks - just use delayed/snapshot data
                ticker = self.ib.reqMktData(contract, '', False, False)
                tickers.append(ticker)
                logger.debug(f"Requested streaming data for option {i+1}/{len(contracts)}")

            # Wait for data to populate
            logger.debug(f"Waiting for option data to populate...")
            await self.ib.sleep(1.5)

            # Check how many tickers have data
            empty_count = sum(1 for t in tickers if not (t.bid or t.ask or t.last))
            if empty_count > 0:
                logger.warning(f"{empty_count}/{len(tickers)} option tickers have no data")

            # Extract data from tickers
            for i, ticker in enumerate(tickers):
                try:
                    _, pos = contract_pos_pairs[i]
                    md = self._extract_market_data(ticker, pos)

                    # Try to extract Greeks for options
                    self._extract_greeks(ticker, md, pos)

                    # Store active ticker
                    self._active_tickers[pos.symbol] = ticker

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
        Extract Greeks from ticker and populate MarketData object.

        Args:
            ticker: IB ticker object
            md: MarketData object to populate
            pos: Position
        """
        if pos.asset_type != AssetType.OPTION:
            return

        if hasattr(ticker, 'modelGreeks') and ticker.modelGreeks:
            greeks = ticker.modelGreeks
            md.delta = float(greeks.delta) if greeks.delta and not isnan(greeks.delta) else None
            md.gamma = float(greeks.gamma) if greeks.gamma and not isnan(greeks.gamma) else None
            md.vega = float(greeks.vega) if greeks.vega and not isnan(greeks.vega) else None
            md.theta = float(greeks.theta) if greeks.theta and not isnan(greeks.theta) else None

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
