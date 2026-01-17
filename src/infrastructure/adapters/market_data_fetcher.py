"""
Market data fetcher with streaming and fallback mechanisms.

Handles fetching market data from IBKR with separate strategies for stocks and options.
Supports event-driven streaming updates and fallback to snapshot data.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from math import isnan
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ...models.market_data import GreeksSource, MarketData
from ...models.position import AssetType, Position
from ...utils.logging_setup import get_logger
from ...utils.timezone import now_utc

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# OPT-010: Immutable update for lock-free queue dispatch
@dataclass(frozen=True, slots=True)
class TickerUpdate:
    """Immutable ticker update for queue dispatch."""

    symbol: str
    ticker_id: int  # id(ticker) for lookup
    timestamp: float


class MarketDataFetcher:
    """
    Fetches market data from IBKR with intelligent fallback mechanisms.

    Supports:
    - Streaming data with Greeks for options (via reqMktData)
    - Event-driven updates via callback on price changes
    - Snapshot data fallback (via reqTickers)
    - Separate handling for stocks vs options
    """

    def __init__(
        self,
        ib: Any,
        data_timeout: float = 5.0,  # Increased from 3.0s for stocks
        option_data_timeout: float = 3.0,  # OPT-003: Reduced from 10s - Greeks typically populate in 2-3s
        poll_interval: float = 0.1,
        poll_interval_max: float = 0.5,  # m1/m8: Max backoff interval
        poll_backoff_factor: float = 1.5,  # m1/m8: Exponential backoff multiplier
        on_price_update: Optional[Callable[[str, MarketData], None]] = None,
    ):
        """
        Initialize market data fetcher.

        Args:
            ib: ib_async.IB instance
            data_timeout: Maximum time to wait for stock data population (seconds)
            option_data_timeout: Maximum time to wait for option data with Greeks (seconds)
            poll_interval: Initial interval between data availability checks (seconds)
            poll_interval_max: Maximum backoff interval for polling (seconds)
            poll_backoff_factor: Multiplier for exponential backoff (default: 1.5x)
            on_price_update: Callback when streaming price updates (symbol, MarketData)
        """
        self.ib = ib
        self._active_tickers: Dict[str, Any] = {}  # symbol -> ticker
        self._ticker_to_symbol: Dict[int, str] = (
            {}
        )  # id(ticker) -> symbol for O(1) streaming lookup
        self._ticker_positions: Dict[str, Position] = {}  # symbol -> Position mapping
        self._ticker_lock = Lock()  # Protects writes to ticker dicts during subscription
        self.data_timeout = data_timeout
        self.option_data_timeout = option_data_timeout
        self.poll_interval = poll_interval
        self._poll_interval_max = poll_interval_max
        self._poll_backoff_factor = poll_backoff_factor
        self._on_price_update = on_price_update
        self._streaming_enabled = False

        # M4: Optional event loop for non-blocking callback dispatch
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # OPT-010: Queue-based lock-free dispatch for streaming callbacks
        self._ticker_queue: asyncio.Queue[TickerUpdate] = asyncio.Queue(maxsize=10000)
        self._dispatch_task: Optional[asyncio.Task] = None
        # OPT-010: Snapshot dicts for lock-free reads (copy-on-write)
        self._ticker_to_symbol_snapshot: Dict[int, str] = {}
        self._ticker_positions_snapshot: Dict[str, Position] = {}
        # CRIT-001: Error counter for rate-limited logging
        self._ticker_error_count: int = 0

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set event loop for non-blocking callback dispatch (M4 fix)."""
        self._event_loop = loop

    # OPT-010: Queue-based dispatch lifecycle methods
    async def start_dispatch(self) -> None:
        """Start the queue-based callback dispatcher."""
        if self._dispatch_task is None or self._dispatch_task.done():
            self._dispatch_task = asyncio.create_task(self._dispatch_loop())
            logger.debug("Started ticker dispatch loop")

    async def stop_dispatch(self) -> None:
        """Stop the callback dispatcher gracefully."""
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
            logger.debug("Stopped ticker dispatch loop")

    def _publish_snapshot(self) -> None:
        """
        OPT-010: Publish atomic snapshot for lock-free reads.

        Copy-on-write pattern: streaming callbacks read from snapshot,
        subscription changes write to main dicts then publish new snapshot.
        """
        # Atomic reference swap - no lock needed for readers
        self._ticker_to_symbol_snapshot = dict(self._ticker_to_symbol)
        self._ticker_positions_snapshot = dict(self._ticker_positions)

    async def _dispatch_loop(self) -> None:
        """
        OPT-010: Process ticker updates from queue with batching.

        Drains queue and processes in batches for efficiency.
        """
        while True:
            try:
                # Get first update (blocking)
                update = await self._ticker_queue.get()
                updates = [update]

                # Drain any additional updates (non-blocking) up to batch limit
                while len(updates) < 100:
                    try:
                        update = self._ticker_queue.get_nowait()
                        updates.append(update)
                    except asyncio.QueueEmpty:
                        break

                # Process batch
                await self._process_ticker_batch(updates)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatch loop error: {e}")

    async def _process_ticker_batch(self, updates: List[TickerUpdate]) -> None:
        """
        OPT-010: Process a batch of ticker updates.

        Uses lock-free snapshot reads for symbol/position lookup.
        """
        if not self._on_price_update:
            return

        errors: list = []

        for update in updates:
            try:
                # OPT-010: Lock-free reads from snapshot
                symbol = self._ticker_to_symbol_snapshot.get(update.ticker_id)
                if not symbol:
                    continue
                pos = self._ticker_positions_snapshot.get(symbol)
                if not pos:
                    continue

                # Get the actual ticker object from active tickers
                ticker = self._active_tickers.get(symbol)
                if not ticker:
                    continue

                # Extract market data (no shared state mutation)
                md = self._extract_market_data(ticker, pos)

                # Extract Greeks if available (for options)
                if pos.asset_type == AssetType.OPTION:
                    self._extract_greeks(ticker, md, pos)
                else:
                    md.delta = 1.0  # Stocks have delta of 1

                # Fire callback
                self._on_price_update(symbol, md)

            except Exception as e:
                errors.append(str(e))

        if errors:
            logger.warning(f"Batch processing errors: {len(errors)} failures. First: {errors[0]}")

    def unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from market data for a single symbol and clean up maps."""
        with self._ticker_lock:
            ticker = self._active_tickers.pop(symbol, None)
            self._ticker_positions.pop(symbol, None)
            if ticker:
                self._ticker_to_symbol.pop(id(ticker), None)
            # OPT-010: Publish snapshot after modification
            self._publish_snapshot()

        if ticker:
            try:
                self.ib.cancelMktData(ticker.contract)
            except Exception as e:
                logger.warning(f"Error cancelling market data for {symbol}: {e}")

    def prune_stale_subscriptions(self, current_symbols: set[str]) -> int:
        """Remove subscriptions for symbols no longer in the portfolio."""
        with self._ticker_lock:
            stale_symbols = set(self._active_tickers.keys()) - current_symbols

        for symbol in stale_symbols:
            self.unsubscribe_symbol(symbol)

        if stale_symbols:
            logger.debug(f"Pruned {len(stale_symbols)} stale subscriptions: {stale_symbols}")

        return len(stale_symbols)

    async def fetch_market_data(
        self, positions: List[Position], qualified_contracts: List, pos_map: Dict[int, Position]
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

        # NOTE: Do NOT prune subscriptions here - it causes subscription churn.
        # Subscriptions persist for portfolio lifetime. Pruning should only happen
        # when positions are explicitly closed, not on every fetch cycle.
        # The _active_tickers cache already handles subscription reuse.

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

        logger.info(
            f"Fetching market data: {len(stock_contracts)} stocks, {len(option_contracts)} options"
        )

        market_data_list = []

        # Fetch stock data using snapshot method (faster, no Greeks needed)
        if stock_contracts:
            stock_md = await self._fetch_stock_snapshot(
                [c for c, _ in stock_contracts], stock_contracts
            )
            market_data_list.extend(stock_md)

        # Fetch option data with streaming (for Greeks) and fallback to snapshot
        if option_contracts:
            option_md = await self._fetch_option_streaming_with_fallback(
                [c for c, _ in option_contracts], option_contracts
            )
            market_data_list.extend(option_md)

        return market_data_list

    async def _fetch_stock_snapshot(
        self, contracts: List, contract_pos_pairs: List[Tuple]
    ) -> List[MarketData]:
        """
        Fetch stock market data using batch streaming subscription.

        Subscribes to ALL stocks first, then waits for data population in aggregate.
        This is dramatically faster than waiting per-symbol (50 stocks: ~3s vs ~150s).

        Symbols that don't receive data within timeout are marked as data_missing.

        Args:
            contracts: List of stock contracts
            contract_pos_pairs: List of (contract, position) tuples

        Returns:
            List of MarketData objects (includes data_missing entries for failed symbols)
        """
        if not contracts:
            return []

        logger.info(f"Batch subscribing to {len(contracts)} stocks...")
        market_data_list = []
        tickers_with_pos = []

        try:
            # Phase 1: Subscribe ALL stocks immediately (non-blocking)
            new_tickers = []  # Only new subscriptions need waiting
            for contract, pos in contract_pos_pairs:
                try:
                    with self._ticker_lock:
                        # Store position mapping for streaming callbacks
                        self._ticker_positions[pos.symbol] = pos

                        # Check if already subscribed
                        if pos.symbol in self._active_tickers:
                            ticker = self._active_tickers[pos.symbol]
                            # Already subscribed - no need to wait again
                        else:
                            # Subscribe to streaming data (no special generic ticks for stocks)
                            ticker = self.ib.reqMktData(contract, "", False, False)
                            self._active_tickers[pos.symbol] = ticker
                            self._ticker_to_symbol[id(ticker)] = pos.symbol
                            new_tickers.append(ticker)  # Track new subscriptions

                    tickers_with_pos.append((ticker, pos))

                except Exception as e:
                    logger.warning(f"Failed to subscribe stock {pos.symbol}: {e}")
                    continue

            # OPT-010: Publish snapshot after all subscriptions are set up
            self._publish_snapshot()

            # Phase 2: Wait ONLY for NEW subscriptions (skip if all cached)
            if new_tickers:
                populated_count = await self._wait_for_batch_population(
                    new_tickers, timeout=self.data_timeout
                )
                logger.info(
                    f"New stock subscriptions populated: {populated_count}/{len(new_tickers)}"
                )
            elif tickers_with_pos:
                logger.debug(
                    f"All {len(tickers_with_pos)} stocks using cached subscriptions (no wait)"
                )

            # Phase 3: Extract data from all tickers (mark missing as data_missing)
            for ticker, pos in tickers_with_pos:
                md = self._extract_market_data(ticker, pos)
                md.delta = 1.0  # Stocks have delta of 1

                # Check if we got valid data
                if not self._has_valid_price(ticker):
                    # Mark as data_missing by logging; quality is already on the md object
                    logger.debug(f"Stock {pos.symbol} marked as data_missing (no price data)")

                market_data_list.append(md)

        except Exception as e:
            logger.error(f"Error in batch stock subscription: {e}")

        return market_data_list

    async def _wait_for_batch_population(
        self, tickers: List, timeout: float = 3.0, target_ratio: float = 0.8
    ) -> int:
        """
        Wait for batch of tickers to have data populated.

        Returns early if target_ratio of tickers have data, or on timeout.

        Args:
            tickers: List of IB ticker objects
            timeout: Maximum time to wait
            target_ratio: Fraction of tickers that must have data to return early

        Returns:
            Number of tickers with valid data
        """
        if not tickers:
            return 0

        target_count = int(len(tickers) * target_ratio)
        start = asyncio.get_running_loop().time()
        # m1/m8: Exponential backoff for polling
        current_interval = self.poll_interval

        while asyncio.get_running_loop().time() - start < timeout:
            populated = sum(1 for t in tickers if self._has_valid_price(t))

            if populated >= target_count:
                return populated

            await asyncio.sleep(current_interval)
            # m1/m8: Exponential backoff - increase interval up to max
            current_interval = min(
                current_interval * self._poll_backoff_factor, self._poll_interval_max
            )

        # Return final count on timeout
        return sum(1 for t in tickers if self._has_valid_price(t))

    def _has_valid_price(self, ticker: Any) -> bool:
        """Check if ticker has valid price data (live or previous close)."""
        if ticker.last and not isnan(ticker.last) and ticker.last > 0:
            return True
        if ticker.bid and not isnan(ticker.bid) and ticker.bid > 0:
            return True
        # Also accept previous close as valid (important for market closed hours)
        if (
            hasattr(ticker, "close")
            and ticker.close
            and not isnan(ticker.close)
            and ticker.close > 0
        ):
            return True
        return False

    async def _wait_for_ticker_data(self, ticker: Any, timeout: float = 3.0) -> bool:
        """
        Wait for ticker to have valid data.

        Args:
            ticker: IB ticker object
            timeout: Maximum time to wait

        Returns:
            True if data received, False if timeout
        """
        start = asyncio.get_running_loop().time()
        # m1/m8: Exponential backoff for polling
        current_interval = self.poll_interval

        while asyncio.get_running_loop().time() - start < timeout:
            # Check if we have valid price data (including previous close for market closed)
            if self._has_valid_price(ticker):
                return True
            await asyncio.sleep(current_interval)
            # m1/m8: Exponential backoff - increase interval up to max
            current_interval = min(
                current_interval * self._poll_backoff_factor, self._poll_interval_max
            )
        return False

    async def _fetch_option_streaming_with_fallback(
        self, contracts: List, contract_pos_pairs: List[Tuple]
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
            logger.warning(
                f"{len(missing_indices)} options have no streaming data, falling back to snapshot..."
            )
            fallback_md = await self._fetch_option_snapshot(
                [contracts[i] for i in missing_indices],
                [contract_pos_pairs[i] for i in missing_indices],
            )

            # Replace missing data with fallback
            fallback_idx = 0
            for i in missing_indices:
                if fallback_idx < len(fallback_md):
                    market_data_list[i] = fallback_md[fallback_idx]
                    fallback_idx += 1

        # Filter out None values
        return [md for md in market_data_list if md is not None]

    async def _wait_for_data_population(self, tickers: List, timeout: float) -> int:
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
        start_time = asyncio.get_running_loop().time()
        last_populated_count = 0
        # m1/m8: Exponential backoff for polling
        current_interval = self.poll_interval

        while True:
            # Check how many tickers have data (including previous close for market closed)
            populated_count = sum(1 for t in tickers if self._has_valid_price(t))

            # Log progress if count changed
            if populated_count > last_populated_count:
                logger.debug(f"Data population progress: {populated_count}/{len(tickers)} tickers")
                last_populated_count = populated_count
                # Reset backoff on progress (data is arriving)
                current_interval = self.poll_interval

            # Exit early if all data populated
            if populated_count == len(tickers):
                elapsed = asyncio.get_running_loop().time() - start_time
                logger.debug(f"All data populated in {elapsed:.2f}s")
                return populated_count

            # Check timeout
            elapsed = asyncio.get_running_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(
                    f"Data population timeout ({timeout}s): {populated_count}/{len(tickers)} populated"
                )
                return populated_count

            # Wait before next poll with exponential backoff
            await asyncio.sleep(current_interval)
            # m1/m8: Exponential backoff - increase interval up to max
            current_interval = min(
                current_interval * self._poll_backoff_factor, self._poll_interval_max
            )

    async def _fetch_option_streaming(
        self, contracts: List[Any], contract_pos_pairs: List[Tuple[Any, Position]]
    ) -> List[Optional[MarketData]]:
        """
        Fetch option market data using streaming (reqMktData with Greeks).

        Args:
            contracts: List of option contracts
            contract_pos_pairs: List of (contract, position) tuples

        Returns:
            List of MarketData objects (may contain None for failed requests)
        """
        market_data_list: List[Optional[MarketData]] = []

        try:
            # NOTE: Do NOT cancel subscriptions here - it causes subscription churn.
            # When only options are passed (stocks filtered as "fresh"), this would
            # incorrectly cancel stock subscriptions. Subscriptions persist for
            # portfolio lifetime. See CLAUDE.md "Market Data Subscription Lifecycle".

            # Request streaming data with Greeks (generic tick 106)
            tickers = []
            new_tickers = []  # Only new subscriptions need waiting
            for i, contract in enumerate(contracts):
                _, pos = contract_pos_pairs[i]

                with self._ticker_lock:
                    # Store position mapping for streaming callbacks
                    self._ticker_positions[pos.symbol] = pos

                    # Generic tick type 106 enables option computations (Greeks + IV)
                    existing = self._active_tickers.get(pos.symbol)
                    if existing:
                        tickers.append(existing)
                    else:
                        ticker = self.ib.reqMktData(contract, "106", False, False)
                        tickers.append(ticker)
                        new_tickers.append(ticker)
                        self._active_tickers[pos.symbol] = ticker
                        self._ticker_to_symbol[id(ticker)] = pos.symbol

            # OPT-010: Publish snapshot after all subscriptions are set up
            self._publish_snapshot()

            # Wait ONLY for NEW subscriptions (skip if all cached)
            # Use longer timeout for options since Greeks take time to populate
            populated_count = len(tickers)  # Assume all populated for cached path
            if new_tickers:
                logger.debug(
                    f"Waiting for {len(new_tickers)} new option subscriptions (timeout={self.option_data_timeout}s)..."
                )
                populated_count = await self._wait_for_data_population(
                    new_tickers, timeout=self.option_data_timeout
                )
                # Adjust count to include cached tickers
                populated_count += len(tickers) - len(new_tickers)
                logger.info(
                    f"New option subscriptions populated: {populated_count}/{len(new_tickers)}"
                )
            elif tickers:
                logger.debug(f"All {len(tickers)} options using cached subscriptions (no wait)")

            # Log results
            empty_count = len(tickers) - populated_count
            if populated_count == len(tickers):
                logger.debug(f"✓ All {len(tickers)} option tickers populated successfully")
            elif populated_count > 0:
                logger.warning(
                    f"⚠ Partial data: {populated_count}/{len(tickers)} tickers populated, {empty_count} empty"
                )
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
        self, contracts: List, contract_pos_pairs: List[Tuple]
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
                    logger.debug(
                        f"✓ Snapshot data for {pos.symbol}: bid={md.bid}, ask={md.ask} (no Greeks)"
                    )

                    market_data_list.append(md)

                except Exception as e:
                    logger.warning(f"Failed to parse snapshot option data: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching option snapshots: {e}")

        return market_data_list

    def _extract_market_data(self, ticker: Any, pos: Position) -> MarketData:
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
            mid=(
                float((ticker.bid + ticker.ask) / 2)
                if ticker.bid and ticker.ask and not isnan(ticker.bid) and not isnan(ticker.ask)
                else None
            ),
            volume=int(ticker.volume) if ticker.volume and not isnan(ticker.volume) else None,
            yesterday_close=(
                float(ticker.close)
                if hasattr(ticker, "close") and ticker.close and not isnan(ticker.close)
                else None
            ),
            timestamp=now_utc(),
        )

    def _extract_greeks(self, ticker: Any, md: MarketData, pos: Position) -> None:
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
        iv = self._safe_float(getattr(ticker, "impliedVolatility", None))
        if iv is not None:
            md.iv = iv

        # Extract Greeks from modelGreeks
        if hasattr(ticker, "modelGreeks") and ticker.modelGreeks:
            greeks = ticker.modelGreeks
            md.delta = self._safe_float(getattr(greeks, "delta", None))
            md.gamma = self._safe_float(getattr(greeks, "gamma", None))
            md.vega = self._safe_float(getattr(greeks, "vega", None))
            md.theta = self._safe_float(getattr(greeks, "theta", None))
            if any(value is not None for value in (md.delta, md.gamma, md.vega, md.theta)):
                md.greeks_source = GreeksSource.IBKR

            # Extract underlying price (critical for delta dollars calculation)
            und_price = self._safe_float(getattr(greeks, "undPrice", None))
            if und_price is not None:
                md.underlying_price = und_price

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convert value to float, treating NaN as None."""
        if value is None:
            return None
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        return None if isnan(value_f) else value_f

    def enable_streaming(self) -> None:
        """
        Enable streaming mode with event callbacks.

        Registers IB pendingTickersEvent to receive real-time price updates.
        """
        if self._streaming_enabled:
            return

        self.ib.pendingTickersEvent += self._on_pending_tickers
        self._streaming_enabled = True
        logger.info("Streaming market data enabled")

    def disable_streaming(self) -> None:
        """Disable streaming mode."""
        if not self._streaming_enabled:
            return

        self.ib.pendingTickersEvent -= self._on_pending_tickers
        self._streaming_enabled = False
        logger.info("Streaming market data disabled")

    def _on_pending_tickers(self, tickers: Any) -> None:
        """
        Handle streaming ticker updates from IB.

        Called by IB when any subscribed ticker has new data.
        OPT-010: Lock-free implementation using queue dispatch.

        This method is called from IB's callback thread, so we use queue to cross
        the thread boundary safely without holding locks.
        """
        if not self._on_price_update:
            return

        timestamp = time.time()
        dropped = 0

        for ticker in tickers:
            try:
                ticker_id = id(ticker)

                # OPT-010: Lock-free read from snapshot (atomic reference)
                symbol = self._ticker_to_symbol_snapshot.get(ticker_id)
                if not symbol:
                    continue

                # Queue update for async processing
                update = TickerUpdate(symbol=symbol, ticker_id=ticker_id, timestamp=timestamp)
                try:
                    # Thread-safe queue put from IB callback thread
                    if self._event_loop and self._event_loop.is_running():
                        self._event_loop.call_soon_threadsafe(self._ticker_queue.put_nowait, update)
                    else:
                        # Fallback for sync context (shouldn't happen in normal operation)
                        self._ticker_queue.put_nowait(update)
                except (asyncio.QueueFull, RuntimeError):
                    dropped += 1

            except Exception as e:
                # CRIT-001: Rate-limited logging for hot path errors
                self._ticker_error_count += 1
                if self._ticker_error_count <= 10 or self._ticker_error_count % 100 == 0:
                    logger.warning(
                        "Ticker processing error (total=%d): %s", self._ticker_error_count, e
                    )

        if dropped > 0:
            logger.warning(f"Ticker queue full, dropped {dropped} updates")

    def cleanup(self) -> None:
        """Cancel all active market data subscriptions (sync version)."""
        self.disable_streaming()

        with self._ticker_lock:
            tickers_to_cancel = list(self._active_tickers.items())
            self._active_tickers.clear()
            self._ticker_to_symbol.clear()
            self._ticker_positions.clear()
            # OPT-010: Clear snapshots
            self._ticker_to_symbol_snapshot = {}
            self._ticker_positions_snapshot = {}

        for symbol, ticker in tickers_to_cancel:
            try:
                self.ib.cancelMktData(ticker.contract)
            except Exception as e:
                logger.warning(f"Error cancelling market data for {symbol}: {e}")

    async def cleanup_async(self) -> None:
        """Cancel all subscriptions and stop dispatch task (async version)."""
        # OPT-010: Stop dispatch task first
        await self.stop_dispatch()
        # Then run sync cleanup
        self.cleanup()
