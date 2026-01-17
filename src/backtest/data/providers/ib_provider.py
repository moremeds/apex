"""
IB Historical Data Provider for Backtesting.

Thin wrapper around existing IbHistoricalAdapter with:
- Backtest-specific client IDs (4-10 from historical_pool)
- Rate limiting for IB pacing rules
- DataFrame conversion for VectorBT compatibility
- Progress callbacks for long fetches
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class IbBacktestDataProvider:
    """
    IB Historical data provider for backtesting.

    Reuses existing IbHistoricalAdapter infrastructure with
    backtest-specific configuration.

    Features:
    - Uses client IDs 4-10 from historical_pool (avoids live trading conflicts)
    - Rate limiting (IB: ~6 requests/min for historical data)
    - Progress callbacks for long fetches
    - Converts BarData -> DataFrame for VectorBT

    Example:
        provider = IbBacktestDataProvider(port=7497)  # TWS Paper
        await provider.connect()

        data = await provider.fetch_bars(
            symbols=["AAPL", "MSFT"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            timeframe="1d",
        )

        await provider.disconnect()
    """

    # IB pacing rules: ~6 requests per minute for historical data
    # Being conservative: 1 request every 11 seconds
    REQUEST_DELAY_SECONDS = 11.0

    # Client IDs reserved for backtest (4-10 from historical_pool)
    # IDs 1-3 are used by live trading (execution, monitoring, historical)
    BACKTEST_CLIENT_IDS = [4, 5, 6, 7, 8, 9, 10]

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4001,
        client_id: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        rate_limit: bool = True,
    ):
        """
        Initialize backtest data provider.

        Args:
            host: IB TWS/Gateway host
            port: IB port (4001=Gateway Live, 4002=Gateway Paper,
                          7496=TWS Live, 7497=TWS Paper)
            client_id: Specific client ID (default: picks from BACKTEST_CLIENT_IDS)
            progress_callback: fn(current, total, symbol) for progress updates
            rate_limit: Whether to apply rate limiting (default True)
        """
        self.host = host
        self.port = port
        self._client_id = client_id if client_id is not None else self._pick_client_id()
        self._progress_callback = progress_callback
        self._rate_limit = rate_limit

        # Reuse existing adapter class
        self._adapter: Optional[IbHistoricalAdapter] = None
        self._last_request_time: Optional[datetime] = None
        self._connected = False
        self._request_count = 0

    def _pick_client_id(self) -> int:
        """Pick default client ID from backtest pool."""
        return self.BACKTEST_CLIENT_IDS[0]  # 4

    async def connect(self) -> None:
        """Connect to IB using existing adapter."""
        if self._connected:
            return

        self._adapter = IbHistoricalAdapter(
            host=self.host,
            port=self.port,
            client_id=self._client_id,
        )
        await self._adapter.connect()
        self._connected = True
        self._request_count = 0
        logger.info(
            f"IB backtest provider connected "
            f"(host={self.host}, port={self.port}, client_id={self._client_id})"
        )

    async def disconnect(self) -> None:
        """Disconnect from IB."""
        if self._adapter and self._connected:
            await self._adapter.disconnect()
            self._connected = False
            logger.info(
                f"IB backtest provider disconnected " f"(requests made: {self._request_count})"
            )

    async def ensure_connected(self) -> None:
        """Ensure connection is alive, reconnect if needed."""
        if not self._connected or self._adapter is None:
            await self.connect()
        elif not self._adapter.is_connected():
            logger.info("IB backtest provider reconnecting...")
            await self._adapter.connect()

    async def _rate_limit_wait(self) -> None:
        """Wait to respect IB rate limits."""
        if not self._rate_limit:
            return

        if self._last_request_time is not None:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.REQUEST_DELAY_SECONDS:
                wait_time = self.REQUEST_DELAY_SECONDS - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        self._last_request_time = datetime.now()
        self._request_count += 1

    async def fetch_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV bars for multiple symbols.

        Args:
            symbols: List of symbols to fetch
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe (1d, 1h, 5m, etc.)

        Returns:
            Dict mapping symbol -> DataFrame with columns:
            [open, high, low, close, volume] indexed by timestamp
        """
        await self.ensure_connected()

        results: Dict[str, pd.DataFrame] = {}
        total = len(symbols)

        logger.info(
            f"Fetching {timeframe} bars for {total} symbols " f"({start.date()} to {end.date()})"
        )

        for i, symbol in enumerate(symbols):
            try:
                # Rate limiting
                await self._rate_limit_wait()

                # Use existing adapter's fetch_bars method
                bars = await self._adapter.fetch_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                )

                # Convert List[BarData] -> DataFrame
                if bars:
                    df = pd.DataFrame(
                        [
                            {
                                "timestamp": bar.timestamp,
                                "open": bar.open,
                                "high": bar.high,
                                "low": bar.low,
                                "close": bar.close,
                                "volume": bar.volume or 0,
                            }
                            for bar in bars
                        ]
                    )
                    df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)
                    results[symbol] = df
                    logger.info(f"  [{i+1}/{total}] {symbol}: {len(df)} bars")
                else:
                    logger.warning(f"  [{i+1}/{total}] {symbol}: no data")
                    results[symbol] = pd.DataFrame()

                # Progress callback
                if self._progress_callback:
                    self._progress_callback(i + 1, total, symbol)

            except Exception as e:
                logger.error(f"  [{i+1}/{total}] {symbol}: failed - {e}")
                results[symbol] = pd.DataFrame()

        successful = sum(1 for df in results.values() if not df.empty)
        logger.info(f"Fetch complete: {successful}/{total} symbols with data")

        return results

    async def fetch_bars_single(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a single symbol.

        Convenience method for single-symbol fetches.

        Args:
            symbol: Symbol to fetch
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe

        Returns:
            DataFrame with OHLCV columns indexed by timestamp
        """
        results = await self.fetch_bars([symbol], start, end, timeframe)
        return results.get(symbol, pd.DataFrame())

    def get_supported_timeframes(self) -> List[str]:
        """Return IB-supported timeframes."""
        if self._adapter:
            return self._adapter.get_supported_timeframes()
        # Fallback from IbHistoricalAdapter.TIMEFRAME_TO_IB_BAR_SIZE
        return ["1s", "5s", "15s", "30s", "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]

    @property
    def client_id(self) -> int:
        """Get the client ID being used."""
        return self._client_id

    @property
    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connected and self._adapter is not None

    @property
    def request_count(self) -> int:
        """Get number of requests made this session."""
        return self._request_count

    # =========================================================================
    # SYNC WRAPPERS (for integration with sync engines like VectorBT)
    # =========================================================================

    def fetch_bars_sync(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Synchronous wrapper for fetch_bars.

        Creates a fresh event loop for each call to avoid ib_async
        event loop caching issues.
        """
        # Create a fresh event loop to avoid "Event loop is closed" errors
        # ib_async caches loop references, so we need a clean loop each time
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._fetch_bars_with_lifecycle(symbols, start, end, timeframe)
            )
        finally:
            loop.close()
            # Reset to avoid polluting other code
            asyncio.set_event_loop(None)

    async def _fetch_bars_with_lifecycle(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch bars with automatic connect/disconnect."""
        try:
            await self.connect()
            return await self.fetch_bars(symbols, start, end, timeframe)
        finally:
            await self.disconnect()

    def fetch_bars_single_sync(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Synchronous wrapper for single-symbol fetch.

        Connects, fetches, disconnects automatically.
        """
        results = self.fetch_bars_sync([symbol], start, end, timeframe)
        return results.get(symbol, pd.DataFrame())


def create_backtest_provider(
    index: int = 0,
    host: str = "127.0.0.1",
    port: int = 4001,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> IbBacktestDataProvider:
    """
    Create a backtest data provider with a unique client ID.

    Factory function for parallel backtests - each provider gets
    a unique client ID to allow concurrent IB connections.

    Args:
        index: Index for parallel backtests (0-6 supported)
        host: IB host
        port: IB port
        progress_callback: Optional progress callback

    Returns:
        IbBacktestDataProvider with unique client ID

    Raises:
        ValueError: If index exceeds available client IDs

    Example:
        # For parallel backtests
        providers = [create_backtest_provider(i) for i in range(4)]
        await asyncio.gather(*[p.connect() for p in providers])
    """
    if index >= len(IbBacktestDataProvider.BACKTEST_CLIENT_IDS):
        raise ValueError(
            f"index {index} exceeds available client IDs "
            f"(max {len(IbBacktestDataProvider.BACKTEST_CLIENT_IDS) - 1})"
        )

    client_id = IbBacktestDataProvider.BACKTEST_CLIENT_IDS[index]
    return IbBacktestDataProvider(
        host=host,
        port=port,
        client_id=client_id,
        progress_callback=progress_callback,
    )
