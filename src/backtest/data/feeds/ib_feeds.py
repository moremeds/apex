"""
Interactive Brokers data feeds for backtesting.

Contains:
- IbHistoricalDataFeed: Loads historical data from IB
- BarCacheDataFeed: Thin wrapper delegating to IbHistoricalDataFeed
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, AsyncIterator, List, Optional

from ....domain.events.domain_events import BarData
from .base import DataFeed
from .models import HistoricalBar

logger = logging.getLogger(__name__)


class IbHistoricalDataFeed(DataFeed):
    """
    Load historical data from Interactive Brokers (legacy).

    Prefer BarCacheDataFeed for production backtests.
    Uses the IbHistoricalAdapter to fetch real market data.

    Requirements:
    - IB TWS or IB Gateway running
    - ib_async package installed
    - Market data subscription for requested symbols

    Usage:
        feed = IbHistoricalDataFeed(
            symbols=["AAPL", "MSFT"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            bar_size="1d",
            host="127.0.0.1",
            port=7497,  # TWS paper: 7497, TWS live: 7496, Gateway: 4001/4002
        )
        await feed.load()

        async for bar in feed.stream_bars():
            print(bar)
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        bar_size: str = "1d",
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 10,
    ):
        """
        Initialize IB historical data feed.

        Args:
            symbols: List of symbols to load.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            bar_size: Bar size (1m, 5m, 15m, 1h, 1d, etc.).
            host: IB TWS/Gateway host.
            port: IB TWS/Gateway port.
            client_id: IB client ID.
        """
        self._symbols = symbols
        self._start_date = start_date
        self._end_date = end_date
        self._bar_size = bar_size
        self._host = host
        self._port = port
        self._client_id = client_id

        self._bars: List[HistoricalBar] = []
        self._loaded = False
        self._adapter: Optional[Any] = None

    async def load(self) -> None:
        """
        Load historical data from IB.

        Connects to IB, fetches data for all symbols, then disconnects.
        """
        from ....infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

        self._bars.clear()

        # Create and connect adapter
        adapter = IbHistoricalAdapter(
            host=self._host,
            port=self._port,
            client_id=self._client_id,
        )
        self._adapter = adapter

        try:
            await adapter.connect()
            logger.info(f"Connected to IB at {self._host}:{self._port}")

            # Convert dates to datetime
            start_dt = datetime.combine(self._start_date, datetime.min.time())
            end_dt = datetime.combine(self._end_date, datetime.max.time())

            # Fetch data for each symbol
            for symbol in self._symbols:
                logger.info(f"Fetching {self._bar_size} bars for {symbol}...")

                try:
                    bars = await adapter.fetch_bars(
                        symbol=symbol,
                        timeframe=self._bar_size,
                        start=start_dt,
                        end=end_dt,
                    )

                    for bar in bars:
                        hist_bar = HistoricalBar(
                            symbol=symbol,
                            timestamp=bar.timestamp,
                            open=bar.open or 0.0,
                            high=bar.high or 0.0,
                            low=bar.low or 0.0,
                            close=bar.close or 0.0,
                            volume=bar.volume or 0,
                            bar_size=self._bar_size,
                        )
                        self._bars.append(hist_bar)

                    logger.info(f"Loaded {len(bars)} bars for {symbol}")

                except Exception as e:
                    logger.error(f"Failed to fetch bars for {symbol}: {e}")

            # Sort all bars by timestamp
            self._bars.sort(key=lambda b: b.timestamp)
            self._loaded = True

            logger.info(
                f"IbHistoricalDataFeed loaded {len(self._bars)} total bars "
                f"for {len(self._symbols)} symbols"
            )

        finally:
            # Always disconnect
            await adapter.disconnect()
            logger.info("Disconnected from IB")

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        if not self._loaded:
            await self.load()

        for bar in self._bars:
            yield bar.to_bar_data(source="IB")

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._symbols

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return len(self._bars)


class BarCacheDataFeed(DataFeed):
    """
    Load historical data from IB for backtesting.

    For standalone backtests, this wraps IbHistoricalDataFeed.
    For in-process backtests (Lab panel), use CachedBarDataFeed instead.

    Note: This is a thin wrapper that delegates to IbHistoricalDataFeed.
    Standalone backtests run in their own process, so no event loop conflicts.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        bar_size: str = "1d",
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 10,
    ):
        """
        Initialize bar cache data feed.

        Args:
            symbols: List of symbols to load.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            bar_size: Bar size (1d, 4h, 1h, etc.).
            host: IB TWS/Gateway host.
            port: IB TWS/Gateway port.
            client_id: IB client ID for historical requests.
        """
        # Delegate to IbHistoricalDataFeed
        self._delegate = IbHistoricalDataFeed(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            bar_size=bar_size,
            host=host,
            port=port,
            client_id=client_id,
        )

    async def load(self) -> None:
        """Load data from IB."""
        await self._delegate.load()

    async def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        async for bar in self._delegate.stream_bars():
            yield bar

    def get_symbols(self) -> List[str]:
        """Get list of symbols."""
        return self._delegate.get_symbols()

    @property
    def bar_count(self) -> int:
        """Get total bar count."""
        return self._delegate.bar_count
