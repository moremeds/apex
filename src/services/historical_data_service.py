"""
Historical Data Service for fetching and caching bar data.

Uses the IbConnectionPool's dedicated historical connection to avoid
blocking the monitoring connection. Provides:
- Batch fetching for multiple symbols (reduces API calls)
- In-memory LRU cache for fast repeated access
- Support for multiple timeframes

Usage:
    service = HistoricalDataService(ib_historical=pool.historical)

    # Single symbol
    bars = await service.fetch_bars("AAPL", "1d", BarPeriod.bars(60))

    # Batch fetch
    results = await service.fetch_bars_batch([
        {"symbol": "AAPL", "timeframe": "1d", "period": BarPeriod.bars(60)},
        {"symbol": "MSFT", "timeframe": "1d", "period": BarPeriod.bars(60)},
    ])
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, TYPE_CHECKING

from ..utils.logging_setup import get_logger
from ..domain.events.domain_events import BarData
from .bar_cache_service import BarPeriod, BarCacheStore

if TYPE_CHECKING:
    from ib_async import IB


logger = get_logger(__name__)


# Mapping from our timeframe strings to IB bar sizes
TIMEFRAME_TO_IB_BAR_SIZE = {
    "1s": "1 secs",
    "5s": "5 secs",
    "15s": "15 secs",
    "30s": "30 secs",
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "1w": "1 week",
    "1M": "1 month",
}


class HistoricalDataService:
    """
    Historical data service using dedicated IB connection.

    Features:
    - In-memory LRU cache for fast repeated access
    - Batch fetching to minimize IB API calls
    - Automatic duration calculation for date ranges
    """

    def __init__(
        self,
        ib_historical: "IB",
        cache_size: int = 512,
        default_daily_lookback: int = 60,
    ):
        """
        Initialize historical data service.

        Args:
            ib_historical: Dedicated IB connection for historical data.
            cache_size: Max entries in LRU cache.
            default_daily_lookback: Default days for daily bars (60 = ~3 months).
        """
        self._ib = ib_historical
        self._cache = BarCacheStore(max_entries=cache_size)
        self._default_daily_lookback = default_daily_lookback

        # Track pending fetches to avoid duplicate requests
        self._pending_fetches: Dict[str, asyncio.Future] = {}

    @property
    def cache(self) -> BarCacheStore:
        """Get the underlying cache store."""
        return self._cache

    def is_connected(self) -> bool:
        """Check if the IB connection is alive."""
        return self._ib is not None and self._ib.isConnected()

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        period: BarPeriod,
        use_cache: bool = True,
    ) -> List[BarData]:
        """
        Fetch historical bars for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL").
            timeframe: Bar timeframe (e.g., "1d", "1h", "5m").
            period: BarPeriod specifying bars count or date range.
            use_cache: Whether to check/update cache.

        Returns:
            List of BarData sorted by timestamp ascending.
        """
        # Check cache first
        if use_cache:
            cached = self._cache.get(symbol, timeframe, period)
            if cached:
                logger.debug(f"Cache hit: {symbol} {timeframe}")
                return cached

        # Avoid duplicate concurrent fetches for same key
        cache_key = f"{symbol}:{timeframe}:{period.mode}:{period.count or period.start}"
        if cache_key in self._pending_fetches:
            logger.debug(f"Waiting for pending fetch: {cache_key}")
            return await self._pending_fetches[cache_key]

        # Create future for this fetch
        future = asyncio.get_event_loop().create_future()
        self._pending_fetches[cache_key] = future

        try:
            bars = await self._fetch_from_ib(symbol, timeframe, period)

            # Update cache
            if use_cache and bars:
                self._cache.put(symbol, timeframe, bars)

            future.set_result(bars)
            return bars

        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._pending_fetches.pop(cache_key, None)

    async def fetch_bars_batch(
        self,
        requests: List[Dict],
        use_cache: bool = True,
    ) -> Dict[str, List[BarData]]:
        """
        Fetch bars for multiple symbols efficiently.

        Batches requests and fetches concurrently to minimize latency.

        Args:
            requests: List of dicts with keys: symbol, timeframe, period.
                Example: [
                    {"symbol": "AAPL", "timeframe": "1d", "period": BarPeriod.bars(60)},
                    {"symbol": "MSFT", "timeframe": "1d", "period": BarPeriod.bars(60)},
                ]
            use_cache: Whether to check/update cache.

        Returns:
            Dict mapping symbol to List[BarData].
        """
        results: Dict[str, List[BarData]] = {}
        to_fetch: List[Dict] = []

        # Check cache for each request
        for req in requests:
            symbol = req["symbol"]
            timeframe = req.get("timeframe", "1d")
            period = req.get("period", BarPeriod.bars(self._default_daily_lookback))

            if use_cache:
                cached = self._cache.get(symbol, timeframe, period)
                if cached:
                    results[symbol] = cached
                    continue

            to_fetch.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "period": period,
            })

        if not to_fetch:
            logger.debug(f"Batch: all {len(requests)} symbols from cache")
            return results

        logger.info(f"Batch: fetching {len(to_fetch)} symbols from IB "
                   f"({len(results)} from cache)")

        # Fetch remaining symbols concurrently
        tasks = [
            self._fetch_from_ib(req["symbol"], req["timeframe"], req["period"])
            for req in to_fetch
        ]

        fetched = await asyncio.gather(*tasks, return_exceptions=True)

        for req, bars in zip(to_fetch, fetched):
            symbol = req["symbol"]
            if isinstance(bars, Exception):
                logger.warning(f"Failed to fetch {symbol}: {bars}")
                results[symbol] = []
            else:
                results[symbol] = bars
                if use_cache and bars:
                    self._cache.put(symbol, req["timeframe"], bars)

        return results

    async def prefetch_daily_bars(
        self,
        symbols: List[str],
        lookback_days: int = 60,
    ) -> int:
        """
        Pre-fetch daily bars for a list of symbols.

        Designed for startup pre-loading. Returns count of successfully fetched.

        Args:
            symbols: List of symbols to prefetch.
            lookback_days: Number of days to fetch (default 60 = ~3 months).

        Returns:
            Number of symbols successfully fetched.
        """
        if not symbols:
            return 0

        period = BarPeriod.bars(lookback_days)
        requests = [
            {"symbol": sym, "timeframe": "1d", "period": period}
            for sym in symbols
        ]

        logger.info(f"Pre-fetching {lookback_days}d daily bars for {len(symbols)} symbols")
        results = await self.fetch_bars_batch(requests, use_cache=True)

        success_count = sum(1 for bars in results.values() if bars)
        logger.info(f"Pre-fetch complete: {success_count}/{len(symbols)} symbols loaded")

        return success_count

    async def _fetch_from_ib(
        self,
        symbol: str,
        timeframe: str,
        period: BarPeriod,
    ) -> List[BarData]:
        """
        Fetch bars from IB API.

        Internal method - use fetch_bars() for caching.
        """
        if not self.is_connected():
            raise ConnectionError("Historical IB connection not available")

        bar_size = TIMEFRAME_TO_IB_BAR_SIZE.get(timeframe)
        if not bar_size:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Calculate duration and end date
        if period.mode == "bars" and period.count:
            duration = self._calculate_duration_for_bars(timeframe, period.count)
            end_dt = datetime.now()
        elif period.mode == "range" and period.start and period.end:
            duration = self._calculate_duration_for_range(period.start, period.end)
            end_dt = period.end
        else:
            # Default: use count-based
            count = period.count or self._default_daily_lookback
            duration = self._calculate_duration_for_bars(timeframe, count)
            end_dt = datetime.now()

        try:
            from ib_async import Stock

            contract = Stock(symbol, "SMART", currency="USD")
            await self._ib.qualifyContractsAsync(contract)

            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            result = []
            for bar in bars:
                bar_ts = bar.date if hasattr(bar, 'date') else datetime.now()
                if isinstance(bar_ts, date) and not isinstance(bar_ts, datetime):
                    bar_ts = datetime.combine(bar_ts, datetime.min.time())

                bar_data = BarData(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=int(bar.volume) if bar.volume else None,
                    bar_start=bar_ts,
                    source="IB",
                    timestamp=bar_ts,
                )
                result.append(bar_data)

            # Filter by start date if range mode
            if period.mode == "range" and period.start:
                start_dt = period.start if isinstance(period.start, datetime) else \
                          datetime.combine(period.start, datetime.min.time())
                result = [b for b in result if b.timestamp and b.timestamp >= start_dt]

            # Apply count limit if specified
            if period.mode == "bars" and period.count and len(result) > period.count:
                result = result[-period.count:]

            logger.debug(f"Fetched {len(result)} {timeframe} bars for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            raise

    def _calculate_duration_for_bars(self, timeframe: str, count: int) -> str:
        """Calculate IB duration string for a given bar count."""
        # Estimate time range needed based on timeframe
        if timeframe == "1d":
            # Add buffer for weekends/holidays
            days = int(count * 1.5) + 10
            if days <= 365:
                return f"{days} D"
            else:
                return f"{(days // 365) + 1} Y"
        elif timeframe == "1w":
            weeks = count + 2
            return f"{weeks * 7} D"
        elif timeframe in ("1h", "4h"):
            hours = count * (1 if timeframe == "1h" else 4)
            days = (hours // 24) + 5
            return f"{min(days, 365)} D"
        elif timeframe in ("1m", "5m", "15m", "30m"):
            # IB limits intraday data
            return "5 D"
        else:
            return "30 D"

    def _calculate_duration_for_range(
        self,
        start: datetime,
        end: datetime,
    ) -> str:
        """Calculate IB duration string for a date range."""
        delta = end - start
        days = delta.days + 1

        if days <= 1:
            return f"{int(delta.total_seconds())} S"
        elif days <= 365:
            return f"{days} D"
        else:
            years = (days // 365) + 1
            return f"{years} Y"

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return {
            "entries": len(self._cache._cache),
            "max_entries": self._cache._max_entries,
        }
