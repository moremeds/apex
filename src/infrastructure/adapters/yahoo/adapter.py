"""
Yahoo Finance Adapter - Market data and fundamentals provider.

Provides:
- Stock prices (bid/ask/last/volume)
- Beta values for SPY-equivalent exposure calculation
- Previous close prices
- Basic fundamentals (market cap, PE ratio, etc.)

Limitations:
- No Greeks (options need IBKR or specialized provider)
- No real-time streaming (polling-based)
- Rate limits apply (use caching)
"""

from __future__ import annotations
import asyncio
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from threading import RLock
from dataclasses import dataclass

try:
    import yfinance as yf
except ImportError:
    yf = None

from ....domain.interfaces.market_data_provider import MarketDataProvider
from ....models.position import Position, AssetType
from ....models.market_data import MarketData, DataQuality, GreeksSource
from ....utils.timezone import now_utc
from ....utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class CachedData:
    """Cached market data entry with timestamp."""
    data: MarketData
    beta: Optional[float]
    fetched_at: datetime


class YahooFinanceAdapter(MarketDataProvider):
    """
    Yahoo Finance market data adapter.

    Provides stock prices and beta values from Yahoo Finance.
    Uses caching to reduce API calls (yfinance has rate limits).

    Features:
    - Price data: bid, ask, last, volume, previous close
    - Beta values for SPY-equivalent exposure (fetched from yfinance)
    - Thread-safe caching with configurable TTL

    Note: Does not support options Greeks - use IBKR for that.
    """

    DEFAULT_BETA = 1.0  # Default beta when yfinance doesn't have data
    MAX_CONCURRENT_REQUESTS = 3  # Limit concurrent requests to avoid rate limiting

    def __init__(
        self,
        price_ttl_seconds: int = 30,
        beta_ttl_hours: int = 24,
        max_cache_size: int = 500,  # m6: Bounded cache size
    ):
        """
        Initialize Yahoo Finance adapter.

        Args:
            price_ttl_seconds: Cache TTL for price data (default: 30s)
            beta_ttl_hours: Cache TTL for beta values (default: 24h)
            max_cache_size: Maximum number of symbols to cache (default: 500)
        """
        self._price_ttl = timedelta(seconds=price_ttl_seconds)
        self._beta_ttl = timedelta(hours=beta_ttl_hours)
        self._max_cache_size = max_cache_size

        self._cache: Dict[str, CachedData] = {}
        self._lock = RLock()
        self._connected = False
        self._yf_available = yf is not None
        self._last_request_time: Optional[datetime] = None
        self._min_request_interval = timedelta(seconds=1)  # Rate limit: 1 req/sec to avoid Yahoo 429

        if not self._yf_available:
            logger.warning("yfinance not installed. YahooFinanceAdapter will return empty data.")

    async def connect(self) -> None:
        """Connect (no-op for yfinance - it's request-based)."""
        if not self._yf_available:
            logger.warning("yfinance not available - Yahoo Finance adapter limited")
        self._connected = True
        logger.info("YahooFinanceAdapter connected")

    async def disconnect(self) -> None:
        """Disconnect (no-op for yfinance)."""
        self._connected = False
        logger.info("YahooFinanceAdapter disconnected")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def _cache_put(self, symbol: str, data: CachedData) -> None:
        """
        Put entry in cache with LRU-style eviction.

        m6: Bounded cache - evicts oldest entries when over max_cache_size.
        Must be called with self._lock held.
        """
        # If symbol already in cache, just update (no size change)
        if symbol in self._cache:
            self._cache[symbol] = data
            return

        # Check if we need to evict before adding new entry
        if len(self._cache) >= self._max_cache_size:
            # Evict oldest entries (by fetched_at timestamp)
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].fetched_at
            )
            # Evict 10% of cache to avoid frequent eviction
            evict_count = max(1, self._max_cache_size // 10)
            for sym, _ in sorted_entries[:evict_count]:
                del self._cache[sym]
            logger.debug("m6: Evicted %d oldest cache entries", evict_count)

        self._cache[symbol] = data

    async def fetch_market_data(self, positions: List[Position]) -> List[MarketData]:
        """
        Fetch market data for positions.

        Only fetches data for stock positions (not options).
        Options require Greeks which yfinance doesn't provide.

        Args:
            positions: List of positions to fetch data for

        Returns:
            List of MarketData objects for stock positions
        """
        if not self._connected:
            raise ConnectionError("YahooFinanceAdapter not connected")

        if not self._yf_available:
            return []

        # Extract unique stock symbols (options handled by IBKR)
        stock_symbols = list(set(
            pos.symbol for pos in positions
            if pos.asset_type == AssetType.STOCK
        ))

        if not stock_symbols:
            return []

        return await self._fetch_batch(stock_symbols)

    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Fetch quotes for symbols.

        Args:
            symbols: List of symbols to fetch

        Returns:
            Dict mapping symbol to MarketData
        """
        if not self._connected:
            raise ConnectionError("YahooFinanceAdapter not connected")

        if not self._yf_available or not symbols:
            return {}

        market_data_list = await self._fetch_batch(symbols)
        return {md.symbol: md for md in market_data_list}

    async def _fetch_batch(self, symbols: List[str]) -> List[MarketData]:
        """
        Fetch market data for a batch of symbols.

        Uses yfinance batch download for efficiency.
        """
        # Check cache first
        fresh_data = []
        stale_symbols = []

        with self._lock:
            for symbol in symbols:
                if symbol in self._cache:
                    entry = self._cache[symbol]
                    if now_utc() - entry.fetched_at < self._price_ttl:
                        fresh_data.append(entry.data)
                        continue
                stale_symbols.append(symbol)

        # Fetch stale/missing symbols - run in thread to avoid blocking event loop
        # (yfinance uses blocking HTTP calls and we have rate limit sleeps)
        if stale_symbols:
            fetched = await asyncio.to_thread(self._fetch_from_yahoo, stale_symbols)
            fresh_data.extend(fetched)

        return fresh_data

    def _fetch_from_yahoo(self, symbols: List[str]) -> List[MarketData]:
        """
        Fetch data from Yahoo Finance API using rate-limited sequential requests.

        Args:
            symbols: Symbols to fetch

        Returns:
            List of MarketData objects
        """
        import time

        market_data_list = []

        for symbol in symbols:
            # Rate limiting: ensure minimum interval between requests
            if self._last_request_time:
                elapsed = now_utc() - self._last_request_time
                if elapsed < self._min_request_interval:
                    sleep_time = (self._min_request_interval - elapsed).total_seconds()
                    time.sleep(sleep_time)

            self._last_request_time = now_utc()

            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                if info:
                    md = self._parse_ticker_info(symbol, info)
                    market_data_list.append(md)

                    # Cache the result (m6: bounded cache with eviction)
                    with self._lock:
                        self._cache_put(symbol, CachedData(
                            data=md,
                            beta=info.get("beta"),
                            fetched_at=now_utc(),
                        ))
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                # On rate limit, cache default to avoid retry storm
                if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                    with self._lock:
                        if symbol not in self._cache:
                            self._cache_put(symbol, CachedData(
                                data=MarketData(symbol=symbol, timestamp=now_utc()),
                                beta=self.DEFAULT_BETA,
                                fetched_at=now_utc(),
                            ))

        return market_data_list

    def _parse_ticker_info(self, symbol: str, info: dict) -> MarketData:
        """
        Parse yfinance ticker info into MarketData.

        Args:
            symbol: Stock symbol
            info: yfinance ticker.info dict

        Returns:
            MarketData object
        """
        # Price data
        bid = info.get("bid")
        ask = info.get("ask")
        last = info.get("regularMarketPrice") or info.get("currentPrice")
        volume = info.get("volume") or info.get("regularMarketVolume")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

        # Calculate mid
        mid = None
        if bid and ask and bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        elif last and last > 0:
            mid = last

        # Beta value
        beta = info.get("beta")

        # Determine data quality
        quality = DataQuality.GOOD
        if not last or last <= 0:
            quality = DataQuality.MISSING
        elif not bid or not ask:
            quality = DataQuality.SUSPICIOUS

        return MarketData(
            symbol=symbol,
            last=float(last) if last else None,
            bid=float(bid) if bid else None,
            ask=float(ask) if ask else None,
            mid=float(mid) if mid else None,
            volume=int(volume) if volume else None,
            yesterday_close=float(prev_close) if prev_close else None,
            timestamp=now_utc(),
            quality=quality,
            greeks_source=GreeksSource.MISSING,  # Yahoo doesn't provide Greeks
            beta=float(beta) if beta else None,
            delta=1.0,  # Stocks have delta of 1
        )

    def get_beta(self, symbol: str) -> float:
        """
        Get beta for a symbol from yfinance.

        Priority:
        1. Fresh cached value from yfinance
        2. Fetch from yfinance
        3. Default beta (1.0)

        Args:
            symbol: Stock symbol

        Returns:
            Beta value
        """
        # Check cache
        with self._lock:
            if symbol in self._cache:
                entry = self._cache[symbol]
                if now_utc() - entry.fetched_at < self._beta_ttl:
                    if entry.beta is not None:
                        return entry.beta

        # Fetch from Yahoo
        if self._yf_available:
            beta = self._fetch_beta(symbol)
            if beta is not None:
                return beta

        return self.DEFAULT_BETA

    def get_betas(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get betas for multiple symbols efficiently using batch fetching.

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol to beta
        """
        if not symbols:
            return {}

        result: Dict[str, float] = {}
        to_fetch: List[str] = []

        # Check cache first for all symbols
        with self._lock:
            for symbol in symbols:
                if symbol in self._cache:
                    entry = self._cache[symbol]
                    if now_utc() - entry.fetched_at < self._beta_ttl:
                        if entry.beta is not None:
                            result[symbol] = entry.beta
                            continue
                to_fetch.append(symbol)

        # Batch fetch missing symbols
        if to_fetch and self._yf_available:
            self._fetch_from_yahoo(to_fetch)
            # Now get from cache
            with self._lock:
                for symbol in to_fetch:
                    if symbol in self._cache and self._cache[symbol].beta is not None:
                        result[symbol] = self._cache[symbol].beta
                    else:
                        result[symbol] = self.DEFAULT_BETA
        else:
            # No yfinance available, use default
            for symbol in to_fetch:
                result[symbol] = self.DEFAULT_BETA

        return result

    def prefetch_betas(self, symbols: List[str]) -> int:
        """
        Prefetch betas for a list of symbols.

        Warms the cache at startup.

        Args:
            symbols: Symbols to prefetch

        Returns:
            Number of successfully fetched betas
        """
        if not self._yf_available:
            return 0

        # Filter to symbols not in cache
        with self._lock:
            to_fetch = [
                s for s in set(symbols)
                if s not in self._cache or
                (now_utc() - self._cache[s].fetched_at >= self._beta_ttl)
            ]

        if not to_fetch:
            return 0

        logger.info(f"Prefetching betas for {len(to_fetch)} symbols...")
        success_count = 0

        # Fetch in batches to avoid rate limits
        batch_size = 50
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i:i + batch_size]
            fetched = self._fetch_from_yahoo(batch)
            success_count += len(fetched)

        logger.info(f"Prefetched {success_count}/{len(to_fetch)} betas")
        return success_count

    def _fetch_beta(self, symbol: str) -> Optional[float]:
        """
        Fetch beta for a single symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Beta value or None
        """
        import time

        # Rate limiting
        if self._last_request_time:
            elapsed = now_utc() - self._last_request_time
            if elapsed < self._min_request_interval:
                sleep_time = (self._min_request_interval - elapsed).total_seconds()
                time.sleep(sleep_time)
        self._last_request_time = now_utc()

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            beta = info.get("beta")

            # Cache the result (m6: bounded cache with eviction)
            with self._lock:
                if symbol in self._cache:
                    self._cache[symbol].beta = float(beta) if beta else None
                    self._cache[symbol].fetched_at = now_utc()
                else:
                    # Create minimal cache entry for beta
                    md = MarketData(symbol=symbol, timestamp=now_utc())
                    self._cache_put(symbol, CachedData(
                        data=md,
                        beta=float(beta) if beta else None,
                        fetched_at=now_utc(),
                    ))

            if beta is not None:
                logger.debug(f"Fetched beta for {symbol}: {beta:.2f}")
                return float(beta)
            return None

        except Exception as e:
            logger.warning(f"Failed to fetch beta for {symbol}: {e}")
            # On rate limit, cache default to avoid retry storm
            if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                with self._lock:
                    if symbol not in self._cache:
                        self._cache_put(symbol, CachedData(
                            data=MarketData(symbol=symbol, timestamp=now_utc()),
                            beta=self.DEFAULT_BETA,
                            fetched_at=now_utc(),
                        ))
            return None

    # -------------------------------------------------------------------------
    # MarketDataProvider interface methods (mostly no-op for yfinance)
    # -------------------------------------------------------------------------

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe (no-op - yfinance doesn't support streaming)."""
        logger.debug(f"YahooFinanceAdapter.subscribe called for {len(symbols)} symbols (no-op)")

    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe (no-op)."""
        pass

    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest cached market data."""
        with self._lock:
            entry = self._cache.get(symbol)
            return entry.data if entry else None

    def set_streaming_callback(
        self,
        callback: Optional[Callable[[str, MarketData], None]]
    ) -> None:
        """Set streaming callback (no-op for yfinance)."""
        pass  # yfinance doesn't support streaming

    def supports_streaming(self) -> bool:
        """Check if streaming is supported (no)."""
        return False

    def supports_greeks(self) -> bool:
        """Check if Greeks are supported (no)."""
        return False

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            fresh_count = sum(
                1 for entry in self._cache.values()
                if now_utc() - entry.fetched_at < self._price_ttl
            )
            return {
                "total_cached": len(self._cache),
                "fresh": fresh_count,
                "stale": len(self._cache) - fresh_count,
            }

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cache for a symbol or all symbols.

        Args:
            symbol: Symbol to clear, or None for all
        """
        with self._lock:
            if symbol:
                self._cache.pop(symbol, None)
            else:
                self._cache.clear()
