"""
Yahoo Finance Historical Data Adapter.

Fetches historical OHLCV bar data from Yahoo Finance.
Implements HistoricalSourcePort for the historical data management system.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

try:
    import yfinance as yf
except ImportError:
    yf = None

from ....domain.events.domain_events import BarData
from ....utils.logging_setup import get_logger
from ....utils.timezone import now_utc

logger = get_logger(__name__)


class YahooHistoricalAdapter:
    """
    Yahoo Finance historical data adapter.

    Fetches historical OHLCV bars using yfinance library.
    Uses adjusted prices (split/dividend adjusted) for backtesting.

    Timeframe support:
    - 5min: Up to 60 days of history
    - 1h: Up to 730 days of history
    - 1d: Full history available

    Note: Yahoo has rate limits. This adapter implements
    rate limiting to avoid 429 errors.
    """

    # Timeframe mapping: APEX format -> yfinance format
    # Supports both short ("5m") and long ("5min") formats for compatibility
    TIMEFRAME_MAP = {
        # Short format (used by BarAggregator and config)
        "1m": "1m",  # Yahoo supports 1m with 7-day limit
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1wk",
        "1M": "1mo",
        # Long format aliases (legacy)
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
    }

    # Maximum history available per timeframe
    # Yahoo returns from IPO (stocks) or inception (ETFs) automatically
    # Intraday limits are hard Yahoo API limits
    MAX_HISTORY_DAYS = {
        "1m": 7,  # 7 days only for 1-minute data
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "1h": 730,
        "4h": 730,
        "1d": 18250,  # ~50 years (Yahoo returns from IPO/inception)
        "1w": 18250,
        "1M": 18250,
        # Long format aliases
        "5min": 60,
        "15min": 60,
        "30min": 60,
    }

    def __init__(
        self,
        rate_limit_per_sec: float = 1.0,
    ) -> None:
        """
        Initialize Yahoo historical adapter.

        Args:
            rate_limit_per_sec: Maximum requests per second.
        """
        self._rate_limit = rate_limit_per_sec
        self._last_request_time: Optional[datetime] = None
        self._yf_available = yf is not None

        if not self._yf_available:
            logger.warning("yfinance not installed. YahooHistoricalAdapter unavailable.")

    @property
    def source_name(self) -> str:
        """Get source identifier."""
        return "yahoo"

    def supports_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is supported."""
        return timeframe in self.TIMEFRAME_MAP

    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        return list(self.TIMEFRAME_MAP.keys())

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        """
        Fetch historical bars from Yahoo Finance.

        Uses adjusted prices (auto_adjust=True) for split/dividend adjustments.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL').
            timeframe: Bar size ('5min', '1h', '1d').
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            List of BarData sorted by timestamp ascending.

        Raises:
            ValueError: If timeframe not supported or invalid date range.
            ConnectionError: If Yahoo Finance unavailable.
        """
        if not self._yf_available:
            raise ConnectionError("yfinance not installed")

        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Validate date range
        max_days = self.MAX_HISTORY_DAYS.get(timeframe, 60)
        requested_days = (end - start).days
        if requested_days > max_days:
            logger.warning(
                f"Requested {requested_days} days of {timeframe} data, "
                f"but Yahoo only provides {max_days} days. Truncating."
            )
            start = end - timedelta(days=max_days)

        # Rate limiting
        await self._rate_limit_wait()

        # Fetch in thread to avoid blocking
        bars = await asyncio.to_thread(
            self._fetch_from_yahoo,
            symbol,
            timeframe,
            start,
            end,
        )

        logger.info(
            f"Fetched {len(bars)} {timeframe} bars for {symbol} "
            f"from {start.date()} to {end.date()}"
        )

        return bars

    async def _rate_limit_wait(self) -> None:
        """Wait for rate limit if needed."""
        if self._last_request_time:
            elapsed = (now_utc() - self._last_request_time).total_seconds()
            min_interval = 1.0 / self._rate_limit
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = now_utc()

    def _fetch_from_yahoo(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        """
        Synchronous fetch from Yahoo Finance.

        Args:
            symbol: Ticker symbol.
            timeframe: APEX timeframe format.
            start: Start datetime.
            end: End datetime.

        Returns:
            List of BarData.
        """
        yf_interval = self.TIMEFRAME_MAP[timeframe]

        try:
            ticker = yf.Ticker(symbol)

            # yfinance end is exclusive for daily bars; intraday uses datetime directly
            intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
            if yf_interval in intraday_intervals:
                yf_end = end
            else:
                yf_end = end + timedelta(days=1)

            # Fetch historical data with adjusted prices
            df = ticker.history(
                start=start,
                end=yf_end,
                interval=yf_interval,
                auto_adjust=True,  # Adjust for splits/dividends
                prepost=False,  # Regular market hours only
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return []

            # Convert to BarData
            bars = []
            for idx, row in df.iterrows():
                # Convert index to datetime
                if hasattr(idx, "to_pydatetime"):
                    bar_time = idx.to_pydatetime()
                else:
                    bar_time = idx

                # Ensure timezone-aware
                if bar_time.tzinfo is None:
                    import pytz

                    bar_time = pytz.UTC.localize(bar_time)

                bar = BarData(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(row["Open"]) if not _isna(row["Open"]) else None,
                    high=float(row["High"]) if not _isna(row["High"]) else None,
                    low=float(row["Low"]) if not _isna(row["Low"]) else None,
                    close=float(row["Close"]) if not _isna(row["Close"]) else None,
                    volume=int(row["Volume"]) if not _isna(row["Volume"]) else None,
                    bar_start=bar_time,
                    bar_end=self._calculate_bar_end(bar_time, timeframe),
                    source=self.source_name,
                    timestamp=bar_time,
                )
                bars.append(bar)

            return bars

        except Exception as e:
            logger.error(f"Error fetching {symbol} from Yahoo: {e}")
            if "Too Many Requests" in str(e) or "429" in str(e):
                raise ConnectionError(f"Yahoo Finance rate limited: {e}")
            raise

    def _calculate_bar_end(
        self,
        bar_start: datetime,
        timeframe: str,
    ) -> datetime:
        """Calculate bar end time based on timeframe."""
        delta_map = {
            # Short format
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1M": timedelta(days=30),
            # Long format aliases
            "5min": timedelta(minutes=5),
            "15min": timedelta(minutes=15),
            "30min": timedelta(minutes=30),
        }
        delta = delta_map.get(timeframe, timedelta(days=1))
        return bar_start + delta

    async def fetch_bars_batch(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, List[BarData]]:
        """
        Fetch bars for multiple symbols.

        Args:
            symbols: List of ticker symbols.
            timeframe: Bar timeframe.
            start: Start datetime.
            end: End datetime.

        Returns:
            Dict mapping symbol to list of bars.
        """
        results = {}

        for symbol in symbols:
            try:
                bars = await self.fetch_bars(symbol, timeframe, start, end)
                results[symbol] = bars
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = []

        return results

    def get_max_history_days(self, timeframe: str) -> int:
        """Get maximum history available for timeframe."""
        return self.MAX_HISTORY_DAYS.get(timeframe, 60)


def _isna(value: object) -> bool:
    """Check if value is NaN/None."""
    if value is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (ImportError, TypeError):
        return False
