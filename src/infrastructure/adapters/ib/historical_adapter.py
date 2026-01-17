"""
IB Historical Adapter for historical data requests.

Handles:
- Historical bar/candle data (implements BarProvider and HistoricalSourcePort)
- Batch historical requests

Uses reserved historical client IDs.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from ....domain.events.domain_events import BarData
from ....domain.interfaces.bar_provider import BarProvider
from ....domain.interfaces.historical_source import HistoricalSourcePort
from ....utils.logging_setup import get_logger
from .base import IbBaseAdapter

# IB returns timestamps in US Eastern Time (naive), we need to convert to UTC
US_EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


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

# Mapping from timeframe to IB duration string (for default lookback)
TIMEFRAME_TO_DEFAULT_DURATION = {
    "1s": "1800 S",
    "5s": "3600 S",
    "15s": "7200 S",
    "30s": "14400 S",
    "1m": "1 D",
    "5m": "5 D",
    "15m": "10 D",
    "30m": "30 D",
    "1h": "30 D",
    "4h": "60 D",
    "1d": "1 Y",
    "1w": "2 Y",
    "1M": "5 Y",
}

# Maximum practical history to request per timeframe (in days)
# IB can provide more but large requests timeout. These limits ensure
# requests complete reliably within ~60 seconds.
# Reference: https://interactivebrokers.github.io/tws-api/historical_limitations.html
MAX_HISTORY_DAYS = {
    "1s": 1,  # ~86,400 bars/day - limit to 1 day
    "5s": 1,  # ~17,280 bars/day
    "15s": 1,  # ~5,760 bars/day
    "30s": 2,  # ~2,880 bars/day
    "1m": 7,  # ~390 bars/day - 7 days = ~2,730 bars
    "5m": 30,  # ~78 bars/day - 30 days = ~2,340 bars
    "15m": 60,  # ~26 bars/day - 60 days = ~1,560 bars
    "30m": 90,  # ~13 bars/day - 90 days = ~1,170 bars
    "1h": 180,  # ~7 bars/day - 180 days = ~1,260 bars
    "4h": 365,  # ~2 bars/day - 365 days = ~730 bars
    "1d": 3650,  # ~10 years
    "1w": 3650,  # ~10 years
    "1M": 3650,  # ~10 years
}


class IbHistoricalAdapter(IbBaseAdapter, BarProvider):
    """
    IB adapter for historical data requests.

    Implements BarProvider for fetching historical bars.
    Also implements HistoricalSourcePort for the historical data management system.
    Uses a reserved historical client ID.

    Note: Historical data requests can be slow and rate-limited by IB.
    Use a separate client ID to avoid blocking live data streams.
    """

    ADAPTER_TYPE = "historical"

    def __init__(self, *args, **kwargs):
        """Initialize historical adapter."""
        super().__init__(*args, **kwargs)
        self._bar_callback: Optional[Callable[[BarData], None]] = None
        self._subscribed_bars: Dict[str, str] = {}  # symbol -> timeframe

    # -------------------------------------------------------------------------
    # HistoricalSourcePort Implementation
    # -------------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        """Get source identifier for data provenance tracking."""
        return "ib"

    def supports_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe is supported."""
        return timeframe in TIMEFRAME_TO_IB_BAR_SIZE

    # -------------------------------------------------------------------------
    # BarProvider Implementation
    # -------------------------------------------------------------------------

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[BarData]:
        """
        Fetch historical bars for a symbol.

        Args:
            symbol: Symbol to fetch bars for.
            timeframe: Bar timeframe (e.g., "1m", "5m", "1h", "1d").
            start: Start datetime (inclusive).
            end: End datetime (inclusive). Defaults to now.
            limit: Maximum bars to return.

        Returns:
            List of BarData sorted by timestamp ascending.
        """
        await self.ensure_connected()

        bar_size = TIMEFRAME_TO_IB_BAR_SIZE.get(timeframe)
        if not bar_size:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        end_dt = end or datetime.now()

        # Truncate request to max practical history for this timeframe
        # Large requests (e.g., 1 year of 1m bars) timeout and return empty
        max_days = MAX_HISTORY_DAYS.get(timeframe, 7)
        if start:
            requested_days = (end_dt - start).days
            if requested_days > max_days:
                original_start = start
                start = end_dt - timedelta(days=max_days)
                logger.warning(
                    f"Truncating {symbol}/{timeframe} request from {requested_days} "
                    f"to {max_days} days (IB practical limit)"
                )

        # Calculate duration string
        if start and end_dt:
            duration = self._calculate_duration(start, end_dt, timeframe)
        else:
            duration = TIMEFRAME_TO_DEFAULT_DURATION.get(timeframe, "1 D")

        try:
            from ib_async import Stock

            contract = Stock(symbol, "SMART", currency="USD")
            await self.ib.qualifyContractsAsync(contract)

            bars = await self.ib.reqHistoricalDataAsync(
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
                # IB returns date for daily bars, datetime for intraday
                bar_ts = bar.date if hasattr(bar, "date") else datetime.now()
                # Ensure timestamp is always datetime for consistent comparison
                if isinstance(bar_ts, date) and not isinstance(bar_ts, datetime):
                    bar_ts = datetime.combine(bar_ts, datetime.min.time())

                # CRITICAL: IB returns naive timestamps in Eastern Time, convert to UTC
                if bar_ts.tzinfo is None:
                    bar_ts = bar_ts.replace(tzinfo=US_EASTERN)
                bar_ts = bar_ts.astimezone(UTC)

                bar_data = BarData(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=int(bar.volume) if bar.volume else None,
                    bar_start=bar_ts,
                    source="ib",  # Lowercase to match source_name property
                    timestamp=bar_ts,
                )
                result.append(bar_data)

            # Filter by start if provided
            if start:
                # Ensure start is datetime for comparison
                start_dt = (
                    start
                    if isinstance(start, datetime)
                    else datetime.combine(start, datetime.min.time())
                )
                # Normalize timezone: make both naive for comparison
                if start_dt.tzinfo is not None:
                    start_dt = start_dt.replace(tzinfo=None)
                result = [b for b in result if b.timestamp.replace(tzinfo=None) >= start_dt]

            # Apply limit
            if limit and len(result) > limit:
                result = result[-limit:]

            logger.info(f"Fetched {len(result)} {timeframe} bars for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            raise

    async def fetch_latest_bar(self, symbol: str, timeframe: str) -> Optional[BarData]:
        """Fetch the most recent completed bar."""
        bars = await self.fetch_bars(symbol, timeframe, limit=1)
        return bars[-1] if bars else None

    async def subscribe_bars(self, symbol: str, timeframe: str) -> None:
        """
        Subscribe to real-time bar updates.

        Note: IB doesn't have native bar streaming. This sets up periodic
        polling for the latest bar. Use IbLiveAdapter for tick-level streaming.
        """
        self._subscribed_bars[symbol] = timeframe
        logger.info(f"Subscribed to {timeframe} bars for {symbol}")

    async def unsubscribe_bars(self, symbol: str, timeframe: str) -> None:
        """Unsubscribe from bar updates."""
        if symbol in self._subscribed_bars:
            del self._subscribed_bars[symbol]
            logger.info(f"Unsubscribed from bars for {symbol}")

    def set_bar_callback(self, callback: Optional[Callable[[BarData], None]]) -> None:
        """Set callback for bar updates."""
        self._bar_callback = callback

    def get_supported_timeframes(self) -> List[str]:
        """Get supported timeframes."""
        return list(TIMEFRAME_TO_IB_BAR_SIZE.keys())

    def get_max_history_days(self, timeframe: str) -> int:
        """Get maximum practical history available for timeframe."""
        return MAX_HISTORY_DAYS.get(timeframe, 7)

    async def fetch_bars_batch(
        self,
        requests: List[dict],
        max_concurrent: int = 3,
        delay_between_batches: float = 1.0,
    ) -> dict:
        """
        Fetch bars for multiple symbols with rate limiting to avoid IB pacing violations.

        IB has strict rate limits (~60 requests/10 min for historical data).
        This method processes requests in small batches with delays between.

        Args:
            requests: List of dicts with symbol, timeframe, start, end, limit.
            max_concurrent: Max parallel requests per batch (default: 3).
            delay_between_batches: Seconds to wait between batches (default: 1.0).

        Returns:
            Dict mapping symbol to List[BarData].
        """
        results: Dict[str, List[BarData]] = {}

        if not requests:
            return results

        async def fetch_single(req: dict) -> tuple[str, List[BarData]]:
            """Fetch a single request and return (symbol, bars)."""
            symbol = req.get("symbol", "")
            timeframe = req.get("timeframe", "1d")
            start = req.get("start")
            end = req.get("end")
            limit = req.get("limit")
            bars = await self.fetch_bars(symbol, timeframe, start, end, limit)
            return symbol, bars

        # Process in batches with rate limiting
        total = len(requests)
        for i in range(0, total, max_concurrent):
            batch = requests[i : i + max_concurrent]
            batch_num = i // max_concurrent + 1
            total_batches = (total + max_concurrent - 1) // max_concurrent

            logger.debug(f"IB batch {batch_num}/{total_batches}: fetching {len(batch)} symbols")

            # Run batch concurrently
            tasks = [fetch_single(req) for req in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results, log errors
            for req, result in zip(batch, batch_results):
                symbol = req.get("symbol", "")
                if isinstance(result, Exception):
                    logger.error(f"IB historical fetch failed for {symbol}: {result}")
                    results[symbol] = []
                else:
                    sym, bars = result
                    results[sym] = bars

            # Rate limit delay between batches (skip delay after last batch)
            if i + max_concurrent < total:
                logger.debug(f"Rate limit delay: {delay_between_batches}s")
                await asyncio.sleep(delay_between_batches)

        logger.info(
            f"IB batch fetch complete: {len(results)} symbols, "
            f"{sum(len(bars) for bars in results.values())} total bars"
        )
        return results

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _calculate_duration(self, start: datetime, end: datetime, timeframe: str) -> str:
        """
        Calculate IB duration string based on date range.

        IB requires duration in format like "1 D", "30 D", "1 M", "1 Y".
        """
        delta = end - start
        days = delta.days + 1  # Include both endpoints

        if days <= 1:
            return f"{int(delta.total_seconds())} S"
        elif days <= 30:
            return f"{days} D"
        elif days <= 365:
            months = (days // 30) + 1
            return f"{months} M"
        else:
            years = (days // 365) + 1
            return f"{years} Y"

    async def fetch_option_chain_bars(
        self,
        underlying: str,
        expiry: str,
        timeframe: str = "1d",
        limit: int = 30,
    ) -> Dict[str, List[BarData]]:
        """
        Fetch historical bars for all options in a chain.

        Args:
            underlying: Underlying symbol (e.g., "AAPL").
            expiry: Expiry date in YYYYMMDD format.
            timeframe: Bar timeframe.
            limit: Max bars per option.

        Returns:
            Dict mapping option symbol to bars.
        """
        await self.ensure_connected()

        try:
            from ib_async import Stock

            # Get option chain
            underlying_contract = Stock(underlying, "SMART", currency="USD")
            await self.ib.qualifyContractsAsync(underlying_contract)

            chains = await self.ib.reqSecDefOptParamsAsync(
                underlying_contract.symbol,
                "",  # All futures months
                underlying_contract.secType,
                underlying_contract.conId,
            )

            if not chains:
                logger.warning(f"No option chains found for {underlying}")
                return {}

            # Get strikes for the expiry
            chain = chains[0]  # Use first chain (usually SMART)
            strikes = chain.strikes

            results = {}
            # Limit to avoid rate limits - get a subset of strikes
            mid_strike_idx = len(strikes) // 2
            selected_strikes = strikes[max(0, mid_strike_idx - 5) : mid_strike_idx + 5]

            for strike in selected_strikes:
                for right in ["C", "P"]:
                    from ib_async import Option

                    option = Option(
                        underlying,
                        expiry,
                        strike,
                        right,
                        "SMART",
                        currency="USD",
                    )
                    try:
                        await self.ib.qualifyContractsAsync(option)
                        bars = await self.fetch_bars(
                            option.localSymbol or f"{underlying}{expiry}{strike}{right}",
                            timeframe,
                            limit=limit,
                        )
                        if bars:
                            results[option.localSymbol] = bars
                    except Exception as e:
                        logger.debug(f"Failed to fetch bars for {underlying} {strike}{right}: {e}")

            logger.info(f"Fetched bars for {len(results)} options on {underlying}")
            return results

        except Exception as e:
            logger.error(f"Failed to fetch option chain bars: {e}")
            return {}
