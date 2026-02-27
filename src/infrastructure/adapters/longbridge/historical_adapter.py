"""Longbridge HistoricalAdapter — implements HistoricalSourcePort for candlestick data."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, List

from src.domain.events.domain_events import BarData

logger = logging.getLogger(__name__)

# Timeframe → Longbridge Period mapping
_TIMEFRAME_TO_PERIOD = {
    "1m": "Min_1",
    "2m": "Min_2",
    "3m": "Min_3",
    "5m": "Min_5",
    "10m": "Min_10",
    "15m": "Min_15",
    "20m": "Min_20",
    "30m": "Min_30",
    "45m": "Min_45",
    "1h": "Min_60",
    "2h": "Min_120",
    "3h": "Min_180",
    "4h": "Min_240",
    "1d": "Day",
    "1w": "Week",
    "1M": "Month",
}

# Max candlesticks per request (Longbridge limit)
_MAX_COUNT = 1000


class LongbridgeHistoricalAdapter:
    """
    Historical bar data provider using Longbridge SDK.

    Implements HistoricalSourcePort. Uses candlesticks() for batch downloads.
    Symbol mapping: internal AAPL → Longbridge AAPL.US.
    """

    def __init__(self, default_market: str = "US") -> None:
        self._ctx: Any = None
        self._connected = False
        self._default_market = default_market

    @property
    def source_name(self) -> str:
        return "longbridge"

    async def connect(self) -> None:
        """Connect to Longbridge for historical data."""
        from longport.openapi import Config, QuoteContext

        config = Config.from_env()
        self._ctx = await asyncio.to_thread(QuoteContext, config)
        self._connected = True
        logger.info("Longbridge HistoricalAdapter connected")

    async def disconnect(self) -> None:
        self._connected = False
        self._ctx = None

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        """Fetch historical bars from Longbridge."""
        if not self._connected or self._ctx is None:
            raise ConnectionError("Not connected to Longbridge")

        from longport.openapi import AdjustType, Period

        period_name = _TIMEFRAME_TO_PERIOD.get(timeframe)
        if period_name is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        period = getattr(Period, period_name)
        lb_symbol = self._to_lb_symbol(symbol)

        raw = await asyncio.to_thread(
            self._ctx.candlesticks,
            lb_symbol,
            period,
            _MAX_COUNT,
            AdjustType.NoAdjust,
        )

        bars: List[BarData] = []
        for c in raw:
            ts = c.timestamp
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            # Filter to requested range
            ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
            start_naive = start.replace(tzinfo=None) if start.tzinfo else start
            end_naive = end.replace(tzinfo=None) if end.tzinfo else end
            if ts_naive < start_naive or ts_naive > end_naive:
                continue

            bars.append(
                BarData(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(c.open),
                    high=float(c.high),
                    low=float(c.low),
                    close=float(c.close),
                    volume=int(c.volume) if c.volume else None,
                    bar_start=ts,
                    source="longbridge",
                    timestamp=ts,
                )
            )

        bars.sort(key=lambda b: b.timestamp)
        return bars

    def supports_timeframe(self, timeframe: str) -> bool:
        return timeframe in _TIMEFRAME_TO_PERIOD

    def get_supported_timeframes(self) -> List[str]:
        return list(_TIMEFRAME_TO_PERIOD.keys())

    def _to_lb_symbol(self, symbol: str) -> str:
        if "." in symbol:
            return symbol
        return f"{symbol}.{self._default_market}"
