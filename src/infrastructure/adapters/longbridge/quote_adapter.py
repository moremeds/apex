"""Longbridge QuoteAdapter — implements QuoteProvider protocol for real-time quotes."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from src.domain.events.domain_events import QuoteTick

logger = logging.getLogger(__name__)


class LongbridgeQuoteAdapter:
    """
    Real-time quote provider using Longbridge (LongPort) SDK.

    Implements the QuoteProvider protocol. Maps internal symbols (AAPL)
    to Longbridge format (AAPL.US) automatically.

    Credentials: LONGPORT_APP_KEY, LONGPORT_APP_SECRET, LONGPORT_ACCESS_TOKEN env vars.
    """

    def __init__(self, default_market: str = "US") -> None:
        self._ctx = None
        self._connected = False
        self._callback: Optional[Callable[[QuoteTick], None]] = None
        self._quotes: Dict[str, QuoteTick] = {}
        self._subscribed: set[str] = set()
        self._lock = threading.Lock()
        self._default_market = default_market

    # ── QuoteProvider protocol ──────────────────────────────

    async def connect(self) -> None:
        """Connect to Longbridge using env var credentials."""
        from longport.openapi import Config, QuoteContext

        config = Config.from_env()
        self._ctx = await asyncio.to_thread(QuoteContext, config)
        self._ctx.set_on_quote(self._on_sdk_quote_batch)
        self._connected = True
        logger.info("Longbridge QuoteContext connected")

    async def disconnect(self) -> None:
        """Disconnect from Longbridge."""
        self._connected = False
        self._ctx = None
        logger.info("Longbridge QuoteContext disconnected")

    def is_connected(self) -> bool:
        return self._connected

    async def subscribe_quotes(self, symbols: List[str]) -> None:
        """Subscribe to real-time quotes. Symbols in internal format (AAPL, SPY)."""
        if not self._connected or self._ctx is None:
            raise ConnectionError("Not connected to Longbridge")

        from longport.openapi import SubType

        lb_symbols = [self._to_lb_symbol(s) for s in symbols]
        sub_types = [SubType.Quote]
        await asyncio.to_thread(self._ctx.subscribe, lb_symbols, sub_types)
        self._subscribed.update(symbols)
        logger.info("Subscribed to quotes: %s", symbols)

    async def unsubscribe_quotes(self, symbols: List[str]) -> None:
        """Unsubscribe from quotes."""
        if not self._connected or self._ctx is None:
            return

        from longport.openapi import SubType

        lb_symbols = [self._to_lb_symbol(s) for s in symbols]
        sub_types = [SubType.Quote]
        await asyncio.to_thread(self._ctx.unsubscribe, lb_symbols, sub_types)
        self._subscribed -= set(symbols)
        logger.info("Unsubscribed from quotes: %s", symbols)

    def set_quote_callback(self, callback: Optional[Callable[[QuoteTick], None]]) -> None:
        self._callback = callback

    def get_latest_quote(self, symbol: str) -> Optional[QuoteTick]:
        with self._lock:
            return self._quotes.get(symbol)

    def get_all_quotes(self) -> Dict[str, QuoteTick]:
        with self._lock:
            return dict(self._quotes)

    def get_subscribed_symbols(self) -> List[str]:
        return list(self._subscribed)

    async def fetch_snapshot(self, symbols: List[str]) -> Dict[str, QuoteTick]:
        """Fetch one-time quote snapshot."""
        if not self._connected or self._ctx is None:
            raise ConnectionError("Not connected to Longbridge")

        lb_symbols = [self._to_lb_symbol(s) for s in symbols]
        raw_quotes = await asyncio.to_thread(self._ctx.quote, lb_symbols)

        result: Dict[str, QuoteTick] = {}
        for q in raw_quotes:
            internal = self._to_internal_symbol(q.symbol)
            tick = self._sdk_quote_to_tick(q.symbol, q)
            result[internal] = tick

        return result

    # ── SDK callbacks ───────────────────────────────────────

    def _on_sdk_quote_batch(self, symbol: str, event) -> None:
        """Called by SDK for batch push — event is a PushQuote."""
        self._on_sdk_quote(symbol, event)

    def _on_sdk_quote(self, symbol: str, quote) -> None:
        """Normalize SDK quote to QuoteTick and dispatch."""
        tick = self._sdk_quote_to_tick(symbol, quote)
        with self._lock:
            self._quotes[tick.symbol] = tick
        if self._callback:
            try:
                self._callback(tick)
            except Exception:
                logger.exception("Error in quote callback for %s", tick.symbol)

    def _sdk_quote_to_tick(self, symbol: str, q) -> QuoteTick:
        """Convert SDK quote object to domain QuoteTick."""
        internal_symbol = self._to_internal_symbol(symbol)
        ts = getattr(q, "timestamp", None)
        if isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return QuoteTick(
            symbol=internal_symbol,
            last=float(q.last_done) if hasattr(q, "last_done") else None,
            volume=int(q.volume) if hasattr(q, "volume") and q.volume else None,
            source="longbridge",
            timestamp=ts or datetime.now(timezone.utc),
        )

    # ── Symbol mapping ──────────────────────────────────────

    def _to_lb_symbol(self, symbol: str) -> str:
        """Internal symbol → Longbridge format. AAPL → AAPL.US"""
        if "." in symbol:
            return symbol
        return f"{symbol}.{self._default_market}"

    def _to_internal_symbol(self, lb_symbol: str) -> str:
        """Longbridge format → internal symbol. AAPL.US → AAPL"""
        if "." in lb_symbol:
            return lb_symbol.rsplit(".", 1)[0]
        return lb_symbol
