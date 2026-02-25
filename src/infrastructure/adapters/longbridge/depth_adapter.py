"""Longbridge DepthAdapter — implements DepthProvider protocol for order book data."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from src.domain.interfaces.depth_provider import DepthLevel, DepthSnapshot

logger = logging.getLogger(__name__)


class LongbridgeDepthAdapter:
    """
    Order book depth provider using Longbridge SDK.

    Implements DepthProvider protocol. Uses set_on_depth callback for push updates
    and depth() for one-time snapshots.
    """

    def __init__(self, default_market: str = "US") -> None:
        self._ctx = None
        self._connected = False
        self._callback: Optional[Callable[[DepthSnapshot], None]] = None
        self._depth: Dict[str, DepthSnapshot] = {}
        self._subscribed: set[str] = set()
        self._lock = threading.Lock()
        self._default_market = default_market

    async def subscribe_depth(self, symbols: List[str]) -> None:
        """Subscribe to depth updates. Requires a connected QuoteContext."""
        if not self._connected or self._ctx is None:
            raise ConnectionError("Not connected to Longbridge")

        from longport.openapi import SubType

        lb_symbols = [self._to_lb_symbol(s) for s in symbols]
        await asyncio.to_thread(self._ctx.subscribe, lb_symbols, [SubType.Depth])
        self._subscribed.update(symbols)
        logger.info("Subscribed to depth: %s", symbols)

    async def unsubscribe_depth(self, symbols: List[str]) -> None:
        if not self._connected or self._ctx is None:
            return

        from longport.openapi import SubType

        lb_symbols = [self._to_lb_symbol(s) for s in symbols]
        await asyncio.to_thread(self._ctx.unsubscribe, lb_symbols, [SubType.Depth])
        self._subscribed -= set(symbols)

    def set_depth_callback(
        self, callback: Optional[Callable[[DepthSnapshot], None]]
    ) -> None:
        self._callback = callback

    def get_latest_depth(self, symbol: str) -> Optional[DepthSnapshot]:
        with self._lock:
            return self._depth.get(symbol)

    # ── Attach to shared QuoteContext ───────────────────────

    def attach_context(self, ctx, connected: bool = True) -> None:
        """Attach to an existing QuoteContext (shared with QuoteAdapter)."""
        self._ctx = ctx
        self._connected = connected
        ctx.set_on_depth(self._on_sdk_depth)

    # ── SDK callback ────────────────────────────────────────

    def _on_sdk_depth(self, symbol: str, event) -> None:
        """Called by SDK on depth push."""
        internal = self._to_internal_symbol(symbol)
        bids = tuple(
            DepthLevel(
                price=float(l.price) if l.price else 0.0,
                volume=float(l.volume),
                order_count=int(l.order_num) if hasattr(l, "order_num") else 0,
            )
            for l in (event.bids if hasattr(event, "bids") else [])
        )
        asks = tuple(
            DepthLevel(
                price=float(l.price) if l.price else 0.0,
                volume=float(l.volume),
                order_count=int(l.order_num) if hasattr(l, "order_num") else 0,
            )
            for l in (event.asks if hasattr(event, "asks") else [])
        )
        snap = DepthSnapshot(
            symbol=internal,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(timezone.utc),
            source="longbridge",
        )
        with self._lock:
            self._depth[internal] = snap
        if self._callback:
            try:
                self._callback(snap)
            except Exception:
                logger.exception("Error in depth callback for %s", internal)

    def _to_lb_symbol(self, symbol: str) -> str:
        if "." in symbol:
            return symbol
        return f"{symbol}.{self._default_market}"

    def _to_internal_symbol(self, lb_symbol: str) -> str:
        if "." in lb_symbol:
            return lb_symbol.rsplit(".", 1)[0]
        return lb_symbol
