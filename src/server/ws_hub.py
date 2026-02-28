"""WebSocket Hub — fan-out to per-symbol subscribed browser clients."""

from __future__ import annotations

import logging
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)


class WebSocketHub:
    """
    Manages WebSocket connections and per-symbol subscriptions.

    Each client subscribes to specific symbols. Broadcasts are routed
    only to clients subscribed to the relevant symbol. Status messages
    are broadcast to all connected clients.
    """

    _max_send_failures: int = 3  # disconnect after N consecutive send failures

    def __init__(self) -> None:
        self._clients: Dict[Any, Set[str]] = {}  # ws → subscribed symbols
        self._fail_counts: Dict[int, int] = {}  # id(ws) → consecutive failure count

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def connect(self, ws: Any) -> None:
        """Register a new WebSocket client."""
        self._clients[ws] = set()
        logger.info("WS client connected (%d total)", self.client_count)

    def disconnect(self, ws: Any) -> None:
        """Remove a WebSocket client."""
        self._clients.pop(ws, None)
        self._fail_counts.pop(id(ws), None)
        logger.info("WS client disconnected (%d total)", self.client_count)

    def get_subscriptions(self, ws: Any) -> Set[str]:
        """Get symbols a client is subscribed to."""
        return self._clients.get(ws, set())

    async def handle_command(self, ws: Any, cmd: dict) -> None:
        """Process a subscribe/unsubscribe command from a client.

        The ``types`` field is accepted for forward-compatibility but currently
        all message types are sent to subscribed symbols (server-side filtering
        by type is not yet implemented).
        """
        action = cmd.get("cmd")
        symbols = cmd.get("symbols", [])
        # types = cmd.get("types", [])  # Accepted but not yet used for filtering

        if action == "subscribe":
            if ws in self._clients:
                self._clients[ws].update(symbols)
                logger.debug("Client subscribed to %s", symbols)
        elif action == "unsubscribe":
            if ws in self._clients:
                self._clients[ws] -= set(symbols)
                logger.debug("Client unsubscribed from %s", symbols)

    # ── Broadcast methods ───────────────────────────────────
    # Message shapes MUST match the TS types in web/src/lib/ws.ts.

    async def broadcast_quote(self, symbol: str, data: dict) -> None:
        """Send quote update to clients subscribed to this symbol.

        Frontend expects: {type, symbol, data: {last, bid, ask, volume, ts}}
        """
        msg = {"type": "quote", "symbol": symbol, "data": data}
        await self._send_to_symbol_subscribers(symbol, msg)

    async def broadcast_bar(self, symbol: str, tf: str, bar: dict) -> None:
        """Send bar close to clients subscribed to this symbol.

        Frontend expects: {type, symbol, timeframe, data: {t, o, h, l, c, v}}
        """
        msg = {"type": "bar", "symbol": symbol, "timeframe": tf, "data": bar}
        await self._send_to_symbol_subscribers(symbol, msg)

    async def broadcast_signal(self, symbol: str, signal: dict) -> None:
        """Send trading signal to clients subscribed to this symbol.

        Frontend expects: {type, data: {symbol, rule, direction, strength, timeframe, timestamp}}
        """
        msg = {"type": "signal", "data": {"symbol": symbol, **signal}}
        await self._send_to_symbol_subscribers(symbol, msg)

    async def broadcast_indicator(self, symbol: str, tf: str, name: str, value: Any) -> None:
        """Send indicator update to clients subscribed to this symbol.

        Frontend expects: {type, symbol, timeframe, name, value}
        """
        msg = {"type": "indicator", "symbol": symbol, "timeframe": tf, "name": name, "value": value}
        await self._send_to_symbol_subscribers(symbol, msg)

    async def broadcast_status(self, providers: list) -> None:
        """Send provider status to ALL connected clients."""
        msg = {"type": "status", "providers": providers}
        dead: list = []
        for ws in list(self._clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_advisor(self, advice: dict) -> None:
        """Send advisor update to ALL connected clients."""
        msg = {"type": "advisor", **advice}
        dead: list = []
        for ws in list(self._clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    # ── Internal ────────────────────────────────────────────

    async def _send_to_symbol_subscribers(self, symbol: str, msg: dict) -> None:
        """Send message to all clients subscribed to the given symbol.

        Clients that fail to receive are tracked; they are disconnected
        after 3 consecutive failures to tolerate transient network hiccups.
        """
        for ws, subs in list(self._clients.items()):
            if symbol in subs:
                try:
                    await ws.send_json(msg)
                    self._fail_counts.pop(id(ws), None)
                except Exception:
                    count = self._fail_counts.get(id(ws), 0) + 1
                    if count >= self._max_send_failures:
                        self._fail_counts.pop(id(ws), None)
                        self.disconnect(ws)
                    else:
                        self._fail_counts[id(ws)] = count
