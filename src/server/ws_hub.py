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

    def __init__(self) -> None:
        self._clients: Dict[Any, Set[str]] = {}  # ws → subscribed symbols

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
        logger.info("WS client disconnected (%d total)", self.client_count)

    def get_subscriptions(self, ws: Any) -> Set[str]:
        """Get symbols a client is subscribed to."""
        return self._clients.get(ws, set())

    async def handle_command(self, ws: Any, cmd: dict) -> None:
        """Process a subscribe/unsubscribe command from a client."""
        action = cmd.get("cmd")
        symbols = cmd.get("symbols", [])

        if action == "subscribe":
            if ws in self._clients:
                self._clients[ws].update(symbols)
                logger.debug("Client subscribed to %s", symbols)
        elif action == "unsubscribe":
            if ws in self._clients:
                self._clients[ws] -= set(symbols)
                logger.debug("Client unsubscribed from %s", symbols)

    # ── Broadcast methods ───────────────────────────────────

    async def broadcast_quote(self, symbol: str, data: dict) -> None:
        """Send quote update to clients subscribed to this symbol."""
        msg = {"type": "quote", "symbol": symbol, **data}
        await self._send_to_symbol_subscribers(symbol, msg)

    async def broadcast_bar(self, symbol: str, tf: str, bar: dict) -> None:
        """Send bar close to clients subscribed to this symbol."""
        msg = {"type": "bar", "symbol": symbol, "tf": tf, "bar": bar}
        await self._send_to_symbol_subscribers(symbol, msg)

    async def broadcast_signal(self, symbol: str, signal: dict) -> None:
        """Send trading signal to clients subscribed to this symbol."""
        msg = {"type": "signal", "symbol": symbol, **signal}
        await self._send_to_symbol_subscribers(symbol, msg)

    async def broadcast_indicator(
        self, symbol: str, tf: str, name: str, value: Any
    ) -> None:
        """Send indicator update to clients subscribed to this symbol."""
        msg = {"type": "indicator", "symbol": symbol, "tf": tf, "name": name, "value": value}
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

    # ── Internal ────────────────────────────────────────────

    async def _send_to_symbol_subscribers(self, symbol: str, msg: dict) -> None:
        """Send message to all clients subscribed to the given symbol."""
        dead: list = []
        for ws, subs in list(self._clients.items()):
            if symbol in subs:
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
