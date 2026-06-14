"""WS client to xenon's ib_realtime server -> apex event bus (Phase 4).

Implements LiveFeedPort. Owns one websockets connection: sends
subscribe/unsubscribe action frames and republishes each received `price`/`batch`
tick (translated) on EventType.MARKET_DATA_TICK. A single malformed frame or a
handler error never kills the receive loop. Keep-alive ping (Task 7) and
reconnect/backoff (Task 8) are layered on after their own tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Set
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from src.domain.events.event_types import EventType
from src.infrastructure.adapters.xenon.auth import AuthProvider, NoAuthProvider
from src.infrastructure.adapters.xenon.translator import translate_price_data

logger = logging.getLogger(__name__)


class XenonTickClient:
    def __init__(
        self,
        url: str,
        event_bus: Any,
        auth: Optional[AuthProvider] = None,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
    ) -> None:
        self._url = url
        self._bus = event_bus
        self._auth: AuthProvider = auth or NoAuthProvider()
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._subscribed: Set[str] = set()
        self._ws: Any = None
        self._task: Optional[asyncio.Task[None]] = None
        self._closing = False

    # ---- LiveFeedPort ----------------------------------------------------
    async def connect(self) -> None:
        """Launch the background receive loop (non-blocking)."""
        if self._task is None:
            self._closing = False
            self._task = asyncio.create_task(self._run())

    async def subscribe(self, symbol: str) -> None:
        self._subscribed.add(symbol)
        await self._send({"action": "subscribe", "symbols": [symbol]})

    async def unsubscribe(self, symbol: str) -> None:
        self._subscribed.discard(symbol)
        await self._send({"action": "unsubscribe", "symbols": [symbol]})

    async def close(self) -> None:
        self._closing = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ---- internals -------------------------------------------------------
    async def _connect_url(self) -> str:
        # SECURITY: the returned URL may carry an auth ticket in its query string
        # (once TicketAuthProvider lands). Never log this value.
        ticket = await self._auth.ticket()
        if not ticket:
            return self._url
        parts = urlsplit(self._url)
        query = dict(parse_qsl(parts.query))
        query["ticket"] = ticket
        return urlunsplit(parts._replace(query=urlencode(query)))

    async def _send(self, payload: dict) -> None:
        ws = self._ws
        if ws is None:
            return  # not connected yet; _resubscribe replays on connect
        try:
            await ws.send(json.dumps(payload))
        except ConnectionClosed:
            pass

    async def _resubscribe(self) -> None:
        if self._subscribed:
            await self._send({"action": "subscribe", "symbols": sorted(self._subscribed)})

    async def _run(self) -> None:
        delay = self._reconnect_delay
        while not self._closing:
            try:
                async with connect(await self._connect_url()) as ws:
                    self._ws = ws
                    delay = self._reconnect_delay  # reset backoff on a good connect
                    await self._resubscribe()
                    async for raw in ws:
                        try:
                            await self._handle(raw)
                        except Exception:
                            logger.exception("xenon WS: error handling frame; continuing")
            except (ConnectionClosed, OSError) as exc:
                logger.warning("xenon WS connection lost: %s", exc)
            except asyncio.CancelledError:
                break
            finally:
                self._ws = None
            if self._closing:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_reconnect_delay)  # capped exponential backoff

    async def _handle(self, raw: Any) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("xenon WS: dropping non-JSON frame")
            return
        if not isinstance(msg, dict):
            logger.warning("xenon WS: dropping non-object frame (%s)", type(msg).__name__)
            return
        mtype = msg.get("type")
        if mtype == "ping":
            await self._send({"action": "pong"})
        elif mtype == "batch":
            updates = msg.get("updates")
            if isinstance(updates, dict):
                for data in updates.values():
                    if isinstance(data, dict):
                        self._publish(data)
        elif mtype == "price":
            data = msg.get("data")
            if isinstance(data, dict):
                self._publish(data)
        elif mtype == "error":
            logger.warning("xenon WS error frame: %s", msg.get("message"))
        # status/subscribed/unsubscribed/unknown: ignored

    def _publish(self, data: dict) -> None:
        tick = translate_price_data(data)
        if tick is not None:
            self._bus.publish(EventType.MARKET_DATA_TICK, tick)
