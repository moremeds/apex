"""In-process fake of xenon's ib_realtime WS server for tests.

Speaks the real protocol surface apex's client uses: records client action
frames, can push server frames (price/batch/ping/error), and can drop
connections to exercise reconnect. Not collected by pytest (filename != test_*).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from websockets.asyncio.server import serve


class FakeXenonServer:
    def __init__(self) -> None:
        self.host = "127.0.0.1"
        self.port: int = 0
        self.received: List[Dict[str, Any]] = []
        self.connections = 0
        self._server: Any = None
        self._clients: set = set()
        self._frame_event = asyncio.Event()

    async def __aenter__(self) -> "FakeXenonServer":
        self._server = await serve(self._handler, self.host, 0)
        self.port = list(self._server.sockets)[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    async def _handler(self, ws: Any) -> None:
        self.connections += 1
        self._clients.add(ws)
        try:
            async for raw in ws:
                self.received.append(json.loads(raw))
                self._frame_event.set()
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    async def wait_for_frames(self, n: int, timeout: float = 2.0) -> None:
        async def _wait() -> None:
            while len(self.received) < n:
                self._frame_event.clear()
                if len(self.received) >= n:
                    return
                await self._frame_event.wait()

        await asyncio.wait_for(_wait(), timeout)

    async def wait_for_connection(self, n: int = 1, timeout: float = 2.0) -> None:
        async def _wait() -> None:
            while self.connections < n:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait(), timeout)

    async def push(self, payload: Dict[str, Any]) -> None:
        for ws in list(self._clients):
            await ws.send(json.dumps(payload))

    async def drop_connections(self) -> None:
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()
