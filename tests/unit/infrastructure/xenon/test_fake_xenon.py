from __future__ import annotations

import json

import pytest
from websockets.asyncio.client import connect

from tests.support.fake_xenon import FakeXenonServer


@pytest.mark.asyncio
async def test_fake_server_records_frames_and_pushes() -> None:
    async with FakeXenonServer() as server:
        async with connect(server.url) as ws:
            await ws.send(json.dumps({"action": "subscribe", "symbols": ["AAPL"]}))
            await server.wait_for_frames(1)
            assert server.received[0] == {"action": "subscribe", "symbols": ["AAPL"]}

            await server.push({"type": "batch", "updates": {"AAPL": {"symbol": "AAPL"}}})
            msg = json.loads(await ws.recv())
            assert msg["type"] == "batch"
