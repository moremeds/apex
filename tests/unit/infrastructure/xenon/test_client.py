from __future__ import annotations

import asyncio
import json

import pytest

from src.domain.events.event_types import EventType
from src.domain.interfaces import LiveFeedPort
from src.infrastructure.adapters.xenon.client import XenonTickClient
from tests.support.fake_xenon import FakeXenonServer


class _RecordingBus:
    """Captures (event_type, payload) and exposes an event-based wait."""

    def __init__(self) -> None:
        self.published: list = []
        self._event = asyncio.Event()

    def publish(self, event_type, payload, priority=None) -> None:
        self.published.append((event_type, payload))
        self._event.set()

    async def wait_for(self, n: int, timeout: float = 2.0) -> None:
        async def _wait() -> None:
            while len(self.published) < n:
                self._event.clear()
                if len(self.published) >= n:
                    return
                await self._event.wait()

        await asyncio.wait_for(_wait(), timeout)


def test_client_satisfies_live_feed_port() -> None:
    assert isinstance(XenonTickClient("ws://x", _RecordingBus()), LiveFeedPort)


@pytest.mark.asyncio
async def test_subscribe_sends_action_frame() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)
        assert server.received[0] == {"action": "subscribe", "symbols": ["AAPL"]}
        await client.close()


@pytest.mark.asyncio
async def test_unsubscribe_sends_action_frame() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)
        await client.unsubscribe("AAPL")
        await server.wait_for_frames(2)
        assert server.received[1] == {"action": "unsubscribe", "symbols": ["AAPL"]}
        await client.close()


@pytest.mark.asyncio
async def test_batch_tick_is_translated_and_published() -> None:
    bus = _RecordingBus()
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, bus, reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)

        await server.push(
            {"type": "batch", "updates": {"AAPL": {
                "symbol": "AAPL", "last": 150.0, "volume": 10,
                "timestamp": "2026-06-14T12:00:00Z"}}}
        )
        await bus.wait_for(1)
        event_type, payload = bus.published[0]
        assert event_type == EventType.MARKET_DATA_TICK
        assert payload["symbol"] == "AAPL" and payload["last"] == 150.0
        await client.close()


@pytest.mark.asyncio
async def test_price_frame_is_published() -> None:
    bus = _RecordingBus()
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, bus, reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await server.push(
            {"type": "price", "symbol": "AAPL", "data": {
                "symbol": "AAPL", "last": 99.0, "timestamp": "2026-06-14T12:00:00Z"}}
        )
        await bus.wait_for(1)
        assert bus.published[0][1]["last"] == 99.0
        await client.close()


@pytest.mark.asyncio
async def test_malformed_frames_do_not_kill_the_client() -> None:
    """A non-JSON / non-dict / wrong-shape frame is dropped; later good ticks still flow."""
    bus = _RecordingBus()
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, bus, reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        # garbage that would crash a naive handler
        for ws in list(server._clients):
            await ws.send("not json")
            await ws.send(json.dumps([1, 2, 3]))           # JSON, but not an object
            await ws.send(json.dumps({"type": "batch", "updates": [1]}))  # updates not a dict
        await server.push(
            {"type": "price", "symbol": "AAPL", "data": {
                "symbol": "AAPL", "last": 1.0, "timestamp": "2026-06-14T12:00:00Z"}}
        )
        await bus.wait_for(1)
        assert bus.published[0][1]["last"] == 1.0
        await client.close()


@pytest.mark.asyncio
async def test_server_ping_is_answered_with_pong() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await server.push({"type": "ping"})
        await server.wait_for_frames(1)
        assert server.received[0] == {"action": "pong"}
        await client.close()


@pytest.mark.asyncio
async def test_reconnects_and_resubscribes_after_drop() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection(1)
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)          # initial subscribe

        await server.drop_connections()          # force a reconnect
        await server.wait_for_connection(2)      # client dialed back in
        await server.wait_for_frames(2)          # active set replayed on reconnect
        assert server.received[-1] == {"action": "subscribe", "symbols": ["AAPL"]}
        assert server.connections >= 2
        await client.close()


@pytest.mark.asyncio
async def test_close_during_reconnect_stops_redialing() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.05)
        await client.connect()
        await server.wait_for_connection(1)
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)

        await server.drop_connections()
        await client.close()                     # close while it would be reconnecting
        before = server.connections
        await asyncio.sleep(0.2)                  # > reconnect_delay
        assert server.connections == before      # no new dial after close()
