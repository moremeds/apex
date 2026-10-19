from __future__ import annotations

import asyncio
import json
import logging
import sys

import pytest

from src.domain.events.event_types import EventType
from src.domain.interfaces import LiveFeedPort
from src.infrastructure.adapters.xenon.auth import ApiKeyAuthProvider
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
            {
                "type": "batch",
                "updates": {
                    "AAPL": {
                        "symbol": "AAPL",
                        "last": 150.0,
                        "volume": 10,
                        "timestamp": "2026-06-14T12:00:00Z",
                    }
                },
            }
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
            {
                "type": "price",
                "symbol": "AAPL",
                "data": {"symbol": "AAPL", "last": 99.0, "timestamp": "2026-06-14T12:00:00Z"},
            }
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
            await ws.send(json.dumps([1, 2, 3]))  # JSON, but not an object
            await ws.send(json.dumps({"type": "batch", "updates": [1]}))  # updates not a dict
        await server.push(
            {
                "type": "price",
                "symbol": "AAPL",
                "data": {"symbol": "AAPL", "last": 1.0, "timestamp": "2026-06-14T12:00:00Z"},
            }
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
        await server.wait_for_frames(1)  # initial subscribe

        await server.drop_connections()  # force a reconnect
        await server.wait_for_connection(2)  # client dialed back in
        await server.wait_for_frames(2)  # active set replayed on reconnect
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
        await client.close()  # close while it would be reconnecting
        before = server.connections
        await asyncio.sleep(0.2)  # > reconnect_delay
        assert server.connections == before  # no new dial after close()


@pytest.mark.asyncio
async def test_api_key_provider_sends_x_api_key_and_no_origin() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(
            server.url, _RecordingBus(), auth=ApiKeyAuthProvider("k-123"), reconnect_delay=0.01
        )
        await client.connect()
        await server.wait_for_connection()
        headers = server.request_headers[0]
        assert headers.get("X-API-Key") == "k-123"
        assert "Origin" not in headers  # xenon's service path rejects an Origin header
        await client.close()


@pytest.mark.asyncio
async def test_default_auth_sends_no_api_key_or_origin() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        headers = server.request_headers[0]
        assert "X-API-Key" not in headers
        assert "Origin" not in headers
        await client.close()


class _TicketAuth:
    async def ticket(self) -> str:
        return "t-1"

    def headers(self) -> dict:
        return {}


@pytest.mark.asyncio
async def test_ticket_provider_still_appends_ticket_query() -> None:
    client = XenonTickClient("ws://x:1/feed?a=1", _RecordingBus(), auth=_TicketAuth())
    assert client._connect_url(await _TicketAuth().ticket()) == "ws://x:1/feed?a=1&ticket=t-1"
    assert client._connect_url(None) == "ws://x:1/feed?a=1"


@pytest.mark.asyncio
async def test_rejected_upgrade_is_retried_not_fatal() -> None:
    """A 401 upgrade (e.g. missing/bad X-API-Key) must back off and redial, not end the loop."""
    async with FakeXenonServer(reject_upgrades=1) as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection(1)  # the second dial got through
        assert server.rejected == 1
        await client.close()


_SECRET = "sk-synthetic-7f3a9c"


def _assert_secret_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    for record in caplog.records:
        assert _SECRET not in record.getMessage(), record.name
        assert _SECRET not in repr(record.args), record.name


@pytest.mark.asyncio
async def test_api_key_never_logged_on_successful_upgrade(caplog) -> None:
    caplog.set_level(logging.DEBUG)  # websockets wire logging on, before connect
    async with FakeXenonServer() as server:
        client = XenonTickClient(
            server.url, _RecordingBus(), auth=ApiKeyAuthProvider(_SECRET), reconnect_delay=0.01
        )
        await client.connect()
        await server.wait_for_connection()
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)
        await client.close()
    assert server.request_headers[0].get("X-API-Key") == _SECRET  # the wire still carries it
    wire = [r.getMessage() for r in caplog.records if r.name == "websockets.client"]
    assert "> X-API-Key: <redacted>" in wire  # DEBUG wire logging was live, value scrubbed
    _assert_secret_never_logged(caplog)


@pytest.mark.asyncio
async def test_api_key_never_logged_on_rejected_upgrade(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    async with FakeXenonServer(reject_upgrades=1) as server:
        client = XenonTickClient(
            server.url, _RecordingBus(), auth=ApiKeyAuthProvider(_SECRET), reconnect_delay=0.01
        )
        await client.connect()
        await server.wait_for_connection(1)  # the redial after the 401 got through
        await client.close()
    assert server.rejected == 1
    messages = [r.getMessage() for r in caplog.records]
    assert any("upgrade rejected" in m and "401" in m for m in messages)
    _assert_secret_never_logged(caplog)


@pytest.mark.asyncio
async def test_ticket_never_logged_in_request_line(caplog) -> None:
    caplog.set_level(logging.DEBUG)

    class _SecretTicket:
        async def ticket(self) -> str:
            return _SECRET + "/+="  # survives only as its URL-encoded form in the path

        def headers(self) -> dict:
            return {}

    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), auth=_SecretTicket())
        await client.connect()
        await server.wait_for_connection()
        await client.close()
    assert any(r.getMessage().startswith("> GET /?ticket=<redacted>") for r in caplog.records)
    _assert_secret_never_logged(caplog)


@pytest.mark.asyncio
async def test_repeated_401_backs_off_exponentially_to_the_cap(monkeypatch) -> None:
    real_sleep = asyncio.sleep
    run_code = XenonTickClient._run.__code__
    delays: list = []

    async def _fake_sleep(delay, *args, **kwargs):
        if sys._getframe(1).f_code is run_code:  # only the client's backoff sleep
            delays.append(delay)
            delay = 0
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    async with FakeXenonServer(reject_upgrades=10_000) as server:
        client = XenonTickClient(
            server.url, _RecordingBus(), reconnect_delay=1.0, max_reconnect_delay=30.0
        )
        await client.connect()

        async def _wait() -> None:
            while server.rejected < 9:
                await real_sleep(0.01)

        await asyncio.wait_for(_wait(), 5.0)
        await client.close()
    assert server.connections == 0  # never got past the upgrade
    assert delays[:8] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]
