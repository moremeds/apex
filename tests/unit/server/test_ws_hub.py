"""Tests for WebSocket Hub — fan-out to per-symbol subscribed clients."""

from typing import Any, List

import pytest

from src.server.ws_hub import WebSocketHub


class MockWebSocket:
    """Fake WebSocket that records sent messages."""

    def __init__(self):
        self.sent: List[dict] = []
        self.closed = False

    async def send_json(self, data: Any) -> None:
        if self.closed:
            raise RuntimeError("WebSocket closed")
        self.sent.append(data)


@pytest.mark.asyncio
async def test_broadcast_quote_to_subscribed():
    hub = WebSocketHub()
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    hub.connect(ws1)
    hub.connect(ws2)
    await hub.handle_command(ws1, {"cmd": "subscribe", "symbols": ["AAPL"], "types": ["quote"]})
    await hub.handle_command(ws2, {"cmd": "subscribe", "symbols": ["SPY"], "types": ["quote"]})

    await hub.broadcast_quote("AAPL", {"last": 185.5, "volume": 1000})
    assert len(ws1.sent) == 1
    assert ws1.sent[0]["type"] == "quote"
    assert ws1.sent[0]["symbol"] == "AAPL"
    # Quote data nested under "data" key (matches frontend WsMessage type)
    assert ws1.sent[0]["data"]["last"] == 185.5
    assert len(ws2.sent) == 0  # not subscribed to AAPL


@pytest.mark.asyncio
async def test_broadcast_bar():
    hub = WebSocketHub()
    ws = MockWebSocket()
    hub.connect(ws)
    await hub.handle_command(ws, {"cmd": "subscribe", "symbols": ["AAPL"], "types": ["quote"]})

    bar = {"t": "2024-01-01T00:00:00", "o": 184.0, "h": 186.0, "l": 183.5, "c": 185.5, "v": 50000}
    await hub.broadcast_bar("AAPL", "1d", bar)
    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "bar"
    # Uses "timeframe" (not "tf") and "data" wrapper (matches frontend WsMessage type)
    assert ws.sent[0]["timeframe"] == "1d"
    assert ws.sent[0]["data"]["o"] == 184.0


@pytest.mark.asyncio
async def test_broadcast_signal():
    hub = WebSocketHub()
    ws = MockWebSocket()
    hub.connect(ws)
    await hub.handle_command(ws, {"cmd": "subscribe", "symbols": ["AAPL"], "types": ["quote"]})

    await hub.broadcast_signal(
        "AAPL", {"rule": "rsi_oversold", "direction": "bullish", "strength": 0.8, "timestamp": "123"}
    )
    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "signal"
    # Signal data nested under "data" key with symbol injected (matches frontend WsMessage type)
    assert ws.sent[0]["data"]["symbol"] == "AAPL"
    assert ws.sent[0]["data"]["rule"] == "rsi_oversold"


@pytest.mark.asyncio
async def test_broadcast_indicator():
    hub = WebSocketHub()
    ws = MockWebSocket()
    hub.connect(ws)
    await hub.handle_command(ws, {"cmd": "subscribe", "symbols": ["AAPL"], "types": ["quote"]})

    await hub.broadcast_indicator("AAPL", "1d", "rsi", 65.3)
    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "indicator"
    # Uses "timeframe" (not "tf") — matches frontend WsMessage type
    assert ws.sent[0]["timeframe"] == "1d"
    assert ws.sent[0]["name"] == "rsi"
    assert ws.sent[0]["value"] == 65.3


@pytest.mark.asyncio
async def test_broadcast_status_to_all():
    hub = WebSocketHub()
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    hub.connect(ws1)
    hub.connect(ws2)
    # No subscriptions needed — status goes to ALL clients

    providers = [{"name": "longbridge", "connected": True, "symbols": 5}]
    await hub.broadcast_status(providers)
    assert len(ws1.sent) == 1
    assert len(ws2.sent) == 1
    assert ws1.sent[0]["type"] == "status"


@pytest.mark.asyncio
async def test_subscribe_command():
    hub = WebSocketHub()
    ws = MockWebSocket()
    hub.connect(ws)
    await hub.handle_command(
        ws, {"cmd": "subscribe", "symbols": ["AAPL", "SPY"], "types": ["quote"]}
    )
    assert hub.get_subscriptions(ws) == {"AAPL", "SPY"}


@pytest.mark.asyncio
async def test_unsubscribe_command():
    hub = WebSocketHub()
    ws = MockWebSocket()
    hub.connect(ws)
    await hub.handle_command(
        ws, {"cmd": "subscribe", "symbols": ["AAPL", "SPY"], "types": ["quote"]}
    )
    await hub.handle_command(ws, {"cmd": "unsubscribe", "symbols": ["AAPL"]})
    assert hub.get_subscriptions(ws) == {"SPY"}


@pytest.mark.asyncio
async def test_disconnect_removes_client():
    hub = WebSocketHub()
    ws = MockWebSocket()
    hub.connect(ws)
    assert hub.client_count == 1
    hub.disconnect(ws)
    assert hub.client_count == 0


@pytest.mark.asyncio
async def test_dead_client_cleanup():
    """If send_json raises repeatedly, the client should be removed."""
    hub = WebSocketHub()
    ws = MockWebSocket()
    ws.closed = True  # Will raise on send
    hub.connect(ws)
    await hub.handle_command(ws, {"cmd": "subscribe", "symbols": ["AAPL"], "types": ["quote"]})

    # Client tolerates up to _max_send_failures-1 consecutive failures
    for _ in range(hub._max_send_failures - 1):
        await hub.broadcast_quote("AAPL", {"last": 185.5})
        assert hub.client_count == 1  # still connected

    # One more failure tips it over
    await hub.broadcast_quote("AAPL", {"last": 185.5})
    assert hub.client_count == 0


@pytest.mark.asyncio
async def test_connect_with_no_subscriptions():
    hub = WebSocketHub()
    ws = MockWebSocket()
    hub.connect(ws)
    assert hub.get_subscriptions(ws) == set()
    # Broadcasting should not fail
    await hub.broadcast_quote("AAPL", {"price": 185.5})
    assert len(ws.sent) == 0
