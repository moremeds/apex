"""Tests for WebSocket route — /ws endpoint with subscribe/unsubscribe."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

from src.server.ws_hub import WebSocketHub


def _make_app_with_hub():
    """Create a minimal FastAPI app with the WS route and a hub."""
    from fastapi import FastAPI

    from src.server.routes.ws import create_ws_router

    hub = WebSocketHub()
    app = FastAPI()
    app.include_router(create_ws_router(hub))
    return app, hub


def test_ws_connect_and_disconnect():
    """Client can connect and disconnect cleanly."""
    app, hub = _make_app_with_hub()
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        assert hub.client_count == 1
    # After context exit, client should be disconnected
    assert hub.client_count == 0


def test_ws_subscribe_command():
    """Client can send subscribe command."""
    app, hub = _make_app_with_hub()
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"cmd": "subscribe", "symbols": ["AAPL", "SPY"]})
        # Give a moment for processing
        # The hub should have our subscriptions
        # We can't easily check subscriptions since we don't have the ws ref,
        # but we can broadcast and see if we receive it
        import time

        time.sleep(0.05)


def test_ws_receives_broadcast():
    """After subscribing, client receives broadcasts for that symbol."""
    app, hub = _make_app_with_hub()
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"cmd": "subscribe", "symbols": ["AAPL"]})
        # Small delay for subscribe to process
        import time

        time.sleep(0.05)

        # Broadcast a quote from the hub (in a background thread)
        import threading

        def broadcast():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(hub.broadcast_quote("AAPL", {"price": 185.5, "volume": 1000}))
            loop.close()

        t = threading.Thread(target=broadcast)
        t.start()
        t.join()

        # Client should receive the message
        msg = ws.receive_json()
        assert msg["type"] == "quote"
        assert msg["symbol"] == "AAPL"
        assert msg["price"] == 185.5


def test_ws_no_message_for_unsubscribed():
    """Client does not receive broadcasts for unsubscribed symbols."""
    app, hub = _make_app_with_hub()
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"cmd": "subscribe", "symbols": ["SPY"]})
        import time

        time.sleep(0.05)

        # Broadcast for AAPL — client subscribed to SPY only
        import threading

        def broadcast():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(hub.broadcast_quote("AAPL", {"price": 185.5}))
            loop.close()

        t = threading.Thread(target=broadcast)
        t.start()
        t.join()

        # Broadcast for SPY — should arrive
        def broadcast_spy():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(hub.broadcast_quote("SPY", {"price": 600.0}))
            loop.close()

        t2 = threading.Thread(target=broadcast_spy)
        t2.start()
        t2.join()

        msg = ws.receive_json()
        assert msg["symbol"] == "SPY"  # Only SPY, not AAPL
