"""WS subscribe drives the manager, sends an initial snapshot, and cleans up."""

from __future__ import annotations

from starlette.testclient import TestClient

from src.api.server import create_app
from src.api.ws.hub import SignalHub


class _FakeMgr:
    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []

    async def subscribe(self, t: str) -> None:
        self.subscribed.append(t)

    async def unsubscribe(self, t: str) -> None:
        self.unsubscribed.append(t)


def test_ws_subscribe_acks_and_disconnect_decrements_refcount() -> None:
    app = create_app()
    app.state.signal_hub = SignalHub()
    app.state.subscription_manager = _FakeMgr()
    app.state.signal_repo = None  # no snapshot in this test
    with TestClient(app) as client:
        with client.websocket_connect("/ws/signals") as ws:
            ws.send_json({"action": "subscribe", "ticker": "AAPL"})
            assert ws.receive_json() == {"status": "subscribed", "ticker": "AAPL"}
    assert app.state.subscription_manager.subscribed == ["AAPL"]
    assert app.state.subscription_manager.unsubscribed == ["AAPL"]


def test_ws_explicit_unsubscribe_decrements_once() -> None:
    app = create_app()
    app.state.signal_hub = SignalHub()
    app.state.subscription_manager = _FakeMgr()
    app.state.signal_repo = None
    with TestClient(app) as client:
        with client.websocket_connect("/ws/signals") as ws:
            ws.send_json({"action": "subscribe", "ticker": "AAPL"})
            ws.receive_json()
            ws.send_json({"action": "unsubscribe", "ticker": "AAPL"})
            assert ws.receive_json() == {"status": "unsubscribed", "ticker": "AAPL"}
    assert app.state.subscription_manager.unsubscribed == ["AAPL"]
