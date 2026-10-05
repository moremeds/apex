"""WS subscribe drives the manager, sends an initial snapshot, and cleans up."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.server import create_app
from src.api.ws.hub import SignalHub
from src.api.ws.signals_ws import signals_ws
from src.application.subscriptions.manager import SubscriptionManager


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


def test_ws_closes_1013_when_no_lake_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("APEX_LIVEWIRE_ROOT", "APEX_LIVEWIRE_SILVER_ROOT", "APEX_PG_URL"):
        monkeypatch.delenv(var, raising=False)
    app = create_app()  # lifespan builds no pipeline -> subscription_manager is None
    with TestClient(app) as client:
        assert app.state.subscription_manager is None
        with client.websocket_connect("/ws/signals") as ws:
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
    assert closed.value.code == 1013


class _Provider:
    async def fetch_bars(self, symbol, timeframe, **kw):
        bar = SimpleNamespace(open=1.0, high=1.0, low=1.0, close=1.0, volume=1, bar_start="t")
        return [bar]


class _Compute:
    async def start(self) -> None:
        pass

    async def inject_historical_bars(self, symbol, timeframe, bars) -> None:
        pass


def test_ws_duplicate_subscribe_releases_refcount_on_disconnect() -> None:
    app = create_app()
    app.state.signal_hub = SignalHub()
    mgr = SubscriptionManager(provider=_Provider(), compute=_Compute(), timeframes=["1d"])
    app.state.subscription_manager = mgr
    app.state.signal_repo = None
    with TestClient(app) as client:
        with client.websocket_connect("/ws/signals") as ws:
            for _ in range(2):
                ws.send_json({"action": "subscribe", "ticker": "AAPL"})
                assert ws.receive_json() == {"status": "subscribed", "ticker": "AAPL"}
            assert mgr.refcount("AAPL") == 1
    assert mgr.refcount("AAPL") == 0


class _FailingMgr(_FakeMgr):
    async def subscribe(self, t: str) -> None:
        if t == "BAD":
            raise RuntimeError("seed failed")
        await super().subscribe(t)


def test_ws_handler_error_still_releases_held_tickers() -> None:
    app = create_app()
    app.state.signal_hub = SignalHub()
    app.state.subscription_manager = _FailingMgr()
    app.state.signal_repo = None
    with TestClient(app) as client:
        # TestClient re-raises the handler's error once the socket closes.
        with pytest.raises(RuntimeError, match="seed failed"):
            with client.websocket_connect("/ws/signals") as ws:
                ws.send_json({"action": "subscribe", "ticker": "AAPL"})
                ws.receive_json()
                ws.send_json({"action": "subscribe", "ticker": "BAD"})
                ws.receive_json()
    # AAPL released; BAD never acquired, so it is not released either.
    assert app.state.subscription_manager.unsubscribed == ["AAPL"]


class _ScriptedWS:
    """Feeds frames to the handler, then blocks like an idle client."""

    def __init__(self, state: SimpleNamespace, frames: list[dict]) -> None:
        self.app = SimpleNamespace(state=state)
        self._frames = frames

    async def accept(self) -> None:
        pass

    async def receive_json(self) -> dict:
        if self._frames:
            return self._frames.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send_json(self, data: dict) -> None:
        pass


@pytest.mark.asyncio
async def test_cancelled_subscribe_does_not_release_another_sockets_refcount() -> None:
    mgr = SubscriptionManager(provider=_Provider(), compute=_Compute(), timeframes=["1d"])
    await mgr.subscribe("AAPL")  # socket A's refcount
    state = SimpleNamespace(signal_hub=SignalHub(), subscription_manager=mgr, signal_repo=None)
    sub = {"action": "subscribe", "ticker": "AAPL"}

    async with mgr._lock:  # a seed in progress: B waits inside mgr.subscribe
        b = asyncio.create_task(signals_ws(_ScriptedWS(state, [sub])))  # type: ignore[arg-type]
        await asyncio.sleep(0.01)
        b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b

    assert mgr.refcount("AAPL") == 1
