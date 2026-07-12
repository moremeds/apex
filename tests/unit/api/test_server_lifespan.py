"""Tests for API server lifespan (PG connection lifecycle)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_server_connects_to_pg_on_startup():
    """Lifespan opens an asyncpg pool when APEX_PG_URL is set."""
    with patch.dict(os.environ, {"APEX_PG_URL": "postgresql://test@localhost/test"}):
        with patch("src.api.server.asyncpg") as mock_asyncpg:
            mock_pool = AsyncMock()
            mock_pool.close = AsyncMock()
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

            from src.api.server import create_app, lifespan

            app = create_app()

            async with lifespan(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/health")

                assert resp.status_code == 200
                assert resp.json()["pg_connected"] is True

            # On shutdown, the pool should be closed
            mock_pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_server_runs_without_pg():
    """Without APEX_PG_URL, /health reports pg_connected=False but app still serves."""
    env = {k: v for k, v in os.environ.items() if k != "APEX_PG_URL"}
    with patch.dict(os.environ, env, clear=True):
        from src.api.server import create_app, lifespan

        app = create_app()
        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")

            assert resp.status_code == 200
            assert resp.json()["pg_connected"] is False


# --- Phase 4: env-gated xenon live-feed wiring --------------------------------

import src.infrastructure.adapters.xenon.client as xenon_client_mod  # noqa: E402
import src.application.subscriptions.revision_watcher as revision_watcher_mod  # noqa: E402
from src.api.server import create_app, lifespan  # noqa: E402


class _FakeBus:
    def publish(self, *a, **k) -> None: ...


class _FakeSM:
    def __init__(self) -> None:
        self.live_feed = None

    def set_live_feed(self, feed) -> None:
        self.live_feed = feed


class _SpyClient:
    last = None

    def __init__(self, url, event_bus=None, **kw) -> None:
        self.url = url
        self.event_bus = event_bus
        self.connected = False
        self.closed = False
        _SpyClient.last = self

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True


class _SpyWatcher:
    last = None

    def __init__(self, reader, manager, poll_seconds) -> None:
        self.reader = reader
        self.manager = manager
        self.poll_seconds = poll_seconds
        self.started = False
        self.stopped = False
        _SpyWatcher.last = self

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def health(self) -> dict:
        return {"enabled": True}


@pytest.mark.asyncio
async def test_lifespan_builds_connects_and_wires_xenon_client(monkeypatch) -> None:
    monkeypatch.setenv("APEX_XENON_WS_URL", "ws://127.0.0.1:1")
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    monkeypatch.setattr(xenon_client_mod, "XenonTickClient", _SpyClient)
    app = create_app()
    app.state.event_bus = _FakeBus()
    app.state.subscription_manager = _FakeSM()
    async with lifespan(app):
        assert isinstance(app.state.xenon_client, _SpyClient)
        assert app.state.xenon_client.connected is True
        assert app.state.subscription_manager.live_feed is app.state.xenon_client
    # after shutdown:
    assert _SpyClient.last.closed is True


@pytest.mark.asyncio
async def test_lifespan_skips_xenon_client_when_no_event_bus(monkeypatch) -> None:
    """No event bus (no pipeline) -> no xenon client, even though the URL now
    defaults to the standard port."""
    monkeypatch.delenv("APEX_XENON_WS_URL", raising=False)
    monkeypatch.delenv("APEX_LIVEWIRE_ROOT", raising=False)
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    monkeypatch.setattr(xenon_client_mod, "XenonTickClient", _SpyClient)
    app = create_app()
    async with lifespan(app):
        assert getattr(app.state, "xenon_client", None) is None


@pytest.mark.asyncio
async def test_lifespan_defaults_xenon_url_to_standard_port(monkeypatch) -> None:
    """APEX_XENON_WS_URL is baked into apex: unset -> connect to ws://127.0.0.1:8765."""
    monkeypatch.delenv("APEX_XENON_WS_URL", raising=False)
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    monkeypatch.setattr(xenon_client_mod, "XenonTickClient", _SpyClient)
    app = create_app()
    app.state.event_bus = _FakeBus()
    app.state.subscription_manager = _FakeSM()
    async with lifespan(app):
        assert isinstance(app.state.xenon_client, _SpyClient)
        assert app.state.xenon_client.url == "ws://127.0.0.1:8765"


# --- Task 13: full streaming pipeline wired in lifespan (env-gated) ------------


@pytest.mark.asyncio
async def test_lifespan_builds_full_pipeline_when_livewire_root_set(monkeypatch, tmp_path) -> None:
    from src.application.services.ta_signal_service import TASignalService
    from src.application.subscriptions.manager import SubscriptionManager
    from src.domain.events.priority_event_bus import PriorityEventBus
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    monkeypatch.setenv("APEX_LIVEWIRE_ROOT", str(tmp_path))
    monkeypatch.setenv("APEX_XENON_WS_URL", "ws://127.0.0.1:1")
    monkeypatch.setenv("APEX_TIMEFRAMES", "1m")
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    monkeypatch.setattr(xenon_client_mod, "XenonTickClient", _SpyClient)

    app = create_app()  # no pre-injected subscription_manager -> build the real graph
    async with lifespan(app):
        bus = app.state.event_bus
        sm = app.state.subscription_manager
        assert isinstance(bus, PriorityEventBus)
        assert isinstance(sm, SubscriptionManager)
        # the graph is wired: provider <- livewire, compute <- ta_service, live_feed <- xenon client
        assert isinstance(sm._provider, LivewireOhlcProvider)
        assert sm._compute is app.state.ta_service
        assert isinstance(app.state.ta_service, TASignalService)
        assert sm._live_feed is app.state.xenon_client
        assert app.state.signal_emitter is not None
        assert app.state.ta_service.is_running is True
    # clean shutdown: service + bus stopped, xenon client closed
    assert app.state.ta_service.is_running is False
    assert _SpyClient.last.closed is True


@pytest.mark.asyncio
async def test_lifespan_passes_signal_repo_as_persistence(monkeypatch, tmp_path) -> None:
    """A configured signal_repo is handed to TASignalService as its persistence
    sink (so fired signals are written to PG), and the pre-injected repo is not
    clobbered by the lifespan."""
    monkeypatch.setenv("APEX_LIVEWIRE_ROOT", str(tmp_path))
    monkeypatch.setenv("APEX_XENON_WS_URL", "ws://127.0.0.1:1")
    monkeypatch.setenv("APEX_TIMEFRAMES", "1m")
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    monkeypatch.setattr(xenon_client_mod, "XenonTickClient", _SpyClient)

    fake_repo = object()
    app = create_app()
    app.state.signal_repo = fake_repo  # pre-injected -> lifespan must not clobber
    async with lifespan(app):
        assert app.state.signal_repo is fake_repo
        assert app.state.ta_service._persistence is fake_repo


@pytest.mark.asyncio
async def test_lifespan_skips_pipeline_when_livewire_root_unset(monkeypatch) -> None:
    monkeypatch.delenv("APEX_LIVEWIRE_ROOT", raising=False)
    monkeypatch.delenv("APEX_XENON_WS_URL", raising=False)
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    app = create_app()
    async with lifespan(app):
        assert getattr(app.state, "subscription_manager", None) is None
        assert getattr(app.state, "ta_service", None) is None


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_revision_watcher(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APEX_LIVEWIRE_SILVER_ROOT", str(tmp_path))
    monkeypatch.setenv("APEX_LIVEWIRE_REVISION_POLL_SECONDS", "7.5")
    monkeypatch.delenv("APEX_LIVEWIRE_ROOT", raising=False)
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    monkeypatch.setattr(revision_watcher_mod, "RevisionWatcher", _SpyWatcher)
    app = create_app()
    app.state.subscription_manager = _FakeSM()

    async with lifespan(app):
        watcher = app.state.revision_watcher
        assert watcher.started is True
        assert watcher.poll_seconds == 7.5

    assert _SpyWatcher.last.stopped is True
