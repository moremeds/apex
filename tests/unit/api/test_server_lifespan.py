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
async def test_lifespan_skips_xenon_client_when_url_unset(monkeypatch) -> None:
    monkeypatch.delenv("APEX_XENON_WS_URL", raising=False)
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    monkeypatch.setattr(xenon_client_mod, "XenonTickClient", _SpyClient)
    app = create_app()
    app.state.event_bus = _FakeBus()
    async with lifespan(app):
        assert getattr(app.state, "xenon_client", None) is None
