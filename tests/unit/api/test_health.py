"""Tests for API health endpoint."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.routes import health as health_module
from src.api.server import APEX_VERSION, create_app


@pytest.fixture(autouse=True)
def _reset_recency_cache():
    """/health's recency cache is module-level; keep tests from leaking into each other."""
    health_module._recency = {"value": None, "as_of": None}
    health_module._recency_task = None
    yield
    health_module._recency_task = None


async def _drain_recency_refresh() -> None:
    """Await the background refresh /health kicked off, if any."""
    task = health_module._recency_task
    if task is not None:
        await task


@pytest.mark.asyncio
async def test_health_returns_ok():
    """GET /health returns status ok with uptime, service name, and running version."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime" in data
    assert data["service"] == "apex-signal-server"
    assert data["silver_revision"] == {"enabled": False}
    assert data["livewire"] == {
        "configured": False,
        "configured_price_mode": "raw",
        "effective_price_mode": None,
        "recency": None,
        "recency_as_of": None,
    }
    # version must reflect the real running build, not a hardcoded literal
    assert data["version"] == APEX_VERSION
    assert data["version"] not in ("", "unknown")


@pytest.mark.asyncio
async def test_health_includes_revision_watcher_state():
    class _Watcher:
        def health(self) -> dict:
            return {
                "enabled": True,
                "observed_revision": 42,
                "last_fully_applied_revision": 41,
            }

    app = create_app()
    app.state.revision_watcher = _Watcher()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.json()["silver_revision"]["observed_revision"] == 42
    assert resp.json()["silver_revision"]["last_fully_applied_revision"] == 41


@pytest.mark.asyncio
async def test_health_reports_configured_and_effective_price_mode(tmp_path):
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    app = create_app()
    app.state.livewire_price_mode = "adjusted"
    app.state.ohlc_provider = LivewireOhlcProvider(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        price_mode="adjusted",
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    # First call serves the empty cache; the lake probe runs in the background.
    assert resp.json()["livewire"] == {
        "configured": True,
        "configured_price_mode": "adjusted",
        "effective_price_mode": "adjusted",
        "recency": None,
        "recency_as_of": None,
    }
    await _drain_recency_refresh()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    # A tmp_path provider whose artifacts do not exist: nulls, not an error.
    assert resp.json()["livewire"]["recency"] == {
        "bronze_last_trade_date": None,
        "silver_last_trade_date": None,
        "lag_days": None,
    }
    assert resp.json()["livewire"]["recency_as_of"] is not None


def test_health_reports_zero_lag_when_silver_matches_bronze(tmp_path: Path) -> None:
    """Real production state 2026-08-23: bronze and silver are both at 2026-08-21."""
    import datetime as dt

    import pandas as pd

    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    dates = [dt.date(2026, 8, 19), dt.date(2026, 8, 20), dt.date(2026, 8, 21)]
    ohlc = {
        "open": [310.140, 317.455, 312.050],
        "high": [319.2799, 320.2800, 312.3800],
        "low": [309.60, 310.65, 307.01],
        "close": [316.83, 311.30, 309.35],
        "volume": [51405496, 40959127, 48591536],
    }
    bronze = tmp_path / "bronze" / "asset_class=equity" / "symbol=AAPL"
    silver = tmp_path / "silver" / "asset_class=equity" / "symbol=AAPL"
    for d in (bronze, silver):
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": dates, **ohlc}).to_parquet(d / "1d.parquet")

    provider = LivewireOhlcProvider(
        bronze_root=tmp_path / "bronze", silver_root=tmp_path / "silver", price_mode="adjusted"
    )
    recency = provider.fetch_recency("AAPL")
    assert recency["bronze_last_trade_date"] == "2026-08-21"
    assert recency["silver_last_trade_date"] == "2026-08-21"
    assert recency["lag_days"] == 0


def test_recency_is_null_when_artifacts_are_absent(tmp_path: Path) -> None:
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    provider = LivewireOhlcProvider(bronze_root=tmp_path)
    assert provider.fetch_recency("AAPL") == {
        "bronze_last_trade_date": None,
        "silver_last_trade_date": None,
        "lag_days": None,
    }


@pytest.mark.asyncio
async def test_health_endpoint_exposes_recency() -> None:
    """A consumer must be able to see how stale the lake is without a second call."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        data = (await client.get("/health")).json()
    assert "recency" in data["livewire"]
    # No provider configured in this app, so recency is null rather than absent.
    assert data["livewire"]["recency"] is None


class _SlowProvider:
    """Stands in for a provider whose lake mount is busy (livewire's 14:00 rewrite)."""

    price_mode = "raw"

    def fetch_recency(self, reference_symbol: str = "AAPL") -> dict:
        time.sleep(1)
        return {
            "bronze_last_trade_date": "2026-08-21",
            "silver_last_trade_date": "2026-08-21",
            "lag_days": 0,
        }


class _DeniedProvider:
    """Stands in for a probe landing on a parquet file mid-replace."""

    price_mode = "raw"

    def fetch_recency(self, reference_symbol: str = "AAPL") -> dict:
        raise PermissionError(
            13, "Permission denied", "/data/livewire/asset_class=equity/symbol=AAPL/1d.parquet"
        )


@pytest.mark.asyncio
async def test_health_does_not_block_on_a_busy_lake() -> None:
    """The 2026-09-03..05 outage: a slow lake made /health miss the 5s probe timeout."""
    app = create_app()
    app.state.ohlc_provider = _SlowProvider()
    transport = ASGITransport(app=app)
    started = time.monotonic()
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    elapsed = time.monotonic() - started

    assert resp.status_code == 200
    assert elapsed < 0.5
    assert resp.json()["livewire"]["recency"] is None
    await _drain_recency_refresh()


@pytest.mark.asyncio
async def test_health_serves_the_refreshed_value_on_the_next_call() -> None:
    app = create_app()
    app.state.ohlc_provider = _SlowProvider()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/health")
        await _drain_recency_refresh()
        data = (await client.get("/health")).json()

    assert data["livewire"]["recency"]["bronze_last_trade_date"] == "2026-08-21"
    assert data["livewire"]["recency_as_of"] is not None


@pytest.mark.asyncio
async def test_health_keeps_the_last_value_when_the_probe_is_denied() -> None:
    """PermissionError off the exfat lake must not 500 /health, nor drop the cache."""
    app = create_app()
    app.state.ohlc_provider = _SlowProvider()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/health")
        await _drain_recency_refresh()

        app.state.ohlc_provider = _DeniedProvider()
        health_module._recency["as_of"] = None  # force a refresh on the next call
        resp = await client.get("/health")
        await _drain_recency_refresh()
        data = (await client.get("/health")).json()

    assert resp.status_code == 200
    assert data["livewire"]["recency"]["bronze_last_trade_date"] == "2026-08-21"


@pytest.mark.asyncio
async def test_health_runs_only_one_lake_probe_at_a_time() -> None:
    """Single-flight: a multi-hour lake stall must not pile up executor threads."""
    app = create_app()
    app.state.ohlc_provider = _SlowProvider()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/health")
        first = health_module._recency_task
        await asyncio.gather(*(client.get("/health") for _ in range(5)))
        assert health_module._recency_task is first
        await _drain_recency_refresh()


def test_last_trade_date_returns_none_on_permission_denied(tmp_path: Path, monkeypatch) -> None:
    """livewire rewrites 1d.parquet in place; a probe mid-replace returns None, not a 500."""
    import duckdb

    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    target = tmp_path / "asset_class=equity" / "symbol=AAPL"
    target.mkdir(parents=True)
    (target / "1d.parquet").touch()

    def _denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(target / "1d.parquet"))

    monkeypatch.setattr(duckdb, "connect", _denied)
    assert LivewireOhlcProvider._last_trade_date(target / "1d.parquet") is None
