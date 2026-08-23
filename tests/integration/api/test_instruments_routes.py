"""Discovery routes: /v1/instruments and the per-instrument detail endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.payload.validate import validate_payload
from src.api.server import create_app
from src.infrastructure.adapters.livewire.coverage import CoverageCatalog


def _app(catalog_db: Path | None) -> object:
    app = create_app()
    app.state.coverage_catalog = CoverageCatalog(catalog_db) if catalog_db else None
    return app


@pytest.mark.asyncio
async def test_instruments_route_returns_schema_valid_payload(catalog_db: Path) -> None:
    app = _app(catalog_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/instruments")
    assert r.status_code == 200
    body = r.json()
    validate_payload(body, "instruments_payload")
    assert body["source"] == "livewire_coverage_snapshot"
    assert {i["symbol"] for i in body["instruments"]} == {"AAPL", "HON", "VIX", "DGS10"}


@pytest.mark.asyncio
async def test_instruments_without_catalog_is_503_with_a_code() -> None:
    app = _app(None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/instruments")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "provider_not_configured"


@pytest.mark.asyncio
async def test_unreadable_catalog_is_503_not_an_empty_universe(tmp_path: Path) -> None:
    """The exact production failure: a bind mount outside colima's VM mount set leaves
    a DIRECTORY at the catalog path. Reporting zero instruments would let a broken
    deployment masquerade as a correct one."""
    fake = tmp_path / "analytics.duckdb"
    fake.mkdir()
    app = _app(fake)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/instruments")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "provider_not_configured"


@pytest.mark.asyncio
async def test_filters_by_asset_class(catalog_db: Path) -> None:
    app = _app(catalog_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/instruments", params={"asset_class": "rates"})
    assert [i["symbol"] for i in r.json()["instruments"]] == ["DGS10"]


@pytest.mark.asyncio
async def test_unknown_asset_class_filter_is_400(catalog_db: Path) -> None:
    app = _app(catalog_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/instruments", params={"asset_class": "crypto"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_asset_class"


@pytest.mark.asyncio
async def test_delisted_discovery_is_a_typed_501(catalog_db: Path) -> None:
    app = _app(catalog_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/instruments", params={"listing": "delisted"})
    assert r.status_code == 501
    assert r.json()["error"]["code"] == "not_yet_available"


@pytest.mark.asyncio
async def test_silver_availability_is_visible_before_requesting_bars(catalog_db: Path) -> None:
    """HON has bronze but no Silver; a consumer must learn that here rather than by
    taking a 503 on /bars."""
    app = _app(catalog_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/instruments")
    by_symbol = {i["symbol"]: i for i in r.json()["instruments"]}
    assert by_symbol["AAPL"]["silver_available"] is True
    assert by_symbol["AAPL"]["price_mode"] == "adjusted"
    assert by_symbol["HON"]["silver_available"] is False
    assert by_symbol["HON"]["price_mode"] == "raw"
