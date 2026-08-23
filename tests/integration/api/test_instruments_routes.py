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


@pytest.mark.asyncio
async def test_instrument_detail_probes_actual_timeframes(tmp_path: Path, catalog_db: Path) -> None:
    """The coverage table measures no equity intraday, so per-symbol timeframes come
    from artifact probes -- five Path.exists() calls, affordable for one symbol."""
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    d = tmp_path / "asset_class=equity" / "symbol=AAPL"
    d.mkdir(parents=True, exist_ok=True)
    for tf in ("1d", "1h"):
        (d / f"{tf}.parquet").write_bytes(b"")

    app = create_app()
    app.state.coverage_catalog = CoverageCatalog(catalog_db)
    app.state.ohlc_provider = LivewireOhlcProvider(bronze_root=tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/equity/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert sorted(body["timeframes"]) == ["1d", "1h"]
    assert body["asset_class"] == "equity"
    assert body["first_date"] == "1980-12-12"


@pytest.mark.asyncio
async def test_instrument_detail_unknown_symbol_is_404(tmp_path: Path, catalog_db: Path) -> None:
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    app = create_app()
    app.state.coverage_catalog = CoverageCatalog(catalog_db)
    app.state.ohlc_provider = LivewireOhlcProvider(bronze_root=tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/equity/NOTAREALTICKER")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_symbol"


@pytest.mark.asyncio
async def test_detail_does_not_shadow_the_list_route(catalog_db: Path) -> None:
    """/v1/instruments is one segment, /v1/{asset_class}/{symbol} is two -- Starlette
    matches whole patterns, so registration order cannot make them collide."""
    app = _app(catalog_db)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/instruments")
    assert r.status_code == 200
    assert "instruments" in r.json()


@pytest.mark.asyncio
async def test_exact_match_not_prefix_match(tmp_path: Path, catalog_db: Path) -> None:
    """get_instrument must not return AAPL when asked for AA."""
    catalog = CoverageCatalog(catalog_db)
    assert catalog.get_instrument("AA", "equity") is None
    assert catalog.get_instrument("AAPL", "equity").symbol == "AAPL"


@pytest.mark.asyncio
async def test_actions_and_delisting_are_typed_501s() -> None:
    """Specified now so the contract is stable; blocked on livewire L1/L2/L4.

    Measured 2026-08-23: no delisted symbol in the lake has correct corporate-action
    data (6,275 have none; the 2,345 that do are ticker reuses whose actions belong
    to a different, living company).
    """
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        for path in ("/v1/equity/BBBY/actions", "/v1/equity/BBBY/delisting"):
            r = await c.get(path)
            assert r.status_code == 501, path
            assert r.json()["error"]["code"] == "not_yet_available", path
            assert r.json()["error"]["symbol"] == "BBBY", path


@pytest.mark.asyncio
async def test_three_segment_routes_do_not_shadow_the_two_segment_detail(
    tmp_path: Path, catalog_db: Path
) -> None:
    """/v1/equity/{symbol}/actions (three segments) and /v1/{asset_class}/{symbol}
    (two) coexist -- Starlette matches whole patterns, not prefixes."""
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    d = tmp_path / "asset_class=equity" / "symbol=AAPL"
    d.mkdir(parents=True, exist_ok=True)
    (d / "1d.parquet").write_bytes(b"")

    app = create_app()
    app.state.coverage_catalog = CoverageCatalog(catalog_db)
    app.state.ohlc_provider = LivewireOhlcProvider(bronze_root=tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        detail = await c.get("/v1/equity/AAPL")
        actions = await c.get("/v1/equity/AAPL/actions")
    assert detail.status_code == 200
    assert detail.json()["timeframes"] == ["1d"]
    assert actions.status_code == 501
