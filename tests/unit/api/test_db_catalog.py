from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import asyncpg
import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from src.api.errors import ApiError, ApiErrorCode, install_error_handlers
from src.api.routes.db_catalog import (
    CatalogCache,
    ColumnInfo,
    DatabaseCatalog,
    TableInfo,
    build_catalog_payload,
    build_database_catalog,
    get_catalog,
    router,
)
from src.infrastructure.persistence.read_pools import create_read_pools, parse_read_urls

AUTH_HEADERS = {"Authorization": "Bearer unit-test-token"}


@pytest.fixture(autouse=True)
def db_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APEX_PG_READ_TOKEN", "unit-test-token")


def _column(schema: str, table: str, name: str, ordinal: int) -> dict[str, Any]:
    return {
        "schema_name": schema,
        "table_name": table,
        "column_name": name,
        "data_type": "integer",
        "value_type": "int4",
        "nullable": False,
        "ordinal": ordinal,
    }


def _constraint(
    name: str,
    kind: str,
    schema: str,
    table: str,
    column: str,
    ordinal: int,
    target: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    return {
        "constraint_name": name,
        "constraint_type": kind,
        "schema_name": schema,
        "table_name": table,
        "column_name": column,
        "ordinality": ordinal,
        "target_schema": target[0] if target else None,
        "target_table": target[1] if target else None,
        "target_column": target[2] if target else None,
    }


def test_catalog_excludes_internal_tables_and_pairs_composite_keys_by_ordinality() -> None:
    columns = [
        _column("public", "child", "left_id", 1),
        _column("public", "child", "right_id", 2),
        _column("public", "parent", "a", 1),
        _column("public", "parent", "b", 2),
        _column("public", "api_request_audit", "id", 1),
        _column("_timescaledb_internal", "chunk", "id", 1),
    ]
    constraints = [
        _constraint("child_pk", "p", "public", "child", "left_id", 1),
        _constraint("child_pk", "p", "public", "child", "right_id", 2),
        _constraint("child_fk", "f", "public", "child", "right_id", 2, ("public", "parent", "b")),
        _constraint("child_fk", "f", "public", "child", "left_id", 1, ("public", "parent", "a")),
        _constraint(
            "leak", "f", "public", "child", "left_id", 1, ("public", "api_request_audit", "id")
        ),
    ]

    catalog = build_database_catalog("warehouse", columns, constraints)
    assert set(catalog.tables) == {("public", "child"), ("public", "parent")}
    child = catalog.tables[("public", "child")]
    assert child.primary_key == ("left_id", "right_id")
    assert len(child.foreign_keys) == 1
    assert child.foreign_keys[0].columns == ("left_id", "right_id")
    assert child.foreign_keys[0].referenced_columns == ("a", "b")


def test_catalog_payload_validates_with_empty_database() -> None:
    payload = build_catalog_payload([DatabaseCatalog("warehouse", {})])
    assert payload["databases"] == [{"name": "warehouse", "schemas": []}]


def test_read_url_parser_uses_path_database_and_rejects_ambiguity() -> None:
    assert parse_read_urls("postgresql://reader:secret@db/one,postgres://reader@db/two") == {
        "one": "postgresql://reader:secret@db/one",
        "two": "postgres://reader@db/two",
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_read_urls("postgresql://db/one,postgresql://other/one")
    with pytest.raises(ValueError, match="invalid"):
        parse_read_urls("postgresql://db/one?database=two")


async def test_read_pools_are_dedicated_lazy_and_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_create_pool(dsn: str, **kwargs: Any) -> object:
        calls.append((dsn, kwargs))
        return object()

    monkeypatch.setattr(
        "src.infrastructure.persistence.read_pools.asyncpg.create_pool", fake_create_pool
    )
    pools = await create_read_pools("postgresql://reader@db/warehouse")
    assert set(pools) == {"warehouse"}
    assert calls[0][1]["min_size"] == 0
    assert calls[0][1]["max_size"] == 3
    assert calls[0][1]["server_settings"] == {
        "default_transaction_read_only": "on",
        "statement_timeout": "30s",
    }
    codecs: dict[str, dict[str, Any]] = {}

    class Connection:
        async def set_type_codec(self, name: str, **kwargs: Any) -> None:
            codecs[name] = kwargs

    await calls[0][1]["init"](Connection())
    assert codecs["jsonb"]["decoder"]('{"price": 1.25}') == {"price": Decimal("1.25")}
    assert codecs["json"]["encoder"]('{"price": 1.25}') == '{"price": 1.25}'


async def test_one_failed_read_pool_does_not_block_other_databases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_pool(dsn: str, **kwargs: Any) -> object:
        if dsn.endswith("/bad"):
            raise ConnectionError("contains-sensitive-host-detail")
        return object()

    monkeypatch.setattr(
        "src.infrastructure.persistence.read_pools.asyncpg.create_pool", fake_create_pool
    )
    pools = await create_read_pools("postgresql://reader@db/bad,postgresql://reader@db/good")
    assert pools["bad"] is None
    assert pools["good"] is not None

    app = FastAPI()
    app.include_router(router)
    install_error_handlers(app)
    app.state.pg_read_pools = pools
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        failed = await client.get("/v1/db/catalog?database=bad")
        never = await client.get("/v1/db/catalog?database=never")
    # Configured but unavailable is an upstream outage, not a bad request.
    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "provider_not_configured"
    assert "sensitive" not in failed.text
    assert never.status_code == 400
    assert never.json()["error"]["code"] == "invalid_parameter"


async def test_catalog_route_reports_missing_source_without_touching_database() -> None:
    app = FastAPI()
    app.include_router(router)
    install_error_handlers(app)
    app.state.pg_read_pools = {}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get("/v1/db/catalog")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"


async def test_catalog_route_rejects_repeated_database_parameter() -> None:
    app = FastAPI()
    app.include_router(router)
    install_error_handlers(app)
    app.state.pg_read_pools = {"one": object(), "two": object()}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get("/v1/db/catalog?database=one&database=two")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_parameter"


async def test_catalog_auth_runs_before_any_pool_access() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.acquires = 0

        def acquire(self, *, timeout: float) -> Any:
            self.acquires += 1
            raise AssertionError("auth must reject before pool acquisition")

    pool = _Pool()
    app = FastAPI()
    app.include_router(router)
    install_error_handlers(app)
    app.state.pg_read_pools = {"warehouse": pool}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        missing = await client.get("/v1/db/catalog")
        wrong = await client.get("/v1/db/catalog", headers={"Authorization": "Bearer nope"})
    for response in (missing, wrong):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["error"]["code"] == "unauthorized"
        assert "unit-test-token" not in response.text
    assert pool.acquires == 0


async def test_catalog_route_is_503_when_token_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APEX_PG_READ_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(router)
    install_error_handlers(app)
    app.state.pg_read_pools = {"warehouse": object()}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/db/catalog", headers=AUTH_HEADERS)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"


def test_db_token_rejects_non_ascii_bearer_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx cannot send non-ASCII headers, so exercise the raw ASGI boundary."""
    from src.api.routes._db_auth import require_db_token

    monkeypatch.setenv("APEX_PG_READ_TOKEN", "unit-test-token")
    request = Request(
        {
            "type": "http",
            "headers": [(b"authorization", "Bearer täken".encode("utf-8"))],
        }
    )
    with pytest.raises(ApiError) as excinfo:
        require_db_token(request)
    assert excinfo.value.code is ApiErrorCode.UNAUTHORIZED


async def test_catalog_cache_reuses_result_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_fetch(database: str, pool: object) -> DatabaseCatalog:
        nonlocal calls
        calls += 1
        return DatabaseCatalog(database, {})

    monkeypatch.setattr("src.api.routes.db_catalog.fetch_database_catalog", fake_fetch)
    cache = CatalogCache(ttl_seconds=600)
    assert await cache.get("warehouse", object()) is await cache.get("warehouse", object())
    assert calls == 1


async def test_catalog_cache_refetches_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    clock = [100.0]

    async def fake_fetch(database: str, pool: object) -> DatabaseCatalog:
        nonlocal calls
        calls += 1
        return DatabaseCatalog(database, {})

    monkeypatch.setattr("src.api.routes.db_catalog.fetch_database_catalog", fake_fetch)
    monkeypatch.setattr("src.api.routes.db_catalog.time.monotonic", lambda: clock[0])
    cache = CatalogCache(ttl_seconds=600)
    await cache.get("warehouse", object())
    clock[0] = 700.0
    await cache.get("warehouse", object())
    assert calls == 2


async def test_catalog_cache_is_independent_per_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_fetch(database: str, pool: object) -> DatabaseCatalog:
        nonlocal calls
        calls += 1
        return DatabaseCatalog(database, {})

    monkeypatch.setattr("src.api.routes.db_catalog.fetch_database_catalog", fake_fetch)
    first, second = FastAPI(), FastAPI()
    first.state.pg_read_pools = {"warehouse": object()}
    second.state.pg_read_pools = {"warehouse": object()}
    first_request = Request({"type": "http", "app": first})
    second_request = Request({"type": "http", "app": second})
    await get_catalog(first_request, "warehouse")
    await get_catalog(first_request, "warehouse")
    await get_catalog(second_request, "warehouse")
    assert calls == 2
    assert first.state.pg_catalog_cache is not second.state.pg_catalog_cache


def _catalog_app(pools: dict[str, Any]) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    install_error_handlers(app)
    app.state.pg_read_pools = pools
    return app


class _FailingPool:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def acquire(self, *, timeout: float) -> Any:
        assert timeout == 30.0
        raise self.error


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (TimeoutError(), 504, "query_timeout"),
        (OSError("secret host detail"), 503, "provider_not_configured"),
        (asyncpg.InvalidPasswordError("secret password rejected"), 503, "provider_not_configured"),
    ],
)
async def test_catalog_pool_failures_map_to_typed_codes(
    error: Exception, status: int, code: str
) -> None:
    app = _catalog_app({"warehouse": _FailingPool(error)})
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get("/v1/db/catalog?database=warehouse")
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "secret" not in response.text


async def test_catalog_unknown_database_is_400_without_pool_access() -> None:
    app = _catalog_app({"warehouse": _FailingPool(AssertionError("must not acquire"))})
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get("/v1/db/catalog?database=postgres")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_parameter"


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Bearer", "Bearer ", "Basic unit-test-token", "Token unit-test-token", "Bearer x"],
)
def test_db_token_rejects_missing_empty_and_foreign_credentials(
    monkeypatch: pytest.MonkeyPatch, authorization: str | None
) -> None:
    from src.api.routes._db_auth import require_db_token

    monkeypatch.setenv("APEX_PG_READ_TOKEN", "unit-test-token")
    headers = [] if authorization is None else [(b"authorization", authorization.encode())]
    with pytest.raises(ApiError) as excinfo:
        require_db_token(Request({"type": "http", "headers": headers}))
    assert excinfo.value.code is ApiErrorCode.UNAUTHORIZED


def test_db_token_empty_env_is_unconfigured_not_an_empty_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api.routes._db_auth import require_db_token

    monkeypatch.setenv("APEX_PG_READ_TOKEN", "")
    request = Request({"type": "http", "headers": [(b"authorization", b"Bearer ")]})
    with pytest.raises(ApiError) as excinfo:
        require_db_token(request)
    assert excinfo.value.code is ApiErrorCode.PROVIDER_NOT_CONFIGURED


async def test_unfiltered_catalog_lists_reachable_databases_and_names_the_rest() -> None:
    # A down host (OSError at acquire) and a pool that failed at startup (None) are both
    # reported in `unavailable`; the reachable database is still listed.
    watchlist = TableInfo(
        "uw_scan", "watchlist", (ColumnInfo("ticker", "text", "string", False),), ("ticker",)
    )
    app = _catalog_app(
        {"option_wizard": object(), "core": _FailingPool(OSError("secret host")), "trading": None}
    )
    cache = CatalogCache()
    cache._entries["option_wizard"] = (
        time.monotonic(),
        DatabaseCatalog("option_wizard", {("uw_scan", "watchlist"): watchlist}),
    )
    app.state.pg_catalog_cache = cache
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get("/v1/db/catalog")
    assert response.status_code == 200
    body = response.json()
    assert [database["name"] for database in body["databases"]] == ["option_wizard"]
    assert body["unavailable"] == ["core", "trading"]
    assert "secret" not in response.text


async def test_catalog_cache_invalidate_forces_a_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    async def fake_fetch(database: str, pool: Any) -> DatabaseCatalog:
        fetched.append(database)
        return DatabaseCatalog(database, {})

    monkeypatch.setattr("src.api.routes.db_catalog.fetch_database_catalog", fake_fetch)
    cache = CatalogCache()
    await cache.get("option_wizard", object())
    await cache.get("option_wizard", object())
    assert fetched == ["option_wizard"]
    cache.invalidate("option_wizard")
    cache.invalidate("never_cached")
    await cache.get("option_wizard", object())
    assert fetched == ["option_wizard", "option_wizard"]
