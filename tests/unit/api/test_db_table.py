from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import QueryParams

from src.api.errors import ApiError, install_error_handlers
from src.api.routes.db_catalog import ColumnInfo, DatabaseCatalog, TableInfo
from src.api.routes.db_table import build_query, router

AUTH_HEADERS = {"Authorization": "Bearer unit-test-token"}


@pytest.fixture(autouse=True)
def db_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APEX_PG_READ_TOKEN", "unit-test-token")


TABLE = TableInfo(
    schema="uw_scan",
    name="facts",
    columns=(
        ColumnInfo("id", "bigint", "int8", False),
        ColumnInfo("ticker", "text", "text", False),
        ColumnInfo("as_of", "date", "date", False),
        ColumnInfo("score", "numeric", "numeric", True),
    ),
    primary_key=("id",),
)

GREEKS_TABLE = TableInfo(
    schema="uw_scan",
    name="greeks_by_expiry_strike",
    columns=(
        ColumnInfo("run_id", "bigint", "int8", False),
        ColumnInfo("ticker", "text", "text", False),
        ColumnInfo("expiry", "date", "date", False),
        ColumnInfo("strike", "numeric", "numeric", False),
        ColumnInfo("call_vanna", "double precision", "float8", True),
    ),
    primary_key=("run_id", "ticker", "expiry", "strike"),
)


def test_query_binds_typed_values_and_defaults_to_primary_key_order() -> None:
    injection = "SPY'; DROP TABLE facts;--"
    plan = build_query(
        TABLE,
        QueryParams(
            [
                ("columns", "ticker,score"),
                ("where", f"ticker:eq:{injection}"),
                ("where", "as_of:ge:2026-09-22"),
                ("where", "score:lt:1.25"),
                ("limit", "10"),
            ]
        ),
    )
    assert injection not in plan.sql
    assert plan.params == (injection, date(2026, 9, 22), Decimal("1.25"), 11, 0)
    assert 'ORDER BY "id"' in plan.sql
    assert plan.sql.endswith("LIMIT $4 OFFSET $5")


@pytest.mark.parametrize(
    ("where", "fragment", "values"),
    [
        ("id:eq:1", '"id" = $1', (1,)),
        ("id:ne:1", '"id" <> $1', (1,)),
        ("score:lt:1.5", '"score" < $1', (Decimal("1.5"),)),
        ("score:le:1.5", '"score" <= $1', (Decimal("1.5"),)),
        ("score:gt:1.5", '"score" > $1', (Decimal("1.5"),)),
        ("score:ge:1.5", '"score" >= $1', (Decimal("1.5"),)),
        ("id:in:1,2", '"id" IN ($1, $2)', (1, 2)),
        ("ticker:like:A%", '"ticker" LIKE $1', ("A%",)),
        ("score:isnull:true", '"score" IS NULL', ()),
        ("score:isnull:false", '"score" IS NOT NULL', ()),
    ],
)
def test_every_supported_filter_operator_builds_bound_sql(
    where: str, fragment: str, values: tuple[Any, ...]
) -> None:
    plan = build_query(TABLE, QueryParams([("where", where)]))
    assert fragment in plan.sql
    assert plan.params[: len(values)] == values
    assert plan.params[-2:] == (501, 0)


def test_query_clamps_limit_and_validates_all_query_shapes() -> None:
    assert build_query(TABLE, QueryParams("limit=999999")).limit == 5000
    invalid = [
        "wat=1",
        "columns=missing",
        "where=missing:eq:1",
        "where=id:nope:1",
        "where=id:isnull:maybe",
        "where=id:in:",
        "where=id:like:1",
        "order=missing",
        "order=id:sideways",
        "limit=0",
        "offset=-1",
        "columns=id&columns=ticker",
    ]
    for query in invalid:
        with pytest.raises(ApiError):
            build_query(TABLE, QueryParams(query))
    with pytest.raises(ApiError, match="offset requires order"):
        build_query(replace(TABLE, primary_key=()), QueryParams("offset=1"))


def test_order_on_nonunique_column_appends_primary_key_tiebreakers() -> None:
    plan = build_query(GREEKS_TABLE, QueryParams("order=strike:desc"))
    assert 'ORDER BY "strike" DESC, "run_id" ASC, "ticker" ASC, "expiry" ASC' in plan.sql
    # A caller-ordered unique key needs no tie-breaker.
    assert 'ORDER BY "id" DESC' in build_query(TABLE, QueryParams("order=id:desc")).sql
    assert "run_id" not in build_query(TABLE, QueryParams("order=id:desc")).sql


def test_offset_is_bounded_by_signed_int64() -> None:
    plan = build_query(TABLE, QueryParams("offset=9223372036854775807"))
    assert plan.params[-1] == 9_223_372_036_854_775_807
    with pytest.raises(ApiError, match="offset is too large"):
        build_query(TABLE, QueryParams("offset=9223372036854775808"))


class _Transaction:
    def __init__(self, connection: "_Connection", readonly: bool) -> None:
        self.connection = connection
        self.readonly = readonly

    async def __aenter__(self) -> None:
        self.connection.readonly_values.append(self.readonly)

    async def __aexit__(self, *args: object) -> None:
        return None


class _Connection:
    def __init__(self, rows: list[dict[str, Any]], *, error: Exception | None = None) -> None:
        self.rows = rows
        self.error = error
        self.readonly_values: list[bool] = []
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self, *, readonly: bool) -> _Transaction:
        return _Transaction(self, readonly)

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.fetches.append((sql, params))
        if self.error is not None:
            raise self.error
        return self.rows


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _Connection:
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        return None


class _Pool:
    def __init__(self, connection: _Connection, *, acquire_error: Exception | None = None) -> None:
        self.connection = connection
        self.acquire_error = acquire_error

    def acquire(self, *, timeout: float) -> _Acquire:
        assert timeout == 30.0
        if self.acquire_error is not None:
            raise self.acquire_error
        return _Acquire(self.connection)


class _CatalogCache:
    def __init__(self) -> None:
        self.calls = 0

    async def get(self, database: str, pool: object) -> DatabaseCatalog:
        self.calls += 1
        return DatabaseCatalog(
            database,
            {
                (TABLE.schema, TABLE.name): TABLE,
                (GREEKS_TABLE.schema, GREEKS_TABLE.name): GREEKS_TABLE,
            },
        )


def _app(connection: _Connection) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    install_error_handlers(app)
    app.state.pg_read_pools = {"warehouse": _Pool(connection)}
    app.state.pg_catalog_cache = _CatalogCache()
    return app


async def test_route_fetches_one_extra_and_marks_a_lower_requested_limit_truncated() -> None:
    connection = _Connection(
        [
            {"id": 1, "ticker": "AAA", "as_of": date(2026, 9, 22), "score": Decimal("1.1")},
            {"id": 2, "ticker": "BBB", "as_of": date(2026, 9, 22), "score": Decimal("2.2")},
        ]
    )
    async with AsyncClient(
        transport=ASGITransport(app=_app(connection)),
        base_url="http://test",
        headers=AUTH_HEADERS,
    ) as client:
        response = await client.get("/v1/db/warehouse/uw_scan/facts?columns=id,score&limit=1")
    assert response.status_code == 200
    assert response.json()["rows"] == [[1, "1.1"]]
    assert response.json()["truncated"] is True
    assert connection.fetches[0][1][-2:] == (2, 0)
    assert connection.readonly_values == [True]


async def test_empty_result_keeps_catalog_column_metadata() -> None:
    connection = _Connection([])
    async with AsyncClient(
        transport=ASGITransport(app=_app(connection)),
        base_url="http://test",
        headers=AUTH_HEADERS,
    ) as client:
        response = await client.get("/v1/db/warehouse/uw_scan/facts?columns=ticker,as_of")
    assert response.status_code == 200
    assert response.json()["columns"] == [
        {"name": "ticker", "type": "text"},
        {"name": "as_of", "type": "date"},
    ]
    assert response.json()["rows"] == []


async def test_generic_table_payload_with_frozen_greeks_source_subset() -> None:
    """The row is a direct-column subset of the 2026-09-22 strike_grid evidence."""
    fixture = Path("tests/fixtures/pg_read_api/rows.jsonl")
    entries = [json.loads(line, parse_float=Decimal) for line in fixture.read_text().splitlines()]
    source = next(entry for entry in entries if entry["query_id"] == "strike_grid")["rows"][0]
    row = {
        "run_id": source["run_id"],
        "ticker": source["ticker"],
        "expiry": date.fromisoformat(source["expiry"]),
        "strike": Decimal(str(source["strike"])),
        "call_vanna": float(source["call_vanna"]),
    }
    connection = _Connection([row])
    path = (
        "/v1/db/warehouse/uw_scan/greeks_by_expiry_strike"
        "?columns=run_id,ticker,expiry,strike,call_vanna&limit=1"
    )
    async with AsyncClient(
        transport=ASGITransport(app=_app(connection)),
        base_url="http://test",
        headers=AUTH_HEADERS,
    ) as client:
        response = await client.get(path)
    assert response.status_code == 200
    assert response.json()["rows"] == [
        [
            source["run_id"],
            source["ticker"],
            source["expiry"],
            str(source["strike"]),
            float(source["call_vanna"]),
        ]
    ]


async def test_unknown_database_table_and_timeout_use_typed_errors() -> None:
    connection = _Connection([], error=asyncpg.QueryCanceledError())
    app = _app(connection)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        unknown_database = await client.get("/v1/db/nope/uw_scan/facts")
        unknown_table = await client.get("/v1/db/warehouse/uw_scan/nope")
        timeout = await client.get("/v1/db/warehouse/uw_scan/facts")
    assert unknown_database.status_code == 400
    assert unknown_database.json()["error"]["code"] == "invalid_parameter"
    assert unknown_table.status_code == 400
    assert timeout.status_code == 504
    assert timeout.json()["error"]["code"] == "query_timeout"


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (asyncpg.NumericValueOutOfRangeError("overflow"), 400, "invalid_parameter"),
        (asyncpg.UndefinedFunctionError("bad operator"), 400, "invalid_parameter"),
        (asyncpg.PostgresConnectionError("secret host detail"), 503, "provider_not_configured"),
    ],
)
async def test_driver_filter_and_connection_errors_are_redacted(
    error: Exception, status: int, code: str
) -> None:
    connection = _Connection([], error=error)
    async with AsyncClient(
        transport=ASGITransport(app=_app(connection)),
        base_url="http://test",
        headers=AUTH_HEADERS,
    ) as client:
        response = await client.get("/v1/db/warehouse/uw_scan/facts?where=id:eq:1")
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "secret" not in response.text


async def test_pool_acquisition_has_a_bounded_timeout() -> None:
    connection = _Connection([])
    app = _app(connection)
    app.state.pg_read_pools = {"warehouse": _Pool(connection, acquire_error=TimeoutError())}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get("/v1/db/warehouse/uw_scan/facts")
    assert response.status_code == 504


async def test_table_auth_runs_before_catalog_and_pool_access() -> None:
    connection = _Connection([])
    cache = _CatalogCache()
    app = _app(connection)
    app.state.pg_catalog_cache = cache
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        missing = await client.get("/v1/db/warehouse/uw_scan/facts")
        wrong = await client.get(
            "/v1/db/warehouse/uw_scan/facts", headers={"Authorization": "Bearer nope"}
        )
    for response in (missing, wrong):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["error"]["code"] == "unauthorized"
        assert "unit-test-token" not in response.text
    assert cache.calls == 0
    assert connection.fetches == []
    assert connection.readonly_values == []
