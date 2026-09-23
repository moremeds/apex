"""Contract checks for the curated UW read-only joins."""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import asyncpg
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import QueryParams

from src.api.errors import ApiError, install_error_handlers
from src.api.routes.db_catalog import _table_allowed
from src.api.routes.uw_joins import router
from src.api.uw_join_registry import JOIN_REGISTRY, build_join_query


@pytest.fixture(autouse=True)
def db_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APEX_PG_READ_TOKEN", "unit-test-token")


class _Context:
    def __init__(self, value: Any = None) -> None:
        self.value = value

    async def __aenter__(self) -> Any:
        return self.value

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _Statement:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.params: tuple[Any, ...] = ()
        names = list(rows[0]) if rows else ["ticker", "has_card", "has_quote", "has_run"]
        self.attributes = [
            SimpleNamespace(name=name, type=SimpleNamespace(name="text")) for name in names
        ]

    def get_attributes(self) -> list[Any]:
        return self.attributes

    async def fetch(self, *params: Any) -> list[dict[str, Any]]:
        self.params = params
        return self.rows


class _Connection:
    def __init__(self, rows: list[dict[str, Any]], *, error: Exception | None = None) -> None:
        self.error = error
        self.statement = _Statement(rows)
        self.readonly: bool | None = None
        self.executed: list[str] = []
        self.sql = ""

    def transaction(self, *, readonly: bool) -> _Context:
        self.readonly = readonly
        return _Context()

    async def execute(self, sql: str) -> None:
        self.executed.append(sql)

    async def prepare(self, sql: str) -> _Statement:
        self.sql = sql
        if self.error is not None:
            raise self.error
        return self.statement


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self, *, timeout: float) -> _Context:
        assert timeout == 30.0
        return _Context(self.connection)


def _app(connection: _Connection | None) -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router)
    app.state.pg_read_pools = {} if connection is None else {"option_wizard": _Pool(connection)}
    return app


def test_registry_binds_filters_and_defaults_latest_run_from_driver() -> None:
    assert len(JOIN_REGISTRY) == 11
    plan = build_join_query(
        "strike_grid",
        QueryParams("ticker=SPY&start=2026-09-01&end=2026-09-22&limit=9000"),
    )
    assert plan.limit == 5000
    assert plan.params == (
        "SPY",
        date(2026, 9, 1),
        date(2026, 9, 22),
        None,
        None,
        5001,
        0,
    )
    assert ":ticker" not in plan.sql
    assert "max(run_id)" in plan.sql
    assert "FROM uw_scan.greeks_by_expiry_strike" in plan.sql
    assert "scan_runs" not in plan.sql
    assert "CASE WHEN call_root=put_root THEN call_root END AS root" in plan.sql
    assert all(spec.coverage for spec in JOIN_REGISTRY.values())
    assert "max(run_id)" in build_join_query("trade_insight_thread", QueryParams("ticker=META")).sql
    assert build_join_query("universe_identity_sector", QueryParams("tier=ranked")).params[:2] == (
        None,
        "ranked",
    )
    assert JOIN_REGISTRY["chain_exposure"].coverage[0] == ("research_chains", "has_chain")
    # macro_domain_states.as_of is timestamptz: range filters must pin the UTC
    # calendar date, or the inclusive end loses the whole tail of that day.
    macro_sql = build_join_query(
        "macro_evidence_chain", QueryParams("start=2026-09-21&end=2026-09-21")
    ).sql
    assert macro_sql.count("(s.as_of AT TIME ZONE 'UTC')::date") == 2


async def test_route_is_read_only_bounded_and_reports_page_coverage() -> None:
    path = Path(__file__).resolve().parents[2] / "fixtures/pg_read_api/rows.jsonl"
    captured = [json.loads(line, parse_float=Decimal) for line in path.read_text().splitlines()]
    rows = next(
        record["rows"] for record in captured if record["query_id"] == "oi_change_with_quote"
    )
    connection = _Connection(rows)
    async with AsyncClient(
        transport=ASGITransport(app=_app(connection)),
        base_url="http://test",
        headers={"Authorization": "Bearer unit-test-token"},
    ) as client:
        response = await client.get("/v1/uw/oi_change_with_quote?ticker=SPY&limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1 and payload["truncated"] is True
    assert payload["coverage"] == {
        "scope": "returned_rows",
        "tables": {
            "option_contract_snapshots": {"matched": 1, "total": 1},
        },
    }
    assert connection.readonly is True
    assert connection.executed == ["SET LOCAL statement_timeout = '30s'"]
    assert connection.statement.params[-2:] == (2, 0)


async def test_empty_result_keeps_prepared_metadata_and_errors_are_typed() -> None:
    connection = _Connection([])
    async with AsyncClient(
        transport=ASGITransport(app=_app(connection)),
        base_url="http://test",
        headers={"Authorization": "Bearer unit-test-token"},
    ) as client:
        empty = await client.get("/v1/uw/watchlist_active_card")
        invalid = await client.get("/v1/uw/daily_ohlc_technical?start=2026-09-22")
        too_large = await client.get(
            "/v1/uw/scan_run_signal_bundle?ticker=SPY&run_id=9223372036854775808"
        )
    assert empty.status_code == 200
    assert [column["name"] for column in empty.json()["columns"]] == [
        "ticker",
        "has_card",
        "has_quote",
        "has_run",
    ]
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_parameter"
    assert too_large.status_code == 400

    async with AsyncClient(
        transport=ASGITransport(app=_app(None)),
        base_url="http://test",
        headers={"Authorization": "Bearer unit-test-token"},
    ) as client:
        missing = await client.get("/v1/uw/watchlist_active_card")
        unknown_join = await client.get("/v1/uw/not_a_join")
    assert missing.status_code == 503
    # The join name is validated before pool availability.
    assert unknown_join.status_code == 400
    assert unknown_join.json()["error"]["code"] == "invalid_parameter"


async def test_auth_runs_before_pool_access() -> None:
    connection = _Connection([])
    async with AsyncClient(
        transport=ASGITransport(app=_app(connection)), base_url="http://test"
    ) as client:
        missing = await client.get("/v1/uw/watchlist_active_card")
        wrong = await client.get(
            "/v1/uw/watchlist_active_card", headers={"Authorization": "Bearer nope"}
        )
    for response in (missing, wrong):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["error"]["code"] == "unauthorized"
        assert "unit-test-token" not in response.text
    assert connection.sql == ""
    assert connection.executed == []
    assert connection.readonly is None


_TABLE_REF = re.compile(r"\buw_scan\.(\w+)")
_LATEST_RUN_DRIVERS = {
    "scan_run_signal_bundle": "signal_gates",
    "oi_change_with_quote": "oi_change_events",
    "strike_grid": "greeks_by_expiry_strike",
    "trade_insight_thread": "trade_insight_snapshots",
}


@pytest.mark.parametrize("name", sorted(JOIN_REGISTRY))
def test_registry_templates_are_closed_and_avoid_excluded_tables(name: str) -> None:
    spec = JOIN_REGISTRY[name]
    tables = set(_TABLE_REF.findall(spec.sql))
    assert tables
    assert all(_table_allowed("uw_scan", table) for table in tables)
    # Every coverage table is referenced and every flag is a projected column.
    for table, flag in spec.coverage:
        assert table in tables
        assert f"AS {flag}" in spec.sql
    # Placeholders and declared filters match exactly; nothing else is bindable.
    tokens = set(re.findall(r":(ticker|tier|start|end|run_id|expiry)\b", spec.sql))
    assert tokens == set(spec.filters)
    assert spec.required <= set(spec.filters)
    compiled = build_join_query(
        name, QueryParams({key: "SPY" for key in spec.required} | {"limit": "1"})
    )
    assert re.search(r"(?<!:):(ticker|tier|start|end|run_id|expiry)\b", compiled.sql) is None


def test_latest_run_is_resolved_from_the_driving_table_not_scan_runs() -> None:
    for name, driver in _LATEST_RUN_DRIVERS.items():
        sql = JOIN_REGISTRY[name].sql
        picked = re.search(
            r"picked AS \(\s*SELECT COALESCE\(:run_id, max\(run_id\)\) AS run_id\s*"
            r"FROM uw_scan\.(\w+)",
            sql,
        )
        assert picked is not None and picked.group(1) == driver
        assert "scan_runs" not in sql
    panel = JOIN_REGISTRY["daily_signal_panel"].sql
    assert "s.basis='eod'" in panel
    assert "'mid'" not in panel


@pytest.mark.parametrize(
    ("name", "query"),
    [
        ("no_such_join", "ticker=SPY"),
        ("strike_grid", ""),
        ("strike_grid", "ticker=%20"),
        ("strike_grid", "ticker=SPY&order=strike"),
        ("strike_grid", "ticker=SPY&tier=ranked"),
        ("strike_grid", "ticker=SPY&run_id=1;DROP"),
        ("strike_grid", "ticker=SPY&run_id=0"),
        ("strike_grid", "ticker=SPY&start=2026-09-22&end=2026-09-01"),
        ("strike_grid", "ticker=SPY&expiry=2026-13-01"),
        ("strike_grid", "ticker=SPY&ticker=QQQ"),
        ("strike_grid", "ticker=SPY&limit=0"),
        ("strike_grid", "ticker=SPY&limit=-1"),
        ("strike_grid", "ticker=SPY&offset=-1"),
        ("strike_grid", "ticker=SPY&offset=2147483648"),
    ],
)
def test_join_query_rejects_bad_names_filters_and_paging(name: str, query: str) -> None:
    with pytest.raises(ApiError) as excinfo:
        build_join_query(name, QueryParams(query))
    assert excinfo.value.status_code == 400


def test_join_ticker_value_is_bound_not_interpolated() -> None:
    injection = "SPY'; DROP TABLE uw_scan.daily_ohlc;--"
    plan = build_join_query("daily_ohlc_technical", QueryParams({"ticker": injection}))
    assert injection not in plan.sql
    assert plan.params[0] == injection
    assert plan.sql == build_join_query("daily_ohlc_technical", QueryParams("ticker=SPY")).sql


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (asyncpg.QueryCanceledError("canceling statement"), 504, "query_timeout"),
        (TimeoutError(), 504, "query_timeout"),
        (asyncpg.DataError("invalid input"), 400, "invalid_parameter"),
        (asyncpg.PostgresConnectionError("secret host detail"), 503, "provider_not_configured"),
        (
            asyncpg.UndefinedTableError('relation "uw_scan.vrp_daily" missing'),
            400,
            "invalid_parameter",
        ),
        (asyncpg.UndefinedColumnError('column "rank" missing'), 400, "invalid_parameter"),
        (asyncpg.InsufficientPrivilegeError("permission denied"), 403, "forbidden"),
        (asyncpg.TooManyConnectionsError("too many clients"), 503, "provider_not_configured"),
        (asyncpg.InvalidPasswordError("password rejected"), 503, "provider_not_configured"),
    ],
)
async def test_join_driver_errors_map_to_typed_redacted_codes(
    error: Exception, status: int, code: str
) -> None:
    connection = _Connection([], error=error)
    async with AsyncClient(
        transport=ASGITransport(app=_app(connection), raise_app_exceptions=False),
        base_url="http://test",
        headers={"Authorization": "Bearer unit-test-token"},
    ) as client:
        response = await client.get("/v1/uw/daily_signal_panel?ticker=XOM")
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    for leaked in ("secret", "relation", "permission", "SELECT", "uw_scan"):
        assert leaked not in response.text


async def test_join_route_clamps_limit_and_validates_frozen_page_coverage() -> None:
    path = Path(__file__).resolve().parents[2] / "fixtures/pg_read_api/rows.jsonl"
    captured = [json.loads(line, parse_float=Decimal) for line in path.read_text().splitlines()]
    rows = next(r["rows"] for r in captured if r["query_id"] == "daily_signal_panel")
    connection = _Connection(rows)
    async with AsyncClient(
        transport=ASGITransport(app=_app(connection)),
        base_url="http://test",
        headers={"Authorization": "Bearer unit-test-token"},
    ) as client:
        response = await client.get("/v1/uw/daily_signal_panel?ticker=XOM&limit=99999")
    assert response.status_code == 200
    assert connection.statement.params[-2:] == (5001, 0)
    tables = response.json()["coverage"]["tables"]
    assert tables["iv_rank_history"] == {"matched": 0, "total": 2}
    assert tables["uw_gex_levels_daily"] == {"matched": 2, "total": 2}
    assert all(0 <= t["matched"] <= t["total"] == 2 for t in tables.values())
