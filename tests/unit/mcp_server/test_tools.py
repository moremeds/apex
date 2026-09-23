"""The 20 MCP tools: registry shape and REST parity, over one real frozen lake.

Real frozen inputs, reused from the REST test suite rather than re-authored here:
SPY January 2025 daily bars and OJ futures contracts (``test_lake_routes``), FSLR/BIIB
Silver + PIT revisions (``tests.support.pit_manifest`` / ``silver_manifest``), TSLA's
real corporate-action splits (``test_reference_routes``), DGS10's real FRED yields
(``test_ohlc_provider``), and the real ``tests/fixtures/livewire_membership`` tree
(djia membership log, the one verified MUNJ security-master row). Nothing here is
invented: every value asserted is read off one of those existing fixtures.

The MCP client is opened with an ``async with`` inside each test rather than through a
``yield``-based pytest fixture: the streamable-HTTP session manager owns an anyio task
group, and its enter/exit must happen in the same asyncio Task -- a fixture that
``yield``s across pytest-asyncio's setup/teardown boundary does not guarantee that.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List

import httpx2
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from src.api.server import create_app
from src.application.lake.services import LakeServices
from src.infrastructure.adapters.livewire.coverage import CoverageCatalog
from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider
from src.infrastructure.adapters.livewire.pit_revisions import PitRevisionReader
from src.infrastructure.adapters.livewire.repairs import RepairsReader
from src.mcp_server import _common
from src.mcp_server.server import build_app, services_from_env
from tests.support.pit_manifest import (
    FSLR_ROWS,
    GENERATION_R76,
    pit_payload,
    publish_pit,
    write_daily,
)
from tests.support.silver_manifest import publish_manifest
from tests.unit.api.test_lake_routes import (
    _OJ,
    _OJ_COLUMNS,
    _SPY_JAN_2025,
    MEMBERSHIP_FIXTURE,
)
from tests.unit.api.test_lake_routes import _daily as _write_bronze_daily
from tests.unit.api.test_reference_routes import _TSLA, _write_actions
from tests.unit.infrastructure.livewire.test_ohlc_provider import _write_dgs10

API_KEY = "mcp-test-key"


class _ThrowawayLabel(_common.Result):
    """A throwaway tool's result shape for the UTF-8 budget test below -- not market
    data, module-scoped only so the SDK can resolve the annotation by name (a class
    defined inside a test function is not reachable from a nested function's
    ``__globals__``, which is the module, not the enclosing test's locals)."""

    label: str


EXPECTED_TOOL_NAMES = frozenset(
    {
        "list_asset_classes",
        "search_instruments",
        "get_instrument",
        "get_coverage",
        "find_gaps",
        "get_lake_status",
        "get_bars",
        "get_bulk_bars",
        "get_rate_series",
        "list_futures_contracts",
        "get_corporate_actions",
        "get_delisting",
        "resolve_security",
        "list_indices",
        "get_index_members",
        "get_membership_history",
        "list_silver_revisions",
        "get_silver_revision",
        "list_pit_revisions",
        "get_pit_revision",
    }
)


# -- fixtures: one real lake, shared by REST and MCP --------------------------------


@pytest.fixture
def lake(tmp_path: Path, catalog_db: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _write_bronze_daily(
        bronze,
        "equity",
        "SPY",
        pd.DataFrame(
            _SPY_JAN_2025,
            columns=["trade_date", "open", "high", "low", "close", "volume"],
        ),
    )
    for contract, rows in _OJ.items():
        _write_bronze_daily(bronze, "futures", contract, pd.DataFrame(rows, columns=_OJ_COLUMNS))
    _write_dgs10(bronze)
    _write_actions(bronze, "TSLA", _TSLA)
    r76 = silver / write_daily(silver, "FSLR", FSLR_ROWS[:4], GENERATION_R76, 75)
    publish_manifest(silver, 76, [r76])
    r77 = silver / write_daily(silver, "FSLR", FSLR_ROWS)
    publish_manifest(silver, 77, [r77])
    publish_pit(silver, pit_payload(silver, 1, index_id="sp500"))
    publish_pit(silver, pit_payload(silver, 2, index_id="ndx100", members=[]))

    monkeypatch.setenv("APEX_LIVEWIRE_ROOT", str(bronze))
    monkeypatch.setenv("APEX_LIVEWIRE_SILVER_ROOT", str(silver))
    monkeypatch.setenv("APEX_LIVEWIRE_PRICE_MODE", "adjusted")
    monkeypatch.setenv("APEX_LIVEWIRE_COVERAGE_DB", str(catalog_db))
    monkeypatch.setenv("APEX_LIVEWIRE_LAKE_ROOT", str(MEMBERSHIP_FIXTURE))
    monkeypatch.delenv("APEX_LIVEWIRE_DELISTED_ROOT", raising=False)

    app = create_app()
    app.state.ohlc_provider = LivewireOhlcProvider(bronze, silver, "adjusted")
    app.state.coverage_catalog = CoverageCatalog(catalog_db)
    app.state.pit_reader = PitRevisionReader(silver)
    app.state.repairs_reader = RepairsReader(None)

    return {
        "bronze": bronze,
        "silver": silver,
        "app": app,
        "services": services_from_env(),
    }


@pytest.fixture
def client(lake: Dict[str, Any]) -> Iterator[TestClient]:
    # No `with`: the production lifespan (PG, xenon, subscriptions) is not this surface.
    yield TestClient(lake["app"])


@asynccontextmanager
async def mcp_session(services: LakeServices) -> AsyncIterator[Client]:
    """Open one MCP server + client, entered and exited in the caller's own task."""
    app = build_app(services, API_KEY, ["testserver"])
    async with app._app.router.lifespan_context(app._app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"authorization": f"Bearer {API_KEY}"},
        ) as http_client:
            async with Client(
                streamable_http_client("http://testserver/mcp", http_client=http_client)
            ) as client:
                yield client


# -- projection helpers ---------------------------------------------------------


def _drop_gen(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "generated_at"}


def _records_from_columns(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = payload["columns"]
    assert len(columns) == len(set(columns)), f"duplicate columns: {columns}"
    return [dict(zip(columns, row, strict=True)) for row in payload["rows"]]


def _project_series(payload: Dict[str, Any], list_key: str) -> Dict[str, Any]:
    """A REST series payload (a list under ``list_key``) and an MCP columns/rows
    payload become the same shape: ``generated_at`` dropped, the series as ``records``.
    """
    payload = _drop_gen(payload)
    if "columns" in payload and "rows" in payload:
        records = _records_from_columns(payload)
        payload = {k: v for k, v in payload.items() if k not in ("columns", "rows")}
    else:
        records = payload[list_key]
        payload = {k: v for k, v in payload.items() if k != list_key}
    payload["records"] = records
    return payload


def _project_bulk(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = _drop_gen(payload)
    payload["symbols"] = {
        symbol: _project_series(series, "bars") for symbol, series in payload["symbols"].items()
    }
    return payload


def _mcp_error(text: str) -> Dict[str, Any]:
    """The tool error's text IS the REST envelope JSON -- no "Error executing tool"
    prefix to strip -- so this is just its ``error`` object."""
    return json.loads(text)["error"]


# -- registry shape ---------------------------------------------------------------


async def test_registered_tools_match_the_20_tool_contract(
    lake: Dict[str, Any],
) -> None:
    async with mcp_session(lake["services"]) as mcp:
        tools = (await mcp.list_tools()).tools
    assert {t.name for t in tools} == EXPECTED_TOOL_NAMES
    for tool in tools:
        assert tool.annotations is not None, tool.name
        assert tool.annotations.read_only_hint is True, tool.name
        assert tool.annotations.idempotent_hint is True, tool.name
        assert tool.annotations.destructive_hint is False, tool.name
        assert tool.output_schema is not None, tool.name
        assert tool.output_schema.get("type") == "object", tool.name


# -- get_bars -----------------------------------------------------------------------


async def test_get_bars_matches_rest_over_an_explicit_window(
    client: TestClient, lake: Dict[str, Any]
) -> None:
    params = {
        "start": "2025-01-07T00:00:00Z",
        "end": "2025-01-15T23:59:59Z",
        "price_mode": "raw",
    }
    rest = client.get("/v1/equity/SPY/bars", params=params).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_bars", {"symbol": "SPY", "limit": 10, **params})
    assert not result.is_error, result.content
    assert _project_series(rest, "bars") == _project_series(result.structured_content, "bars")
    # Confirms the window was not empty, so the equality above is not vacuous.
    assert len(rest["bars"]) == 5


async def test_get_bars_bounded_limit_truncates_even_over_an_explicit_window(
    lake: Dict[str, Any],
) -> None:
    """Unlike REST legacy (previous test: an explicit start ignores ``limit``
    entirely), MCP's bounded policy applies ``limit`` regardless of an explicit
    window, so a limit under the 5 real rows truncates."""
    params = {
        "start": "2025-01-07T00:00:00Z",
        "end": "2025-01-15T23:59:59Z",
        "price_mode": "raw",
        "limit": 2,
    }
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_bars", {"symbol": "SPY", **params})
    assert not result.is_error, result.content
    body = result.structured_content
    assert body["truncated"] is True
    assert len(body["rows"]) == 2
    assert _records_from_columns(body)[-1]["close"] == 592.78  # 2025-01-15, the real last row


async def test_get_bars_pit_revision_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    params: Dict[str, Any] = {"pit_revision": 1, "start": "2026-09-15T00:00:00Z"}
    rest = client.get("/v1/equity/FSLR/bars", params=params).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_bars", {"symbol": "FSLR", **params})
    assert not result.is_error, result.content
    assert _project_series(rest, "bars") == _project_series(result.structured_content, "bars")
    assert [r["close"] for r in rest["bars"]] == [
        202.34,
        191.07,
        201.16,
        195.96,
        199.84,
    ]


# -- get_bulk_bars --------------------------------------------------------------------


async def test_get_bulk_bars_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    params = {
        "symbols": "SPY",
        "start": "2025-01-07T00:00:00Z",
        "end": "2025-01-15T23:59:59Z",
        "price_mode": "raw",
        "limit": 10,
    }
    rest = client.get("/v1/equity/bars", params=params).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool(
            "get_bulk_bars",
            {
                "symbols": ["SPY"],
                "start": params["start"],
                "end": params["end"],
                "price_mode": "raw",
                "limit": 10,
            },
        )
    assert not result.is_error, result.content
    assert _project_bulk(rest) == _project_bulk(result.structured_content)
    assert len(rest["symbols"]["SPY"]["bars"]) == 5


# -- get_rate_series --------------------------------------------------------------


async def test_get_rate_series_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    params = {
        "start": "2026-08-18T00:00:00Z",
        "end": "2026-08-20T23:59:59Z",
        "limit": 10,
    }
    rest = client.get("/v1/rates/DGS10/series", params=params).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_rate_series", {"symbol": "DGS10", **params})
    assert not result.is_error, result.content
    assert _project_series(rest, "points") == _project_series(result.structured_content, "points")
    assert [r["yield_pct"] for r in rest["points"]] == [4.71, 4.65, 4.69]


# -- discovery ------------------------------------------------------------------


async def test_get_instrument_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/equity/SPY").json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_instrument", {"symbol": "SPY", "asset_class": "equity"})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)


async def test_get_coverage_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/lake/coverage", params={"limit": 2, "offset": 0}).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_coverage", {"limit": 2, "offset": 0})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["truncated"] is True


async def test_find_gaps_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    params = {"start": "2025-01-06", "end": "2025-01-17"}
    rest = client.get("/v1/equity/SPY/gaps", params=params).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("find_gaps", {"symbol": "SPY", **params})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["gaps"] == [{"start": "2025-01-14", "end": "2025-01-14", "sessions": 1}]


async def test_list_futures_contracts_matches_rest(
    client: TestClient, lake: Dict[str, Any]
) -> None:
    rest = client.get("/v1/futures/oj/contracts", params={"limit": 1}).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("list_futures_contracts", {"root": "oj", "limit": 1})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["truncated"] is True


# -- identity / membership -------------------------------------------------------


async def test_resolve_security_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/security/MUNJ", params={"as_of": "2026-09-14"}).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("resolve_security", {"symbol": "MUNJ", "as_of": "2026-09-14"})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["security_id"] == "sec_405d12b544ef24fee4a9ef06b721d90e"


async def test_list_indices_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/membership/indices", params={"limit": 10, "offset": 0}).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("list_indices", {"limit": 10, "offset": 0})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["indices"] == ["djia"]


async def test_get_index_members_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    params = {
        "as_of": "2026-09-14",
        "include_candidates": "true",
        "limit": 10,
        "offset": 0,
    }
    rest = client.get("/v1/membership/djia", params=params).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool(
            "get_index_members",
            {
                "index_id": "djia",
                "as_of": "2026-09-14",
                "include_candidates": True,
                "limit": 10,
                "offset": 0,
            },
        )
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["total"] == 30


async def test_get_membership_history_matches_rest(
    client: TestClient, lake: Dict[str, Any]
) -> None:
    params = {"symbol": "aa", "as_of": "2026-09-14", "limit": 10, "offset": 0}
    rest = client.get("/v1/membership/history", params=params).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_membership_history", params)
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["security_id"] == "unresolved:AA"


async def test_get_corporate_actions_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    params = {"limit": 10, "offset": 0}
    rest = client.get("/v1/equity/TSLA/actions", params=params).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_corporate_actions", {"symbol": "TSLA", **params})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert [a["ex_date"] for a in rest["actions"]] == ["2020-08-31", "2022-08-25"]


async def test_get_delisting_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    params = {"limit": 10, "offset": 0}
    rest = client.get("/v1/equity/MUNJ/delisting", params=params).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_delisting", {"symbol": "MUNJ", **params})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["symbol"] == "MUNJ"


# -- revisions ----------------------------------------------------------------------


async def test_list_silver_revisions_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/lake/silver-revisions").json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("list_silver_revisions", {})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["current"] == 77


async def test_get_silver_revision_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/lake/silver-revisions/76").json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_silver_revision", {"revision": 76})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["affected"][0]["symbol"] == "FSLR"


async def test_list_pit_revisions_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/lake/pit-revisions", params={"index_id": "sp500"}).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("list_pit_revisions", {"index_id": "sp500"})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert [r["revision"] for r in rest["revisions"]] == [1]


async def test_get_pit_revision_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/lake/pit-revisions/1", params={"limit": 1}).json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_pit_revision", {"revision": 1, "limit": 1})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)
    assert rest["publisher_status"] == "PARTIAL" and rest["truncated"] is True


# -- errors -------------------------------------------------------------------------


async def test_unknown_symbol_error_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/equity/NOPE/gaps").json()
    assert rest["error"]["code"] == "unknown_symbol"
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("find_gaps", {"symbol": "NOPE"})
    assert result.is_error
    # Full envelope equality (code, message, symbol, asset_class, details) -- both
    # sides build the same LakeError through the shared application-layer function.
    assert _mcp_error(result.content[0].text) == rest["error"]


async def test_bad_revision_error_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/lake/pit-revisions/99").json()
    assert rest["error"]["code"] == "unknown_revision"
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_pit_revision", {"revision": 99})
    assert result.is_error
    assert _mcp_error(result.content[0].text) == rest["error"]


async def test_malformed_as_of_is_invalid_parameter_on_both(
    client: TestClient, lake: Dict[str, Any]
) -> None:
    rest = client.get("/v1/security/MUNJ", params={"as_of": "14-09-2026"}).json()
    assert rest["error"]["code"] == "invalid_parameter"
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("resolve_security", {"symbol": "MUNJ", "as_of": "14-09-2026"})
    assert result.is_error
    assert _mcp_error(result.content[0].text) == rest["error"]


async def test_unknown_listing_is_invalid_parameter_on_both(
    client: TestClient, lake: Dict[str, Any]
) -> None:
    rest = client.get("/v1/equity/SPY/bars", params={"listing": "maybe"}).json()
    assert rest["error"]["code"] == "invalid_parameter"
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_bars", {"symbol": "SPY", "listing": "maybe"})
    assert result.is_error
    assert _mcp_error(result.content[0].text) == rest["error"]


async def test_malformed_bars_start_is_an_argument_schema_error(
    lake: Dict[str, Any],
) -> None:
    """A value the SDK's own input schema rejects (an unparseable datetime) never
    reaches ``query_bars``/``parse_day`` at all -- it fails as a pydantic
    ValidationError at the tool-call boundary, REST's typed-validation class (422)."""
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_bars", {"symbol": "SPY", "start": "yesterday"})
    assert result.is_error
    error = _mcp_error(result.content[0].text)
    assert error["code"] == "invalid_parameter"
    assert error["details"]["source"] == "arguments"
    assert error["details"]["problems"], error["details"]


async def test_list_tool_limit_zero_is_an_argument_schema_error(
    lake: Dict[str, Any],
) -> None:
    """``limit`` is schema-bounded (ge=1) now, so 0 never reaches the tool body."""
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("list_silver_revisions", {"limit": 0})
    assert result.is_error
    error = _mcp_error(result.content[0].text)
    assert error["code"] == "invalid_parameter"
    assert error["details"]["source"] == "arguments"


async def test_unknown_tool_is_invalid_parameter(lake: Dict[str, Any]) -> None:
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_nonexistent_tool", {})
    assert result.is_error
    error = _mcp_error(result.content[0].text)
    assert error["code"] == "invalid_parameter"
    assert error["details"]["source"] == "tool"


async def test_result_too_large(lake: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "BUDGET_BYTES", 10)
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_bars", {"symbol": "SPY", "price_mode": "raw"})
    assert result.is_error
    assert _mcp_error(result.content[0].text)["code"] == "result_too_large"


async def test_result_too_large_counts_utf8_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The budget counts UTF-8 bytes, not characters, so a result built entirely from
    multi-byte characters can cross it while its character count would not. Not
    market data: a throwaway tool on its own server, exercising only the budget."""
    from mcp.server.transport_security import TransportSecuritySettings

    from src.mcp_server import server as server_mod
    from src.mcp_server._common import LakeMCPServer, lake_tool

    label = "肥皂" * 20  # 40 non-ASCII characters, 3 UTF-8 bytes each

    probe = _ThrowawayLabel(label=label)
    wire = CallToolResult(
        content=[TextContent(type="text", text=probe.model_dump_json())],
        structured_content=probe.model_dump(mode="json"),
    ).model_dump_json(by_alias=True)
    char_len, byte_len = len(wire), len(wire.encode("utf-8"))
    assert byte_len > char_len, "the probe must cost more bytes than characters"
    budget = (char_len + byte_len) // 2
    assert char_len < budget < byte_len

    throwaway = LakeMCPServer("throwaway")
    tool = lake_tool(throwaway)

    @tool
    async def get_label() -> _ThrowawayLabel:
        return _ThrowawayLabel(label=label)

    monkeypatch.setattr(_common, "BUDGET_BYTES", budget)
    app = server_mod.BearerAuth(
        throwaway.streamable_http_app(
            streamable_http_path=server_mod.MCP_PATH,
            json_response=True,
            stateless_http=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=["testserver"],
                allowed_origins=["http://testserver", "https://testserver"],
            ),
        ),
        API_KEY,
    )
    async with app._app.router.lifespan_context(app._app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"authorization": f"Bearer {API_KEY}"},
        ) as http_client:
            async with Client(
                streamable_http_client("http://testserver/mcp", http_client=http_client)
            ) as client:
                result = await client.call_tool("get_label", {})
    assert result.is_error, result.structured_content
    assert _mcp_error(result.content[0].text)["code"] == "result_too_large"


# -- more REST parity: no-argument tools, and a documented default-limit difference --


async def test_list_asset_classes_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/lake/asset-classes").json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("list_asset_classes", {})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)


async def test_get_lake_status_matches_rest(client: TestClient, lake: Dict[str, Any]) -> None:
    rest = client.get("/v1/lake/status").json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("get_lake_status", {})
    assert not result.is_error, result.content
    assert _drop_gen(rest) == _drop_gen(result.structured_content)


async def test_search_instruments_default_limit_differs_from_rest(
    client: TestClient, lake: Dict[str, Any]
) -> None:
    """Documented, deliberate difference: with no ``limit``, REST's
    ``/v1/instruments`` defaults to 500 rows (``src/api/routes/instruments.py``,
    ``Query(default=500, ...)``), while MCP's ``search_instruments`` defaults to the
    shared lake page default, ``src.application.lake.services.PAGE_DEFAULT`` (100).
    No fixture in this repo has >100 real coverage rows to make the cap bind
    observably (this one has 5), so the two defaults are asserted directly."""
    from src.application.lake.services import PAGE_DEFAULT

    rest_default_limit = 500  # src/api/routes/instruments.py: Query(default=500, ...)
    assert PAGE_DEFAULT == 100
    assert PAGE_DEFAULT != rest_default_limit

    rest = client.get("/v1/instruments").json()
    async with mcp_session(lake["services"]) as mcp:
        result = await mcp.call_tool("search_instruments", {})
    assert not result.is_error, result.content
    mcp_body = result.structured_content
    # Neither cap binds on this small fixture, so the bodies still agree here --
    # the difference above is in the (uncapped-here) default, not in these 5 rows.
    assert rest["count"] == mcp_body["count"]
    assert rest["count"] < min(PAGE_DEFAULT, rest_default_limit)
