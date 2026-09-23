"""Read-only integration check; run on the database host with explicit read DSNs.

Usage: APEX_PG_READ_URLS=... APEX_PG_READ_TOKEN=... \
    uv run python scripts/check_pg_read_api.py --output result.json
No role creation, grants, writes, production server changes or external market calls.
The token is sent as the Bearer credential and is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from src.api.errors import install_error_handlers  # noqa: E402
from src.api.routes.db_catalog import router as catalog_router  # noqa: E402
from src.api.routes.db_table import router as table_router  # noqa: E402
from src.api.routes.uw_joins import router as joins_router  # noqa: E402
from src.api.uw_join_registry import JOIN_REGISTRY  # noqa: E402
from src.infrastructure.persistence.read_pools import (  # noqa: E402
    close_read_pools,
    create_read_pools,
)


async def check() -> dict:
    if not os.environ.get("APEX_PG_READ_URLS"):
        raise RuntimeError("APEX_PG_READ_URLS must explicitly name the verification databases")
    token = os.environ.get("APEX_PG_READ_TOKEN")
    if not token:
        raise RuntimeError("APEX_PG_READ_TOKEN must be set for the verification run")
    report: dict = {"captured_at": datetime.now(timezone.utc).isoformat(), "databases": {}}
    app = FastAPI()
    for router in (catalog_router, table_router, joins_router):
        app.include_router(router)
    install_error_handlers(app)
    app.state.pg_read_pools = {}
    auth = {"Authorization": f"Bearer {token}"}
    pools: dict = {}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://check") as client:
            # Auth boundary on all three namespaces BEFORE any source reads:
            # missing and wrong credentials must both 401 with the challenge.
            for path in (
                "/v1/db/catalog",
                "/v1/db/option_wizard/uw_scan/greeks_by_expiry_strike",
                "/v1/uw/strike_grid?ticker=SPY",
            ):
                missing = await client.get(path)
                assert missing.status_code == 401, (path, missing.text)
                assert missing.headers.get("www-authenticate") == "Bearer", path
                wrong = await client.get(path, headers={"Authorization": "Bearer wrong"})
                assert wrong.status_code == 401, (path, wrong.text)
            report["auth_boundary"] = True

        pools = await create_read_pools()
        app.state.pg_read_pools = pools
        for name, pool in pools.items():
            async with pool.acquire(timeout=30) as conn, conn.transaction(readonly=True):
                row = await conn.fetchrow(
                    "SELECT current_database() AS database, current_user AS role, "
                    "current_setting('transaction_read_only') AS readonly, "
                    "current_setting('statement_timeout') AS statement_timeout"
                )
                assert row["readonly"] == "on", row
                assert row["statement_timeout"] == "30s", row
                report["databases"][name] = dict(row)
        assert set(report["databases"]) == {"core", "option_chain", "option_wizard", "apex_signals"}
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://check", headers=auth
        ) as client:
            response = await client.get("/v1/db/catalog")
            assert response.status_code == 200, response.text
            catalog = response.json()
            report["catalog_tables"] = {
                db["name"]: sum(len(schema["tables"]) for schema in db["schemas"])
                for db in catalog["databases"]
            }
            response = await client.get(
                "/v1/db/option_wizard/uw_scan/greeks_by_expiry_strike",
                params={"columns": "ticker,run_id,strike", "where": "ticker:eq:SPY", "limit": 2},
            )
            assert response.status_code == 200, response.text
            report["table_page"] = response.json()
            assert response.json()["count"] == 2
            assert isinstance(response.json()["rows"][0][2], str)
            empty = await client.get(
                "/v1/db/option_wizard/uw_scan/greeks_by_expiry_strike",
                params={"where": "ticker:isnull:true", "limit": 2},
            )
            assert empty.status_code == 200, empty.text
            assert empty.json()["rows"] == [] and empty.json()["columns"]
            report["empty_result_preserves_columns"] = True
            for path, params in (
                ("/v1/db/option_wizard/uw_scan/raw_payloads", {}),
                (
                    "/v1/db/option_wizard/uw_scan/greeks_by_expiry_strike",
                    {"columns": "ticker;SELECT 1"},
                ),
                ("/v1/uw/strike_grid", {"ticker": "SPY", "run_id": "9223372036854775808"}),
            ):
                rejected = await client.get(path, params=params)
                assert rejected.status_code == 400, rejected.text
            report["invalid_requests_rejected"] = True
            samples = {
                "chain_exposure": "APP",
                "daily_ohlc_technical": "AMD",
                "fundamental_evidence_chain": "ZM",
                "daily_signal_panel": "XOM",
                "trade_insight_thread": "META",
            }
            report["joins"] = {}
            for name, spec in JOIN_REGISTRY.items():
                join_params: dict = {"limit": 2}
                if "ticker" in spec.filters:
                    join_params["ticker"] = samples.get(name, "SPY")
                result = await client.get(f"/v1/uw/{name}", params=join_params)
                assert result.status_code == 200, (name, result.text)
                page = result.json()
                assert page["count"] > 0, name
                assert page["coverage"]["scope"] == "returned_rows"
                assert all(
                    0 <= count["matched"] <= count["total"] == page["count"]
                    for count in page["coverage"]["tables"].values()
                ), name
                report["joins"][name] = page

            # macro_domain_states.as_of is timestamptz: the route must count the
            # same states as an independent UTC-date predicate, in any session zone.
            page = (
                await client.get(
                    "/v1/uw/macro_evidence_chain",
                    params={"start": "2026-09-21", "end": "2026-09-21", "limit": 5000},
                )
            ).json()
            assert not page["truncated"], "macro page truncated; raise limit or filter"
            state_col = next(
                i for i, column in enumerate(page["columns"]) if column["name"] == "state_id"
            )
            route_states = {row[state_col] for row in page["rows"]}
            boundary_sql = (
                "SELECT count(DISTINCT state_id) FROM uw_scan.macro_domain_states "
                "WHERE status='published' "
                "AND (as_of AT TIME ZONE 'UTC')::date BETWEEN '2026-09-21' AND '2026-09-21'"
            )
            zone_counts: dict[str, int] = {}
            async with pools["option_wizard"].acquire(timeout=30) as conn:
                for zone in ("UTC", "Asia/Hong_Kong"):
                    async with conn.transaction(readonly=True):
                        await conn.execute(f"SET LOCAL timezone = '{zone}'")
                        zone_counts[zone] = await conn.fetchval(boundary_sql)
            assert zone_counts["UTC"] == zone_counts["Asia/Hong_Kong"] == len(route_states), (
                len(route_states),
                zone_counts,
            )
            report["macro_utc_boundary"] = {
                "route_states": len(route_states),
                "direct_by_zone": zone_counts,
            }
            report["passed"] = True
    finally:
        await close_read_pools(pools)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(check())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: four catalogs, generic reads, rejection checks, {len(result['joins'])} joins")
