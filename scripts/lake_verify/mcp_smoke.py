"""Call every MCP tool once over a real socket, the way a client does (plan PR2 step 5).

    python scripts/lake_verify/mcp_smoke.py --url http://127.0.0.1:8334/mcp --out RUN.json
    (key from APEX_MCP_API_KEY)

Arguments are discovered from the lake through the tools themselves (an index, a PIT
revision, a rates symbol), never invented. Writes one JSON record per call; exits 1
unless the tool set is exactly the 20 names and every call succeeds.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

EXPECTED = {
    "list_asset_classes", "search_instruments", "get_instrument", "get_coverage",
    "find_gaps", "get_lake_status", "get_bars", "get_bulk_bars", "get_rate_series",
    "list_futures_contracts", "get_corporate_actions", "get_delisting", "resolve_security",
    "list_indices", "get_index_members", "get_membership_history", "list_silver_revisions",
    "get_silver_revision", "list_pit_revisions", "get_pit_revision",
}  # fmt: skip


async def main(url: str, key: str, out: str) -> int:
    records: List[Dict[str, Any]] = []
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx2.AsyncClient(headers=headers, timeout=120) as http:
        async with Client(streamable_http_client(url, http_client=http)) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}

            async def call(name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
                t0 = time.perf_counter()
                result = await client.call_tool(name, args)
                ms = round((time.perf_counter() - t0) * 1000, 1)
                ok = not result.is_error
                body = result.structured_content if ok else result.content[0].text
                records.append({"tool": name, "args": args, "ok": ok, "ms": ms,
                                "summary": str(body)[:300]})  # fmt: skip
                return ok, body

            await call("list_asset_classes", {})
            await call("get_lake_status", {})
            await call("search_instruments", {"q": "SP", "asset_class": "equity", "limit": 5})
            await call("get_instrument", {"symbol": "SPY", "asset_class": "equity"})
            await call("get_coverage", {"symbol": "SPY", "limit": 5})
            await call("find_gaps", {"symbol": "SPY", "asset_class": "equity"})
            await call("get_bars", {"symbol": "SPY", "timeframe": "1d", "limit": 5})
            await call("get_bulk_bars", {"symbols": ["SPY", "QQQ"], "limit": 5})
            ok, rates = await call("search_instruments", {"asset_class": "rates", "limit": 1})
            if ok and rates["instruments"]:
                sym = rates["instruments"][0]["symbol"]
                await call("get_rate_series", {"symbol": sym, "limit": 5})
            await call("list_futures_contracts", {"root": "OJ", "limit": 5})
            await call("get_corporate_actions", {"symbol": "SPY", "limit": 5})
            await call("get_delisting", {"symbol": "SPY"})
            await call("resolve_security", {"symbol": "SPY"})
            ok, indices = await call("list_indices", {})
            if ok and indices["indices"]:
                await call("get_index_members", {"index_id": indices["indices"][0], "limit": 5})
            await call("get_membership_history", {"symbol": "SPY", "limit": 5})
            await call("list_silver_revisions", {"limit": 3})
            await call("get_silver_revision", {"limit": 3})
            ok, pits = await call("list_pit_revisions", {})
            if ok and pits["revisions"]:
                await call("get_pit_revision", {"revision": pits["revisions"][0]["revision"],
                                                "limit": 3})  # fmt: skip
    called = {r["tool"] for r in records}
    failed = [r for r in records if not r["ok"]]
    report = {"url_host": url.split("/")[2], "tool_names_exact": names == EXPECTED,
              "tools_listed": sorted(names), "tools_called": sorted(called),
              "uncalled": sorted(EXPECTED - called), "failed": len(failed), "calls": records}  # fmt: skip
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("tool_names_exact", "uncalled", "failed")}))
    for r in records:
        print(f"{'ok ' if r['ok'] else 'ERR'} {r['ms']:>8}ms {r['tool']} {r['summary'][:120]}")
    return 0 if report["tool_names_exact"] and not failed and called == EXPECTED else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.url, os.environ["APEX_MCP_API_KEY"], args.out)))
