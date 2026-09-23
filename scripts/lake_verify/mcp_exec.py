"""Candidate-side MCP target for the matrix (plan PR2 step 5): every case goes through
the real MCP app -- bearer auth, Host checks, Streamable HTTP JSON-RPC -- as a tool
call, and the result is mapped back to the REST response shape so the same oracle
checkers settle it.

Mapping back is mechanical and lossless: columns + rows become the REST records, an
error's JSON envelope becomes its REST status. Where REST and MCP differ by design
the case is NOT_APPLICABLE, never silently passed:
- legacy-policy series cells (REST only; their bounded twins are the ``inproc`` cells);
- instruments with ``listing`` != listed, or a limit MCP's page cap (2000) refuses.
Unpaged legacy list routes are answered by following ``next_offset`` to the end.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import unquote

from model import NotApplicable
from src.api.errors import STATUS_BY_CODE, ApiErrorCode

API_KEY = "matrix-local-key"  # in-process only; never leaves this process
HOST = "testserver"
PAGE_MAX = 2000
_LEGACY_PAGED = {"get_corporate_actions", "get_delisting", "list_indices",
                 "get_membership_history", "get_index_members"}  # fmt: skip
_PAGE_KEYS = ("limit", "offset", "returned", "truncated", "next_offset", "total")


_ENV_LOCK = threading.Lock()


def _int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value  # sent as-is: the tool's input schema rejects it, as REST's 422 does


def _bool(value: Any) -> Any:
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _pick(params: Dict[str, Any], **rename: str) -> Dict[str, Any]:
    """REST query params -> tool arguments (``rename`` maps tool name -> REST name)."""
    out = {}
    for tool_name, rest_name in rename.items():
        if rest_name in params and params[rest_name] is not None:
            out[tool_name] = params[rest_name]
    for key in ("limit", "offset", "max_gaps", "revision"):
        if key in out:
            out[key] = _int(out[key])
    for key in ("include_silver", "include_candidates"):
        if key in out:
            out[key] = _bool(out[key])
    return out


_ROUTES: List[Tuple[str, str]] = [
    (r"/v1/lake/asset-classes", "list_asset_classes"),
    (r"/v1/lake/status", "get_lake_status"),
    (r"/v1/lake/coverage", "get_coverage"),
    (r"/v1/lake/silver-revisions", "list_silver_revisions"),
    (r"/v1/lake/silver-revisions/(?P<revision>[^/]+)", "get_silver_revision"),
    (r"/v1/lake/pit-revisions", "list_pit_revisions"),
    (r"/v1/lake/pit-revisions/(?P<revision>[^/]+)", "get_pit_revision"),
    (r"/v1/security/(?P<symbol>[^/]+)", "resolve_security"),
    (r"/v1/futures/(?P<root>[^/]+)/contracts", "list_futures_contracts"),
    (r"/v1/instruments", "search_instruments"),
    (r"/v1/membership/indices", "list_indices"),
    (r"/v1/membership/history", "get_membership_history"),
    (r"/v1/membership/(?P<index_id>[^/]+)", "get_index_members"),
    (r"/v1/equity/bars", "_legacy_series"),
    (r"/v1/rates/(?P<symbol>[^/]+)/series", "_legacy_series"),
    (r"/v1/equity/(?P<symbol>[^/]+)/actions", "get_corporate_actions"),
    (r"/v1/equity/(?P<symbol>[^/]+)/delisting", "get_delisting"),
    (r"/v1/(?P<asset_class>[^/]+)/(?P<symbol>[^/]+)/gaps", "find_gaps"),
    (r"/v1/(?P<asset_class>[^/]+)/(?P<symbol>[^/]+)/bars", "_legacy_series"),
    (r"/v1/(?P<asset_class>[^/]+)/(?P<symbol>[^/]+)", "get_instrument"),
]

_ARGS = {
    "get_coverage": dict(symbol="symbol", asset_class="asset_class",
                         include_silver="include_silver", limit="limit", offset="offset"),
    "list_silver_revisions": dict(limit="limit", offset="offset"),
    "get_silver_revision": dict(limit="limit", offset="offset"),
    "list_pit_revisions": dict(index_id="index_id", limit="limit", offset="offset"),
    "get_pit_revision": dict(limit="limit", offset="offset"),
    "resolve_security": dict(as_of="as_of", known_at="known_at"),
    "list_futures_contracts": dict(limit="limit", offset="offset"),
    "search_instruments": dict(q="q", asset_class="asset_class", limit="limit"),
    "list_indices": dict(limit="limit", offset="offset"),
    "get_membership_history": dict(symbol="symbol", index_id="index_id", as_of="as_of",
                                   limit="limit", offset="offset"),
    "get_index_members": dict(as_of="as_of", known_at="known_at",
                              include_candidates="include_candidates",
                              limit="limit", offset="offset"),
    "get_corporate_actions": dict(action_type="type", start="start", end="end",
                                  limit="limit", offset="offset"),
    "get_delisting": dict(limit="limit", offset="offset"),
    "find_gaps": dict(timeframe="timeframe", start="start", end="end", listing="listing",
                      max_gaps="max_gaps", calendar="calendar"),
    "list_asset_classes": {},
    "get_lake_status": {},
    "get_instrument": {},
}  # fmt: skip


def translate(request: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """A matrix request -> (tool, arguments)."""
    if request["transport"] == "inproc":
        kwargs = dict(request["kwargs"])
        if "silver_revision_pin" in kwargs:
            kwargs["silver_revision"] = kwargs.pop("silver_revision_pin")
        tool = {"bars": "get_bars", "bulk": "get_bulk_bars", "rates": "get_rate_series"}
        return tool[request["call"]], {k: v for k, v in kwargs.items() if v is not None}
    path, params = request["path"], dict(request.get("params") or {})
    for pattern, tool in _ROUTES:
        match = re.fullmatch(pattern, path)
        if match is None:
            continue
        if tool == "_legacy_series":
            raise NotApplicable("legacy REST output policy; MCP serves the bounded twin")
        args = _pick(params, **_ARGS[tool])
        args.update({k: _int(unquote(v)) if k == "revision" else unquote(v)
                     for k, v in match.groupdict().items()})  # fmt: skip
        if tool == "search_instruments":
            if params.get("listing", "listed") != "listed":
                raise NotApplicable("instruments listing != listed is REST-only (501)")
            args.setdefault("limit", 500)  # REST's default page
            limit = args["limit"]
            if isinstance(limit, int) and PAGE_MAX < limit <= 5000:
                # REST serves 2001..5000; MCP's page cap refuses it: a real difference.
                raise NotApplicable(f"instruments limit {limit} is REST-only (MCP max 2000)")
        return tool, args
    raise NotApplicable(f"no MCP twin for {path}")


_CORE = frozenset(("time", "open", "high", "low", "close", "volume", "yield_pct"))


def _records(block: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Rows -> REST records. REST omits a null per-class extra field rather than
    emitting it; a column holds null there, so null extras are dropped again."""
    out = {k: v for k, v in block.items() if k not in ("columns", "rows")}
    out[key] = [
        {c: v for c, v in zip(block["columns"], row) if v is not None or c in _CORE}
        for row in block["rows"]
    ]
    return out


def to_rest(tool: str, content: Dict[str, Any]) -> Dict[str, Any]:
    if tool == "get_bars":
        return _records(content, "bars")
    if tool == "get_rate_series":
        return _records(content, "points")
    if tool == "get_bulk_bars":
        return {
            **content,
            "symbols": {s: _records(b, "bars") for s, b in content["symbols"].items()},
        }
    return content


class McpTarget:
    """One in-process MCP app per price-mode process, reached through its HTTP app."""

    def __init__(self, process: str) -> None:
        from starlette.testclient import TestClient

        from src.mcp_server.server import build_app, services_from_env

        with _ENV_LOCK:  # services_from_env reads the price mode from the environment
            os.environ["APEX_LIVEWIRE_PRICE_MODE"] = process
            services = services_from_env()
        app = build_app(services, API_KEY, [HOST])
        self._client = TestClient(app, base_url=f"http://{HOST}")
        self._client.__enter__()  # runs the SDK session-manager lifespan
        self._id = 0

    def call(self, tool: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        self._id += 1
        response = self._client.post(
            "/mcp",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": self._id,
                "method": "tools/call",
                "params": {"name": tool, "arguments": args},
            },  # fmt: skip
        )
        body = response.json()
        if response.status_code != 200 or "error" in body:
            raise RuntimeError(f"MCP transport {response.status_code}: {str(body)[:300]}")
        result = body["result"]
        if result.get("isError"):
            return False, result["content"][0]["text"]
        return True, result["structuredContent"]

    def execute(self, request: Dict[str, Any]) -> Tuple[int, Any]:
        tool, args = translate(request)
        paged = tool in _LEGACY_PAGED and "limit" not in args and "offset" not in args
        if paged:
            args = {**args, "limit": PAGE_MAX, "offset": 0}
        ok, content = self.call(tool, args)
        if not ok:
            return _error(content)
        if not paged:
            return 200, to_rest(tool, content)
        return 200, self._drain(tool, args, content)

    def _drain(self, tool: str, args: Dict[str, Any], first: Dict[str, Any]) -> Dict[str, Any]:
        """Follow next_offset to the end, then drop the page fields: REST's unpaged body."""
        key = {"get_corporate_actions": "actions", "get_delisting": "intervals",
               "list_indices": "indices", "get_membership_history": "events",
               "get_index_members": "members"}[tool]  # fmt: skip
        merged = dict(first)
        page = first
        while page.get("next_offset") is not None:
            ok, page = self.call(tool, {**args, "offset": page["next_offset"]})
            if not ok:
                raise RuntimeError(f"page after offset failed: {page}")
            merged[key] = merged[key] + page[key]
        if "count" in merged:
            merged["count"] = len(merged[key])
        return {k: v for k, v in merged.items() if k not in _PAGE_KEYS}


_PREFIX = re.compile(r"^Error executing tool \w+: ")


def _error(text: str) -> Tuple[int, Any]:
    """The tool error's REST envelope -> its REST status. A failure of the tool's input
    schema (no envelope) is REST's typed-validation 422 invalid_parameter."""
    try:
        envelope = json.loads(_PREFIX.sub("", text))
        code = envelope["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return 422, {"error": {"code": "invalid_parameter", "message": text[:300]}}
    if code == "result_too_large":  # MCP-only: the serialized-result budget
        return 400, envelope
    return STATUS_BY_CODE[ApiErrorCode(code)], envelope
