"""Pure-function tests for ``scripts/lake_verify/mcp_exec.py::_error``.

These exercise the tool-error-envelope -> REST-status mapping directly, with hand-built
JSON envelopes (status records, not market data -- no fixture needed).
"""

from __future__ import annotations

import json

import mcp_exec  # type: ignore[import-not-found]  # sys.path set by conftest.py
import pytest


def _envelope(code: str, details: dict | None = None) -> str:
    error = {"code": code, "message": "x"}
    if details is not None:
        error["details"] = details
    return json.dumps({"error": error})


def test_no_envelope_raises() -> None:
    with pytest.raises(RuntimeError, match="no REST envelope"):
        mcp_exec._error("not json at all")


def test_argument_source_maps_to_422() -> None:
    status, body = mcp_exec._error(_envelope("invalid_parameter", {"source": "arguments"}))
    assert status == 422
    assert body["error"]["code"] == "invalid_parameter"


def test_tool_source_raises() -> None:
    """An unknown tool name is a harness/mapping bug, never a REST-equivalent
    rejection: it must never quietly satisfy an expected-rejection case."""
    with pytest.raises(RuntimeError, match="unknown tool"):
        mcp_exec._error(_envelope("invalid_parameter", {"source": "tool"}))


def test_result_too_large_maps_to_400() -> None:
    """Not an ``ApiErrorCode`` at all -- MCP-only, so it cannot come from
    ``STATUS_BY_CODE``."""
    status, body = mcp_exec._error(_envelope("result_too_large"))
    assert status == 400
    assert body["error"]["code"] == "result_too_large"


def test_ordinary_code_maps_through_status_by_code() -> None:
    status, body = mcp_exec._error(_envelope("unknown_symbol"))
    assert status == 404
    assert body["error"]["code"] == "unknown_symbol"
