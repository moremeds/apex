"""Shared MCP tool plumbing: read-only registration, stable errors, the response budget.

Every failure a client sees is an ``is_error`` result whose text is exactly the REST
error envelope, ``{"error": {"code", "message", "symbol"?, "asset_class"?, "details"?}}``
-- lake failures from ``lake_tool``, and the SDK's own (argument schema, unknown tool,
output conversion) from ``LakeMCPServer.call_tool``. Raising ``ToolError`` instead would
reach the client prefixed with "Error executing tool ...", which is not JSON.

No ``from __future__ import annotations`` here or in the tool modules: the SDK builds
each tool's input and output schema from its real annotations, and ``lake_tool``'s
wrapper would otherwise resolve string annotations against this module's globals.
"""

import asyncio
import functools
import json
import logging
import os
import uuid
from datetime import date
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult, InputRequiredResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, ValidationError

from src.application.lake.errors import LakeError, redact_paths
from src.infrastructure.adapters.livewire.parquet_reads import QueryTimeout

logger = logging.getLogger(__name__)

# Serialized result budget (design §3.2), in UTF-8 bytes of the whole wire result: over
# it, fail with instructions to narrow, never emit truncated JSON.
BUDGET_BYTES = 2 * 1024 * 1024

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

Fn = TypeVar("Fn", bound=Callable[..., Awaitable[Any]])

DEFAULT_CALL_TIMEOUT_SECONDS = 60.0


def call_timeout() -> float:
    """Whole-call deadline. The lake's per-query deadline bounds one parquet read; a
    call may make many (bulk, futures) or none (status, catalog), and mcp 2.2.0 does
    not cancel a stateless call its client abandoned -- so the server bounds it."""
    raw = os.environ.get("APEX_MCP_CALL_TIMEOUT_SECONDS", "").strip()
    return float(raw) if raw else DEFAULT_CALL_TIMEOUT_SECONDS


class Result(BaseModel):
    """Tool results declare their main fields and keep the payload's others."""

    model_config = ConfigDict(extra="allow")


def error_result(
    code: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
    *,
    symbol: Optional[str] = None,
    asset_class: Optional[str] = None,
) -> CallToolResult:
    """The REST error envelope (``api_error_response``) as an error result."""
    body: Dict[str, Any] = {"code": code, "message": redact_paths(message)}
    if symbol is not None:
        body["symbol"] = symbol
    if asset_class is not None:
        body["asset_class"] = asset_class
    if details:
        body["details"] = dict(details)
    text = json.dumps({"error": body}, default=str)
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def _internal(tool: str) -> CallToolResult:
    incident = uuid.uuid4().hex[:12]
    logger.exception("tool %s failed (incident %s)", tool, incident)
    return error_result("internal_error", f"internal error; see server logs (incident {incident})")


class LakeMCPServer(MCPServer):
    """Turns the SDK's own tool failures into the same envelope as lake failures."""

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Optional[Context[Any, Any]] = None,
    ) -> CallToolResult | InputRequiredResult:
        try:
            return await super().call_tool(name, arguments, context)
        except UnexpectedToolError:
            return _internal(name)
        except ToolError as exc:
            cause = exc.__cause__
            if isinstance(cause, ValidationError):
                # Field names and pydantic's reasons: REST's typed-validation class (422).
                problems = [
                    {
                        "field": ".".join(str(p) for p in err["loc"]),
                        "reason": err["msg"],
                    }
                    for err in cause.errors()
                ]
                return error_result(
                    "invalid_parameter",
                    f"invalid arguments for {name}",
                    {"source": "arguments", "problems": problems},
                )
            if name not in {tool.name for tool in await self.list_tools()}:
                return error_result(
                    "invalid_parameter", f"unknown tool {name!r}", {"source": "tool"}
                )
            return _internal(name)


def parse_day(value: Optional[str], name: str) -> Optional[date]:
    """Dates arrive as strings, as on REST, so a malformed one is the same stable
    ``invalid_parameter`` rather than an SDK schema error."""
    if value is None or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise LakeError(
            "invalid_parameter", f"malformed {name} {value!r}; expected YYYY-MM-DD"
        ) from exc


def lake_tool(server: MCPServer) -> Callable[[Fn], Fn]:
    """Register ``fn`` as a read-only tool whose lake failures are error results."""

    def register(fn: Fn) -> Fn:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            deadline = call_timeout()
            try:
                # Cancelling the call interrupts an in-flight LakeDb (bar/yield) read;
                # the other readers (catalog, membership, reference, manifests, repairs)
                # run in worker threads and finish on their own.
                result = await asyncio.wait_for(fn(*args, **kwargs), timeout=deadline)
            except TimeoutError:
                return error_result(
                    "query_timeout",
                    f"tool call exceeded {deadline:g}s; narrow the request",
                )
            except LakeError as exc:
                return error_result(
                    exc.code,
                    exc.message,
                    exc.details,
                    symbol=exc.symbol,
                    asset_class=exc.asset_class,
                )
            except QueryTimeout:
                return error_result("query_timeout", "lake read exceeded its deadline")
            except Exception:
                return _internal(fn.__name__)
            # Compact text beside the structured result (the SDK's own rendering is
            # indented JSON, 2-3x larger); the budget counts the whole wire result.
            call = CallToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result.model_dump(mode="json"),
            )
            size = len(call.model_dump_json(by_alias=True).encode("utf-8"))
            if size > BUDGET_BYTES:
                return error_result(
                    "result_too_large",
                    f"result is {size} bytes, over the {BUDGET_BYTES}-byte budget; "
                    "narrow the window, lower limit, or page with offset",
                )
            return call

        server.tool(annotations=READ_ONLY)(wrapper)
        return fn

    return register


def columnar(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Any]]:
    """Records -> columns + rows. Columns in first-seen order; a key a record omits
    (the payload drops null extra fields) is null in that row."""
    columns: List[str] = []
    for record in records:
        columns.extend(key for key in record if key not in columns)
    return {"columns": columns, "rows": [[r.get(c) for c in columns] for r in records]}
