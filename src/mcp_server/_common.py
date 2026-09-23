"""Shared MCP tool plumbing: read-only registration, stable errors, the response budget.

No ``from __future__ import annotations`` here or in the tool modules: the SDK builds
each tool's input and output schema from its real annotations, and ``lake_tool``'s
wrapper would otherwise resolve string annotations against this module's globals.
"""

import functools
import json
import logging
import uuid
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
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict

from src.application.lake.errors import LakeError, redact_paths
from src.infrastructure.adapters.livewire.parquet_reads import QueryTimeout

logger = logging.getLogger(__name__)

# Serialized result budget (design §3.2): over it, fail with instructions to narrow,
# never emit truncated JSON.
BUDGET_BYTES = 2 * 1024 * 1024

READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)

M = TypeVar("M", bound=BaseModel)
Fn = TypeVar("Fn", bound=Callable[..., Awaitable[Any]])


class Result(BaseModel):
    """Tool results declare their main fields and keep the payload's others."""

    model_config = ConfigDict(extra="allow")


def tool_error(code: str, message: str, details: Optional[Mapping[str, Any]] = None) -> ToolError:
    """The REST error envelope, as the tool error's text: same codes, same details."""
    body: Dict[str, Any] = {"code": code, "message": redact_paths(message)}
    if details:
        body["details"] = dict(details)
    return ToolError(json.dumps({"error": body}, default=str))


def lake_tool(server: MCPServer) -> Callable[[Fn], Fn]:
    """Register ``fn`` as a read-only tool, mapping lake failures to stable tool errors."""

    def register(fn: Fn) -> Fn:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = await fn(*args, **kwargs)
            except LakeError as exc:
                raise tool_error(exc.code, exc.message, exc.details) from exc
            except QueryTimeout as exc:
                raise tool_error("query_timeout", "lake read exceeded its deadline") from exc
            except ToolError:
                raise
            except Exception as exc:
                incident = uuid.uuid4().hex[:12]
                logger.exception("tool %s failed (incident %s)", fn.__name__, incident)
                raise tool_error(
                    "internal_error",
                    f"internal error; see server logs (incident {incident})",
                ) from exc
            size = len(result.model_dump_json())
            if size > BUDGET_BYTES:
                raise tool_error(
                    "result_too_large",
                    f"result is {size} bytes, over the {BUDGET_BYTES}-byte budget; "
                    "narrow the window, lower limit, or page with offset",
                )
            return result

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
