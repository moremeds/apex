"""Case/outcome records shared by the matrix runner and the operation modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Case:
    id: str
    operation: str
    dims: Dict[str, Any]
    # {"transport": "http", "process": "raw"|"adjusted", "path": str, "params": {...}}
    # or {"transport": "inproc", "process": ..., "call": str, "kwargs": {...}}
    request: Dict[str, Any]
    # {"kind": "rejection", "status": int, "code": str}
    # {"kind": "value", "check": str, "args": {...}}   -- the oracle decides at run time
    # {"kind": "blocked_data" | "blocked_dependency", "reason": str}
    expect: Dict[str, Any]


@dataclass
class Outcome:
    status: str
    detail: str = ""
    facts: Dict[str, Any] = field(default_factory=dict)


class Invalidated(RuntimeError):
    """A mutable source changed between the candidate read and the oracle read."""


class NotApplicable(Exception):
    """mcp target: the cell has no MCP twin by design (REST-only); the reason is recorded."""


# The one remaining NotApplicable reason mcp_exec.translate() may raise (design §3.2:
# legacy REST output policy is a deliberate compatibility difference from MCP's
# bounded policy; the bounded twin runs as an inproc cell instead). Lives here, not in
# mcp_exec.py, so matrix.py's summarize() gate can check every NOT_APPLICABLE record
# against it without importing the candidate-side (starlette/mcp_server) module.
NOT_APPLICABLE_LEGACY_SERIES_REASON = (
    "legacy-policy series path (design §3.2): legacy REST is a deliberate compatibility "
    "difference from MCP's bounded policy; the bounded twin runs as an inproc cell"
)
