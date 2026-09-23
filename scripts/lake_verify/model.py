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
