"""Shared lossless JSON representation for PostgreSQL read results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from src.api.payload.validate import validate_payload


def json_value(value: Any) -> Any:
    """Preserve decimal precision and make non-finite database values valid JSON."""
    if isinstance(value, (Decimal, UUID)):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, bytes):
        return "\\x" + value.hex()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_tabular(
    database: str,
    schema: str,
    table: str,
    columns: list[dict[str, str]],
    rows: Sequence[Any],
    limit: int,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a page; callers fetch one extra row for truncation."""
    names = [column["name"] for column in columns]
    payload = {
        "database": database,
        "schema": schema,
        "table": table,
        "columns": columns,
        "rows": [[json_value(row[name]) for name in names] for row in rows[:limit]],
        "count": min(len(rows), limit),
        "truncated": len(rows) > limit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if coverage is not None:
        payload["coverage"] = coverage
    validate_payload(payload, "tabular_payload")
    return payload
