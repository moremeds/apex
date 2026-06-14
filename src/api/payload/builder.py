"""Build signal_service_payload dicts from ta_signals rows.

Maps the DB column `time` -> schema field `timestamp`; drops DB-only columns
(`created_at`). Keys not in the schema's signal object are simply omitted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

# ta_signals columns that map 1:1 onto the schema's trading_signal object.
# NOTE: lifecycle fields (status/invalidated_by/invalidated_at) are intentionally
# OMITTED -- migration 005 has no such columns, so apex cannot source them yet.
# The schema does not require them (status has a default), so omitting keeps
# payloads valid. Lifecycle persistence is a deferred item.
_SIGNAL_FIELDS = (
    "signal_id",
    "symbol",
    "category",
    "indicator",
    "direction",
    "strength",
    "priority",
    "timeframe",
    "trigger_rule",
    "current_value",
    "threshold",
    "previous_value",
    "message",
    "cooldown_until",
    "metadata",
)

# Schema-required numeric fields that must NOT be null (else the row is invalid).
_REQUIRED_NUMERIC = ("current_value",)


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def signal_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _SIGNAL_FIELDS:
        if key in row and row[key] is not None:
            out[key] = _iso(row[key])
    # DB stores the event time as `time`; the contract field is `timestamp`.
    out["timestamp"] = _iso(row["time"])
    return out


def is_emittable(row: dict[str, Any]) -> bool:
    """A row is emittable only if its schema-required numeric fields are non-null.

    ta_signals.current_value is nullable, but the contract requires a number -- so
    rows missing it are dropped (and logged by the caller) rather than emitted invalid.
    """
    return all(row.get(k) is not None for k in _REQUIRED_NUMERIC)


def build_payload(rows: Iterable[dict[str, Any]], generated_at: datetime) -> dict[str, Any]:
    # Drop rows that would be schema-invalid (e.g. null current_value).
    emittable = [r for r in rows if is_emittable(r)]
    signals = [signal_row_to_dict(r) for r in emittable]
    return {
        "signals": signals,
        "timestamp": generated_at.isoformat(),
        "symbol_count": len({s["symbol"] for s in signals}),
    }
