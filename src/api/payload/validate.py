"""Validate outgoing payloads against signal_service_payload.schema.json.

The schema is the contract with argon (argon-adaptation.md 5). Every payload
apex sends -- REST or WS -- passes through here first.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_DIR = (
    Path(__file__).resolve().parents[3] / "config" / "verification" / "schemas"
)
_DEFAULT_SCHEMA = "signal_service_payload"


class ValidationFailure(ValueError):
    """Raised when a payload does not satisfy the contract schema."""


@lru_cache(maxsize=None)
def _schema(name: str = _DEFAULT_SCHEMA) -> dict:
    return json.loads((_SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def validate_payload(payload: dict[str, Any], schema: str = _DEFAULT_SCHEMA) -> None:
    """Validate ``payload`` against ``<schema>.schema.json`` (signal contract by default)."""
    try:
        jsonschema.validate(instance=payload, schema=_schema(schema))
    except jsonschema.ValidationError as exc:
        raise ValidationFailure(str(exc)) from exc
