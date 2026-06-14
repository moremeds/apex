"""Coerce numpy scalars / NaN into JSON-native types, recursively.

Indicator ``state`` dicts come straight out of pandas/numpy compute, so they carry
numpy scalars -- numpy bool (e.g. bollinger's ``squeeze``), numpy int/float -- which
stdlib ``json`` cannot serialize (raises ``TypeError``), plus ``NaN`` (a bare ``NaN`` is
invalid JSON and Postgres JSONB rejects it). Both egress paths that JSON-encode that
state -- the chart compute-on-read service and the live-persist repository -- share this
one coercer so they encode identically and can't drift.
"""

from __future__ import annotations

import math
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars to Python scalars and NaN to None.

    Leaves JSON-native values untouched. ``dict``/``list``/``tuple`` are recursed
    (tuples become lists). numpy scalars are unwrapped via ``.item()``; any float NaN
    (numpy or Python) becomes ``None``.
    """
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    item = value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            item = value.item()  # numpy scalar -> python scalar
        except (ValueError, AttributeError):
            item = value
    if isinstance(item, float) and math.isnan(item):
        return None
    return item
