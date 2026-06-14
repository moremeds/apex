"""to_jsonable: coerce numpy scalars / NaN to JSON-native types, recursively.

Both egress paths that serialize indicator state to JSON -- the chart compute-on-read
service (``src/application/chart/indicator_compute.py``) and the live-persist repository
(``ta_signal_repository.save_indicator``) -- must encode identically, so they share this
one coercer. The key cases: numpy bool/int/float scalars are NOT JSON-serializable by
stdlib ``json`` (raises ``TypeError``), and numpy/float ``NaN`` must become ``null`` (a
bare ``NaN`` is invalid JSON and Postgres JSONB rejects it).
"""

from __future__ import annotations

import json
import math

import numpy as np

from src.utils.jsonable import to_jsonable


def test_numpy_bool_becomes_python_bool() -> None:
    out = to_jsonable(np.bool_(True))
    assert out is True
    assert isinstance(out, bool)
    assert json.dumps(out) == "true"


def test_numpy_int_and_float_become_python_scalars() -> None:
    assert to_jsonable(np.int64(7)) == 7
    assert isinstance(to_jsonable(np.int64(7)), int)
    assert to_jsonable(np.float64(1.5)) == 1.5
    assert isinstance(to_jsonable(np.float64(1.5)), float)


def test_nan_becomes_none() -> None:
    assert to_jsonable(np.float64("nan")) is None
    assert to_jsonable(float("nan")) is None


def test_recurses_into_dicts_and_lists() -> None:
    state = {
        "squeeze": np.bool_(True),  # the bollinger field that crashed persist
        "value": np.float64(42.5),
        "history": [np.int64(1), np.float64("nan"), np.bool_(False)],
        "nested": {"flag": np.bool_(False)},
    }
    out = to_jsonable(state)

    # Whole structure is now stdlib-json serializable (the original raised TypeError).
    encoded = json.dumps(out)
    assert json.loads(encoded) == {
        "squeeze": True,
        "value": 42.5,
        "history": [1, None, False],
        "nested": {"flag": False},
    }


def test_plain_python_values_pass_through() -> None:
    state = {"a": 1, "b": "x", "c": 2.0, "d": True, "e": None}
    assert to_jsonable(state) == state
    # math.nan still maps to None even when nested in a plain dict
    assert to_jsonable({"k": math.nan}) == {"k": None}
