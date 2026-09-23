"""Serialization checks use type examples, not observed market data."""

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from src.api.payload.tabular import build_tabular, json_value
from src.api.payload.validate import ValidationFailure


def test_lossless_values_and_page_boundary() -> None:
    columns = [{"name": "value", "type": "numeric"}]
    rows = [{"value": Decimal("0.1234567890123456789")}]
    payload = build_tabular("db", "public", "example", columns, rows, 1)
    assert payload["rows"] == [["0.1234567890123456789"]]
    assert payload["truncated"] is False
    assert build_tabular("db", "public", "example", columns, rows * 2, 1)["truncated"]
    assert build_tabular("db", "public", "example", columns, [], 1)["columns"] == columns
    assert json_value({"date": date(2026, 9, 22), "nested": [Decimal("0.1")]}) == {
        "date": "2026-09-22",
        "nested": ["0.1"],
    }
    assert json_value(float("inf")) == "inf"
    assert json_value(b"\x00\xff") == "\\x00ff"


def test_empty_page_and_null_cells_validate() -> None:
    columns = [{"name": "value", "type": "numeric"}, {"name": "note", "type": "text"}]
    empty = build_tabular("db", "public", "example", columns, [], 500)
    assert (empty["rows"], empty["count"], empty["truncated"]) == ([], 0, False)
    nulls = build_tabular("db", "public", "example", columns, [{"value": None, "note": None}], 5)
    assert nulls["rows"] == [[None, None]]


def test_coverage_shape_is_part_of_the_schema() -> None:
    columns = [{"name": "has_quote", "type": "bool"}]
    rows = [{"has_quote": True}]
    good = {"scope": "returned_rows", "tables": {"quotes": {"matched": 1, "total": 1}}}
    assert build_tabular("db", "s", "j", columns, rows, 1, coverage=good)["coverage"] == good
    bad_coverages: list[dict[str, Any]] = [
        {"scope": "database", "tables": {}},
        {"scope": "returned_rows"},
        {"scope": "returned_rows", "tables": {"quotes": {"matched": -1, "total": 1}}},
        {"scope": "returned_rows", "tables": {"quotes": {"matched": 0.5, "total": 1}}},
        {"scope": "returned_rows", "tables": {"quotes": {"matched": 1}}},
        {"scope": "returned_rows", "tables": {"quotes": {"matched": 1, "total": 5001}}},
    ]
    for coverage in bad_coverages:
        with pytest.raises(ValidationFailure):
            build_tabular("db", "s", "j", columns, rows, 1, coverage=coverage)
