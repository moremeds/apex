"""Serialization checks use type examples, not observed market data."""

from datetime import date
from decimal import Decimal

from src.api.payload.tabular import build_tabular, json_value


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
