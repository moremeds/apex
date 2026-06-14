from __future__ import annotations

from datetime import datetime, timezone

from src.api.payload.builder import build_payload, signal_row_to_dict
from src.api.payload.validate import validate_payload


def _row() -> dict:
    return {
        "time": datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc),
        "signal_id": "trend:macd:AAPL:1d",
        "symbol": "AAPL",
        "timeframe": "1d",
        "category": "trend",
        "indicator": "MACD",
        "direction": "buy",
        "strength": 65,
        "priority": "medium",
        "trigger_rule": "macd_bull_cross",
        "current_value": 1.23,
        "threshold": 0.0,
        "previous_value": -0.4,
        "message": "MACD bullish cross",
        "cooldown_until": None,
        "metadata": {"fast": 12, "slow": 26},
    }


def test_row_to_dict_maps_time_to_timestamp() -> None:
    d = signal_row_to_dict(_row())
    assert d["timestamp"] == "2026-06-14T12:00:00+00:00"
    assert "time" not in d


def test_build_payload_is_schema_valid() -> None:
    payload = build_payload([_row()], generated_at=datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc))
    validate_payload(payload)
    assert payload["symbol_count"] == 1
    assert payload["signals"][0]["symbol"] == "AAPL"


def test_null_current_value_row_is_dropped() -> None:
    bad = _row()
    bad["current_value"] = None
    payload = build_payload([bad, _row()], generated_at=datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc))
    validate_payload(payload)
    assert len(payload["signals"]) == 1
