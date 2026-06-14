from __future__ import annotations

import pytest

from src.api.payload.validate import ValidationFailure, validate_payload


def _valid_signal() -> dict:
    return {
        "signal_id": "momentum:rsi:AAPL:1d",
        "symbol": "AAPL",
        "category": "momentum",
        "indicator": "RSI",
        "direction": "buy",
        "strength": 72,
        "priority": "high",
        "timeframe": "1d",
        "trigger_rule": "rsi_oversold_cross",
        "current_value": 28.4,
        "timestamp": "2026-06-14T12:00:00Z",
    }


def test_valid_payload_passes() -> None:
    validate_payload({"signals": [_valid_signal()], "timestamp": "2026-06-14T12:00:00Z"})


def test_missing_required_field_fails() -> None:
    bad = _valid_signal()
    del bad["strength"]
    with pytest.raises(ValidationFailure):
        validate_payload({"signals": [bad], "timestamp": "2026-06-14T12:00:00Z"})


def test_bad_signal_id_pattern_fails() -> None:
    bad = _valid_signal()
    bad["signal_id"] = "NOT A VALID ID"
    with pytest.raises(ValidationFailure):
        validate_payload({"signals": [bad], "timestamp": "2026-06-14T12:00:00Z"})
