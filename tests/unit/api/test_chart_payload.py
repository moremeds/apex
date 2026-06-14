"""Chart payload builders produce schema-valid dicts (the argon chart contract)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.api.payload.chart import (
    build_bars_payload,
    build_confluence_payload,
    build_indicator_payload,
)
from src.api.payload.validate import validate_payload
from src.domain.events.domain_events import BarData

_NOW = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)


def _bar(close: float) -> BarData:
    return BarData(
        symbol="AAPL",
        timeframe="1d",
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=1_000_000,
        vwap=close + 0.1,
        timestamp=_NOW,
        bar_start=_NOW,
    )


def test_build_bars_payload_is_schema_valid() -> None:
    payload = build_bars_payload("AAPL", "1d", [_bar(150.0), _bar(151.0)], generated_at=_NOW)
    validate_payload(payload, "bars_payload")
    assert payload["symbol"] == "AAPL"
    assert payload["timeframe"] == "1d"
    assert payload["count"] == 2
    assert payload["bars"][0]["close"] == 150.0
    assert payload["bars"][0]["time"].endswith("+00:00")


def test_build_indicator_payload_is_schema_valid() -> None:
    points = [
        {"time": _NOW, "state": {"value": 65.3, "zone": "neutral"}, "bar_close": 150.0},
        {"time": _NOW, "state": {"value": 70.1, "zone": "overbought"}, "bar_close": 151.0},
    ]
    payload = build_indicator_payload("AAPL", "1d", "rsi", points, generated_at=_NOW)
    validate_payload(payload, "indicator_series_payload")
    assert payload["indicator"] == "rsi"
    assert payload["count"] == 2
    assert payload["points"][0]["state"]["value"] == 65.3


def test_build_confluence_payload_is_schema_valid() -> None:
    rows = [
        {
            "time": _NOW,
            "alignment_score": 0.4,
            "bullish_count": 3,
            "bearish_count": 1,
            "neutral_count": 2,
            "total_indicators": 6,
            "dominant_direction": "bullish",
        }
    ]
    payload = build_confluence_payload("AAPL", "1d", rows, generated_at=_NOW)
    validate_payload(payload, "confluence_payload")
    assert payload["points"][0]["alignment_score"] == 0.4
    assert payload["points"][0]["dominant_direction"] == "bullish"


def test_signal_payload_validation_still_defaults_to_signal_schema() -> None:
    """Backward-compat: validate_payload with no schema arg uses the signal contract."""
    payload = {"signals": [], "timestamp": _NOW.isoformat(), "symbol_count": 0}
    validate_payload(payload)
