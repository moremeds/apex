from __future__ import annotations

from datetime import datetime, timezone

from src.infrastructure.adapters.xenon.translator import translate_price_data


def test_translates_full_price_data_with_iso_timestamp() -> None:
    out = translate_price_data(
        {
            "symbol": "AAPL",
            "last": 150.25,
            "bid": 150.2,
            "ask": 150.3,
            "volume": 1000,
            "timestamp": "2026-06-14T12:00:00.000Z",
        }
    )
    assert out is not None
    assert out["symbol"] == "AAPL"
    assert out["last"] == 150.25
    assert out["volume"] == 1000
    assert out["timestamp"] == datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)


def test_null_last_but_bid_ask_present_is_kept() -> None:
    out = translate_price_data(
        {
            "symbol": "AAPL",
            "last": None,
            "bid": 10.0,
            "ask": 10.2,
            "timestamp": "2026-06-14T12:00:00Z",
        }
    )
    assert out is not None
    assert out["bid"] == 10.0 and out["ask"] == 10.2


def test_no_usable_price_is_dropped() -> None:
    out = translate_price_data(
        {
            "symbol": "AAPL",
            "last": None,
            "bid": None,
            "ask": None,
            "timestamp": "2026-06-14T12:00:00Z",
        }
    )
    assert out is None


def test_missing_symbol_is_dropped() -> None:
    assert translate_price_data({"last": 1.0, "timestamp": "2026-06-14T12:00:00Z"}) is None


def test_missing_timestamp_yields_none_timestamp_key() -> None:
    out = translate_price_data({"symbol": "AAPL", "last": 5.0})
    assert out is not None
    assert out["timestamp"] is None  # aggregator will default to now()
