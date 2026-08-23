"""Chart payload builders produce schema-valid dicts (the argon chart contract)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.api.payload.chart import (
    build_bars_payload,
    build_confluence_payload,
    build_indicator_payload,
    build_rates_series_payload,
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


def test_build_confluence_payload_orders_points_ascending() -> None:
    """get_confluence_history returns newest-first; the chart contract is a time series,
    so points must come out oldest-first to match /bars and /indicators."""
    older = datetime(2026, 6, 10, tzinfo=timezone.utc)
    newer = datetime(2026, 6, 12, tzinfo=timezone.utc)

    def _row(ts: datetime, score: float) -> dict:
        return {
            "time": ts,
            "alignment_score": score,
            "bullish_count": 1,
            "bearish_count": 0,
            "neutral_count": 0,
            "total_indicators": 1,
            "dominant_direction": "bullish",
        }

    rows_desc = [_row(newer, 0.9), _row(older, 0.1)]  # as the repo returns them (DESC)
    payload = build_confluence_payload("AAPL", "1d", rows_desc, generated_at=_NOW)
    times = [p["time"] for p in payload["points"]]
    assert times == sorted(times)  # ascending
    assert payload["points"][0]["alignment_score"] == 0.1  # oldest first


def test_signal_payload_validation_still_defaults_to_signal_schema() -> None:
    """Backward-compat: validate_payload with no schema arg uses the signal contract."""
    payload = {"signals": [], "timestamp": _NOW.isoformat(), "symbol_count": 0}
    validate_payload(payload)


# --- Task 6: basis, listing status and futures identity ----------------------------

_GEN = datetime(2026, 8, 22, tzinfo=timezone.utc)


def test_bars_payload_states_basis_and_listing_explicitly() -> None:
    """price_mode and listing_status are REQUIRED and non-null. A consumer must never
    have to infer the adjustment basis, and must not be silently handed delisted bars."""
    payload = build_bars_payload("AAPL", "1d", [], generated_at=_GEN)
    assert payload["price_mode"] == "raw"
    assert payload["listing_status"] == "listed"
    assert payload["asset_class"] == "equity"
    assert payload["adjustment_revision"] is None


def test_bars_payload_carries_adjustment_revision_when_adjusted() -> None:
    payload = build_bars_payload(
        "AAPL", "1d", [], generated_at=_GEN, price_mode="adjusted", adjustment_revision=33
    )
    assert (payload["price_mode"], payload["adjustment_revision"]) == ("adjusted", 33)


def test_bars_payload_drops_vwap() -> None:
    """No parquet in the lake carries a vwap column; it served null forever.

    Real AAPL bar, frozen 2026-08-23 from bronze/asset_class=equity.
    """
    bar = BarData(
        symbol="AAPL",
        timeframe="1d",
        open=312.05,
        high=312.38,
        low=307.01,
        close=309.35,
        volume=48591536,
        timestamp=datetime(2026, 8, 21, tzinfo=timezone.utc),
        bar_start=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )
    payload = build_bars_payload("AAPL", "1d", [bar], generated_at=_GEN)
    assert "vwap" not in payload["bars"][0]
    assert payload["bars"][0]["close"] == 309.35


def test_futures_payload_carries_contract_identity() -> None:
    payload = build_bars_payload(
        "BZ_202609",
        "1d",
        [],
        generated_at=_GEN,
        asset_class="futures",
        contract={
            "contract_id": 3871332472656972,
            "root_symbol": "BZ",
            "expiry_date": "2026-09-01",
        },
    )
    assert payload["contract"]["root_symbol"] == "BZ"


def test_futures_bar_carries_settlement_and_open_interest() -> None:
    """Real ICE Brent bar, frozen 2026-08-23 from bronze/asset_class=futures."""
    bar = BarData(
        symbol="BZ_202609",
        timeframe="1d",
        open=71.86,
        high=72.23,
        low=71.04,
        close=71.57,
        volume=7303,
        settlement=71.57,
        open_interest=0,
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        bar_start=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    row = build_bars_payload("BZ_202609", "1d", [bar], generated_at=_GEN, asset_class="futures")[
        "bars"
    ][0]
    assert row["settlement"] == 71.57
    assert row["open_interest"] == 0


def test_equity_bars_omit_futures_columns_entirely() -> None:
    """Emitting them as null would add noise to every equity bar in the lake."""
    bar = BarData(
        symbol="AAPL",
        timeframe="1d",
        open=312.05,
        high=312.38,
        low=307.01,
        close=309.35,
        volume=48591536,
        timestamp=datetime(2026, 8, 21, tzinfo=timezone.utc),
        bar_start=datetime(2026, 8, 21, tzinfo=timezone.utc),
    )
    row = build_bars_payload("AAPL", "1d", [bar], generated_at=_GEN)["bars"][0]
    assert "settlement" not in row
    assert "open_interest" not in row


def test_every_shape_validates_against_the_schema() -> None:
    for kwargs in (
        {},
        {"price_mode": "adjusted", "adjustment_revision": 33},
        {
            "asset_class": "futures",
            "contract": {"contract_id": 1, "root_symbol": "BZ", "expiry_date": "2026-09-01"},
        },
        {"listing_status": "delisted"},
        {"asset_class": "fx", "timeframe": "1h"},
    ):
        timeframe = kwargs.pop("timeframe", "1d")
        payload = build_bars_payload("AAPL", timeframe, [], generated_at=_GEN, **kwargs)
        validate_payload(payload, "bars_payload")


def test_rates_payload_validates() -> None:
    """Real FRED DGS10 observations, frozen 2026-08-23."""
    from src.infrastructure.adapters.livewire.ohlc_provider import RatePoint

    points = [
        RatePoint(
            time=datetime(2026, 8, 18, tzinfo=timezone.utc), tenor_years=10.0, yield_pct=4.71
        ),
        RatePoint(
            time=datetime(2026, 8, 19, tzinfo=timezone.utc), tenor_years=10.0, yield_pct=4.65
        ),
    ]
    payload = build_rates_series_payload("DGS10", points, generated_at=_GEN)
    assert payload["tenor_years"] == 10.0
    assert payload["points"][0]["yield_pct"] == 4.71
    validate_payload(payload, "rates_series_payload")


def test_rates_payload_accepts_a_generator() -> None:
    """points is an Iterable; iterating it twice would silently null tenor_years."""
    from src.infrastructure.adapters.livewire.ohlc_provider import RatePoint

    def gen():
        yield RatePoint(
            time=datetime(2026, 8, 20, tzinfo=timezone.utc), tenor_years=10.0, yield_pct=4.69
        )

    payload = build_rates_series_payload("DGS10", gen(), generated_at=_GEN)
    assert payload["tenor_years"] == 10.0
    assert payload["count"] == 1


def test_rates_payload_nulls_tenor_when_mixed() -> None:
    from src.infrastructure.adapters.livewire.ohlc_provider import RatePoint

    points = [
        RatePoint(
            time=datetime(2026, 8, 18, tzinfo=timezone.utc), tenor_years=10.0, yield_pct=4.71
        ),
        RatePoint(time=datetime(2026, 8, 18, tzinfo=timezone.utc), tenor_years=5.0, yield_pct=3.9),
    ]
    payload = build_rates_series_payload("MIXED", points, generated_at=_GEN)
    assert payload["tenor_years"] is None
    validate_payload(payload, "rates_series_payload")
