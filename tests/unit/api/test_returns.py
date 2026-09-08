"""GET /v1/equity/returns: the bulk weekly return table (issue #160).

Closes below are REAL adjusted 1d closes pulled from the production apex instance on
2026-09-08 and frozen here; the expected returns are hand-computed from them, so a
change in the arithmetic fails rather than re-deriving itself.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app
from src.domain.events.domain_events import BarData

# symbol -> {day: adjusted close}, from /v1/equity/{symbol}/bars on 2026-09-08.
_CLOSES: Dict[str, Dict[str, float]] = {
    "SPY": {
        "2025-12-31": 678.3156842283097,
        "2026-08-28": 769.35,
        "2026-08-31": 767.05,
        "2026-09-01": 761.78,
        "2026-09-02": 765.16,
        "2026-09-03": 773.17,
        "2026-09-04": 770.19,
    },
    "QQQ": {
        "2025-12-31": 612.8626741215079,
        "2026-08-28": 716.43,
        "2026-08-31": 716.76,
        "2026-09-01": 707.64,
        "2026-09-02": 709.24,
        "2026-09-03": 717.67,
        "2026-09-04": 718.96,
    },
    "MU": {
        "2025-12-31": 285.2462880885288,
        "2026-08-28": 932.86,
        "2026-08-31": 958.73,
        "2026-09-01": 933.44,
        "2026-09-02": 956.08,
        "2026-09-03": 958.16,
        "2026-09-04": 1016.59,
    },
}


class _FakeProvider:
    """Serves the frozen closes and counts fetches, so "one read per symbol" is testable."""

    def __init__(self) -> None:
        # A path that does not exist: _artifact_exists probes it to tell "no artifact"
        # apart from "artifact exists, window empty".
        self.bronze_root = Path("/nonexistent")
        self.silver_root = None
        self.snapshot = None
        self.calls: List[str] = []

    def effective_price_mode(self, asset_class: str = "equity") -> str:
        return "raw"

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        asset_class: str = "equity",
        price_mode: str | None = None,
    ) -> List[BarData]:
        self.calls.append(symbol)
        bars: List[BarData] = []
        for day, close in sorted(_CLOSES.get(symbol, {}).items()):
            ts = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
            if not start <= ts <= end:
                continue
            bars.append(
                BarData(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=1,
                    timestamp=ts,
                    bar_start=ts,
                )
            )
        return bars


async def _get(provider: Any, query: str) -> Any:
    app = create_app()
    app.state.ohlc_provider = provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(f"/v1/equity/returns?{query}")


_WEEK = "start=2026-08-31&end=2026-09-04"


async def test_returns_match_hand_computed_values() -> None:
    resp = await _get(_FakeProvider(), f"symbols=MU,SPY&{_WEEK}")
    assert resp.status_code == 200
    body = resp.json()

    assert body["start"] == "2026-08-31"
    assert body["end"] == "2026-09-04"
    assert body["price_mode"] == "raw"
    assert body["missing"] == []
    assert body["benchmarks"]["SPY"]["window_return"] == pytest.approx(0.0010918307662313165)
    assert body["benchmarks"]["QQQ"]["window_return"] == pytest.approx(0.003531398740979741)

    mu = body["results"][0]
    assert mu["symbol"] == "MU"
    # 1016.59 / 932.86 - 1, against 2026-08-28 as the last close BEFORE the window.
    assert mu["window_return"] == pytest.approx(0.08975623351842721)
    assert mu["ytd_return"] == pytest.approx(2.5639026429135936)
    assert mu["excess_vs_spy"] == pytest.approx(0.08866440275219589)
    assert mu["excess_vs_qqq"] == pytest.approx(0.08622483477744747)
    # MU closed the window at its highest close, so it sits exactly on the 52w high.
    assert mu["pct_from_52w_high"] == pytest.approx(0.0)

    assert [row["date"] for row in mu["daily"]] == [
        "2026-08-31",
        "2026-09-01",
        "2026-09-02",
        "2026-09-03",
        "2026-09-04",
    ]
    assert mu["daily"][0]["close"] == 958.73
    assert mu["daily"][0]["return"] == pytest.approx(0.027731921188602904)
    assert mu["daily"][1]["return"] == pytest.approx(-0.026378646751431534)
    assert mu["daily"][-1]["return"] == pytest.approx(0.06098146447357444)

    spy = body["results"][1]
    # 770.19 / 773.17 - 1: SPY peaked on 09-03 and gave some back on 09-04.
    assert spy["pct_from_52w_high"] == pytest.approx(-0.0038542623226456296)
    assert spy["ytd_return"] == pytest.approx(0.13544477579965108)
    assert spy["excess_vs_spy"] == pytest.approx(0.0)


async def test_symbol_without_an_artifact_is_listed_missing() -> None:
    """One unservable symbol must not sink the other six on the weekly table."""
    resp = await _get(_FakeProvider(), f"symbols=SOXX,MU&{_WEEK}")
    assert resp.status_code == 200
    body = resp.json()

    assert [row["symbol"] for row in body["results"]] == ["MU"]
    assert body["missing"] == [
        {"symbol": "SOXX", "reason": "no artifact for SOXX under asset_class=equity"}
    ]


async def test_benchmarks_are_fetched_once_even_when_also_requested() -> None:
    provider = _FakeProvider()
    resp = await _get(provider, f"symbols=SPY,QQQ,MU,SPY&{_WEEK}")
    assert resp.status_code == 200
    assert sorted(provider.calls) == ["MU", "QQQ", "SPY"]


async def test_symbols_are_uppercased_and_deduplicated_in_order() -> None:
    resp = await _get(_FakeProvider(), f"symbols=mu, spy ,MU&{_WEEK}")
    assert [row["symbol"] for row in resp.json()["results"]] == ["MU", "SPY"]


async def test_too_many_symbols_is_rejected() -> None:
    symbols = ",".join(f"MU{i}" for i in range(201))
    resp = await _get(_FakeProvider(), f"symbols={symbols}&{_WEEK}")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_parameter"


@pytest.mark.parametrize(
    "query",
    [
        f"symbols=&{_WEEK}",
        "symbols=MU&end=2026-09-04",
        "symbols=MU&start=2026-08-31",
        "symbols=MU&start=08%2F31%2F2026&end=2026-09-04",
        "symbols=MU&start=2026-09-04&end=2026-08-31",
    ],
    ids=["no-symbols", "no-start", "no-end", "malformed-start", "start-after-end"],
)
async def test_malformed_requests_are_rejected(query: str) -> None:
    resp = await _get(_FakeProvider(), query)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_parameter"


async def test_returns_is_not_shadowed_by_the_instrument_route() -> None:
    """/v1/{asset_class}/{symbol} would match symbol="returns"; registration order wins."""
    resp = await _get(_FakeProvider(), f"symbols=MU&{_WEEK}")
    assert resp.status_code == 200
    assert "results" in resp.json()


def test_first_daily_return_is_null_without_a_prior_close() -> None:
    from src.api.routes.returns import _metrics

    series = [(date(2026, 8, 31), 958.73), (date(2026, 9, 1), 933.44)]
    metrics = _metrics(series, date(2026, 8, 31), date(2026, 9, 4))
    assert metrics["daily"][0]["return"] is None
    assert metrics["window_return"] is None
