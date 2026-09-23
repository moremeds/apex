"""Bulk equity return table for the Flash weekly (issue #160).

- GET /v1/equity/returns?symbols=A,B,C&start=YYYY-MM-DD&end=YYYY-MM-DD

Every derived number (daily return, window return, YTD, distance from the 52-week
high, excess vs SPY/QQQ) is computed here, from one fetch per symbol on one price
basis. The consumer does no arithmetic: a model that computes its own derived
figures gets them wrong, and a second implementation downstream is a second
definition. Symbols that cannot be served are listed in ``missing`` with a reason
rather than dropped, so a downstream guard has something to reject.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, Query, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.chart import basis_for
from src.api.routes._lake import provider_or_raise
from src.application.lake.guards import artifact_exists, spec_or_raise
from src.domain.events.domain_events import BarData
from src.infrastructure.adapters.livewire.ohlc_provider import AdjustedDataUnavailable
from src.infrastructure.adapters.livewire.parquet_reads import QueryTimeout

router = APIRouter(tags=["chart"])

_MAX_SYMBOLS = 200
_BENCHMARKS = ("SPY", "QQQ")
# Enough calendar days to clear a long weekend plus a holiday, so the first day in
# the window always has a prior close to return against.
_PRIOR_CLOSE_LOOKBACK = timedelta(days=10)
_YEAR = timedelta(days=365)

# (day, close) pairs, ascending.
_Series = List[Tuple[date, float]]


def _norm_symbols(raw: str) -> List[str]:
    ordered: Dict[str, None] = {}
    for part in raw.split(","):
        symbol = part.strip().upper()
        if symbol:
            ordered[symbol] = None
    symbols = list(ordered)
    if not symbols:
        raise ApiError(ApiErrorCode.INVALID_PARAMETER, "symbols is required and must be non-empty")
    if len(symbols) > _MAX_SYMBOLS:
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"{len(symbols)} symbols requested; at most {_MAX_SYMBOLS} per call",
        )
    return symbols


def _parse_day(value: Optional[str], name: str) -> date:
    if not value:
        raise ApiError(ApiErrorCode.INVALID_PARAMETER, f"{name} is required (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"malformed {name} {value!r}; expected YYYY-MM-DD",
        ) from exc


def _closes(bars: Sequence[BarData]) -> _Series:
    """Bars reduced to the only two fields this endpoint reads, ascending by day."""
    series: _Series = []
    for bar in bars:
        when = bar.timestamp if bar.timestamp is not None else bar.bar_start
        if when is None or bar.close is None:
            continue
        if when.tzinfo is not None:
            when = when.astimezone(timezone.utc)
        series.append((when.date(), float(bar.close)))
    series.sort(key=lambda pair: pair[0])
    return series


def _pct(last: float, base: Optional[float]) -> Optional[float]:
    if base is None or base == 0:
        return None
    return last / base - 1


def _last_close_before(series: _Series, day: date) -> Optional[float]:
    prior = [close for when, close in series if when < day]
    return prior[-1] if prior else None


def _window_return(series: _Series, start: date, end: date) -> Optional[float]:
    window = [(when, close) for when, close in series if start <= when <= end]
    if not window:
        return None
    return _pct(window[-1][1], _last_close_before(series, start))


def _metrics(series: _Series, start: date, end: date) -> Dict[str, Any]:
    window = [(when, close) for when, close in series if start <= when <= end]
    prev = _last_close_before(series, start)
    daily: List[Dict[str, Any]] = []
    for when, close in window:
        daily.append({"date": when.isoformat(), "close": close, "return": _pct(close, prev)})
        prev = close
    last = window[-1][1]
    prior_year = [close for when, close in series if when.year < end.year]
    high_52w = [close for when, close in series if end - _YEAR <= when <= end]
    return {
        "daily": daily,
        "window_return": _pct(last, _last_close_before(series, start)),
        "ytd_return": _pct(last, prior_year[-1] if prior_year else None),
        "pct_from_52w_high": _pct(last, max(high_52w)) if high_52w else None,
    }


async def _load(
    provider: Any,
    spec: Any,
    symbols: Sequence[str],
    lo: datetime,
    hi: datetime,
    price_mode: str,
) -> Tuple[Dict[str, _Series], Dict[str, str]]:
    """One fetch per distinct symbol, so a benchmark that is also requested is read once."""
    series: Dict[str, _Series] = {}
    failures: Dict[str, str] = {}
    for symbol in symbols:
        try:
            bars = await provider.fetch_bars(
                symbol, "1d", lo, hi, asset_class=spec.name, price_mode=price_mode
            )
        except (AdjustedDataUnavailable, QueryTimeout) as exc:
            failures[symbol] = str(exc)
            continue
        series[symbol] = _closes(bars)
    return series, failures


@router.get("/v1/equity/returns")
async def equity_returns(
    request: Request,
    symbols: str = Query("", description="Comma-separated tickers, at most 200"),
    start: Optional[str] = Query(None, description="First day of the window, YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="Last day of the window, inclusive"),
) -> Dict[str, Any]:
    requested = _norm_symbols(symbols)
    start_day = _parse_day(start, "start")
    end_day = _parse_day(end, "end")
    if start_day > end_day:
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"start {start_day.isoformat()} is after end {end_day.isoformat()}",
        )

    spec = spec_or_raise("equity")
    provider = provider_or_raise(request)
    # One basis for every number on the page: the window, YTD, the 52-week high and both
    # benchmarks are all read in the provider's configured mode, and it is echoed back.
    price_mode = provider.effective_price_mode(spec.name)
    if price_mode == "adjusted":
        # Pin one Silver revision for the whole table: a revision landing mid-request
        # would adjust some symbols on one corporate-action set and the rest on another.
        provider = await asyncio.to_thread(provider.pin_snapshot)

    # YTD needs the prior year's last close and the 52-week high needs a year of history,
    # so one fetch per symbol spans the widest of the three lookbacks and is sliced here.
    lo_day = min(
        start_day - _PRIOR_CLOSE_LOOKBACK,
        date(end_day.year - 1, 12, 1),
        end_day - _YEAR,
    )
    lo = datetime.combine(lo_day, time.min, tzinfo=timezone.utc)
    hi = datetime.combine(end_day, time.max, tzinfo=timezone.utc)

    to_fetch = list(dict.fromkeys((*_BENCHMARKS, *requested)))
    series, failures = await _load(provider, spec, to_fetch, lo, hi, price_mode)

    bench_returns = {
        name: _window_return(series.get(name, []), start_day, end_day) for name in _BENCHMARKS
    }

    results: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    for symbol in requested:
        if symbol in failures:
            missing.append({"symbol": symbol, "reason": failures[symbol]})
            continue
        rows = series.get(symbol, [])
        if not any(start_day <= when <= end_day for when, _ in rows):
            reason = (
                f"no bars between {start_day.isoformat()} and {end_day.isoformat()}"
                if artifact_exists(provider, symbol, "1d", spec, price_mode)
                else f"no artifact for {symbol} under {spec.partition}"
            )
            missing.append({"symbol": symbol, "reason": reason})
            continue
        row = {"symbol": symbol, **_metrics(rows, start_day, end_day)}
        window_return = row["window_return"]
        for name, key in (("SPY", "excess_vs_spy"), ("QQQ", "excess_vs_qqq")):
            benchmark = bench_returns[name]
            row[key] = (
                None if window_return is None or benchmark is None else window_return - benchmark
            )
        results.append(row)

    return {
        "start": start_day.isoformat(),
        "end": end_day.isoformat(),
        "price_mode": price_mode,
        "basis": basis_for(price_mode),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # A benchmark that could not be read (quarantined Silver, lake timeout) says why,
        # so a null excess is never mistaken for "no bars in the window".
        "benchmarks": {
            name: {
                "window_return": bench_returns[name],
                **({"failure": failures[name]} if name in failures else {}),
            }
            for name in _BENCHMARKS
        },
        "results": results,
        "missing": missing,
    }
