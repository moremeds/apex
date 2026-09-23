"""Build chart read-surface payloads: bars, indicator series, confluence.

These mirror ``payload/builder.py`` (ISO timestamps, ``count``, ``generated_at``) and
are validated on egress against their schemas under config/verification/schemas/.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from src.application.lake.bars import BarsResult, BulkResult, PitProvenance, RatesResult
from src.infrastructure.adapters.livewire.asset_classes import get_asset_class

# The two adjustment bases apex can serve. `adjusted` reads livewire Silver, whose
# factor chain compounds splits AND cash dividends (verified against revision 76 on
# 2026-09-21: SPY carries a 0.99752... factor across a dividend-only interval), so
# "split-adjusted" would understate what the numbers are. Named, not inferred: a
# consumer must never have to guess the basis from the price_mode label.
_BASIS_BY_MODE = {"adjusted": "split+dividend", "raw": "unadjusted"}


def basis_for(price_mode: str) -> str:
    """The adjustment basis a ``price_mode`` is served on."""
    try:
        return _BASIS_BY_MODE[price_mode]
    except KeyError as exc:  # pragma: no cover - routes validate before calling
        raise ValueError(f"unknown price_mode {price_mode!r}") from exc


def _iso(value: Any) -> Any:
    """ISO-8601 string, normalised to UTC so the chart contract matches the signal one.

    DuckDB returns bar timestamps in the session timezone; convert tz-aware values to
    UTC (+00:00) for a consistent contract. Naive datetimes are emitted as-is.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.isoformat()
    return value


def _bar_to_dict(
    bar: Any, extra_fields: tuple[str, ...] = (), source_basis: bool = False
) -> Dict[str, Any]:
    # livewire bars set timestamp == bar_start; prefer timestamp, fall back to bar_start.
    when = bar.timestamp if getattr(bar, "timestamp", None) is not None else bar.bar_start
    row: Dict[str, Any] = {
        "time": _iso(when),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }
    # Per-class extra columns, read from the registry rather than hardcoded -- the
    # whole point of the registry is that a seventh class is a row, not an edit here.
    # Omitted entirely where absent rather than emitted as null noise on ~20M bars.
    for extra in extra_fields:
        value = getattr(bar, extra, None)
        if value is not None:
            row[extra] = value
    if source_basis:
        # The serving mode says "unadjusted"; this says what livewire recorded for the
        # row itself (raw / split_adjusted / unknown). Absent column -> unknown, never
        # inferred from the mode.
        row["source_price_basis"] = getattr(bar, "source_price_basis", None) or "unknown"
    return row


def _carries_source_basis(asset_class: str, price_mode: str) -> bool:
    return asset_class == "equity" and price_mode == "raw"


def build_bar_rows(
    bars: Iterable[Any], asset_class: str = "equity", price_mode: str = "raw"
) -> List[Dict[str, Any]]:
    """Bars as contract rows, for a payload that carries several series at once."""
    extra_fields = get_asset_class(asset_class).extra_bar_fields
    source_basis = _carries_source_basis(asset_class, price_mode)
    return [_bar_to_dict(bar, extra_fields, source_basis) for bar in bars]


def build_bars_payload(
    symbol: str,
    timeframe: str,
    bars: Iterable[Any],
    *,
    generated_at: datetime,
    asset_class: str = "equity",
    price_mode: str = "raw",
    listing_status: str = "listed",
    adjustment_revision: int | None = None,
    contract: Dict[str, Any] | None = None,
    window: Dict[str, Any] | None = None,
    truncated: bool | None = None,
    provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the bars contract.

    ``price_mode`` and ``listing_status`` are required and non-null by design: a
    consumer written against ``listing_status == "listed"`` cannot later be handed
    delisted bars silently, and the adjustment basis is never left to inference.
    """
    rows = build_bar_rows(bars, asset_class, price_mode)
    payload = {
        "symbol": symbol,
        "asset_class": asset_class,
        "timeframe": timeframe,
        "price_mode": price_mode,
        "basis": basis_for(price_mode),
        "listing_status": listing_status,
        "adjustment_revision": adjustment_revision,
        "contract": contract,
        "bars": rows,
        "count": len(rows),
        "generated_at": generated_at.isoformat(),
    }
    # Additive fields: present only when the caller has them, so the legacy shape
    # of a hand-built payload is unchanged.
    for key, value in (("window", window), ("truncated", truncated), ("provenance", provenance)):
        if value is not None:
            payload[key] = value
    return payload


def _window(start: datetime, end: datetime) -> Dict[str, Any]:
    return {"start": _iso(start), "end": _iso(end)}


def pit_provenance_dict(pit: PitProvenance) -> Dict[str, Any]:
    return {
        "revision": pit.revision,
        "index_id": pit.index_id,
        "publisher_status": pit.publisher_status,
        "policy_version": pit.policy_version,
        "as_of": pit.as_of.isoformat(),
        "published_at": pit.published_at.isoformat(),
        "daily_bar_cutoff": pit.daily_bar_cutoff.isoformat(),
        "silver_revision": pit.silver_revision,
        "membership_revision": pit.membership_revision,
        "scopes": [
            {
                "security_id": scope.security_id,
                "session_from": scope.session_from.isoformat(),
                "session_to": None if scope.session_to is None else scope.session_to.isoformat(),
            }
            for scope in pit.scopes
        ],
    }


def bars_payload_from(result: BarsResult, *, generated_at: datetime) -> Dict[str, Any]:
    """The bars contract for a shared-query result, with window/provenance fields."""
    return build_bars_payload(
        result.symbol,
        result.timeframe,
        result.bars,
        generated_at=generated_at,
        asset_class=result.asset_class,
        contract=result.contract,
        price_mode=result.price_mode,
        listing_status=result.listing_status,
        adjustment_revision=result.adjustment_revision,
        window=_window(result.window_start, result.window_end),
        truncated=result.truncated,
        provenance={
            "immutable_history": result.immutable_history,
            "silver_revision": result.pinned_silver_revision,
            "pit": None if result.pit is None else pit_provenance_dict(result.pit),
        },
    )


def bulk_bars_payload_from(result: BulkResult, *, generated_at: datetime) -> Dict[str, Any]:
    return {
        "price_mode": result.price_mode,
        "basis": basis_for(result.price_mode),
        "adjustment_revision": result.adjustment_revision,
        "silver_revision": result.pinned_silver_revision,
        "timeframe": result.timeframe,
        "window": _window(result.window_start, result.window_end),
        "symbols": {
            symbol: {
                "listing_status": series.listing_status,
                "truncated": series.truncated,
                "bars": build_bar_rows(series.bars, "equity", result.price_mode),
            }
            for symbol, series in result.series.items()
        },
        "missing": result.missing,
        "generated_at": generated_at.isoformat(),
    }


def rates_payload_from(
    result: RatesResult, *, generated_at: datetime, bounded: bool
) -> Dict[str, Any]:
    payload = build_rates_series_payload(result.symbol, result.points, generated_at=generated_at)
    if bounded:
        payload["window"] = _window(result.window_start, result.window_end)
        payload["truncated"] = result.truncated
    return payload


def build_rates_series_payload(
    symbol: str, points: Iterable[Any], *, generated_at: datetime
) -> Dict[str, Any]:
    """Build the rates contract. Separate from bars because a yield has no OHLC and
    cannot satisfy bars_payload.schema.json's numeric open/high/low/close."""
    # Materialize once: `points` is an Iterable, and iterating it twice would leave
    # tenor_years silently None for any generator caller.
    materialized = list(points)
    rows = [{"time": _iso(p.time), "yield_pct": p.yield_pct} for p in materialized]
    tenors = {p.tenor_years for p in materialized}
    return {
        "symbol": symbol,
        "asset_class": "rates",
        "tenor_years": next(iter(tenors)) if len(tenors) == 1 else None,
        "points": rows,
        "count": len(rows),
        "generated_at": generated_at.isoformat(),
    }


def build_indicator_payload(
    symbol: str,
    timeframe: str,
    indicator: str,
    points: Iterable[Dict[str, Any]],
    *,
    generated_at: datetime,
) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = [
        {"time": _iso(p["time"]), "state": p["state"], "bar_close": p.get("bar_close")}
        for p in points
    ]
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "indicator": indicator,
        "points": out,
        "count": len(out),
        "generated_at": generated_at.isoformat(),
    }


_CONFLUENCE_FIELDS = (
    "alignment_score",
    "bullish_count",
    "bearish_count",
    "neutral_count",
    "total_indicators",
    "dominant_direction",
)


def build_confluence_payload(
    symbol: str, timeframe: str, rows: Iterable[Dict[str, Any]], *, generated_at: datetime
) -> Dict[str, Any]:
    # get_confluence_history returns newest-first; emit oldest-first so the chart
    # contract is a consistent ascending time series (like /bars and /indicators).
    ordered = sorted(rows, key=lambda r: r["time"])
    out: List[Dict[str, Any]] = []
    for r in ordered:
        point: Dict[str, Any] = {"time": _iso(r["time"])}
        for field in _CONFLUENCE_FIELDS:
            if field in r:
                point[field] = r[field]
        out.append(point)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "points": out,
        "count": len(out),
        "generated_at": generated_at.isoformat(),
    }
