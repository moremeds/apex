"""Build chart read-surface payloads: bars, indicator series, confluence.

These mirror ``payload/builder.py`` (ISO timestamps, ``count``, ``generated_at``) and
are validated on egress against their schemas under config/verification/schemas/.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from src.infrastructure.adapters.livewire.asset_classes import get_asset_class


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


def _bar_to_dict(bar: Any, extra_fields: tuple[str, ...] = ()) -> Dict[str, Any]:
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
    return row


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
) -> Dict[str, Any]:
    """Build the bars contract.

    ``price_mode`` and ``listing_status`` are required and non-null by design: a
    consumer written against ``listing_status == "listed"`` cannot later be handed
    delisted bars silently, and the adjustment basis is never left to inference.
    """
    extra_fields = get_asset_class(asset_class).extra_bar_fields
    rows = [_bar_to_dict(b, extra_fields) for b in bars]
    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "timeframe": timeframe,
        "price_mode": price_mode,
        "listing_status": listing_status,
        "adjustment_revision": adjustment_revision,
        "contract": contract,
        "bars": rows,
        "count": len(rows),
        "generated_at": generated_at.isoformat(),
    }


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
