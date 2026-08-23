"""Build chart read-surface payloads: bars, indicator series, confluence.

These mirror ``payload/builder.py`` (ISO timestamps, ``count``, ``generated_at``) and
are validated on egress against their schemas under config/verification/schemas/.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List


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


def _bar_to_dict(bar: Any) -> Dict[str, Any]:
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
    # Futures-only columns. Omitted entirely for every other class rather than
    # emitted as null noise on ~20M equity bars.
    for extra in ("settlement", "open_interest"):
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
    rows = [_bar_to_dict(b) for b in bars]
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
