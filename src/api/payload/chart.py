"""Build chart read-surface payloads: bars, indicator series, confluence.

These mirror ``payload/builder.py`` (ISO timestamps, ``count``, ``generated_at``) and
are validated on egress against their schemas under config/verification/schemas/.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List

_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w")


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _bar_to_dict(bar: Any) -> Dict[str, Any]:
    # livewire bars set timestamp == bar_start; prefer timestamp, fall back to bar_start.
    when = bar.timestamp if getattr(bar, "timestamp", None) is not None else bar.bar_start
    return {
        "time": _iso(when),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "vwap": bar.vwap,
    }


def build_bars_payload(
    symbol: str, timeframe: str, bars: Iterable[Any], *, generated_at: datetime
) -> Dict[str, Any]:
    rows = [_bar_to_dict(b) for b in bars]
    return {
        "symbol": symbol,
        "timeframe": timeframe,
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
    out: List[Dict[str, Any]] = []
    for r in rows:
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
