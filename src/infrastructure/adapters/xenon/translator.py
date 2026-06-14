"""Translate a xenon ``PriceData`` dict to an apex tick dict.

The only semantic conversion is ISO-string timestamp -> tz-aware datetime; the
aggregator's dict path reads ``symbol``/``last``/``bid``/``ask``/``volume``/
``timestamp`` directly (see BarAggregator._extract_*). Ticks with no usable
price (last/mid/bid+ask all absent) are dropped so the bus is not spammed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def _parse_ts(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _has_price(data: Dict[str, Any]) -> bool:
    if data.get("last") is not None:
        return True
    if data.get("mid") is not None:
        return True
    return data.get("bid") is not None and data.get("ask") is not None


def translate_price_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return an apex tick dict, or ``None`` if the tick is unusable."""
    symbol = data.get("symbol")
    if not symbol:
        return None
    if not _has_price(data):
        return None
    return {
        "symbol": symbol,
        "last": data.get("last"),
        "bid": data.get("bid"),
        "ask": data.get("ask"),
        "volume": data.get("volume"),
        "timestamp": _parse_ts(data.get("timestamp")),
    }
