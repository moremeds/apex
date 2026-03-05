"""Shared constants and helpers for the server signal pipeline.

Extracted from ServerPipeline to be reusable by WebBridge and route handlers.
"""

from __future__ import annotations

from typing import Any

from src.domain.events.domain_events import QuoteTick

# Strategy indicators whose full state dict should be broadcast via WS
STRATEGY_INDICATORS = {"dual_macd", "trend_pulse", "regime_detector"}

# Regime short-code → target exposure for RegimeFlex mapping
_REGIME_EXPOSURE = {"R0": 1.0, "R1": 0.5, "R2": 0.0, "R3": 0.25}

# Direction mapping: upstream → frontend-friendly labels
DIRECTION_MAP = {
    "buy": "bullish",
    "sell": "bearish",
    "alert": "neutral",
    "LONG": "bullish",
    "SHORT": "bearish",
    "FLAT": "neutral",
    "long": "bullish",
    "short": "bearish",
    "flat": "neutral",
}


def map_regime_to_flex(state: dict) -> dict:
    """Map regime_detector state → RegimeFlexRow format for frontend."""
    regime_full = state.get("regime", "R1_CHOPPY_EXTENDED")
    regime_short = regime_full.split("_")[0] if "_" in str(regime_full) else str(regime_full)
    signal = "NONE"
    if state.get("regime_changed"):
        prev = (state.get("previous_regime") or "").split("_")[0]
        signal = f"{prev}→{regime_short}"
    return {
        "date": state.get("date", ""),
        "regime": regime_short,
        "target_exposure": _REGIME_EXPOSURE.get(regime_short, 0.5),
        "signal": signal,
    }


def tick_to_dict(tick: QuoteTick) -> dict[str, Any]:
    """Convert QuoteTick to JSON-serializable dict for WS broadcast."""
    return {
        "last": tick.last,
        "bid": tick.bid,
        "ask": tick.ask,
        "volume": tick.volume,
        "ts": tick.timestamp.isoformat() if tick.timestamp else None,
    }


def format_ts(ts: Any) -> str:
    """Format a timestamp to ISO string."""
    if hasattr(ts, "isoformat"):
        return str(ts.isoformat())
    return str(ts)
