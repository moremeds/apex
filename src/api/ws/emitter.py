"""Subscribe to the event bus and push fired signals to the WS hub.

Does NOT persist -- TASignalService._persist_signal already does. This is the
broadcast-only fan-out from the event bus to connected argon clients.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from src.api.payload.validate import ValidationFailure, validate_payload

logger = logging.getLogger(__name__)

# Fields copied straight off the fired event (already contract-valued: category
# and priority arrive as enum .value strings, see TradingSignalEvent.from_signal).
_PASSTHROUGH = (
    "signal_id",
    "symbol",
    "category",
    "indicator",
    "priority",
    "timeframe",
    "trigger_rule",
    "current_value",
    "threshold",
    "previous_value",
    "message",
)

# TradingSignalEvent.from_signal converts the contract direction into an event
# direction via {"buy": "LONG", "sell": "SHORT", "alert": "FLAT"} (domain_events.py).
# To emit the contract again we apply the exact inverse.
_DIRECTION_TO_CONTRACT = {"LONG": "buy", "SHORT": "sell", "FLAT": "alert"}
_CONTRACT_DIRECTIONS = {"buy", "sell", "alert"}


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, datetime) else v


def _enum_value(v: Any) -> Any:
    """Unwrap an Enum to its .value; pass plain values through."""
    return v.value if hasattr(v, "value") else v


def _to_contract_direction(raw: Any) -> Any:
    """Normalise a fired-event direction to the contract's buy/sell/alert.

    Handles SignalDirection enums (already buy/sell/alert), the event-native
    LONG/SHORT/FLAT strings, and already-contract strings. Unknown values are
    returned unchanged so the schema validator rejects them (no silent guess).
    """
    val = _enum_value(raw)
    if not isinstance(val, str):
        return val
    if val in _CONTRACT_DIRECTIONS:
        return val
    return _DIRECTION_TO_CONTRACT.get(val.upper(), val)


def _to_contract_strength(raw: Any) -> Any:
    """The contract requires an integer 0-100; the event casts strength to float."""
    try:
        return max(0, min(100, int(round(float(raw)))))
    except (TypeError, ValueError):
        return raw  # let the validator reject a non-numeric strength


def signal_to_payload(signal: Any) -> dict:
    """Map a fired TradingSignalEvent to a one-signal signal_service_payload.

    The bus event carries event-native vocabulary (direction LONG/SHORT/FLAT,
    strength as float); this translates it back to the argon contract before
    validation. Null/absent optional fields are omitted.
    """
    sig: dict = {
        f: _iso(getattr(signal, f, None))
        for f in _PASSTHROUGH
        if getattr(signal, f, None) is not None
    }
    direction = _to_contract_direction(getattr(signal, "direction", None))
    if direction is not None:
        sig["direction"] = direction
    strength = getattr(signal, "strength", None)
    if strength is not None:
        sig["strength"] = _to_contract_strength(strength)
    ts = getattr(signal, "timestamp", None) or datetime.now(timezone.utc)
    sig["timestamp"] = _iso(ts)
    return {"signals": [sig], "timestamp": _iso(ts), "symbol_count": 1}


class SignalEmitter:
    def __init__(self, hub: Any) -> None:
        self._hub = hub

    async def on_trading_signal(self, payload: Any) -> None:
        signal = getattr(payload, "signal", payload)  # unwrap event wrapper
        try:
            out = signal_to_payload(signal)
            validate_payload(out)
        except ValidationFailure as exc:
            logger.warning("dropping invalid signal payload: %s", exc)
            return
        symbol = getattr(signal, "symbol", None)
        if symbol:
            await self._hub.broadcast(symbol, out)

    def subscribe(self, event_bus: Any) -> None:
        """Subscribe on_trading_signal to the bus's TRADING_SIGNAL event."""
        from src.domain.events.event_types import EventType

        event_bus.subscribe(EventType.TRADING_SIGNAL, self._dispatch)

    def _dispatch(self, payload: Any) -> None:
        # Bus handlers are sync; schedule the async broadcast.
        asyncio.create_task(self.on_trading_signal(payload))
