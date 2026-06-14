"""Shared signal_service_payload field normalization.

Both egress paths must coerce domain/DB-native values into the argon contract
vocabulary before ``validate_payload``:

* the WS emitter (live) translates bus *events* (direction "LONG"/"SHORT"/"FLAT",
  strength as float);
* the REST/snapshot builder translates ``ta_signals`` *rows* (same event-native
  direction, plus JSONB ``metadata`` that asyncpg hands back as a JSON *string*).

Keeping the coercion here stops the two paths from drifting -- a past drift let
the REST/snapshot path emit event-native "LONG", which the schema rejects.
"""

from __future__ import annotations

import json
from typing import Any

# TradingSignalEvent.from_signal maps contract buy/sell/alert -> event
# LONG/SHORT/FLAT (domain_events.py). To re-emit the contract we invert it.
_DIRECTION_TO_CONTRACT = {"LONG": "buy", "SHORT": "sell", "FLAT": "alert"}
_CONTRACT_DIRECTIONS = {"buy", "sell", "alert"}


def _enum_value(value: Any) -> Any:
    """Unwrap an Enum to its ``.value``; pass plain values through."""
    return value.value if hasattr(value, "value") else value


def to_contract_direction(raw: Any) -> Any:
    """Normalise any direction to the contract's buy/sell/alert.

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


def to_contract_strength(raw: Any) -> Any:
    """The contract requires an integer 0-100; events cast strength to float."""
    try:
        return max(0, min(100, int(round(float(raw)))))
    except (TypeError, ValueError, OverflowError):
        # Non-numeric or non-finite (inf/nan) -> return as-is so the schema
        # validator rejects it, rather than raising past the emitter's handler.
        return raw


def decode_metadata(raw: Any) -> Any:
    """asyncpg returns JSONB as a str; the contract requires an object.

    Decode JSON strings to dicts; pass dicts/None through; leave anything
    undecodable for the validator to reject (no silent guess).
    """
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw
    return raw
