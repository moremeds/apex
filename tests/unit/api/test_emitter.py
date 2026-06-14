"""Emitter maps a fired signal to a valid payload and broadcasts it."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.api.payload.validate import validate_payload
from src.api.ws.emitter import SignalEmitter, signal_to_payload
from src.api.ws.hub import SignalHub


def _signal():
    return SimpleNamespace(
        signal_id="momentum:rsi:AAPL:1d",
        symbol="AAPL",
        timeframe="1d",
        category="momentum",
        indicator="RSI",
        direction="buy",
        strength=70,
        priority="high",
        trigger_rule="rsi_cross",
        current_value=28.0,
        threshold=30.0,
        previous_value=32.0,
        message="",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc),
    )


def test_signal_to_payload_is_schema_valid() -> None:
    payload = signal_to_payload(_signal())
    validate_payload(payload)
    assert payload["signals"][0]["signal_id"] == "momentum:rsi:AAPL:1d"


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, d: dict) -> None:
        self.sent.append(d)


@pytest.mark.asyncio
async def test_emitter_broadcasts_on_event() -> None:
    hub = SignalHub()
    ws = _FakeWS()
    hub.register(ws, "AAPL")
    emitter = SignalEmitter(hub)
    await emitter.on_trading_signal(_signal())
    assert ws.sent[0]["signals"][0]["symbol"] == "AAPL"


def _real_event():
    """A real TradingSignalEvent as published on the bus.

    Built via TradingSignalEvent.from_signal from a real domain TradingSignal, so
    it carries the event-native vocabulary the emitter must translate back to the
    contract: direction "LONG" (from "buy"), strength as float, priority/category
    as enum .value strings.
    """
    from src.domain.events.domain_events import TradingSignalEvent
    from src.domain.signals.models import (
        SignalCategory,
        SignalDirection,
        SignalPriority,
        TradingSignal,
    )

    signal = TradingSignal(
        signal_id="momentum:rsi:AAPL:1d",
        symbol="AAPL",
        category=SignalCategory.MOMENTUM,
        indicator="rsi",
        direction=SignalDirection.BUY,
        strength=72,
        priority=SignalPriority.HIGH,
        timeframe="1d",
        trigger_rule="rsi_oversold_exit",
        current_value=31.4,
        threshold=30.0,
        previous_value=28.0,
        message="RSI exits oversold",
        timestamp=datetime(2026, 6, 14, 12, 5, tzinfo=timezone.utc),
    )
    return TradingSignalEvent.from_signal(signal, source="rule_engine")


def test_real_trading_signal_event_maps_to_valid_payload() -> None:
    """The emitter must translate a real bus event into a contract-valid payload."""
    event = _real_event()
    # Precondition: the event really does carry event-native vocabulary.
    assert event.direction == "LONG"
    assert isinstance(event.strength, float)

    payload = signal_to_payload(event)
    validate_payload(payload)  # would raise (and was dropped) before the fix

    sig = payload["signals"][0]
    assert sig["direction"] == "buy"
    assert sig["strength"] == 72 and isinstance(sig["strength"], int)
    assert sig["priority"] == "high"
    assert sig["category"] == "momentum"
    assert sig["signal_id"] == "momentum:rsi:AAPL:1d"


def test_enum_direction_is_unwrapped_to_contract() -> None:
    """A SignalDirection enum (raw-signal wiring) maps to its contract string."""
    from src.domain.signals.models import SignalDirection

    sig = _signal()
    sig.direction = SignalDirection.SELL
    payload = signal_to_payload(sig)
    validate_payload(payload)
    assert payload["signals"][0]["direction"] == "sell"


@pytest.mark.asyncio
async def test_emitter_delivers_real_event_to_socket() -> None:
    """End-to-end: a real fired event reaches a subscribed socket as a valid frame."""
    hub = SignalHub()
    ws = _FakeWS()
    hub.register(ws, "AAPL")
    emitter = SignalEmitter(hub)
    await emitter.on_trading_signal(_real_event())
    assert len(ws.sent) == 1, "real event was dropped instead of broadcast"
    validate_payload(ws.sent[0])
    assert ws.sent[0]["signals"][0]["direction"] == "buy"
