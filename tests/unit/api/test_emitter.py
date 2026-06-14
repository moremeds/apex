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
