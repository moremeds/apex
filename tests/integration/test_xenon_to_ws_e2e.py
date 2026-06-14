"""End-to-end: a client on /ws/signals gets live TA signals from a running apex.

Boots the REAL ASGI app via uvicorn on an ephemeral port (so lifespan runs and
builds the full pipeline), points it at a fake xenon server + a temp livewire
bronze root, then drives the whole chain through real sockets:

  * subscribe over /ws/signals  -> real SubscriptionManager seeds history from the
    real LivewireOhlcProvider AND opens the xenon live sub (fake xenon sees it);
  * a live xenon tick           -> real pipeline -> BAR_CLOSE on the app's bus;
  * a fired TRADING_SIGNAL      -> real SignalEmitter -> hub -> a signal_service_payload
    frame on the client's socket.

The tick->indicator->rule->TRADING_SIGNAL hop is covered by the rule-engine unit
tests and test_xenon_live_e2e (tick->BAR_CLOSE); this test proves the *wiring* of
the deployed server end-to-end.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import uvicorn
from websockets.asyncio.client import connect

from src.api.payload.validate import validate_payload
from src.api.server import create_app
from src.domain.events.event_types import EventType
from tests.support.fake_xenon import FakeXenonServer


def _write_seed(root, symbol: str = "AAPL") -> None:
    """A few recent 1m bars so the real provider seed returns data (within lookback)."""
    sym_dir = root / "asset_class=equity" / f"symbol={symbol}"
    sym_dir.mkdir(parents=True, exist_ok=True)
    base = datetime.now(timezone.utc) - timedelta(minutes=5)
    ts = [base + timedelta(minutes=i) for i in range(3)]
    pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, utc=True),
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [11.0, 12.0, 13.0],
            "volume": [100, 200, 300],
            "asset_class": ["equity"] * 3,
            "symbol": [symbol] * 3,
        }
    ).to_parquet(sym_dir / "1m.parquet")


def _trading_signal_event():
    from src.domain.events.domain_events import TradingSignalEvent
    from src.domain.signals.models import (
        SignalCategory,
        SignalDirection,
        SignalPriority,
        TradingSignal,
    )

    sig = TradingSignal(
        signal_id="momentum:rsi:AAPL:1m",
        symbol="AAPL",
        category=SignalCategory.MOMENTUM,
        indicator="rsi",
        direction=SignalDirection.BUY,
        strength=72,
        priority=SignalPriority.HIGH,
        timeframe="1m",
        trigger_rule="rsi_oversold_exit",
        current_value=31.4,
        threshold=30.0,
        previous_value=28.0,
        message="RSI exits oversold",
        timestamp=datetime.now(timezone.utc),
    )
    return TradingSignalEvent.from_signal(sig, source="rule_engine")


async def _recv_json(ws, timeout: float = 5.0) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout))


@pytest.mark.asyncio
async def test_xenon_tick_to_ws_signal_through_running_server(monkeypatch, tmp_path) -> None:
    _write_seed(tmp_path, "AAPL")

    async with FakeXenonServer() as xenon:
        monkeypatch.setenv("APEX_LIVEWIRE_ROOT", str(tmp_path))
        monkeypatch.setenv("APEX_XENON_WS_URL", xenon.url)
        monkeypatch.setenv("APEX_TIMEFRAMES", "1m")
        monkeypatch.delenv("APEX_PG_URL", raising=False)

        app = create_app()
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = uvicorn.Server(config)
        serve_task = asyncio.create_task(server.serve())
        try:
            # wait until the server is up, then discover its ephemeral port
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(0.05)
            assert server.started, "uvicorn did not start"
            port = server.servers[0].sockets[0].getsockname()[1]

            # probe the running app's bus to prove live ticks reach compute
            closed: list = []
            app.state.event_bus.subscribe(EventType.BAR_CLOSE, lambda ev: closed.append(ev))

            async with connect(f"ws://127.0.0.1:{port}/ws/signals") as ws:
                await ws.send(json.dumps({"action": "subscribe", "ticker": "AAPL"}))
                ack = await _recv_json(ws)
                assert ack == {"status": "subscribed", "ticker": "AAPL"}

                # subscribe drove the real SubscriptionManager -> opened the xenon sub
                await xenon.wait_for_frames(1)
                assert {"action": "subscribe", "symbols": ["AAPL"]} in xenon.received

                # a live xenon tick reaches the real compute pipeline (-> BAR_CLOSE)
                await xenon.push(
                    {
                        "type": "batch",
                        "updates": {
                            "AAPL": {
                                "symbol": "AAPL",
                                "last": 100.0,
                                "volume": 5,
                                "timestamp": "2026-06-14T15:00:10Z",
                            }
                        },
                    }
                )
                await xenon.push(
                    {
                        "type": "batch",
                        "updates": {
                            "AAPL": {
                                "symbol": "AAPL",
                                "last": 101.0,
                                "volume": 7,
                                "timestamp": "2026-06-14T15:01:10Z",
                            }
                        },
                    }
                )
                for _ in range(200):
                    if closed:
                        break
                    await asyncio.sleep(0.01)
                assert closed, "live xenon tick did not reach the compute pipeline"
                assert closed[0].symbol == "AAPL"

                # a fired signal is delivered to the subscribed socket as a contract frame
                app.state.event_bus.publish(EventType.TRADING_SIGNAL, _trading_signal_event())
                frame = await _recv_json(ws)
                validate_payload(frame)
                sig = frame["signals"][0]
                assert sig["symbol"] == "AAPL"
                assert sig["direction"] == "buy"  # translated back from event "LONG"
                assert sig["signal_id"] == "momentum:rsi:AAPL:1m"
        finally:
            server.should_exit = True
            await serve_task
