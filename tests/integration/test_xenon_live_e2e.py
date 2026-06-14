from __future__ import annotations

import asyncio

import pytest

from src.application.services.ta_signal_service import TASignalService
from src.domain.events.event_types import EventType
from src.domain.events.priority_event_bus import PriorityEventBus
from src.infrastructure.adapters.xenon.client import XenonTickClient
from tests.support.fake_xenon import FakeXenonServer


@pytest.mark.asyncio
async def test_live_tick_drives_bar_close_through_real_pipeline() -> None:
    # The bus is intentionally NOT started: PriorityEventBus.publish has a
    # documented sync fallback (`if not self._running: self._dispatch_sync(...)`,
    # priority_event_bus.py:195) so every publish dispatches synchronously to
    # subscribers -- deterministic, exercising the real handler chain. Do NOT
    # call `await bus.start()`: that switches to async lanes and makes this racy.
    bus = PriorityEventBus()
    closed: list = []
    bus.subscribe(EventType.BAR_CLOSE, lambda ev: closed.append(ev))

    service = TASignalService(event_bus=bus, timeframes=["1m"])
    await service.start()

    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, bus, reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)

        # Two ticks two minutes apart -> the first 1m bar closes when the second
        # tick opens a new bar window.
        await server.push(
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
        await server.push(
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

        await client.close()

    await service.stop()

    assert closed, "no BAR_CLOSE produced from live ticks (publish->pipeline seam broken)"
    assert closed[0].symbol == "AAPL"
    assert closed[0].timeframe == "1m"
