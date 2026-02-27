"""Tests for ServerPipeline — wires tick → bar → indicator → signal → WS hub."""

import asyncio
from datetime import datetime, timezone

import pytest

from src.domain.events.domain_events import QuoteTick
from src.server.pipeline import ServerPipeline


class MockHub:
    """Tracks broadcast calls."""

    def __init__(self):
        self.bars_broadcast = 0
        self.indicators_broadcast = 0
        self.signals_broadcast = 0

    async def broadcast_bar(self, symbol, tf, bar):
        self.bars_broadcast += 1

    async def broadcast_indicator(self, symbol, tf, name, value):
        self.indicators_broadcast += 1

    async def broadcast_signal(self, symbol, signal):
        self.signals_broadcast += 1


def make_tick(symbol: str, price: float, ts: datetime) -> QuoteTick:
    return QuoteTick(
        symbol=symbol,
        last=price,
        volume=100,
        source="test",
        timestamp=ts,
    )


def test_pipeline_creates():
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1m"])
    assert pipeline is not None
    assert len(pipeline._aggregators) == 1


@pytest.mark.asyncio
async def test_pipeline_on_tick_feeds_aggregators():
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1m", "5m"])
    await pipeline.start()

    tick = make_tick("AAPL", 185.5, datetime(2026, 2, 25, 14, 30, 0, tzinfo=timezone.utc))
    pipeline.on_tick(tick)
    # Pipeline should have aggregators for each timeframe
    assert len(pipeline._aggregators) == 2

    await pipeline.stop()


@pytest.mark.asyncio
async def test_pipeline_start_stop():
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1m"])
    await pipeline.start()
    assert pipeline._started
    await pipeline.stop()
    assert not pipeline._started


@pytest.mark.asyncio
async def test_pipeline_bar_close_triggers_broadcast():
    """When enough ticks close a bar, the hub should get a broadcast."""
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1m"])
    await pipeline.start()

    # Generate ticks spanning a full minute to close a bar
    base = datetime(2026, 2, 25, 14, 30, 0, tzinfo=timezone.utc)
    for i in range(5):
        tick = make_tick("AAPL", 185.0 + i * 0.1, base.replace(second=i * 12))
        pipeline.on_tick(tick)

    # Now send a tick in the NEXT minute to close the first bar
    next_min = base.replace(minute=31, second=0)
    pipeline.on_tick(make_tick("AAPL", 186.0, next_min))

    # Give the async event bus time to process events and schedule broadcasts
    await asyncio.sleep(0.5)

    assert hub.bars_broadcast >= 1

    await pipeline.stop()
