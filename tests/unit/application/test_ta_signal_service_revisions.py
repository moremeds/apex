from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.application.services.ta_signal_service import TASignalService


class _FakeIndicatorEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def replace_symbol_histories(self, symbol: str, histories: dict) -> dict[str, int]:
        self.calls.append((symbol, histories))
        return {timeframe: len(rows) for timeframe, rows in histories.items()}

    async def compute_on_history(self, symbol: str, timeframe: str) -> int:
        return 0


@pytest.mark.asyncio
async def test_service_delegates_symbol_history_replacement() -> None:
    service = TASignalService(event_bus=object())
    engine = _FakeIndicatorEngine()
    service._indicator_engine = engine
    histories = {"1d": [{"timestamp": "2026-01-02", "close": 100.0}]}

    result = await service.replace_symbol_histories("NVDA", histories)

    assert result == {"1d": 1}
    assert engine.calls == [("NVDA", histories)]


@pytest.mark.asyncio
async def test_service_replacement_requires_initialized_engine() -> None:
    service = TASignalService(event_bus=object())

    with pytest.raises(RuntimeError, match="IndicatorEngine not initialized"):
        await service.replace_symbol_histories("NVDA", {"1d": []})


class _RecordingAggregator:
    def __init__(self) -> None:
        self.ticks: list[dict] = []

    def on_tick(self, tick: dict) -> None:
        self.ticks.append(tick)


def _running_service(*, max_ticks: int = 10_000) -> tuple[TASignalService, _RecordingAggregator]:
    service = TASignalService(event_bus=object(), refresh_buffer_max_ticks=max_ticks)
    aggregator = _RecordingAggregator()
    service._bar_aggregators = {"1m": aggregator}
    service._running = True
    return service, aggregator


def test_refresh_buffers_and_replays_ticks_in_event_time_order() -> None:
    service, aggregator = _running_service()
    base = datetime(2026, 7, 12, tzinfo=timezone.utc)
    service.begin_symbol_refresh("NVDA")

    service._on_market_data_tick({"symbol": "NVDA", "timestamp": base + timedelta(seconds=2)})
    service._on_market_data_tick({"symbol": "NVDA", "timestamp": base + timedelta(seconds=1)})

    assert aggregator.ticks == []
    service.commit_symbol_refresh("NVDA")
    assert [tick["timestamp"] for tick in aggregator.ticks] == [
        base + timedelta(seconds=1),
        base + timedelta(seconds=2),
    ]


def test_refresh_passes_unrelated_ticks_through_immediately() -> None:
    service, aggregator = _running_service()
    service.begin_symbol_refresh("NVDA")

    service._on_market_data_tick({"symbol": "AAPL", "timestamp": datetime.now(timezone.utc)})

    assert [tick["symbol"] for tick in aggregator.ticks] == ["AAPL"]
    service.abort_symbol_refresh("NVDA")


def test_refresh_buffer_overflow_is_explicit() -> None:
    service, _ = _running_service(max_ticks=1)
    now = datetime.now(timezone.utc)
    service.begin_symbol_refresh("NVDA")
    service._on_market_data_tick({"symbol": "NVDA", "timestamp": now})
    service._on_market_data_tick({"symbol": "NVDA", "timestamp": now + timedelta(seconds=1)})

    with pytest.raises(RuntimeError, match="tick buffer exceeded"):
        service.commit_symbol_refresh("NVDA")
