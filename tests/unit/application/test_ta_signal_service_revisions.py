from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, cast

import pytest

from src.application.services.ta_signal_service import TASignalService
from src.domain.events.domain_events import BarCloseEvent
from src.domain.events.event_types import EventType
from src.domain.signals.data.bar_aggregator import BarAggregator
from src.domain.signals.indicator_engine import IndicatorEngine
from src.domain.signals.rule_engine import RuleEngine, RuleRegistry
from src.domain.signals.rules.short_timeframe_rules import SHORT_TIMEFRAME_RULES


class _FakeIndicatorEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def replace_symbol_histories(
        self, symbol: str, histories: dict, live_tail_timeframes: Any = None
    ) -> dict[str, int]:
        self.calls.append((symbol, histories))
        return {timeframe: len(rows) for timeframe, rows in histories.items()}

    async def compute_on_history(self, symbol: str, timeframe: str, publish: bool = True) -> int:
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
    service._bar_aggregators = {"1m": cast(Any, aggregator)}
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


def test_refresh_overflow_then_abort_replays_captured_ticks() -> None:
    service, aggregator = _running_service(max_ticks=1)
    now = datetime.now(timezone.utc)
    service.begin_symbol_refresh("NVDA")
    service._on_market_data_tick({"symbol": "NVDA", "timestamp": now})
    service._on_market_data_tick({"symbol": "NVDA", "timestamp": now + timedelta(seconds=1)})

    with pytest.raises(RuntimeError, match="tick buffer exceeded"):
        service.commit_symbol_refresh("NVDA")
    service.abort_symbol_refresh("NVDA")  # what SubscriptionManager does on failure

    assert [tick["timestamp"] for tick in aggregator.ticks] == [now]
    service.begin_symbol_refresh("NVDA")  # refresh state was cleared


class _SyncBus:
    def __init__(self) -> None:
        self._subs: dict[Any, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: Any, callback: Callable[[Any], None]) -> None:
        self._subs[event_type].append(callback)

    def publish(self, event_type: Any, payload: Any) -> None:
        for callback in self._subs[event_type]:
            callback(payload)


def test_overflow_abort_replay_after_replace_keeps_tail_without_duplicates() -> None:
    """Real aggregator + engine: Silver replace keeps the live tail, then an
    overflowed refresh aborts and its replayed ticks close each bar exactly once."""
    bus = _SyncBus()
    engine = IndicatorEngine(bus, max_workers=1)
    engine._indicators = []  # history only; skip indicator compute
    engine.start()
    service = TASignalService(event_bus=bus, refresh_buffer_max_ticks=1)
    service._bar_aggregators = {"1d": BarAggregator("1d", bus)}
    service._running = True
    d = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def tick(day: int, hour: int) -> None:
        ts = d + timedelta(days=day, hours=hour)
        service._on_market_data_tick({"symbol": "NVDA", "price": 100.0 + day, "timestamp": ts})

    for day in (0, 1, 2):
        tick(day, 15)  # live closes day D and D+1; D+2 stays open
    service.begin_symbol_refresh("NVDA")
    tick(2, 16)  # captured
    tick(3, 15)  # overflow: dropped
    lake = [{"timestamp": d + timedelta(days=k), "close": 1.0} for k in (-1, 0)]
    engine.replace_symbol_histories("NVDA", {"1d": lake})
    with pytest.raises(RuntimeError, match="tick buffer exceeded"):
        service.commit_symbol_refresh("NVDA")
    service.abort_symbol_refresh("NVDA")
    tick(3, 15)  # first post-refresh tick closes live D+2

    stamps = [bar["timestamp"] for bar in engine.get_history("NVDA", "1d") or []]
    # lake D-1, lake D (live D dropped as its duplicate), live D+1, live D+2
    assert stamps == [d + timedelta(days=k) for k in (-1, 0, 2, 3)]


async def test_seed_sets_a_baseline_without_evaluating_rules() -> None:
    """A subscribe emits no INDICATOR_UPDATE, so no rule runs: no TRADING_SIGNAL, hence no
    persisted signal row and no WS frame. A detect_initial rule would fire on any first
    evaluation; the next live close sees the seed baseline as its previous state."""
    bus = _SyncBus()
    updates: list[Any] = []
    signals: list[Any] = []
    bus.subscribe(EventType.INDICATOR_UPDATE, updates.append)
    bus.subscribe(EventType.TRADING_SIGNAL, signals.append)
    engine = IndicatorEngine(bus, max_workers=1)
    engine._indicators = [ind for ind in engine._indicators if ind.name == "rsi"]
    engine.start()
    always = replace(
        next(r for r in SHORT_TIMEFRAME_RULES if r.name == "rsi_st_extreme_overbought"),
        timeframes=("1d",),
        condition_config={"field": "value", "threshold": 0, "detect_initial": True},
        cooldown_seconds=0,
    )
    registry = RuleRegistry()
    registry.add_rule(always)
    RuleEngine(bus, registry).start()
    service = TASignalService(event_bus=bus)
    service._indicator_engine = engine
    d = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [
        {"timestamp": d + timedelta(days=i), "open": c, "high": c + 1, "low": c - 1, "close": c}
        for i, c in enumerate(100.0 + (i % 7) for i in range(40))
    ]

    await service.inject_historical_bars("NVDA", "1d", bars[:-1])
    await service.inject_historical_bars("NVDA", "1d", bars)  # resubscribe adds one bar
    covered = bars[-1]["timestamp"] + timedelta(days=1)  # live close the lake holds
    await engine._process_bar_async(
        BarCloseEvent(symbol="NVDA", timeframe="1d", close=1.0, bar_end=covered)
    )
    await asyncio.sleep(0.05)  # let rule-evaluation tasks run

    assert updates == [] and signals == []
    assert ("NVDA", "1d", "rsi") in engine._previous_states


def test_buffer_is_thread_safe_under_concurrent_ticks() -> None:
    """Ticks arriving from many threads (as they would if the tick handler is
    ever dispatched via the event bus' heavy-callback thread pool) must not
    corrupt the buffer. All captured ticks replay exactly once on commit."""
    import threading

    service, aggregator = _running_service(max_ticks=10_000)
    base = datetime(2026, 7, 12, tzinfo=timezone.utc)
    service.begin_symbol_refresh("NVDA")

    tick_count = 500
    barrier = threading.Barrier(8)

    def fire(start: int) -> None:
        barrier.wait()  # maximize overlap
        for i in range(start, tick_count, 8):
            service._on_market_data_tick(
                {"symbol": "NVDA", "timestamp": base + timedelta(seconds=i)}
            )

    threads = [threading.Thread(target=fire, args=(offset,)) for offset in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert aggregator.ticks == []  # nothing dispatched while buffering
    service.commit_symbol_refresh("NVDA")
    # Every tick captured exactly once, and replayed in event-time order.
    seconds = [tick["timestamp"] for tick in aggregator.ticks]
    assert len(seconds) == tick_count
    assert seconds == sorted(seconds)
