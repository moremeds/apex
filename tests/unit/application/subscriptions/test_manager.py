from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.application.subscriptions.manager import Subscription, SubscriptionManager


def test_subscription_refcount_increments_and_decrements() -> None:
    sub = Subscription(symbol="AAPL")
    assert sub.refcount == 0
    assert sub.acquire() == 1
    assert sub.acquire() == 2
    assert sub.release() == 1
    assert sub.release() == 0


def test_release_below_zero_raises() -> None:
    sub = Subscription(symbol="AAPL")
    with pytest.raises(ValueError, match="refcount underflow"):
        sub.release()


def _fake_bar(close: float = 11.0):
    """A BarData-shaped object the manager can convert to a dict."""
    return SimpleNamespace(
        open=10.0,
        high=10.5,
        low=9.5,
        close=close,
        volume=100,
        bar_start="2026-01-02T00:00:00Z",
    )


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def fetch_bars(self, symbol, timeframe, **kw):
        self.calls.append((symbol, timeframe))
        return [_fake_bar()]


class _FakeCompute:
    """Stands in for TASignalService (matches the real async 3-arg signature)."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.injected: list[tuple[str, str, int]] = []

    async def start(self) -> None:
        self.started.append("all")

    async def stop(self) -> None:
        self.stopped.append("all")

    async def inject_historical_bars(self, symbol, timeframe, bars) -> int:
        self.injected.append((symbol, timeframe, len(bars)))
        return len(bars)


@pytest.mark.asyncio
async def test_subscribe_seeds_history_once_and_fans_out() -> None:
    provider, compute = _FakeProvider(), _FakeCompute()
    mgr = SubscriptionManager(provider=provider, compute=compute, timeframes=["1d"])

    await mgr.subscribe("AAPL")
    await mgr.subscribe("AAPL")  # second client, same ticker

    assert mgr.refcount("AAPL") == 2
    assert provider.calls == [("AAPL", "1d")]
    assert compute.injected == [("AAPL", "1d", 1)]
    assert mgr.active_symbols() == {"AAPL"}


@pytest.mark.asyncio
async def test_unsubscribe_teardown_at_zero() -> None:
    provider, compute = _FakeProvider(), _FakeCompute()
    mgr = SubscriptionManager(provider=provider, compute=compute, timeframes=["1d"])

    await mgr.subscribe("AAPL")
    await mgr.subscribe("AAPL")
    await mgr.unsubscribe("AAPL")
    assert mgr.refcount("AAPL") == 1
    assert "AAPL" in mgr.active_symbols()

    await mgr.unsubscribe("AAPL")
    assert mgr.refcount("AAPL") == 0
    assert "AAPL" not in mgr.active_symbols()


@pytest.mark.asyncio
async def test_unsubscribe_unknown_symbol_is_noop() -> None:
    mgr = SubscriptionManager(provider=_FakeProvider(), compute=_FakeCompute(), timeframes=["1d"])
    await mgr.unsubscribe("NOPE")  # must not raise
    assert mgr.refcount("NOPE") == 0


@pytest.mark.asyncio
async def test_concurrent_subscribe_seeds_exactly_once() -> None:
    import asyncio as _aio

    provider, compute = _FakeProvider(), _FakeCompute()
    mgr = SubscriptionManager(provider=provider, compute=compute, timeframes=["1d"])

    await _aio.gather(*[mgr.subscribe("AAPL") for _ in range(20)])
    assert mgr.refcount("AAPL") == 20
    assert provider.calls == [("AAPL", "1d")]  # seeded exactly once despite 20 racers


@pytest.mark.asyncio
async def test_failed_seed_does_not_poison_subscription() -> None:
    class _FlakyProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch_bars(self, symbol, timeframe, **kw):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("livewire unavailable")
            return [_fake_bar()]

    provider, compute = _FlakyProvider(), _FakeCompute()
    mgr = SubscriptionManager(provider=provider, compute=compute, timeframes=["1d"])

    with pytest.raises(RuntimeError):
        await mgr.subscribe("AAPL")
    assert mgr.refcount("AAPL") == 0
    assert "AAPL" not in mgr.active_symbols()

    await mgr.subscribe("AAPL")  # retry succeeds
    assert mgr.refcount("AAPL") == 1
    assert compute.injected == [("AAPL", "1d", 1)]


# --- Phase 4: live-feed wiring -------------------------------------------------


class _FakeLiveFeed:
    def __init__(self) -> None:
        self.subscribed: list = []
        self.unsubscribed: list = []
        self.fail_subscribe = False
        self.fail_unsubscribe = False

    async def connect(self) -> None: ...

    async def subscribe(self, symbol: str) -> None:
        if self.fail_subscribe:
            raise RuntimeError("feed down")
        self.subscribed.append(symbol)

    async def unsubscribe(self, symbol: str) -> None:
        if self.fail_unsubscribe:
            raise RuntimeError("feed down")
        self.unsubscribed.append(symbol)

    async def close(self) -> None: ...


@pytest.mark.asyncio
async def test_subscribe_opens_live_feed() -> None:
    feed = _FakeLiveFeed()
    mgr = SubscriptionManager(provider=_FakeProvider(), compute=_FakeCompute(), timeframes=["1d"])
    mgr.set_live_feed(feed)
    await mgr.subscribe("AAPL")
    assert feed.subscribed == ["AAPL"]


@pytest.mark.asyncio
async def test_second_subscriber_does_not_reopen_feed() -> None:
    feed = _FakeLiveFeed()
    mgr = SubscriptionManager(provider=_FakeProvider(), compute=_FakeCompute(), timeframes=["1d"])
    mgr.set_live_feed(feed)
    await mgr.subscribe("AAPL")
    await mgr.subscribe("AAPL")  # refcount 1 -> 2
    assert feed.subscribed == ["AAPL"]  # opened once only


@pytest.mark.asyncio
async def test_unsubscribe_drops_live_feed_at_refcount_zero() -> None:
    feed = _FakeLiveFeed()
    mgr = SubscriptionManager(provider=_FakeProvider(), compute=_FakeCompute(), timeframes=["1d"])
    mgr.set_live_feed(feed)
    await mgr.subscribe("AAPL")
    await mgr.unsubscribe("AAPL")
    assert feed.unsubscribed == ["AAPL"]


@pytest.mark.asyncio
async def test_subscribe_feed_failure_leaves_no_poisoned_entry() -> None:
    feed = _FakeLiveFeed()
    feed.fail_subscribe = True
    mgr = SubscriptionManager(provider=_FakeProvider(), compute=_FakeCompute(), timeframes=["1d"])
    mgr.set_live_feed(feed)
    with pytest.raises(RuntimeError):
        await mgr.subscribe("AAPL")
    assert mgr.refcount("AAPL") == 0
    assert "AAPL" not in mgr.active_symbols()


@pytest.mark.asyncio
async def test_unsubscribe_feed_failure_still_clears_local_state() -> None:
    feed = _FakeLiveFeed()
    feed.fail_unsubscribe = True
    mgr = SubscriptionManager(provider=_FakeProvider(), compute=_FakeCompute(), timeframes=["1d"])
    mgr.set_live_feed(feed)
    await mgr.subscribe("AAPL")
    await mgr.unsubscribe("AAPL")  # remote drop raises, but must not propagate
    assert mgr.refcount("AAPL") == 0
    assert "AAPL" not in mgr.active_symbols()
