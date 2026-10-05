from __future__ import annotations

import pytest

from src.api.ws.hub import SignalHub


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)


@pytest.mark.asyncio
async def test_broadcast_only_to_subscribers() -> None:
    hub = SignalHub()
    a, b = _FakeWS(), _FakeWS()
    hub.register(a, "AAPL")
    hub.register(b, "TSLA")

    await hub.broadcast("AAPL", {"signals": [], "timestamp": "t"})
    assert len(a.sent) == 1
    assert len(b.sent) == 0


@pytest.mark.asyncio
async def test_unregister_one_ticker_returns_it_and_keeps_others() -> None:
    hub = SignalHub()
    a = _FakeWS()
    hub.register(a, "AAPL")
    hub.register(a, "TSLA")

    removed = hub.unregister(a, "AAPL")
    assert removed == {"AAPL"}
    await hub.broadcast("AAPL", {"signals": [], "timestamp": "t"})
    await hub.broadcast("TSLA", {"signals": [], "timestamp": "t"})
    assert len(a.sent) == 1  # still subscribed to TSLA only


@pytest.mark.asyncio
async def test_unregister_all_returns_full_ticker_set() -> None:
    hub = SignalHub()
    a = _FakeWS()
    hub.register(a, "AAPL")
    hub.register(a, "TSLA")

    removed = hub.unregister(a)  # ticker=None -> remove everything (disconnect path)
    assert removed == {"AAPL", "TSLA"}
    await hub.broadcast("AAPL", {"signals": [], "timestamp": "t"})
    assert a.sent == []


def test_register_reports_only_new_registrations() -> None:
    hub = SignalHub()
    a = _FakeWS()
    assert hub.register(a, "AAPL") is True
    assert hub.register(a, "AAPL") is False
    assert hub.unregister(a) == {"AAPL"}


class _DeadWS(_FakeWS):
    async def send_json(self, data: dict) -> None:
        raise RuntimeError("socket closed")


@pytest.mark.asyncio
async def test_dead_socket_is_muted_but_still_reported_on_unregister() -> None:
    hub = SignalHub()
    dead = _DeadWS()
    hub.register(dead, "AAPL")

    await hub.broadcast("AAPL", {"signals": [], "timestamp": "t"})
    assert hub._by_ticker["AAPL"] == set()  # no more fan-out to it
    # The handler's cleanup still sees AAPL, so its manager refcount is released.
    assert hub.unregister(dead) == {"AAPL"}


@pytest.mark.asyncio
async def test_repeat_register_unmutes_a_muted_socket() -> None:
    hub = SignalHub()
    flaky = _DeadWS()
    hub.register(flaky, "AAPL")
    await hub.broadcast("AAPL", {"signals": [], "timestamp": "t"})  # send fails -> muted

    assert hub.register(flaky, "AAPL") is False  # no second refcount
    assert hub._by_ticker["AAPL"] == {flaky}  # but fan-out resumes
