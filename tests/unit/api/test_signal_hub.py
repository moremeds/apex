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
