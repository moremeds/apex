from __future__ import annotations

from src.domain.interfaces import LiveFeedPort


class _Conforming:
    async def connect(self) -> None: ...
    async def subscribe(self, symbol: str) -> None: ...
    async def unsubscribe(self, symbol: str) -> None: ...
    async def close(self) -> None: ...


class _Missing:
    async def connect(self) -> None: ...


def test_conforming_object_is_a_live_feed_port() -> None:
    assert isinstance(_Conforming(), LiveFeedPort)


def test_incomplete_object_is_not_a_live_feed_port() -> None:
    assert not isinstance(_Missing(), LiveFeedPort)
