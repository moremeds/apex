"""Tracks WS connections per ticker and fans out payloads to subscribers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Optional, Set


class SignalHub:
    def __init__(self) -> None:
        self._by_ticker: DefaultDict[str, Set[Any]] = defaultdict(set)
        self._tickers_of: Dict[Any, Set[str]] = {}

    def register(self, ws: Any, ticker: str) -> None:
        self._by_ticker[ticker].add(ws)
        self._tickers_of.setdefault(ws, set()).add(ticker)

    def unregister(self, ws: Any, ticker: Optional[str] = None) -> Set[str]:
        """Remove `ws` from one ticker (if given) or all tickers (disconnect).

        Returns the set of tickers actually removed, so the caller can decrement
        the matching SubscriptionManager refcounts exactly once each.
        """
        held = self._tickers_of.get(ws, set())
        if ticker is None:
            removed = set(held)
            self._tickers_of.pop(ws, None)
        else:
            removed = {ticker} & held
            held.discard(ticker)
            if not held:
                self._tickers_of.pop(ws, None)
        for t in removed:
            self._by_ticker[t].discard(ws)
        return removed

    async def broadcast(self, ticker: str, payload: dict) -> None:
        dead = []
        for ws in list(self._by_ticker.get(ticker, ())):
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001 -- drop broken sockets
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)  # full removal of a dead socket
