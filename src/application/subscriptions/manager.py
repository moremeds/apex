"""Ref-counted subscription lifecycle (spec 3.1).

subscribe(ticker)   -> seed history from livewire, start compute, publish.
unsubscribe(ticker) -> refcount--; at 0 stop compute and TTL-prune.
Many argon clients on one ticker -> compute once, fan out.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Protocol, Set


@dataclass
class Subscription:
    """Per-symbol refcount and run state."""

    symbol: str
    refcount: int = 0
    started: bool = False
    seeded: bool = field(default=False)

    def acquire(self) -> int:
        self.refcount += 1
        return self.refcount

    def release(self) -> int:
        if self.refcount <= 0:
            raise ValueError(f"refcount underflow for {self.symbol}")
        self.refcount -= 1
        return self.refcount


class _ComputeService(Protocol):
    """Matches the real TASignalService (verified 2026-06-14): async, 3-arg, dict bars."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def inject_historical_bars(
        self, symbol: str, timeframe: str, bars: List[Dict[str, Any]]
    ) -> int: ...


class SubscriptionManager:
    """Ref-counted, subscription-driven compute coordinator (spec 3.1)."""

    def __init__(
        self,
        provider: Any,
        compute: _ComputeService,
        timeframes: List[str],
        seed_lookback_days: int = 365,
    ) -> None:
        self._provider = provider
        self._compute = compute
        self._timeframes = timeframes
        self._lookback_days = seed_lookback_days
        self._subs: Dict[str, Subscription] = {}
        self._lock = asyncio.Lock()
        self._compute_started = False

    def active_symbols(self) -> Set[str]:
        return {s for s, sub in self._subs.items() if sub.refcount > 0}

    def refcount(self, symbol: str) -> int:
        sub = self._subs.get(symbol)
        return sub.refcount if sub else 0

    async def subscribe(self, symbol: str) -> None:
        async with self._lock:
            sub = self._subs.setdefault(symbol, Subscription(symbol=symbol))
            # Do start/seed BEFORE incrementing the refcount, so a failure leaves
            # no poisoned half-initialized entry (the next subscribe retries).
            try:
                if not self._compute_started:
                    await self._compute.start()
                    self._compute_started = True
                if not sub.seeded:
                    await self._seed(symbol)
                    sub.seeded = True
                    sub.started = True
            except Exception:
                if sub.refcount == 0:
                    self._subs.pop(symbol, None)  # drop poisoned entry; allow retry
                raise
            sub.acquire()

    async def unsubscribe(self, symbol: str) -> None:
        async with self._lock:
            sub = self._subs.get(symbol)
            if sub is None or sub.refcount == 0:
                return
            remaining = sub.release()
            if remaining == 0:
                sub.started = False
                # Retain persisted rows for a short TTL (fast re-subscribe / audit),
                # then prune. Pruning is delegated to the persistence layer (Phase 3);
                # here we only drop the in-memory run state.
                self._subs.pop(symbol, None)

    async def _seed(self, symbol: str) -> None:
        """Pull history for each timeframe from livewire and inject it once.

        HistoricalSourcePort.fetch_bars requires start/end, so we pass an explicit
        lookback window. The provider returns BarData; TASignalService.inject_
        historical_bars wants dicts and is async -- convert + await. start/end are
        passed as keywords so test fakes (**kw) capture them.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self._lookback_days)
        for tf in self._timeframes:
            bars = await self._provider.fetch_bars(symbol, tf, start=start, end=end)
            payload = [self._bar_to_dict(b) for b in bars]
            await self._compute.inject_historical_bars(symbol, tf, payload)

    @staticmethod
    def _bar_to_dict(bar: Any) -> Dict[str, Any]:
        return {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "timestamp": bar.bar_start,
        }
