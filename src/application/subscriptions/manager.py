"""Ref-counted subscription lifecycle (spec 3.1).

subscribe(ticker)   -> seed history from livewire, start compute, publish.
unsubscribe(ticker) -> refcount--; at 0 stop compute and TTL-prune.
Many argon clients on one ticker -> compute once, fan out.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Set

if TYPE_CHECKING:
    from src.infrastructure.adapters.livewire.revisions import SilverRevision

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class RefreshResult:
    """Per-symbol result of applying one Silver revision."""

    applied: Dict[str, int]
    failed: Dict[str, str]


class _ComputeService(Protocol):
    """Matches the real TASignalService (verified 2026-06-14): async, 3-arg, dict bars."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def inject_historical_bars(
        self, symbol: str, timeframe: str, bars: List[Dict[str, Any]]
    ) -> int: ...

    def begin_symbol_refresh(self, symbol: str) -> None: ...

    async def replace_symbol_histories(
        self, symbol: str, histories: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, int]: ...

    def commit_symbol_refresh(self, symbol: str) -> None: ...

    def abort_symbol_refresh(self, symbol: str) -> None: ...


class SubscriptionManager:
    """Ref-counted, subscription-driven compute coordinator (spec 3.1)."""

    def __init__(
        self,
        provider: Any,
        compute: _ComputeService,
        timeframes: List[str],
        seed_lookback_days: int = 365,
        live_feed: Any = None,
    ) -> None:
        self._provider = provider
        self._compute = compute
        self._timeframes = timeframes
        self._lookback_days = seed_lookback_days
        self._live_feed = live_feed
        self._subs: Dict[str, Subscription] = {}
        self._lock = asyncio.Lock()
        self._compute_started = False
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._applied_revisions: Dict[str, int] = {}

    def set_live_feed(self, live_feed: Any) -> None:
        """Attach a LiveFeedPort after construction (used by the app lifespan)."""
        self._live_feed = live_feed

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
                if self._live_feed is not None and sub.refcount == 0:
                    await self._live_feed.subscribe(symbol)
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
                # Drop the live feed for this symbol. Best-effort: a feed error
                # must not block the in-memory cleanup below.
                if self._live_feed is not None:
                    try:
                        await self._live_feed.unsubscribe(symbol)
                    except Exception:  # noqa: BLE001 - best-effort remote drop
                        logger.warning("live feed unsubscribe failed for %s", symbol)
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

    async def refresh_revision(self, revision: "SilverRevision") -> RefreshResult:
        """Reseed active symbols affected by a validated Silver revision."""
        active = self.active_symbols()
        affected = [item for item in revision.affected if item.symbol in active]
        outcomes = await asyncio.gather(
            *(self._refresh_symbol(item, revision.revision) for item in affected),
            return_exceptions=False,
        )
        applied: Dict[str, int] = {}
        failed: Dict[str, str] = {}
        for symbol, error in outcomes:
            if error is None:
                applied[symbol] = revision.revision
                self._applied_revisions[symbol] = revision.revision
            else:
                failed[symbol] = error
        return RefreshResult(applied=applied, failed=failed)

    async def _refresh_symbol(self, affected: Any, revision: int) -> tuple[str, str | None]:
        symbol = affected.symbol
        lock = self._refresh_locks.setdefault(symbol, asyncio.Lock())
        async with lock:
            self._compute.begin_symbol_refresh(symbol)
            try:
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=self._lookback_days)
                requested = set(affected.timeframes)
                histories: Dict[str, List[Dict[str, Any]]] = {}
                for timeframe in self._timeframes:
                    if timeframe not in requested:
                        continue
                    bars = await self._provider.fetch_bars(symbol, timeframe, start=start, end=end)
                    histories[timeframe] = [self._bar_to_dict(bar) for bar in bars]
                await self._compute.replace_symbol_histories(symbol, histories)
                self._compute.commit_symbol_refresh(symbol)
                return symbol, None
            except Exception as exc:  # noqa: BLE001 - isolate one revised symbol
                self._compute.abort_symbol_refresh(symbol)
                logger.exception("Silver revision %s failed for %s", revision, symbol)
                return symbol, str(exc)

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
