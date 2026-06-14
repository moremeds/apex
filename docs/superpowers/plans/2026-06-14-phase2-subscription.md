# Phase 2 — Subscription + Compute Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ref-counted `SubscriptionManager` that computes/persists TA **only** for the tickers argon subscribes to: `subscribe(ticker)` seeds history from livewire and starts compute; `unsubscribe(ticker)` decrements a refcount and tears down at zero — so the signal store stays MB, not GB.

**Architecture:** `SubscriptionManager` holds a `dict[symbol -> Subscription]`. Each `Subscription` owns a refcount and (lazily) a running compute context that seeds history via `LivewireOhlcProvider` (Phase 1) and feeds it into the existing `TASignalService.inject_historical_bars(...)`. Subscribe is idempotent per symbol (compute once, fan out); unsubscribe at refcount 0 stops compute and schedules a TTL prune of persisted rows. The manager is async and guarded by an `asyncio.Lock` so concurrent argon clients can't race the refcount.

**Tech Stack:** Python 3.13, asyncio, `TASignalService` (`src/application/services/ta_signal_service.py`), `LivewireOhlcProvider` (Phase 1).

**Spec:** `docs/superpowers/specs/2026-06-14-apex-adaptation-design.md` §5 (Phase 2) + source doc §3.1. **Depends on:** Phase 1 (`LivewireOhlcProvider`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/application/subscriptions/__init__.py` | Package marker. |
| `src/application/subscriptions/manager.py` | `SubscriptionManager`, `Subscription` dataclass — refcount + lifecycle. |
| `tests/unit/application/subscriptions/test_manager.py` | Refcount, idempotency, seed-on-subscribe, teardown. |

---

## Task 1: `Subscription` refcount dataclass

**Files:**
- Create: `src/application/subscriptions/__init__.py`
- Create: `src/application/subscriptions/manager.py`
- Test: `tests/unit/application/subscriptions/test_manager.py`

- [ ] **Step 1: Create package marker**

Create `src/application/subscriptions/__init__.py`:

```python
"""Ref-counted, subscription-driven TA compute (spec §3.1)."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/application/subscriptions/test_manager.py`:

```python
from __future__ import annotations

from src.application.subscriptions.manager import Subscription


def test_subscription_refcount_increments_and_decrements() -> None:
    sub = Subscription(symbol="AAPL")
    assert sub.refcount == 0
    assert sub.acquire() == 1
    assert sub.acquire() == 2
    assert sub.release() == 1
    assert sub.release() == 0


def test_release_below_zero_raises() -> None:
    sub = Subscription(symbol="AAPL")
    import pytest
    with pytest.raises(ValueError, match="refcount underflow"):
        sub.release()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/application/subscriptions/test_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the dataclass**

Create `src/application/subscriptions/manager.py`:

```python
"""Ref-counted subscription lifecycle (spec §3.1).

subscribe(ticker)   -> seed history from livewire, start compute, publish.
unsubscribe(ticker) -> refcount--; at 0 stop compute and TTL-prune.
Many argon clients on one ticker -> compute once, fan out.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/application/subscriptions/test_manager.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/application/subscriptions/ tests/unit/application/subscriptions/test_manager.py
git commit -m "feat(subs): Subscription refcount primitive"
```

---

## Task 2: `SubscriptionManager.subscribe` — idempotent seed + start

**Files:**
- Modify: `src/application/subscriptions/manager.py`
- Test: `tests/unit/application/subscriptions/test_manager.py`

- [ ] **Step 1: Write the failing test**

Append to the test file:

The fakes mirror the REAL `TASignalService` contract (verified 2026-06-14):
`async def inject_historical_bars(self, symbol, timeframe, bars: list[dict]) -> int`
— it is **async**, takes **(symbol, timeframe, bars)**, and `bars` are **dicts**
(not `BarData`). The provider returns `BarData`, so the manager converts.

```python
import pytest
from types import SimpleNamespace


def _fake_bar(close: float = 11.0):
    """A BarData-shaped object the manager can convert to a dict."""
    return SimpleNamespace(
        open=10.0, high=10.5, low=9.5, close=close, volume=100, bar_start="2026-01-02T00:00:00Z"
    )


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def fetch_bars(self, symbol, timeframe, **kw):
        self.calls.append((symbol, timeframe))
        return [_fake_bar()]  # one bar


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
    from src.application.subscriptions.manager import SubscriptionManager

    provider, compute = _FakeProvider(), _FakeCompute()
    mgr = SubscriptionManager(provider=provider, compute=compute, timeframes=["1d"])

    await mgr.subscribe("AAPL")
    await mgr.subscribe("AAPL")  # second client, same ticker

    assert mgr.refcount("AAPL") == 2
    assert provider.calls == [("AAPL", "1d")]          # seeded ONCE
    assert compute.injected == [("AAPL", "1d", 1)]     # injected ONCE, with timeframe
    # the injected payload must be plain dicts, not BarData/SimpleNamespace
    assert mgr.active_symbols() == {"AAPL"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/application/subscriptions/test_manager.py::test_subscribe_seeds_history_once_and_fans_out -v`
Expected: FAIL — `SubscriptionManager` not defined.

- [ ] **Step 3: Implement subscribe**

Append to `src/application/subscriptions/manager.py`:

```python
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Protocol, Set


class _ComputeService(Protocol):
    # Matches the real TASignalService (verified 2026-06-14): async, 3-arg, dict bars.
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def inject_historical_bars(
        self, symbol: str, timeframe: str, bars: List[Dict[str, Any]]
    ) -> int: ...


class SubscriptionManager:
    """Ref-counted, subscription-driven compute coordinator (spec §3.1)."""

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

    async def _seed(self, symbol: str) -> None:
        """Pull history for each timeframe from livewire and inject it once.

        HistoricalSourcePort.fetch_bars requires start/end, so we pass an explicit
        lookback window. The provider returns BarData; TASignalService.inject_
        historical_bars wants dicts and is async — convert + await. start/end are
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/application/subscriptions/test_manager.py::test_subscribe_seeds_history_once_and_fans_out -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/application/subscriptions/manager.py tests/unit/application/subscriptions/test_manager.py
git commit -m "feat(subs): idempotent subscribe with seed-on-first"
```

---

## Task 3: `unsubscribe` — refcount teardown at zero

**Files:**
- Modify: `src/application/subscriptions/manager.py`
- Test: `tests/unit/application/subscriptions/test_manager.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.asyncio
async def test_unsubscribe_teardown_at_zero() -> None:
    from src.application.subscriptions.manager import SubscriptionManager

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
    from src.application.subscriptions.manager import SubscriptionManager
    mgr = SubscriptionManager(provider=_FakeProvider(), compute=_FakeCompute(), timeframes=["1d"])
    await mgr.unsubscribe("NOPE")  # must not raise
    assert mgr.refcount("NOPE") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/application/subscriptions/test_manager.py -k unsubscribe -v`
Expected: FAIL — `unsubscribe` not defined.

- [ ] **Step 3: Implement unsubscribe**

Append to `SubscriptionManager`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/application/subscriptions/test_manager.py -k unsubscribe -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/application/subscriptions/manager.py tests/unit/application/subscriptions/test_manager.py
git commit -m "feat(subs): refcount teardown on unsubscribe"
```

---

## Task 4: Concurrency safety (no refcount race)

**Files:**
- Test: `tests/unit/application/subscriptions/test_manager.py`

- [ ] **Step 1: Write the verifying test**

Append:

```python
@pytest.mark.asyncio
async def test_concurrent_subscribe_seeds_exactly_once() -> None:
    import asyncio as _aio
    from src.application.subscriptions.manager import SubscriptionManager

    provider, compute = _FakeProvider(), _FakeCompute()
    mgr = SubscriptionManager(provider=provider, compute=compute, timeframes=["1d"])

    await _aio.gather(*[mgr.subscribe("AAPL") for _ in range(20)])
    assert mgr.refcount("AAPL") == 20
    assert provider.calls == [("AAPL", "1d")]   # seeded exactly once despite 20 racers
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/unit/application/subscriptions/test_manager.py::test_concurrent_subscribe_seeds_exactly_once -v`
Expected: PASS (the `asyncio.Lock` in `subscribe` serializes the first-seed check). If FAIL, the lock is missing around the `first`/seed block — fix and re-run.

- [ ] **Step 3: Add a seed-failure / retry test**

Append:

```python
@pytest.mark.asyncio
async def test_failed_seed_does_not_poison_subscription() -> None:
    from src.application.subscriptions.manager import SubscriptionManager

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
    assert mgr.refcount("AAPL") == 0          # not left half-initialized
    assert "AAPL" not in mgr.active_symbols()

    await mgr.subscribe("AAPL")                # retry succeeds
    assert mgr.refcount("AAPL") == 1
    assert compute.injected == [("AAPL", "1d", 1)]
```

- [ ] **Step 4: Run the edge tests**

Run: `uv run pytest tests/unit/application/subscriptions/test_manager.py -k "concurrent or poison" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/application/subscriptions/test_manager.py
git commit -m "test(subs): concurrent seeds-once + failed-seed retry"
```

---

## Task 5: Adapt the real `TASignalService` to the `_ComputeService` shape

**Files:**
- Read: `src/application/services/ta_signal_service.py` (verify `start`, `stop`, `inject_historical_bars` signatures)
- Possibly Modify: `src/application/subscriptions/manager.py` (only if real signatures differ)
- Test: `tests/integration/test_subscription_with_real_service.py`

- [ ] **Step 1: Verify the real method signatures**

Run: `uv run python -c "import inspect; from src.application.services.ta_signal_service import TASignalService as T; print(inspect.signature(T.inject_historical_bars)); print(inspect.signature(T.start)); print(inspect.signature(T.stop))"`
Expected (verified 2026-06-14): `inject_historical_bars(self, symbol: str, timeframe: str, bars: List[Dict[str, Any]]) -> int` — **async**. `start()`/`stop()` are async no-arg. The `_ComputeService` Protocol and `_seed` in Tasks 2–3 already match this exactly. If a future refactor changes the signature, update both in lockstep. Also confirm the `TASignalService.__init__` required args (event bus, persistence, config) so Step 2's constructor call is correct.

- [ ] **Step 2: Write an integration test (skipped without PG)**

Create `tests/integration/test_subscription_with_real_service.py`:

```python
"""SubscriptionManager drives the real TASignalService for one ticker."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("APEX_LIVEWIRE_ROOT"),
    reason="needs a livewire bronze root to seed real history",
)


@pytest.mark.asyncio
async def test_subscribe_one_real_ticker() -> None:
    from src.application.services.ta_signal_service import TASignalService
    from src.application.subscriptions.manager import SubscriptionManager
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    provider = LivewireOhlcProvider(bronze_root=Path(os.environ["APEX_LIVEWIRE_ROOT"]))
    service = TASignalService(...)  # construct per its real __init__ (see Step 1)
    mgr = SubscriptionManager(provider=provider, compute=service, timeframes=["1d"])

    await mgr.subscribe("AAPL")
    assert mgr.refcount("AAPL") == 1
    await mgr.unsubscribe("AAPL")
    assert "AAPL" not in mgr.active_symbols()
```

- [ ] **Step 3: Run the unit suite (integration skips without env)**

Run: `uv run pytest tests/unit/application/subscriptions tests/integration/test_subscription_with_real_service.py -v`
Expected: unit tests PASS; integration SKIPPED (no `APEX_LIVEWIRE_ROOT`). Fill the `TASignalService(...)` constructor args from Step 1 before running with the env set.

- [ ] **Step 4: Commit**

```bash
git add src/application/subscriptions/manager.py tests/integration/test_subscription_with_real_service.py
git commit -m "feat(subs): adapt SubscriptionManager to real TASignalService"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** §3.1 subscribe(seed history + start) → Task 2; unsubscribe(refcount→0, free) → Task 3; ref-counted fan-out → Tasks 2,4; "compute once" → Task 4; failed-subscribe safety → Task 4 Step 3. TTL prune of persisted rows → handed to Phase 3 (noted in Task 3 comment). ✅
- **Scope boundary (steady-state, ISSUE-2):** §3.1's *steady-state* — filtering live ticks to active symbols, incremental recompute per new tick/bar, and per-symbol compute teardown at refcount 0 — inherently requires the **live feed, which is Phase 4 (xenon)**. Phase 2 has no live tick source, so it delivers only the historical **seed** + the **refcount lifecycle**. The hook where Phase 4 will gate incoming ticks by `active_symbols()` and tear down per-symbol state is called out here as the Phase-4 integration point; it is **deliberately not** implemented in Phase 2. This is a sequencing boundary, not a coverage gap. ⚠️ documented
- **Honest gaps:** real `TASignalService.__init__`/`inject_historical_bars` signatures verified in Task 5 Step 1 before integration. Compute is injected behind a Protocol so unit tests don't need PG/event-bus. ✅
- **Type consistency:** `Subscription`, `SubscriptionManager`, `_ComputeService`, `acquire/release/refcount/active_symbols/subscribe/unsubscribe/_seed` consistent across tasks. ✅
