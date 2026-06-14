# Phase 4 — Live-in Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the live-feed *adapter* and wiring that let a subscribed ticker stream real xenon IB ticks into apex's existing compute pipeline — behind a feature flag that is OFF by default. The end-to-end seam (xenon tick → bus → `BarAggregator` → bar/signal) is proven by an integration test.

**Scope boundary (read first):** This phase delivers the WS client, tick translation, `SubscriptionManager` live-feed wiring, and env-gated lifespan construction. It does **not** stand up the full pipeline (`PriorityEventBus` + `TASignalService` + `SubscriptionManager`) inside the server `lifespan` — Phase 3 deliberately left that env-gated/unbuilt (it needs the livewire provider + PG). So setting `APEX_XENON_WS_URL` in production today builds and connects the client only if something has populated `app.state.event_bus`/`app.state.subscription_manager`. Phase 4 correctly hooks into that seam and proves the data path with a real integration test; constructing the pipeline in `lifespan` is a separate, tracked follow-up (see Task 13, optional). This boundary matches the approved spec (§8, §12d).

**Architecture:** A thin live-feed *adapter*, not a bar engine. apex's `BarAggregator` already does the tick→bar stitch, so Phase 4 only (1) speaks the xenon WS protocol, (2) translates each tick onto `EventType.MARKET_DATA_TICK`, and (3) wires subscribe/unsubscribe so a watched ticker opens/drops its xenon sub. Everything downstream (bars→indicators→rules→signals→argon) is unchanged.

**Tech Stack:** Python 3.13 async, `websockets` 16 (`websockets.asyncio.client.connect` / `websockets.asyncio.server.serve`), the existing `PriorityEventBus`, `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`).

**Spec:** `docs/superpowers/specs/2026-06-14-phase4-live-in-design.md`

**Verified contract facts (do not re-derive):**
- xenon server→client steady-state ticks: `{"type":"batch","updates":{SYM:PriceData}}`; on-subscribe: `{"type":"price","symbol":SYM,"data":PriceData}`; keep-alive: `{"type":"ping"}` → client MUST reply `{"action":"pong"}` (else dropped at 65s). (`~/projects/xenon/scripts/infra/ib_realtime/ib_realtime_server.js`)
- client→server: `{"action":"subscribe","symbols":[...]}` / `{"action":"unsubscribe","symbols":[...]}`.
- `PriceData` = `{symbol, last, bid, ask, volume, timestamp(ISO-8601 string), ...}`; `last` may be null. (`ib_tick_handler.js:createPriceData`)
- Loopback / no-`CLERK_JWKS_URL` connections skip ticket auth → no ticket needed for the trusted-network path (`ib_realtime_server.js:362-373`).
- apex ingest: `BarAggregator.on_tick` accepts a dict `{symbol, last|mid|price|bid+ask, volume, timestamp}`; `timestamp` is returned as-is so it must be a `datetime` (`src/domain/signals/data/bar_aggregator.py:221-264`).
- event bus: `PriorityEventBus.publish(EventType.MARKET_DATA_TICK, payload)` (sync). When the bus is **not** started (`_running` False) `publish` dispatches **synchronously** to subscribers via `_dispatch_sync` (`priority_event_bus.py:195,539`) — this is what the e2e test relies on. `EventType.MARKET_DATA_TICK = "market_data_tick"` (`event_types.py:52`); `EventType.BAR_CLOSE = "bar_close"` (`event_types.py:80`).
- `SubscriptionManager.__init__(provider, compute, timeframes, seed_lookback_days=365)` (`src/application/subscriptions/manager.py:51`); `subscribe`/`unsubscribe` already await I/O (history seed) under `self._lock` — the lock serializes per-symbol setup. Phase 4 adds the live-feed call to that same critical section, consistent with the existing design (decoupling lock from I/O is out of scope).
- websockets 16: `connect()` defaults `ping_interval=20s`, `ping_timeout=20s` — it sends protocol PING frames and closes the socket (→ `ConnectionClosed`) if the peer stops responding, so a dead-but-open connection is detected without a manual read-timeout. xenon's Node `ws` server auto-answers protocol PING frames.
- ports live in `src/domain/interfaces/` as `@runtime_checkable Protocol` (mirror `historical_source.py`); adapters live in `src/infrastructure/adapters/<source>/`.
- CI test jobs install `.[dev,server,api]` → `websockets` must be declared in the `api` extra to be importable in tests.

**Toolchain rules:** run tests with `uv run python -m pytest ... --no-cov` (NEVER bare `pytest`). Format with `uv run black` + `uv run isort`; lint `uv run flake8`. Do NOT touch `src/backtest` or `domain/backtest`.

---

## File Structure

**Create:**
- `src/domain/interfaces/live_feed.py` — `LiveFeedPort` (`@runtime_checkable Protocol`): `connect()`, `subscribe(symbol)`, `unsubscribe(symbol)`, `close()`.
- `src/infrastructure/adapters/xenon/__init__.py`
- `src/infrastructure/adapters/xenon/auth.py` — `AuthProvider` Protocol + `NoAuthProvider`.
- `src/infrastructure/adapters/xenon/translator.py` — `translate_price_data(data) -> dict | None` (pure).
- `src/infrastructure/adapters/xenon/client.py` — `XenonTickClient` (implements `LiveFeedPort`).
- `tests/support/__init__.py`
- `tests/support/fake_xenon.py` — `FakeXenonServer` test double (real in-process WS server).
- `tests/unit/infrastructure/xenon/__init__.py`
- `tests/unit/infrastructure/xenon/test_live_feed_port.py`
- `tests/unit/infrastructure/xenon/test_auth.py`
- `tests/unit/infrastructure/xenon/test_translator.py`
- `tests/unit/infrastructure/xenon/test_fake_xenon.py`
- `tests/unit/infrastructure/xenon/test_client.py`
- `tests/integration/test_xenon_live_e2e.py`

**Modify:**
- `pyproject.toml` — add `websockets>=16.0` to the `api` extra.
- `src/domain/interfaces/__init__.py` — export `LiveFeedPort`.
- `src/application/subscriptions/manager.py` — optional `live_feed`; open on subscribe, best-effort drop on unsubscribe; `set_live_feed()`.
- `tests/unit/application/subscriptions/test_manager.py` — live-feed wiring tests.
- `src/api/server.py` — env-gated `XenonTickClient` construction in `lifespan` + `close()` on shutdown.
- `tests/unit/api/test_server_lifespan.py` — lifespan wiring tests.

---

## Task 1: Declare the `websockets` dependency

**Files:** Modify `pyproject.toml` (the `api` optional-dependencies group)

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, under `[project.optional-dependencies]` in the `api = [ ... ]` list, add a line after the `httpx` entry:

```toml
api = [
    "fastapi>=0.115.0",        # REST API framework
    "uvicorn[standard]>=0.30.0", # ASGI server
    "httpx>=0.27.0",           # Async HTTP client (for tests)
    "websockets>=16.0",        # WS client for the xenon live tick feed (Phase 4)
]
```

- [ ] **Step 2: Verify it resolves and imports**

Run: `uv run python -c "from websockets.asyncio.client import connect; from websockets.asyncio.server import serve; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: declare websockets dep for xenon live feed (Phase 4)"
```

---

## Task 2: `LiveFeedPort` interface

**Files:**
- Create: `src/domain/interfaces/live_feed.py`, `tests/unit/infrastructure/xenon/__init__.py`, `tests/unit/infrastructure/xenon/test_live_feed_port.py`
- Modify: `src/domain/interfaces/__init__.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/unit/infrastructure/xenon/__init__.py` (empty file).

- [ ] **Step 2: Write the failing test**

Create `tests/unit/infrastructure/xenon/test_live_feed_port.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_live_feed_port.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'LiveFeedPort'`.

- [ ] **Step 4: Create the port**

Create `src/domain/interfaces/live_feed.py`:

```python
"""Port for a live market-data feed (e.g. xenon's IB tick WS).

A feed-agnostic surface so the subscription layer can open/drop a live
subscription per ticker without knowing the transport (xenon, a fake, etc.).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LiveFeedPort(Protocol):
    """Open a connection, (un)subscribe per symbol, and close it."""

    async def connect(self) -> None:
        """Start the feed connection (non-blocking: launches the background loop)."""
        ...

    async def subscribe(self, symbol: str) -> None:
        """Begin receiving live ticks for ``symbol``."""
        ...

    async def unsubscribe(self, symbol: str) -> None:
        """Stop receiving live ticks for ``symbol``."""
        ...

    async def close(self) -> None:
        """Tear down the feed connection and stop the background loop."""
        ...
```

- [ ] **Step 5: Export it**

In `src/domain/interfaces/__init__.py`, add an import line next to the other port imports:

```python
from .live_feed import LiveFeedPort
```

and add `"LiveFeedPort",` to the `__all__` list.

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_live_feed_port.py -v --no-cov`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add src/domain/interfaces/live_feed.py src/domain/interfaces/__init__.py tests/unit/infrastructure/xenon/__init__.py tests/unit/infrastructure/xenon/test_live_feed_port.py
git commit -m "feat(domain): add LiveFeedPort interface (Phase 4)"
```

---

## Task 3: Auth seam (`AuthProvider` + `NoAuthProvider`)

**Files:**
- Create: `src/infrastructure/adapters/xenon/__init__.py`, `src/infrastructure/adapters/xenon/auth.py`, `tests/unit/infrastructure/xenon/test_auth.py`

- [ ] **Step 1: Create the package marker**

Create `src/infrastructure/adapters/xenon/__init__.py` (empty file).

- [ ] **Step 2: Write the failing test**

Create `tests/unit/infrastructure/xenon/test_auth.py`:

```python
from __future__ import annotations

import pytest

from src.infrastructure.adapters.xenon.auth import AuthProvider, NoAuthProvider


@pytest.mark.asyncio
async def test_no_auth_provider_returns_none() -> None:
    assert await NoAuthProvider().ticket() is None


def test_no_auth_provider_satisfies_protocol() -> None:
    assert isinstance(NoAuthProvider(), AuthProvider)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_auth.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: ...xenon.auth`.

- [ ] **Step 4: Implement**

Create `src/infrastructure/adapters/xenon/auth.py`:

```python
"""Pluggable auth for the xenon WS connection (decision D2).

Default ``NoAuthProvider`` is the trusted-network / loopback path xenon already
permits (it skips ticket validation for localhost or when CLERK is unconfigured).
A ``TicketAuthProvider`` (service-JWT) can drop in later with no client change.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class AuthProvider(Protocol):
    """Yield a connection ticket, or ``None`` for the no-auth path."""

    async def ticket(self) -> Optional[str]:
        ...


class NoAuthProvider:
    """Trusted-network path: no ticket. Returns ``None``."""

    async def ticket(self) -> Optional[str]:
        return None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_auth.py -v --no-cov`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/infrastructure/adapters/xenon/__init__.py src/infrastructure/adapters/xenon/auth.py tests/unit/infrastructure/xenon/test_auth.py
git commit -m "feat(xenon): add pluggable auth seam (NoAuthProvider) (Phase 4)"
```

---

## Task 4: Tick translator (`translate_price_data`)

**Files:** Create `src/infrastructure/adapters/xenon/translator.py`, `tests/unit/infrastructure/xenon/test_translator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/infrastructure/xenon/test_translator.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from src.infrastructure.adapters.xenon.translator import translate_price_data


def test_translates_full_price_data_with_iso_timestamp() -> None:
    out = translate_price_data(
        {
            "symbol": "AAPL",
            "last": 150.25,
            "bid": 150.2,
            "ask": 150.3,
            "volume": 1000,
            "timestamp": "2026-06-14T12:00:00.000Z",
        }
    )
    assert out is not None
    assert out["symbol"] == "AAPL"
    assert out["last"] == 150.25
    assert out["volume"] == 1000
    assert out["timestamp"] == datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)


def test_null_last_but_bid_ask_present_is_kept() -> None:
    out = translate_price_data(
        {"symbol": "AAPL", "last": None, "bid": 10.0, "ask": 10.2,
         "timestamp": "2026-06-14T12:00:00Z"}
    )
    assert out is not None
    assert out["bid"] == 10.0 and out["ask"] == 10.2


def test_no_usable_price_is_dropped() -> None:
    out = translate_price_data(
        {"symbol": "AAPL", "last": None, "bid": None, "ask": None,
         "timestamp": "2026-06-14T12:00:00Z"}
    )
    assert out is None


def test_missing_symbol_is_dropped() -> None:
    assert translate_price_data({"last": 1.0, "timestamp": "2026-06-14T12:00:00Z"}) is None


def test_missing_timestamp_yields_none_timestamp_key() -> None:
    out = translate_price_data({"symbol": "AAPL", "last": 5.0})
    assert out is not None
    assert out["timestamp"] is None  # aggregator will default to now()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_translator.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: ...xenon.translator`.

- [ ] **Step 3: Implement**

Create `src/infrastructure/adapters/xenon/translator.py`:

```python
"""Translate a xenon ``PriceData`` dict to an apex tick dict.

The only semantic conversion is ISO-string timestamp -> tz-aware datetime; the
aggregator's dict path reads ``symbol``/``last``/``bid``/``ask``/``volume``/
``timestamp`` directly (see BarAggregator._extract_*). Ticks with no usable
price (last/mid/bid+ask all absent) are dropped so the bus is not spammed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def _parse_ts(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _has_price(data: Dict[str, Any]) -> bool:
    if data.get("last") is not None:
        return True
    if data.get("mid") is not None:
        return True
    return data.get("bid") is not None and data.get("ask") is not None


def translate_price_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return an apex tick dict, or ``None`` if the tick is unusable."""
    symbol = data.get("symbol")
    if not symbol:
        return None
    if not _has_price(data):
        return None
    return {
        "symbol": symbol,
        "last": data.get("last"),
        "bid": data.get("bid"),
        "ask": data.get("ask"),
        "volume": data.get("volume"),
        "timestamp": _parse_ts(data.get("timestamp")),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_translator.py -v --no-cov`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/adapters/xenon/translator.py tests/unit/infrastructure/xenon/test_translator.py
git commit -m "feat(xenon): translate PriceData tick to apex tick dict (Phase 4)"
```

---

## Task 5: `FakeXenonServer` test double

**Files:** Create `tests/support/__init__.py`, `tests/support/fake_xenon.py`, `tests/unit/infrastructure/xenon/test_fake_xenon.py`

- [ ] **Step 1: Create the support package marker**

Create `tests/support/__init__.py` (empty file).

- [ ] **Step 2: Write the failing self-test**

Create `tests/unit/infrastructure/xenon/test_fake_xenon.py`:

```python
from __future__ import annotations

import json

import pytest
from websockets.asyncio.client import connect

from tests.support.fake_xenon import FakeXenonServer


@pytest.mark.asyncio
async def test_fake_server_records_frames_and_pushes() -> None:
    async with FakeXenonServer() as server:
        async with connect(server.url) as ws:
            await ws.send(json.dumps({"action": "subscribe", "symbols": ["AAPL"]}))
            await server.wait_for_frames(1)
            assert server.received[0] == {"action": "subscribe", "symbols": ["AAPL"]}

            await server.push({"type": "batch", "updates": {"AAPL": {"symbol": "AAPL"}}})
            msg = json.loads(await ws.recv())
            assert msg["type"] == "batch"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_fake_xenon.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: tests.support.fake_xenon`.

- [ ] **Step 4: Implement the double**

Create `tests/support/fake_xenon.py`:

```python
"""In-process fake of xenon's ib_realtime WS server for tests.

Speaks the real protocol surface apex's client uses: records client action
frames, can push server frames (price/batch/ping/error), and can drop
connections to exercise reconnect. Not collected by pytest (filename != test_*).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from websockets.asyncio.server import serve


class FakeXenonServer:
    def __init__(self) -> None:
        self.host = "127.0.0.1"
        self.port: int = 0
        self.received: List[Dict[str, Any]] = []
        self.connections = 0
        self._server: Any = None
        self._clients: set = set()
        self._frame_event = asyncio.Event()

    async def __aenter__(self) -> "FakeXenonServer":
        self._server = await serve(self._handler, self.host, 0)
        self.port = list(self._server.sockets)[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    async def _handler(self, ws: Any) -> None:
        self.connections += 1
        self._clients.add(ws)
        try:
            async for raw in ws:
                self.received.append(json.loads(raw))
                self._frame_event.set()
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    async def wait_for_frames(self, n: int, timeout: float = 2.0) -> None:
        async def _wait() -> None:
            while len(self.received) < n:
                self._frame_event.clear()
                if len(self.received) >= n:
                    return
                await self._frame_event.wait()

        await asyncio.wait_for(_wait(), timeout)

    async def wait_for_connection(self, n: int = 1, timeout: float = 2.0) -> None:
        async def _wait() -> None:
            while self.connections < n:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait(), timeout)

    async def push(self, payload: Dict[str, Any]) -> None:
        for ws in list(self._clients):
            await ws.send(json.dumps(payload))

    async def drop_connections(self) -> None:
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()
```

> Note on `wait_for_connection`: a short `sleep(0.01)` poll on a monotonic counter is acceptable here — it is bounded by `timeout` and cannot hang. Frame waits use an `asyncio.Event` (no polling).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_fake_xenon.py -v --no-cov`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add tests/support/__init__.py tests/support/fake_xenon.py tests/unit/infrastructure/xenon/test_fake_xenon.py
git commit -m "test(xenon): add FakeXenonServer test double (Phase 4)"
```

---

## Task 6: `XenonTickClient` — connect, subscribe, publish (single connection)

This task builds the client with **only** connect/subscribe/publish + robust per-frame handling. Ping (Task 7) and reconnect (Task 8) are added **after** their own failing tests, to keep each behavior test-first.

**Files:** Create `src/infrastructure/adapters/xenon/client.py`, `tests/unit/infrastructure/xenon/test_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/infrastructure/xenon/test_client.py`:

```python
from __future__ import annotations

import asyncio

import pytest

from src.domain.events.event_types import EventType
from src.domain.interfaces import LiveFeedPort
from src.infrastructure.adapters.xenon.client import XenonTickClient
from tests.support.fake_xenon import FakeXenonServer


class _RecordingBus:
    """Captures (event_type, payload) and exposes an event-based wait."""

    def __init__(self) -> None:
        self.published: list = []
        self._event = asyncio.Event()

    def publish(self, event_type, payload, priority=None) -> None:
        self.published.append((event_type, payload))
        self._event.set()

    async def wait_for(self, n: int, timeout: float = 2.0) -> None:
        async def _wait() -> None:
            while len(self.published) < n:
                self._event.clear()
                if len(self.published) >= n:
                    return
                await self._event.wait()

        await asyncio.wait_for(_wait(), timeout)


def test_client_satisfies_live_feed_port() -> None:
    assert isinstance(XenonTickClient("ws://x", _RecordingBus()), LiveFeedPort)


@pytest.mark.asyncio
async def test_subscribe_sends_action_frame() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)
        assert server.received[0] == {"action": "subscribe", "symbols": ["AAPL"]}
        await client.close()


@pytest.mark.asyncio
async def test_unsubscribe_sends_action_frame() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)
        await client.unsubscribe("AAPL")
        await server.wait_for_frames(2)
        assert server.received[1] == {"action": "unsubscribe", "symbols": ["AAPL"]}
        await client.close()


@pytest.mark.asyncio
async def test_batch_tick_is_translated_and_published() -> None:
    bus = _RecordingBus()
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, bus, reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)

        await server.push(
            {"type": "batch", "updates": {"AAPL": {
                "symbol": "AAPL", "last": 150.0, "volume": 10,
                "timestamp": "2026-06-14T12:00:00Z"}}}
        )
        await bus.wait_for(1)
        event_type, payload = bus.published[0]
        assert event_type == EventType.MARKET_DATA_TICK
        assert payload["symbol"] == "AAPL" and payload["last"] == 150.0
        await client.close()


@pytest.mark.asyncio
async def test_price_frame_is_published() -> None:
    bus = _RecordingBus()
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, bus, reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await server.push(
            {"type": "price", "symbol": "AAPL", "data": {
                "symbol": "AAPL", "last": 99.0, "timestamp": "2026-06-14T12:00:00Z"}}
        )
        await bus.wait_for(1)
        assert bus.published[0][1]["last"] == 99.0
        await client.close()


@pytest.mark.asyncio
async def test_malformed_frames_do_not_kill_the_client() -> None:
    """A non-JSON / non-dict / wrong-shape frame is dropped; later good ticks still flow."""
    bus = _RecordingBus()
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, bus, reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        # garbage that would crash a naive handler
        for ws in list(server._clients):
            await ws.send("not json")
            await ws.send(json.dumps([1, 2, 3]))           # JSON, but not an object
            await ws.send(json.dumps({"type": "batch", "updates": [1]}))  # updates not a dict
        await server.push(
            {"type": "price", "symbol": "AAPL", "data": {
                "symbol": "AAPL", "last": 1.0, "timestamp": "2026-06-14T12:00:00Z"}}
        )
        await bus.wait_for(1)
        assert bus.published[0][1]["last"] == 1.0
        await client.close()
```

Add `import json` at the top of the test file (used by the malformed-frame test).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_client.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: ...xenon.client`.

- [ ] **Step 3: Implement the client (connect + subscribe + publish only)**

Create `src/infrastructure/adapters/xenon/client.py`:

```python
"""WS client to xenon's ib_realtime server -> apex event bus (Phase 4).

Implements LiveFeedPort. Owns one websockets connection: sends
subscribe/unsubscribe action frames and republishes each received `price`/`batch`
tick (translated) on EventType.MARKET_DATA_TICK. A single malformed frame or a
handler error never kills the receive loop. (Keep-alive ping is added in Task 7;
reconnect/backoff in Task 8.)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Set
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from src.domain.events.event_types import EventType
from src.infrastructure.adapters.xenon.auth import AuthProvider, NoAuthProvider
from src.infrastructure.adapters.xenon.translator import translate_price_data

logger = logging.getLogger(__name__)


class XenonTickClient:
    def __init__(
        self,
        url: str,
        event_bus: Any,
        auth: Optional[AuthProvider] = None,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
    ) -> None:
        self._url = url
        self._bus = event_bus
        self._auth: AuthProvider = auth or NoAuthProvider()
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._subscribed: Set[str] = set()
        self._ws: Any = None
        self._task: Optional[asyncio.Task[None]] = None
        self._closing = False

    # ---- LiveFeedPort ----------------------------------------------------
    async def connect(self) -> None:
        """Launch the background receive loop (non-blocking)."""
        if self._task is None:
            self._closing = False
            self._task = asyncio.create_task(self._run())

    async def subscribe(self, symbol: str) -> None:
        self._subscribed.add(symbol)
        await self._send({"action": "subscribe", "symbols": [symbol]})

    async def unsubscribe(self, symbol: str) -> None:
        self._subscribed.discard(symbol)
        await self._send({"action": "unsubscribe", "symbols": [symbol]})

    async def close(self) -> None:
        self._closing = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ---- internals -------------------------------------------------------
    async def _connect_url(self) -> str:
        # SECURITY: the returned URL may carry an auth ticket in its query string
        # (once TicketAuthProvider lands). Never log this value.
        ticket = await self._auth.ticket()
        if not ticket:
            return self._url
        parts = urlsplit(self._url)
        query = dict(parse_qsl(parts.query))
        query["ticket"] = ticket
        return urlunsplit(parts._replace(query=urlencode(query)))

    async def _send(self, payload: dict) -> None:
        ws = self._ws
        if ws is None:
            return  # not connected yet; _resubscribe replays on connect
        try:
            await ws.send(json.dumps(payload))
        except ConnectionClosed:
            pass

    async def _resubscribe(self) -> None:
        if self._subscribed:
            await self._send({"action": "subscribe", "symbols": sorted(self._subscribed)})

    async def _run(self) -> None:
        # NOTE (Task 6): single connection, no reconnect. Task 8 wraps this body
        # in a reconnect loop with capped exponential backoff.
        try:
            async with connect(await self._connect_url()) as ws:
                self._ws = ws
                await self._resubscribe()
                async for raw in ws:
                    try:
                        await self._handle(raw)
                    except Exception:
                        logger.exception("xenon WS: error handling frame; continuing")
        except (ConnectionClosed, OSError) as exc:
            logger.warning("xenon WS connection lost: %s", exc)
        except asyncio.CancelledError:
            pass
        finally:
            self._ws = None

    async def _handle(self, raw: Any) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("xenon WS: dropping non-JSON frame")
            return
        if not isinstance(msg, dict):
            logger.warning("xenon WS: dropping non-object frame (%s)", type(msg).__name__)
            return
        mtype = msg.get("type")
        if mtype == "batch":
            updates = msg.get("updates")
            if isinstance(updates, dict):
                for data in updates.values():
                    if isinstance(data, dict):
                        self._publish(data)
        elif mtype == "price":
            data = msg.get("data")
            if isinstance(data, dict):
                self._publish(data)
        # ping/error/status/subscribed/unsubscribed/unknown handled in later tasks or ignored

    def _publish(self, data: dict) -> None:
        tick = translate_price_data(data)
        if tick is not None:
            self._bus.publish(EventType.MARKET_DATA_TICK, tick)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_client.py -v --no-cov`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/adapters/xenon/client.py tests/unit/infrastructure/xenon/test_client.py
git commit -m "feat(xenon): WS client publishes live ticks to the event bus (Phase 4)"
```

---

## Task 7: `XenonTickClient` — keep-alive ping/pong + error logging

**Files:** Modify `src/infrastructure/adapters/xenon/client.py`, `tests/unit/infrastructure/xenon/test_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/infrastructure/xenon/test_client.py`:

```python
@pytest.mark.asyncio
async def test_server_ping_is_answered_with_pong() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await server.push({"type": "ping"})
        await server.wait_for_frames(1)
        assert server.received[0] == {"action": "pong"}
        await client.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest "tests/unit/infrastructure/xenon/test_client.py::test_server_ping_is_answered_with_pong" -v --no-cov`
Expected: FAIL — the Task 6 `_handle` ignores `{"type":"ping"}`, so no `pong` frame is sent and `wait_for_frames(1)` times out.

- [ ] **Step 3: Implement ping + error handling**

In `src/infrastructure/adapters/xenon/client.py` `_handle`, add a `ping` branch (before `batch`) and an `error` branch:

```python
        mtype = msg.get("type")
        if mtype == "ping":
            await self._send({"action": "pong"})
        elif mtype == "batch":
            updates = msg.get("updates")
            if isinstance(updates, dict):
                for data in updates.values():
                    if isinstance(data, dict):
                        self._publish(data)
        elif mtype == "price":
            data = msg.get("data")
            if isinstance(data, dict):
                self._publish(data)
        elif mtype == "error":
            logger.warning("xenon WS error frame: %s", msg.get("message"))
        # status/subscribed/unsubscribed/unknown: ignored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_client.py -v --no-cov`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/adapters/xenon/client.py tests/unit/infrastructure/xenon/test_client.py
git commit -m "feat(xenon): answer keep-alive ping, log error frames (Phase 4)"
```

---

## Task 8: `XenonTickClient` — reconnect, backoff, re-subscribe, close-cancels

**Files:** Modify `src/infrastructure/adapters/xenon/client.py`, `tests/unit/infrastructure/xenon/test_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/infrastructure/xenon/test_client.py`:

```python
@pytest.mark.asyncio
async def test_reconnects_and_resubscribes_after_drop() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection(1)
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)          # initial subscribe

        await server.drop_connections()          # force a reconnect
        await server.wait_for_connection(2)      # client dialed back in
        await server.wait_for_frames(2)          # active set replayed on reconnect
        assert server.received[-1] == {"action": "subscribe", "symbols": ["AAPL"]}
        assert server.connections >= 2
        await client.close()


@pytest.mark.asyncio
async def test_close_during_reconnect_stops_redialing() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.05)
        await client.connect()
        await server.wait_for_connection(1)
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)

        await server.drop_connections()
        await client.close()                     # close while it would be reconnecting
        before = server.connections
        await asyncio.sleep(0.2)                  # > reconnect_delay
        assert server.connections == before      # no new dial after close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest "tests/unit/infrastructure/xenon/test_client.py::test_reconnects_and_resubscribes_after_drop" -v --no-cov`
Expected: FAIL — the Task 6 `_run` connects once and returns after a drop, so `wait_for_connection(2)` times out.

- [ ] **Step 3: Implement reconnect with capped exponential backoff**

Replace the `_run` body in `src/infrastructure/adapters/xenon/client.py` with a reconnect loop (delete the Task-6 NOTE comment):

```python
    async def _run(self) -> None:
        delay = self._reconnect_delay
        while not self._closing:
            try:
                async with connect(await self._connect_url()) as ws:
                    self._ws = ws
                    delay = self._reconnect_delay  # reset backoff on a good connect
                    await self._resubscribe()
                    async for raw in ws:
                        try:
                            await self._handle(raw)
                        except Exception:
                            logger.exception("xenon WS: error handling frame; continuing")
            except (ConnectionClosed, OSError) as exc:
                logger.warning("xenon WS connection lost: %s", exc)
            except asyncio.CancelledError:
                break
            finally:
                self._ws = None
            if self._closing:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_reconnect_delay)  # capped exponential backoff
```

> Dead-but-open connections are handled by websockets' built-in keepalive
> (`ping_interval`/`ping_timeout` defaults) which raises `ConnectionClosed` →
> the loop reconnects. No manual read-timeout is needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/ -v --no-cov`
Expected: all PASS (translator + auth + fake-server + port + client incl. both new tests).

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/adapters/xenon/client.py tests/unit/infrastructure/xenon/test_client.py
git commit -m "feat(xenon): reconnect with capped backoff + re-subscribe (Phase 4)"
```

---

## Task 9: Wire `LiveFeedPort` into `SubscriptionManager`

**Files:** Modify `src/application/subscriptions/manager.py`, `tests/unit/application/subscriptions/test_manager.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/application/subscriptions/test_manager.py` (the file already defines `_FakeProvider`, `_FakeCompute` at module level and constructs `SubscriptionManager(provider=..., compute=..., timeframes=["1d"])` inline — verified; follow that):

```python
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
    await mgr.subscribe("AAPL")           # refcount 1 -> 2
    assert feed.subscribed == ["AAPL"]    # opened once only


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
    await mgr.unsubscribe("AAPL")          # remote drop raises, but must not propagate
    assert mgr.refcount("AAPL") == 0
    assert "AAPL" not in mgr.active_symbols()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/application/subscriptions/test_manager.py -k "live_feed or feed" -v --no-cov`
Expected: FAIL with `AttributeError: 'SubscriptionManager' object has no attribute 'set_live_feed'`.

- [ ] **Step 3: Implement the wiring**

In `src/application/subscriptions/manager.py`:

3a. Add a module-level logger after the imports (if not already present):

```python
import logging

logger = logging.getLogger(__name__)
```

3b. Add `live_feed` to `__init__` (keyword, default `None`) and store it:

```python
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
```

3c. Add a setter just below `__init__`:

```python
    def set_live_feed(self, live_feed: Any) -> None:
        """Attach a LiveFeedPort after construction (used by the app lifespan)."""
        self._live_feed = live_feed
```

3d. In `subscribe`, open the live feed inside the existing try (so a failure follows the same poison-cleanup path), guarded so it opens only on the first subscriber (refcount still 0):

```python
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
```

3e. In `unsubscribe`, drop the live feed when refcount hits zero — **best-effort**, so a feed error never blocks local cleanup:

```python
            remaining = sub.release()
            if remaining == 0:
                sub.started = False
                if self._live_feed is not None:
                    try:
                        await self._live_feed.unsubscribe(symbol)
                    except Exception:  # noqa: BLE001 - best-effort remote drop
                        logger.warning("live feed unsubscribe failed for %s", symbol)
                self._subs.pop(symbol, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/application/subscriptions/test_manager.py -v --no-cov`
Expected: all PASS (existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add src/application/subscriptions/manager.py tests/unit/application/subscriptions/test_manager.py
git commit -m "feat(subscriptions): open/best-effort-drop live feed on (un)subscribe (Phase 4)"
```

---

## Task 10: Env-gated lifespan construction

**Files:** Modify `src/api/server.py` (the `lifespan` function), `tests/unit/api/test_server_lifespan.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/api/test_server_lifespan.py` (the existing tests enter the lifespan via `async with lifespan(app):` — follow that). These monkeypatch the client class with a spy so they assert `connect()`/`close()` actually ran (no real socket):

```python
import pytest

import src.infrastructure.adapters.xenon.client as xenon_client_mod
from src.api.server import create_app, lifespan


class _FakeBus:
    def publish(self, *a, **k) -> None: ...


class _FakeSM:
    def __init__(self) -> None:
        self.live_feed = None

    def set_live_feed(self, feed) -> None:
        self.live_feed = feed


class _SpyClient:
    last = None

    def __init__(self, url, event_bus=None, **kw) -> None:
        self.url = url
        self.event_bus = event_bus
        self.connected = False
        self.closed = False
        _SpyClient.last = self

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_lifespan_builds_connects_and_wires_xenon_client(monkeypatch) -> None:
    monkeypatch.setenv("APEX_XENON_WS_URL", "ws://127.0.0.1:1")
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    monkeypatch.setattr(xenon_client_mod, "XenonTickClient", _SpyClient)
    app = create_app()
    app.state.event_bus = _FakeBus()
    app.state.subscription_manager = _FakeSM()
    async with lifespan(app):
        assert isinstance(app.state.xenon_client, _SpyClient)
        assert app.state.xenon_client.connected is True
        assert app.state.subscription_manager.live_feed is app.state.xenon_client
    # after shutdown:
    assert _SpyClient.last.closed is True


@pytest.mark.asyncio
async def test_lifespan_skips_xenon_client_when_url_unset(monkeypatch) -> None:
    monkeypatch.delenv("APEX_XENON_WS_URL", raising=False)
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    monkeypatch.setattr(xenon_client_mod, "XenonTickClient", _SpyClient)
    app = create_app()
    app.state.event_bus = _FakeBus()
    async with lifespan(app):
        assert getattr(app.state, "xenon_client", None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/api/test_server_lifespan.py -k "xenon" -v --no-cov`
Expected: FAIL (`xenon_client` never set / AttributeError).

- [ ] **Step 3: Implement the lifespan wiring**

In `src/api/server.py`, inside `lifespan`, after the Phase 3 signal-surface block (after the `signal_repo` lines, before `try: yield`), add:

```python
    # Phase 4 live-in (env-gated): connect to xenon's tick feed when configured.
    # Mirrors the pre-injection guard above so tests can inject a fake/spy client.
    if getattr(app.state, "xenon_client", None) is None:
        xenon_url = os.environ.get("APEX_XENON_WS_URL")
        bus = getattr(app.state, "event_bus", None)
        if xenon_url and bus is not None:
            from src.infrastructure.adapters.xenon.client import XenonTickClient

            client = XenonTickClient(xenon_url, event_bus=bus)
            app.state.xenon_client = client
            sm = getattr(app.state, "subscription_manager", None)
            if sm is not None and hasattr(sm, "set_live_feed"):
                sm.set_live_feed(client)
            await client.connect()  # non-blocking: launches the background loop
        else:
            app.state.xenon_client = None
```

> The import is inside the `if` so the spy monkeypatch on
> `src.infrastructure.adapters.xenon.client.XenonTickClient` is picked up.

And replace the `finally:` block so the client is closed on shutdown:

```python
    try:
        yield
    finally:
        client = getattr(app.state, "xenon_client", None)
        if client is not None:
            await client.close()
        if app.state.pg_pool is not None:
            await app.state.pg_pool.close()
            logger.info("PostgreSQL pool closed")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/api/test_server_lifespan.py -v --no-cov`
Expected: all PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/api/server.py tests/unit/api/test_server_lifespan.py
git commit -m "feat(api): env-gated xenon live-feed wiring in lifespan (Phase 4)"
```

---

## Task 11: End-to-end — live tick reaches the real compute pipeline

Proves the NEW Phase-4 seam: a tick pushed by the fake xenon server flows through the real `XenonTickClient` → real `PriorityEventBus` → real `TASignalService` `BarAggregator` and produces a `BAR_CLOSE` when ticks cross a bar boundary. The `closed` assertion IS the CI regression guard — if the publish→pipeline seam breaks, no `BAR_CLOSE` is produced and the test fails. (Downstream bar→indicator→rule→signal→emitter→socket is already covered by Phase 2/3 tests and not re-proven here.)

**Files:** Create `tests/integration/test_xenon_live_e2e.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/test_xenon_live_e2e.py`:

```python
from __future__ import annotations

import asyncio

import pytest

from src.application.services.ta_signal_service import TASignalService
from src.domain.events.event_types import EventType
from src.domain.events.priority_event_bus import PriorityEventBus
from src.infrastructure.adapters.xenon.client import XenonTickClient
from tests.support.fake_xenon import FakeXenonServer


@pytest.mark.asyncio
async def test_live_tick_drives_bar_close_through_real_pipeline() -> None:
    # The bus is intentionally NOT started: PriorityEventBus.publish has a
    # documented sync fallback (`if not self._running: self._dispatch_sync(...)`,
    # priority_event_bus.py:195) so every publish dispatches synchronously to
    # subscribers -- deterministic, exercising the real handler chain. Do NOT
    # call `await bus.start()`: that switches to async lanes and makes this racy.
    bus = PriorityEventBus()
    closed: list = []
    bus.subscribe(EventType.BAR_CLOSE, lambda ev: closed.append(ev))

    service = TASignalService(event_bus=bus, timeframes=["1m"])
    await service.start()

    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, bus, reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection()
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)

        # Two ticks two minutes apart -> the first 1m bar closes when the second
        # tick opens a new bar window.
        await server.push({"type": "batch", "updates": {"AAPL": {
            "symbol": "AAPL", "last": 100.0, "volume": 5,
            "timestamp": "2026-06-14T15:00:10Z"}}})
        await server.push({"type": "batch", "updates": {"AAPL": {
            "symbol": "AAPL", "last": 101.0, "volume": 7,
            "timestamp": "2026-06-14T15:01:10Z"}}})

        for _ in range(200):
            if closed:
                break
            await asyncio.sleep(0.01)

        await client.close()

    await service.stop()

    assert closed, "no BAR_CLOSE produced from live ticks (publish->pipeline seam broken)"
    assert closed[0].symbol == "AAPL"
    assert closed[0].timeframe == "1m"
```

- [ ] **Step 2: Confirm the assertion is load-bearing (one-time sanity check)**

The `closed` assertion fails in CI whenever the seam regresses, so it is the real guard. To convince yourself once: temporarily change `_publish` in `client.py` to `pass`, run the test, see it FAIL with the `no BAR_CLOSE` message, then restore `_publish`.

- [ ] **Step 3: Run test to verify it passes**

Run: `uv run python -m pytest tests/integration/test_xenon_live_e2e.py -v --no-cov`
Expected: PASS (1 passed).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_xenon_live_e2e.py
git commit -m "test(xenon): e2e live tick -> real pipeline -> BAR_CLOSE (Phase 4)"
```

---

## Task 12: Full verification & quality gates

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit + integration suite**

Run: `uv run python -m pytest tests/unit tests/integration -q --no-cov`
Expected: all PASS, no errors. Record the pass count.

- [ ] **Step 2: Format**

Run: `uv run black src tests && uv run isort src tests`
Expected: files reformatted/clean.

- [ ] **Step 3: Lint**

Run: `uv run flake8 src/infrastructure/adapters/xenon src/domain/interfaces/live_feed.py src/application/subscriptions/manager.py src/api/server.py tests/support/fake_xenon.py tests/unit/infrastructure/xenon tests/integration/test_xenon_live_e2e.py`
Expected: no findings.

- [ ] **Step 4: Type check the new modules**

Run: `uv run mypy src/infrastructure/adapters/xenon src/domain/interfaces/live_feed.py 2>&1 | tail -20`
Expected: no new errors in the Phase 4 modules (pre-existing repo-wide errors, if any, are out of scope).

- [ ] **Step 5: Confirm the flag-OFF default behavior**

Run: `uv run python -c "import os; os.environ.pop('APEX_XENON_WS_URL', None); from src.api.server import create_app; create_app(); print('app builds with no xenon flag')"`
Expected: prints the line — proves the service constructs with the feature OFF.

- [ ] **Step 6: Final commit (if formatting changed anything)**

```bash
git add -A
git commit -m "chore: format + lint Phase 4 live-in" || echo "nothing to commit"
```

---

## Task 13 (OPTIONAL — out of default scope): stand up the pipeline in lifespan

Only do this if the goal is for the **server entrypoint itself** to stream live (not just the adapter + e2e proof). It requires a livewire bronze root and turns the flag into a true production switch. Left optional because it depends on livewire/PG config and broadens scope beyond the approved Phase 4 spec.

Sketch (TDD each piece): in `lifespan`, when `APEX_LIVEWIRE_ROOT` (+ optionally `APEX_PG_URL`) is set, construct `PriorityEventBus()` → `app.state.event_bus`; `TASignalService(event_bus=bus, timeframes=[...])`; `LivewireOhlcProvider(bronze_root=...)`; `SubscriptionManager(provider, compute=service, timeframes=[...])` → `app.state.subscription_manager`; then the existing Task 10 block attaches the xenon client as its `live_feed`. Gate everything so absence of config = today's behavior. Add a lifespan test that asserts the wired graph. **Do not** start the bus's async lanes unless you also verify the WS route/emitter expectations.

---

## Self-Review (completed by plan author, post-tribunal)

**Spec coverage:** §3 contract → Tasks 4/6/7 (frames, PriceData, ping). §3.1 auth → Task 3 + Task 6 `_connect_url` (safe ticket merge). §4 ingest/translation → Task 4. §5 components → Tasks 2,3,4,6. §5.1 dict-tick → Task 6 `_publish`. §7 subscription wiring → Task 9. §8 lifespan → Task 10. §9 resilience: reconnect+capped backoff (Task 8), keep-alive (Task 7 + websockets built-in ping), malformed-frame survival (Task 6), null-drop (Task 4), best-effort drop (Task 9). §10 testing → Tasks 5,4,6–11. §11 scope: service-JWT & reconnect-reseed deferred (matches spec); pipeline construction in lifespan deferred to optional Task 13 (Scope boundary note). §12 success: (a)→Task 11, (b)→Task 9, (c)→Task 8, (d)→Task 10 + Task 12 step 5, (e)→every frame/field traced to a verified source.

**Tribunal fixes folded in:** client-death-on-bad-frame hardening (Task 6 + test); fail-first split of ping/reconnect (Tasks 6/7/8); capped exponential backoff (Task 8); spy-client lifespan tests asserting connect/close (Task 10); unsubscribe-failure + second-subscriber tests (Task 9); close-cancels-reconnect test (Task 8); error-frame logging (Task 7); safe URL query merge (Task 6); event-based test waits (Task 6 `_RecordingBus`); overclaim corrected via Scope boundary note + Task 13. Deferred with reason: feed I/O under the manager lock (consistent with existing Phase-2 seed-under-lock design; refactor out of scope).

**Placeholder scan:** no TBD/TODO; every code step has complete code; commands have expected output.

**Type consistency:** `LiveFeedPort` methods match `_FakeLiveFeed`, `_SpyClient`(connect/close), `XenonTickClient`. `translate_price_data -> dict | None` consumed in `_publish`. `set_live_feed` defined in Task 9, used in Task 10. `reconnect_delay`/`max_reconnect_delay`/`_subscribed`/`_ws` consistent across Tasks 6–8. `EventType.MARKET_DATA_TICK`/`BAR_CLOSE` used consistently.
