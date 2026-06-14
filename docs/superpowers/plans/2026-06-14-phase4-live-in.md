# Phase 4 — Live-in Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect apex to xenon's live IB tick WS feed so a subscribed ticker streams real ticks into the existing compute pipeline, producing live TA signals — behind a feature flag that is OFF by default.

**Architecture:** A thin live-feed *adapter*, not a bar engine. apex's `BarAggregator` already does the tick→bar stitch, so Phase 4 only (1) speaks the xenon WS protocol, (2) translates each tick onto `EventType.MARKET_DATA_TICK`, and (3) wires subscribe/unsubscribe so a watched ticker opens/drops its xenon sub. Everything downstream (bars→indicators→rules→signals→argon) is unchanged.

**Tech Stack:** Python 3.13 async, `websockets` 16 (`websockets.asyncio.client.connect` / `websockets.asyncio.server.serve`), the existing `PriorityEventBus`, `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`).

**Spec:** `docs/superpowers/specs/2026-06-14-phase4-live-in-design.md`

**Verified contract facts (do not re-derive):**
- xenon server→client steady-state ticks: `{"type":"batch","updates":{SYM:PriceData}}`; on-subscribe: `{"type":"price","symbol":SYM,"data":PriceData}`; keep-alive: `{"type":"ping"}` → client MUST reply `{"action":"pong"}` (else dropped at 65s). (`~/projects/xenon/scripts/infra/ib_realtime/ib_realtime_server.js`)
- client→server: `{"action":"subscribe","symbols":[...]}` / `{"action":"unsubscribe","symbols":[...]}`.
- `PriceData` = `{symbol, last, bid, ask, volume, timestamp(ISO-8601 string), ...}`; `last` may be null. (`ib_tick_handler.js:createPriceData`)
- Loopback / no-`CLERK_JWKS_URL` connections skip ticket auth → no ticket needed for the trusted-network path (`ib_realtime_server.js:362-373`).
- apex ingest: `BarAggregator.on_tick` accepts a dict `{symbol, last|mid|price|bid+ask, volume, timestamp}`; `timestamp` is returned as-is so it must be a `datetime` (`src/domain/signals/data/bar_aggregator.py:221-264`).
- event bus: `PriorityEventBus.publish(EventType.MARKET_DATA_TICK, payload)` (sync) (`src/domain/events/priority_event_bus.py`, `src/domain/interfaces/event_bus.py:40`); `EventType.MARKET_DATA_TICK = "market_data_tick"` (`src/domain/events/event_types.py:52`).
- `SubscriptionManager.__init__(provider, compute, timeframes, seed_lookback_days=365)` (`src/application/subscriptions/manager.py:51`).
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
- `tests/unit/infrastructure/xenon/test_translator.py`
- `tests/unit/infrastructure/xenon/test_auth.py`
- `tests/unit/infrastructure/xenon/test_client.py`
- `tests/integration/test_xenon_live_e2e.py`

**Modify:**
- `pyproject.toml` — add `websockets>=16.0` to the `api` extra.
- `src/domain/interfaces/__init__.py` — export `LiveFeedPort`.
- `src/application/subscriptions/manager.py` — optional `live_feed`; open on subscribe, drop on unsubscribe; `set_live_feed()`.
- `tests/unit/application/subscriptions/test_manager.py` — live-feed wiring tests.
- `src/api/server.py` — env-gated `XenonTickClient` construction in `lifespan`.
- `tests/unit/api/test_server_lifespan.py` — lifespan wiring tests.

---

## Task 1: Declare the `websockets` dependency

**Files:**
- Modify: `pyproject.toml` (the `api` optional-dependencies group)

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
- Create: `src/domain/interfaces/live_feed.py`
- Modify: `src/domain/interfaces/__init__.py`
- Test: `tests/unit/infrastructure/xenon/test_client.py` (the port-satisfaction check is added with the client in Task 6; here we only add a focused interface test)
- Test: `tests/unit/infrastructure/xenon/__init__.py` (empty package marker)

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
- Create: `src/infrastructure/adapters/xenon/__init__.py`
- Create: `src/infrastructure/adapters/xenon/auth.py`
- Test: `tests/unit/infrastructure/xenon/test_auth.py`

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

**Files:**
- Create: `src/infrastructure/adapters/xenon/translator.py`
- Test: `tests/unit/infrastructure/xenon/test_translator.py`

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
    # ISO string parsed to a tz-aware datetime (aggregator compares datetimes)
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

**Files:**
- Create: `tests/support/__init__.py`
- Create: `tests/support/fake_xenon.py`
- Test: `tests/unit/infrastructure/xenon/test_fake_xenon.py` (self-test of the double)

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
frames, can push server frames (price/batch/ping), and can drop connections to
exercise reconnect. Not collected by pytest (filename is not test_*).
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

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_fake_xenon.py -v --no-cov`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add tests/support/__init__.py tests/support/fake_xenon.py tests/unit/infrastructure/xenon/test_fake_xenon.py
git commit -m "test(xenon): add FakeXenonServer test double (Phase 4)"
```

---

## Task 6: `XenonTickClient` — connect, subscribe, publish ticks

**Files:**
- Create: `src/infrastructure/adapters/xenon/client.py`
- Test: `tests/unit/infrastructure/xenon/test_client.py`

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
    """Captures (event_type, payload) published to it."""

    def __init__(self) -> None:
        self.published: list = []

    def publish(self, event_type, payload, priority=None) -> None:
        self.published.append((event_type, payload))


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
        # give the receive loop a moment to process + publish
        for _ in range(100):
            if bus.published:
                break
            await asyncio.sleep(0.01)

        assert len(bus.published) == 1
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
        for _ in range(100):
            if bus.published:
                break
            await asyncio.sleep(0.01)
        assert bus.published[0][1]["last"] == 99.0
        await client.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_client.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: ...xenon.client`.

- [ ] **Step 3: Implement the client (connect/subscribe/publish; ping & reconnect added in later tasks but included now to avoid churn)**

Create `src/infrastructure/adapters/xenon/client.py`:

```python
"""WS client to xenon's ib_realtime server -> apex event bus (Phase 4).

Implements LiveFeedPort. Owns one resilient websockets connection: sends
subscribe/unsubscribe action frames, receives `price`/`batch` ticks and
republishes each (translated) on EventType.MARKET_DATA_TICK, answers the
server's application-level `{"type":"ping"}` keep-alive, and reconnects with a
fixed delay while re-subscribing the active set.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Set

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
    ) -> None:
        self._url = url
        self._bus = event_bus
        self._auth: AuthProvider = auth or NoAuthProvider()
        self._reconnect_delay = reconnect_delay
        self._subscribed: Set[str] = set()
        self._ws: Any = None
        self._task: Optional[asyncio.Task[None]] = None
        self._closing = False
        self._connected = asyncio.Event()

    # ---- LiveFeedPort ----------------------------------------------------
    async def connect(self) -> None:
        """Launch the background receive/reconnect loop (non-blocking)."""
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
        ticket = await self._auth.ticket()
        return f"{self._url}?ticket={ticket}" if ticket else self._url

    async def _send(self, payload: dict) -> None:
        ws = self._ws
        if ws is None:
            return  # not connected yet; _resubscribe will replay on (re)connect
        try:
            await ws.send(json.dumps(payload))
        except ConnectionClosed:
            pass

    async def _resubscribe(self) -> None:
        if self._subscribed:
            await self._send({"action": "subscribe", "symbols": sorted(self._subscribed)})

    async def _run(self) -> None:
        while not self._closing:
            try:
                async with connect(await self._connect_url()) as ws:
                    self._ws = ws
                    self._connected.set()
                    await self._resubscribe()
                    async for raw in ws:
                        await self._handle(raw)
            except (ConnectionClosed, OSError) as exc:
                logger.warning("xenon WS connection lost: %s", exc)
            except asyncio.CancelledError:
                break
            finally:
                self._ws = None
                self._connected.clear()
            if self._closing:
                break
            await asyncio.sleep(self._reconnect_delay)

    async def _handle(self, raw: Any) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("xenon WS: dropping non-JSON frame")
            return
        mtype = msg.get("type")
        if mtype == "ping":
            await self._send({"action": "pong"})
        elif mtype == "batch":
            for data in (msg.get("updates") or {}).values():
                self._publish(data)
        elif mtype == "price":
            self._publish(msg.get("data") or {})
        # status / subscribed / unsubscribed / error: ignored (no action needed)

    def _publish(self, data: dict) -> None:
        tick = translate_price_data(data)
        if tick is not None:
            self._bus.publish(EventType.MARKET_DATA_TICK, tick)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/test_client.py -v --no-cov`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/infrastructure/adapters/xenon/client.py tests/unit/infrastructure/xenon/test_client.py
git commit -m "feat(xenon): WS client publishes live ticks to the event bus (Phase 4)"
```

---

## Task 7: `XenonTickClient` — keep-alive ping/pong

**Files:**
- Modify: `tests/unit/infrastructure/xenon/test_client.py` (add a test; implementation already present from Task 6)

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

- [ ] **Step 2: Run test to verify it passes (implementation already handles ping)**

Run: `uv run python -m pytest "tests/unit/infrastructure/xenon/test_client.py::test_server_ping_is_answered_with_pong" -v --no-cov`
Expected: PASS.

> Note: the Task 6 implementation already replies to `{"type":"ping"}`. This task exists to lock that behavior behind an explicit test. If for any reason it had NOT been implemented, the test would fail first (RED) — confirming the test is real.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/infrastructure/xenon/test_client.py
git commit -m "test(xenon): lock keep-alive ping/pong behavior (Phase 4)"
```

---

## Task 8: `XenonTickClient` — reconnect + re-subscribe

**Files:**
- Modify: `tests/unit/infrastructure/xenon/test_client.py` (add a test; implementation already present from Task 6)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/infrastructure/xenon/test_client.py`:

```python
@pytest.mark.asyncio
async def test_reconnects_and_resubscribes_after_drop() -> None:
    async with FakeXenonServer() as server:
        client = XenonTickClient(server.url, _RecordingBus(), reconnect_delay=0.01)
        await client.connect()
        await server.wait_for_connection(1)
        await client.subscribe("AAPL")
        await server.wait_for_frames(1)  # initial subscribe

        await server.drop_connections()          # force a reconnect
        await server.wait_for_connection(2)       # client dialed back in
        # on reconnect the client replays its active set as one subscribe frame
        await server.wait_for_frames(2)
        assert server.received[-1] == {"action": "subscribe", "symbols": ["AAPL"]}
        assert server.connections >= 2
        await client.close()
```

- [ ] **Step 2: Run test to verify it passes (implementation already reconnects)**

Run: `uv run python -m pytest "tests/unit/infrastructure/xenon/test_client.py::test_reconnects_and_resubscribes_after_drop" -v --no-cov`
Expected: PASS.

> Note: the Task 6 `_run` loop already reconnects with `reconnect_delay` and calls `_resubscribe`. If reconnect/resubscribe were missing, `wait_for_connection(2)` or `wait_for_frames(2)` would time out (RED).

- [ ] **Step 3: Run the full client + translator + auth suite**

Run: `uv run python -m pytest tests/unit/infrastructure/xenon/ -v --no-cov`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/infrastructure/xenon/test_client.py
git commit -m "test(xenon): lock reconnect + re-subscribe behavior (Phase 4)"
```

---

## Task 9: Wire `LiveFeedPort` into `SubscriptionManager`

**Files:**
- Modify: `src/application/subscriptions/manager.py`
- Test: `tests/unit/application/subscriptions/test_manager.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/application/subscriptions/test_manager.py` (keep existing imports; add what's missing):

```python
class _FakeLiveFeed:
    def __init__(self) -> None:
        self.subscribed: list = []
        self.unsubscribed: list = []
        self.fail = False

    async def connect(self) -> None: ...

    async def subscribe(self, symbol: str) -> None:
        if self.fail:
            raise RuntimeError("feed down")
        self.subscribed.append(symbol)

    async def unsubscribe(self, symbol: str) -> None:
        self.unsubscribed.append(symbol)

    async def close(self) -> None: ...


@pytest.mark.asyncio
async def test_subscribe_opens_live_feed() -> None:
    feed = _FakeLiveFeed()
    mgr = _make_manager()          # existing helper used elsewhere in this file
    mgr.set_live_feed(feed)
    await mgr.subscribe("AAPL")
    assert feed.subscribed == ["AAPL"]


@pytest.mark.asyncio
async def test_unsubscribe_drops_live_feed_at_refcount_zero() -> None:
    feed = _FakeLiveFeed()
    mgr = _make_manager()
    mgr.set_live_feed(feed)
    await mgr.subscribe("AAPL")
    await mgr.unsubscribe("AAPL")
    assert feed.unsubscribed == ["AAPL"]


@pytest.mark.asyncio
async def test_live_feed_failure_leaves_no_poisoned_entry() -> None:
    feed = _FakeLiveFeed()
    feed.fail = True
    mgr = _make_manager()
    mgr.set_live_feed(feed)
    with pytest.raises(RuntimeError):
        await mgr.subscribe("AAPL")
    assert mgr.refcount("AAPL") == 0
    assert "AAPL" not in mgr.active_symbols()
```

If `_make_manager()` does not already exist in the test file, add this helper near the top (after imports), using the existing fakes in that file (a fake provider + fake compute). If the file already builds a manager inline per test, mirror that construction instead — the key is a manager with a fake provider/compute and the timeframes the other tests use.

```python
def _make_manager():
    # Mirror the construction the existing tests in this file already use.
    from src.application.subscriptions.manager import SubscriptionManager
    return SubscriptionManager(provider=_FakeProvider(), compute=_FakeCompute(), timeframes=["1d"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/application/subscriptions/test_manager.py -k "live_feed" -v --no-cov`
Expected: FAIL with `AttributeError: 'SubscriptionManager' object has no attribute 'set_live_feed'`.

- [ ] **Step 3: Implement the wiring**

In `src/application/subscriptions/manager.py`:

3a. Add `live_feed` to `__init__` (keyword, default `None`) and store it. Change the signature and body:

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

3b. Add a setter just below `__init__`:

```python
    def set_live_feed(self, live_feed: Any) -> None:
        """Attach a LiveFeedPort after construction (used by the app lifespan)."""
        self._live_feed = live_feed
```

3c. In `subscribe`, open the live feed inside the existing try (so a failure follows the same poison-cleanup path), right after `sub.started = True`:

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

3d. In `unsubscribe`, drop the live feed when refcount hits zero, before popping the entry:

```python
            remaining = sub.release()
            if remaining == 0:
                sub.started = False
                if self._live_feed is not None:
                    await self._live_feed.unsubscribe(symbol)
                self._subs.pop(symbol, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/application/subscriptions/test_manager.py -v --no-cov`
Expected: all PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/application/subscriptions/manager.py tests/unit/application/subscriptions/test_manager.py
git commit -m "feat(subscriptions): open/drop live feed on subscribe/unsubscribe (Phase 4)"
```

---

## Task 10: Env-gated lifespan construction

**Files:**
- Modify: `src/api/server.py` (the `lifespan` function)
- Test: `tests/unit/api/test_server_lifespan.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/api/test_server_lifespan.py` (mirror the existing pattern in that file for entering the lifespan; the canonical pattern is shown below):

```python
import pytest

from src.api.server import create_app, lifespan


class _FakeBus:
    def publish(self, *a, **k) -> None: ...


class _FakeSM:
    def __init__(self) -> None:
        self.live_feed = None

    def set_live_feed(self, feed) -> None:
        self.live_feed = feed


@pytest.mark.asyncio
async def test_lifespan_builds_xenon_client_when_url_set(monkeypatch) -> None:
    monkeypatch.setenv("APEX_XENON_WS_URL", "ws://127.0.0.1:1")
    monkeypatch.delenv("APEX_PG_URL", raising=False)
    app = create_app()
    app.state.event_bus = _FakeBus()
    app.state.subscription_manager = _FakeSM()
    async with lifespan(app):
        from src.infrastructure.adapters.xenon.client import XenonTickClient

        assert isinstance(app.state.xenon_client, XenonTickClient)
        assert app.state.subscription_manager.live_feed is app.state.xenon_client


@pytest.mark.asyncio
async def test_lifespan_skips_xenon_client_when_url_unset(monkeypatch) -> None:
    monkeypatch.delenv("APEX_XENON_WS_URL", raising=False)
    monkeypatch.delenv("APEX_PG_URL", raising=False)
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
    # Mirrors the pre-injection guard above so tests can inject a fake client.
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

And in the `finally:` block, close it on shutdown (before/after the PG pool close):

```python
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

**Files:**
- Create: `tests/integration/test_xenon_live_e2e.py`

This proves the NEW Phase-4 seam deterministically: a tick pushed by the fake xenon server flows through the real `XenonTickClient` → real `PriorityEventBus` → real `TASignalService` `BarAggregator` and produces a `BAR_CLOSE` when ticks cross a bar boundary. (Downstream bar→indicator→rule→signal→emitter→socket is already covered by Phase 2/3 tests and is not re-proven here.)

- [ ] **Step 1: Write the failing test**

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

        # Two ticks two minutes apart -> the first 1m bar closes when the
        # second tick opens a new bar window.
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

    assert closed, "no BAR_CLOSE produced from live ticks"
    assert closed[0].symbol == "AAPL"
    assert closed[0].timeframe == "1m"
```

- [ ] **Step 2: Run test to verify it fails first**

Temporarily break the chain to confirm the test is real: it should already PASS once the client is implemented, so to satisfy RED-first discipline, first run it with the client's `_publish` short-circuited is NOT necessary — instead confirm the test fails before Task 6 exists. Since Task 6 is already done by now, run it and confirm it PASSES, then **prove the test bites** by running:

Run: `uv run python -m pytest tests/integration/test_xenon_live_e2e.py -v --no-cov`
Expected: PASS. Then, as the RED check, comment out the `bus.subscribe(...closed.append...)` line and re-run → it must FAIL with the `no BAR_CLOSE` assertion, proving the assertion is load-bearing. Restore the line.

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

## Self-Review (completed by plan author)

**Spec coverage:** §3 contract → Tasks 4/5/6 (frames, PriceData, ping). §3.1 auth → Task 3 (NoAuthProvider) + Task 6 (`_connect_url` ticket append). §4 ingest/translation → Task 4. §5 components → Tasks 2,3,4,6. §5.1 dict-tick → Task 6 `_publish`. §7 subscription wiring → Task 9. §8 lifespan → Task 10. §9 resilience (reconnect/ping/null-drop) → Tasks 6,7,8 + Task 4 null-drop. §10 testing (fake server, translator, client, wiring, e2e) → Tasks 5,4,6–8,9–11. §11 scope: service-JWT & reconnect-reseed explicitly NOT implemented (deferred, matches spec). §12 success criteria (a)→Task 11, (b)→Task 9, (c)→Task 8, (d)→Task 10 + Task 12 step 5, (e)→every frame/field traced to a verified source.

**Placeholder scan:** no TBD/TODO; every code step has complete code; commands have expected output.

**Type consistency:** `LiveFeedPort` methods (connect/subscribe/unsubscribe/close) match `_FakeLiveFeed`, `XenonTickClient`, and the `_Conforming` test. `translate_price_data` returns `dict | None`, consumed as such in `_publish`. `set_live_feed` defined in Task 9, used in Task 10. `EventType.MARKET_DATA_TICK` used consistently. `reconnect_delay`, `_subscribed`, `_ws` names consistent across Tasks 6–8.

**Known dependency:** Tasks 7, 8, 11 rely on the Task 6 implementation already containing ping + reconnect (intentional — they are written into the client once to avoid edit churn, then locked by tests). Each such task documents how to confirm the test is load-bearing (RED check).
