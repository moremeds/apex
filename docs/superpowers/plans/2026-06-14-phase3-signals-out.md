# Phase 3 — Signals-out (WS + REST) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve apex's TA signals to argon two ways — a **WS push** stream and a **REST pull** (`GET /signals/{ticker}?since=…`) — with every payload validated against `signal_service_payload.schema.json` and persisted to the authoritative `ta_signals` table.

**Architecture:** A `SignalPayloadBuilder` turns domain signals into schema-valid `signal_service_payload` dicts. A REST route reads recent rows from `ta_signals` and returns a payload. A WS route accepts argon's `{action: subscribe/unsubscribe, ticker}` frames, drives the Phase 2 `SubscriptionManager`, and pushes payloads to connected clients via a `SignalHub` fan-out. Persistence reuses the existing `ta_signal_repository`. All wired into the existing FastAPI app factory (`src/api/server.py`).

**Tech Stack:** Python 3.13, FastAPI (incl. `WebSocket`), asyncpg pool (`app.state.pg_pool`), `jsonschema`, existing `ta_signals` table (migration 005).

**Spec:** `docs/superpowers/specs/2026-06-14-apex-adaptation-design.md` §5 (Phase 3). **Depends on:** Phase 2 (`SubscriptionManager`). **D1 (own vs shared PG)** stays open — this plan uses `app.state.pg_pool` regardless of which DB it points at.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/api/payload/builder.py` | `build_payload(signals, ...)` → schema-valid `signal_service_payload` dict. |
| `src/api/payload/validate.py` | `validate_payload(dict)` against the JSON schema (load once). |
| `src/api/routes/signals.py` | `GET /signals/{ticker}` REST pull. |
| `src/api/ws/hub.py` | `SignalHub` — connection registry + fan-out. |
| `src/api/ws/signals_ws.py` | `WS /ws/signals` — subscribe protocol + push. |
| `tests/unit/api/test_payload_builder.py` | builder produces schema-valid output. |
| `tests/unit/api/test_signal_hub.py` | hub register/unregister/broadcast. |
| `tests/integration/api/test_signals_rest.py` | REST endpoint via `httpx.AsyncClient`. |
| `tests/integration/api/test_signals_ws.py` | WS subscribe→receive via `TestClient`. |
| `src/api/server.py` | register the new router + WS route + hub (modify). |

**Reuse (do not reinvent):** `src/infrastructure/persistence/repositories/ta_signal_repository.py` for reads/writes. Verify its method names in Task 3 Step 1.

**`ta_signals` columns (migration 005, verified):** `time, signal_id, symbol, timeframe, category, indicator, direction, strength, priority, trigger_rule, current_value, threshold, previous_value, message, cooldown_until, metadata, created_at`.

---

## Task 1: Payload schema validator

**Files:**
- Create: `src/api/payload/__init__.py`
- Create: `src/api/payload/validate.py`
- Test: `tests/unit/api/test_payload_validate.py`

- [ ] **Step 1: Confirm jsonschema is available**

Run: `uv run python -c "import jsonschema; print(jsonschema.__version__)"`
Expected: a version prints. If `ModuleNotFoundError`, add `jsonschema` to `pyproject.toml` `[project].dependencies` and run `uv sync`, then commit that change first.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/api/test_payload_validate.py`:

```python
from __future__ import annotations

import pytest

from src.api.payload.validate import ValidationFailure, validate_payload


def _valid_signal() -> dict:
    return {
        "signal_id": "momentum:rsi:AAPL:1d",
        "symbol": "AAPL",
        "category": "momentum",
        "indicator": "RSI",
        "direction": "buy",
        "strength": 72,
        "priority": "high",
        "timeframe": "1d",
        "trigger_rule": "rsi_oversold_cross",
        "current_value": 28.4,
        "timestamp": "2026-06-14T12:00:00Z",
    }


def test_valid_payload_passes() -> None:
    validate_payload({"signals": [_valid_signal()], "timestamp": "2026-06-14T12:00:00Z"})


def test_missing_required_field_fails() -> None:
    bad = _valid_signal()
    del bad["strength"]
    with pytest.raises(ValidationFailure):
        validate_payload({"signals": [bad], "timestamp": "2026-06-14T12:00:00Z"})


def test_bad_signal_id_pattern_fails() -> None:
    bad = _valid_signal()
    bad["signal_id"] = "NOT A VALID ID"
    with pytest.raises(ValidationFailure):
        validate_payload({"signals": [bad], "timestamp": "2026-06-14T12:00:00Z"})
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_payload_validate.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Implement the validator**

Create `src/api/payload/__init__.py`:

```python
"""Signal-service payload construction and validation."""
```

Create `src/api/payload/validate.py`:

```python
"""Validate outgoing payloads against signal_service_payload.schema.json.

The schema is the contract with argon (argon-adaptation.md §5). Every payload
apex sends — REST or WS — passes through here first.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "config" / "verification" / "schemas" / "signal_service_payload.schema.json"
)


class ValidationFailure(ValueError):
    """Raised when a payload does not satisfy the contract schema."""


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_payload(payload: dict[str, Any]) -> None:
    try:
        jsonschema.validate(instance=payload, schema=_schema())
    except jsonschema.ValidationError as exc:  # noqa: PERF203
        raise ValidationFailure(str(exc)) from exc
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/test_payload_validate.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/api/payload/ tests/unit/api/test_payload_validate.py
git commit -m "feat(api): payload schema validator"
```

---

## Task 2: Payload builder (domain signal → contract dict)

**Files:**
- Create: `src/api/payload/builder.py`
- Test: `tests/unit/api/test_payload_builder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/api/test_payload_builder.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from src.api.payload.builder import build_payload, signal_row_to_dict
from src.api.payload.validate import validate_payload


def _row() -> dict:
    return {
        "time": datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc),
        "signal_id": "trend:macd:AAPL:1d",
        "symbol": "AAPL",
        "timeframe": "1d",
        "category": "trend",
        "indicator": "MACD",
        "direction": "buy",
        "strength": 65,
        "priority": "medium",
        "trigger_rule": "macd_bull_cross",
        "current_value": 1.23,
        "threshold": 0.0,
        "previous_value": -0.4,
        "message": "MACD bullish cross",
        "cooldown_until": None,
        "metadata": {"fast": 12, "slow": 26},
    }


def test_row_to_dict_maps_time_to_timestamp() -> None:
    d = signal_row_to_dict(_row())
    assert d["timestamp"] == "2026-06-14T12:00:00+00:00"
    assert "time" not in d


def test_build_payload_is_schema_valid() -> None:
    payload = build_payload([_row()], generated_at=datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc))
    validate_payload(payload)  # must not raise
    assert payload["symbol_count"] == 1
    assert payload["signals"][0]["symbol"] == "AAPL"


def test_null_current_value_row_is_dropped() -> None:
    bad = _row()
    bad["current_value"] = None  # schema requires a number -> drop, not emit invalid
    payload = build_payload([bad, _row()], generated_at=datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc))
    validate_payload(payload)
    assert len(payload["signals"]) == 1  # the null-current_value row was dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_payload_builder.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the builder**

Create `src/api/payload/builder.py`:

```python
"""Build signal_service_payload dicts from ta_signals rows.

Maps the DB column `time` → schema field `timestamp`; drops DB-only columns
(`created_at`). Keys not in the schema's signal object are simply omitted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

# ta_signals columns that map 1:1 onto the schema's trading_signal object.
# NOTE: lifecycle fields (status/invalidated_by/invalidated_at) are intentionally
# OMITTED — migration 005 has no such columns, so apex cannot source them yet.
# Lifecycle persistence is a deferred item (see self-review). The schema does not
# require them (status has a default), so omitting keeps payloads valid.
_SIGNAL_FIELDS = (
    "signal_id", "symbol", "category", "indicator", "direction", "strength",
    "priority", "timeframe", "trigger_rule", "current_value", "threshold",
    "previous_value", "message", "cooldown_until", "metadata",
)

# Schema-required numeric fields that must NOT be null (else the row is invalid).
_REQUIRED_NUMERIC = ("current_value",)


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def signal_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _SIGNAL_FIELDS:
        if key in row and row[key] is not None:
            out[key] = _iso(row[key])
    # DB stores the event time as `time`; the contract field is `timestamp`.
    out["timestamp"] = _iso(row["time"])
    return out


def is_emittable(row: dict[str, Any]) -> bool:
    """A row is emittable only if its schema-required numeric fields are non-null.

    ta_signals.current_value is nullable, but the contract requires a number — so
    rows missing it are dropped (and logged by the caller) rather than emitted invalid.
    """
    return all(row.get(k) is not None for k in _REQUIRED_NUMERIC)


def build_payload(rows: Iterable[dict[str, Any]], generated_at: datetime) -> dict[str, Any]:
    # Drop rows that would be schema-invalid (e.g. null current_value).
    emittable = [r for r in rows if is_emittable(r)]
    signals = [signal_row_to_dict(r) for r in emittable]
    return {
        "signals": signals,
        "timestamp": generated_at.isoformat(),
        "symbol_count": len({s["symbol"] for s in signals}),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/test_payload_builder.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/api/payload/builder.py tests/unit/api/test_payload_builder.py
git commit -m "feat(api): signal payload builder"
```

---

## Task 3: REST pull endpoint `GET /signals/{ticker}`

**Files:**
- Read: `src/infrastructure/persistence/repositories/ta_signal_repository.py` (Step 1)
- Create: `src/api/routes/signals.py`
- Modify: `src/api/server.py` (register router)
- Test: `tests/integration/api/test_signals_rest.py`

- [ ] **Step 1: Verify the repository read method**

Run: `grep -nE "class |async def |def " src/infrastructure/persistence/repositories/ta_signal_repository.py | head -40`
Expected: a read method (e.g. `fetch_signals(symbol, since=...)`). Use its real name in Step 3; if no read method exists, add one that runs:
`SELECT * FROM ta_signals WHERE symbol = $1 AND time >= $2 ORDER BY time DESC LIMIT $3`.

- [ ] **Step 2: Write the failing test**

Create `tests/integration/api/test_signals_rest.py`:

```python
"""REST pull endpoint returns a schema-valid payload."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.payload.validate import validate_payload
from src.api.server import create_app


class _FakeRepo:
    async def fetch_signals(self, symbol, since=None, limit=500):
        return [{
            "time": __import__("datetime").datetime(2026, 6, 14, 12, 0,
                     tzinfo=__import__("datetime").timezone.utc),
            "signal_id": "trend:macd:AAPL:1d", "symbol": symbol, "timeframe": "1d",
            "category": "trend", "indicator": "MACD", "direction": "buy",
            "strength": 65, "priority": "medium", "trigger_rule": "macd_bull_cross",
            "current_value": 1.23, "threshold": 0.0, "previous_value": -0.4,
            "message": "x", "cooldown_until": None, "metadata": {},
        }]


@pytest.mark.asyncio
async def test_get_signals_returns_valid_payload() -> None:
    app = create_app()
    app.state.signal_repo = _FakeRepo()  # injected stand-in
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/signals/AAPL")
    assert resp.status_code == 200
    payload = resp.json()
    validate_payload(payload)
    assert payload["signals"][0]["symbol"] == "AAPL"
```

- [ ] **Step 3: Implement the route**

Create `src/api/routes/signals.py`:

```python
"""REST pull endpoint for TA signals (argon backfill on load/reconnect/?asof)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request

from src.api.payload.builder import build_payload
from src.api.payload.validate import validate_payload

router = APIRouter(tags=["signals"])


@router.get("/signals/{ticker}")
async def get_signals(ticker: str, request: Request, since: Optional[datetime] = None) -> dict:
    repo = request.app.state.signal_repo
    rows = await repo.fetch_signals(ticker, since=since)
    payload = build_payload(rows, generated_at=datetime.now(timezone.utc))
    validate_payload(payload)   # contract guarantee on every REST response
    return payload
```

Modify `src/api/server.py` `create_app()` to register it (follow the existing router-include pattern):

```python
    from src.api.routes.signals import router as signals_router
    app.include_router(signals_router)
```

**Repository wiring (ISSUE-4).** `TASignalRepository.__init__(self, db: Database)` takes
apex's `Database` wrapper (verified 2026-06-14), **not** an `asyncpg.Pool`. In `lifespan`,
construct and connect a `Database`, then the repo — guarded so tests can pre-inject fakes:

```python
    if getattr(app.state, "signal_repo", None) is None and pg_url:
        from src.infrastructure.persistence.database import Database
        from src.infrastructure.persistence.repositories.ta_signal_repository import TASignalRepository
        db = Database(config.database)   # config.database = DatabaseConfig (verify the attr in Step 1)
        await db.connect()
        app.state.signal_db = db
        app.state.signal_repo = TASignalRepository(db)
```

Close `app.state.signal_db` in the lifespan `finally` block alongside the existing pool teardown.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/api/test_signals_rest.py -v`
Expected: PASS (`now(timezone.utc)` is allowed in app code; only workflow scripts forbid it).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/signals.py src/api/server.py tests/integration/api/test_signals_rest.py
git commit -m "feat(api): GET /signals/{ticker} REST pull"
```

---

## Task 4: `SignalHub` fan-out

**Files:**
- Create: `src/api/ws/__init__.py`
- Create: `src/api/ws/hub.py`
- Test: `tests/unit/api/test_signal_hub.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/api/test_signal_hub.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_signal_hub.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the hub**

Create `src/api/ws/__init__.py`:

```python
"""WebSocket signal push surface."""
```

Create `src/api/ws/hub.py`:

```python
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
            except Exception:  # noqa: BLE001 — drop broken sockets
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)  # full removal of a dead socket
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/test_signal_hub.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/api/ws/ tests/unit/api/test_signal_hub.py
git commit -m "feat(api): SignalHub WS fan-out"
```

---

## Task 5: WS endpoint `WS /ws/signals` (subscribe protocol)

**Files:**
- Create: `src/api/ws/signals_ws.py`
- Modify: `src/api/server.py` (register WS route + hub + manager on app.state)
- Test: `tests/integration/api/test_signals_ws.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/api/test_signals_ws.py`:

```python
"""WS subscribe drives the manager, sends an initial snapshot, and cleans up."""

from __future__ import annotations

from starlette.testclient import TestClient

from src.api.server import create_app
from src.api.ws.hub import SignalHub


class _FakeMgr:
    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []

    async def subscribe(self, t: str) -> None:
        self.subscribed.append(t)

    async def unsubscribe(self, t: str) -> None:
        self.unsubscribed.append(t)


def test_ws_subscribe_acks_and_disconnect_decrements_refcount() -> None:
    app = create_app()
    # Pre-inject fakes; the guarded lifespan (run by `with TestClient`) won't clobber them.
    app.state.signal_hub = SignalHub()
    app.state.subscription_manager = _FakeMgr()
    app.state.signal_repo = None  # no snapshot in this test
    with TestClient(app) as client:               # context form runs lifespan (ISSUE-13)
        with client.websocket_connect("/ws/signals") as ws:
            ws.send_json({"action": "subscribe", "ticker": "AAPL"})
            assert ws.receive_json() == {"status": "subscribed", "ticker": "AAPL"}
        # leaving the ws context triggers WebSocketDisconnect on the server
    assert app.state.subscription_manager.subscribed == ["AAPL"]
    # disconnect must decrement the manager for every ticker the socket held (ISSUE-3)
    assert app.state.subscription_manager.unsubscribed == ["AAPL"]


def test_ws_explicit_unsubscribe_decrements_once() -> None:
    app = create_app()
    app.state.signal_hub = SignalHub()
    app.state.subscription_manager = _FakeMgr()
    app.state.signal_repo = None
    with TestClient(app) as client:
        with client.websocket_connect("/ws/signals") as ws:
            ws.send_json({"action": "subscribe", "ticker": "AAPL"})
            ws.receive_json()
            ws.send_json({"action": "unsubscribe", "ticker": "AAPL"})
            assert ws.receive_json() == {"status": "unsubscribed", "ticker": "AAPL"}
    # one explicit unsubscribe + a no-op disconnect (socket already removed) = exactly one
    assert app.state.subscription_manager.unsubscribed == ["AAPL"]
```

- [ ] **Step 2: Implement the WS route**

Create `src/api/ws/signals_ws.py`:

```python
"""WS endpoint: argon subscribes to tickers; apex pushes signal payloads.

Frame protocol (argon → apex):
    {"action": "subscribe",   "ticker": "AAPL"}
    {"action": "unsubscribe", "ticker": "AAPL"}
apex → argon: ack {"status": ..., "ticker": ...}, then an initial snapshot payload,
then live signal_service_payload frames as signals fire.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.payload.builder import build_payload
from src.api.payload.validate import validate_payload

router = APIRouter()


@router.websocket("/ws/signals")
async def signals_ws(ws: WebSocket) -> None:
    await ws.accept()
    hub = ws.app.state.signal_hub
    mgr = ws.app.state.subscription_manager
    repo = getattr(ws.app.state, "signal_repo", None)
    try:
        while True:
            msg = await ws.receive_json()
            ticker = msg.get("ticker", "")
            action = msg.get("action")
            if action == "subscribe" and ticker:
                hub.register(ws, ticker)
                await mgr.subscribe(ticker)
                await ws.send_json({"status": "subscribed", "ticker": ticker})
                # Initial snapshot so argon can render immediately (spec §3.1).
                # MVP: recent persisted signals. NOTE: enriching this with the full
                # historical bars + indicator series requires an indicator-snapshot
                # API on TASignalService — deferred (see self-review).
                if repo is not None:
                    rows = await repo.fetch_signals(ticker)
                    snapshot = build_payload(rows, generated_at=datetime.now(timezone.utc))
                    validate_payload(snapshot)
                    await ws.send_json(snapshot)
            elif action == "unsubscribe" and ticker:
                # Decrement the manager once per ticker actually removed from the hub.
                for removed in hub.unregister(ws, ticker):
                    await mgr.unsubscribe(removed)
                await ws.send_json({"status": "unsubscribed", "ticker": ticker})
            else:
                await ws.send_json({"status": "error", "detail": "bad frame"})
    except WebSocketDisconnect:
        # Decrement EVERY ticker the socket still held (ISSUE-3: no leak).
        for removed in hub.unregister(ws):
            await mgr.unsubscribe(removed)
```

Modify `src/api/server.py`:
- include the WS router: `app.include_router(signals_ws_router)`;
- in `lifespan`, construct dependencies **only if not already set**, so tests
  (and the REST test in Task 3) can pre-inject fakes that the lifespan will not
  clobber. Starlette `TestClient` runs the lifespan, so an unconditional assign
  would overwrite injected fakes and then fail constructing the real ones
  without livewire/PG:

```python
    if getattr(app.state, "signal_hub", None) is None:
        app.state.signal_hub = SignalHub()
    if getattr(app.state, "subscription_manager", None) is None and app.state.pg_pool is not None:
        # real wiring: LivewireOhlcProvider (Phase 1) + TASignalService -> SubscriptionManager (Phase 2)
        app.state.subscription_manager = _build_subscription_manager(app)
```

  Apply the same `getattr(..., None) is None` guard to `app.state.signal_repo`
  from Task 3. Factor the real construction into `_build_subscription_manager(app)`
  so the lifespan stays readable.

- [ ] **Step 3: Run test to verify it passes**

Run: `uv run pytest tests/integration/api/test_signals_ws.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/api/ws/signals_ws.py src/api/server.py tests/integration/api/test_signals_ws.py
git commit -m "feat(api): WS /ws/signals subscribe protocol"
```

---

## Task 6: Emit fired signals to the WS hub (via the event bus)

`TASignalService` exposes **no callback registration** — fired signals flow through
the **event bus** (`_on_trading_signal` is subscribed to `EventType.TRADING_SIGNAL`;
persistence happens there via `_persist_signal`). So the WS emitter subscribes a
NEW handler to the same bus and broadcasts; it does **not** persist (no double-write).

**Files:**
- Read: `src/domain/events/event_types.py` + the bus interface (Step 1)
- Create: `src/api/ws/emitter.py` (signal→payload mapper + bus subscription)
- Modify: `src/api/server.py` (subscribe the emitter in lifespan)
- Test: `tests/unit/api/test_emitter.py`

- [ ] **Step 1: Verify the bus + signal object shape**

Run: `grep -nE "TRADING_SIGNAL|class EventType" src/domain/events/event_types.py`
Run: `uv run python -c "import inspect; from src.domain.interfaces.event_bus import EventBus; print([m for m in dir(EventBus) if not m.startswith('__')])"`
Run: `grep -nE "class TradingSignal|signal_id|self\.symbol|self\.category|self\.direction|self\.strength" src/domain/signals/*.py | head`
Expected: confirm `EventType.TRADING_SIGNAL`, the bus `subscribe(event_type, handler)` method name, and that the fired signal object carries `signal_id, symbol, category, indicator, direction, strength, priority, timeframe, trigger_rule, current_value, timestamp`. Adjust the mapper attribute names in Step 3 to the verified ones.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/api/test_emitter.py`:

```python
"""Emitter maps a fired signal to a valid payload and broadcasts it."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.api.payload.validate import validate_payload
from src.api.ws.emitter import signal_to_payload, SignalEmitter
from src.api.ws.hub import SignalHub


def _signal():
    return SimpleNamespace(
        signal_id="momentum:rsi:AAPL:1d", symbol="AAPL", timeframe="1d",
        category="momentum", indicator="RSI", direction="buy", strength=70,
        priority="high", trigger_rule="rsi_cross", current_value=28.0,
        threshold=30.0, previous_value=32.0, message="",
        timestamp=datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc),
    )


def test_signal_to_payload_is_schema_valid() -> None:
    payload = signal_to_payload(_signal())
    validate_payload(payload)
    assert payload["signals"][0]["signal_id"] == "momentum:rsi:AAPL:1d"


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, d: dict) -> None:
        self.sent.append(d)


@pytest.mark.asyncio
async def test_emitter_broadcasts_on_event() -> None:
    hub = SignalHub()
    ws = _FakeWS()
    hub.register(ws, "AAPL")
    emitter = SignalEmitter(hub)
    # Simulate the bus delivering a TRADING_SIGNAL event payload.
    await emitter.on_trading_signal(_signal())
    assert ws.sent[0]["signals"][0]["symbol"] == "AAPL"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/api/test_emitter.py -v`
Expected: FAIL — `ModuleNotFoundError: src.api.ws.emitter`.

- [ ] **Step 4: Implement the emitter**

Create `src/api/ws/emitter.py`:

```python
"""Subscribe to the event bus and push fired signals to the WS hub.

Does NOT persist — TASignalService._persist_signal already does. This is the
broadcast-only fan-out from the event bus to connected argon clients.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.api.payload.builder import build_payload, signal_row_to_dict  # noqa: F401
from src.api.payload.validate import ValidationFailure, validate_payload

logger = logging.getLogger(__name__)

# schema signal fields read off the fired signal object (verify names in Step 1).
_FIELDS = (
    "signal_id", "symbol", "category", "indicator", "direction", "strength",
    "priority", "timeframe", "trigger_rule", "current_value", "threshold",
    "previous_value", "message",
)


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, datetime) else v


def signal_to_payload(signal: Any) -> dict:
    """Map a fired TradingSignal object to a one-signal signal_service_payload."""
    sig = {f: _iso(getattr(signal, f, None)) for f in _FIELDS if getattr(signal, f, None) is not None}
    ts = getattr(signal, "timestamp", None) or datetime.now(timezone.utc)
    sig["timestamp"] = _iso(ts)
    return {"signals": [sig], "timestamp": _iso(ts), "symbol_count": 1}


class SignalEmitter:
    def __init__(self, hub: Any) -> None:
        self._hub = hub

    async def on_trading_signal(self, payload: Any) -> None:
        signal = getattr(payload, "signal", payload)  # unwrap event wrapper
        try:
            out = signal_to_payload(signal)
            validate_payload(out)
        except ValidationFailure as exc:
            logger.warning("dropping invalid signal payload: %s", exc)
            return
        symbol = getattr(signal, "symbol", None)
        if symbol:
            await self._hub.broadcast(symbol, out)

    def subscribe(self, event_bus: Any) -> None:
        """Subscribe on_trading_signal to the bus's TRADING_SIGNAL event."""
        from src.domain.events.event_types import EventType
        event_bus.subscribe(EventType.TRADING_SIGNAL, self._dispatch)

    def _dispatch(self, payload: Any) -> None:
        # Bus handlers are sync; schedule the async broadcast.
        import asyncio

        asyncio.create_task(self.on_trading_signal(payload))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/api/test_emitter.py -v`
Expected: PASS (2 passed). If `event_bus.subscribe` is named differently (Step 1), fix `subscribe`; the mapper/broadcast tests do not depend on the bus name.

- [ ] **Step 6: Wire the emitter in lifespan**

In `src/api/server.py` lifespan, after the hub + subscription manager + the real
`TASignalService`/event bus exist: `SignalEmitter(app.state.signal_hub).subscribe(event_bus)`.
Guard so it only runs when the real bus exists (skipped in fake-injected tests).

- [ ] **Step 7: Run the full API + new-feature suites**

Run: `uv run pytest tests/unit/api tests/integration/api -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/api/ws/emitter.py src/api/server.py tests/unit/api/test_emitter.py
git commit -m "feat(api): emit fired signals to WS hub via event bus"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** §5 Phase 3 — WS server → Tasks 4,5; REST pull `GET /signals/{ticker}?since=` → Task 3; persist to `ta_signals` (authoritative) → reuse `ta_signal_repository` (Task 3; emitter does NOT double-write, Task 6); validate against schema → Tasks 1,2,3,6; first-payload snapshot (§3.1) → Task 5 (recent signals; full bars+indicators deferred). ✅
- **Repo type fixed (ISSUE-4):** `TASignalRepository(db: Database)`, not `pg_pool`; lifespan builds + connects a `Database` (Task 3). ✅
- **Emission fixed (ISSUE-1):** emitter subscribes to the event bus `TRADING_SIGNAL` event (no fictitious callback); test publishes a real signal object (Task 6). ✅
- **Refcount leak fixed (ISSUE-3):** `hub.unregister(ws, ticker)` returns removed tickers; WS unsubscribe decrements one, disconnect decrements all (Task 5). ✅
- **Validation fixed (ISSUE-9):** REST + WS snapshot both `validate_payload`; null `current_value` rows dropped via `is_emittable` (Tasks 2,3,5). ✅
- **Test wiring fixed (ISSUE-13):** WS tests use `with TestClient(app) as client:`; lifespan construction guarded with `getattr(..., None) is None` so injected fakes survive (Tasks 3,5). ✅
- **Deferred (recorded, out of MVP scope):**
  - Lifecycle persistence (ISSUE-10): `status`/`invalidated_*` need new `ta_signals` columns + an invalidation update method — omitted from the builder until then; schema-valid because those fields are optional.
  - Full first-payload chart seed (ISSUE-6): bars + indicator series need an indicator-snapshot API on `TASignalService`; Task 5 currently sends recent persisted signals as the initial snapshot.
- **Honest gaps flagged:** `ta_signal_repository.fetch_signals` read-method name (Task 3 Step 1), `config.database` attr (Task 3), event-bus `subscribe` method name + `TradingSignal` field names (Task 6 Step 1). D1 PG-isolation deferred but non-blocking. `jsonschema 4.26.0` confirmed installed (Task 1 Step 1 dep-add branch is a no-op). ✅
- **Type consistency:** `validate_payload`/`ValidationFailure`, `build_payload`/`signal_row_to_dict`/`is_emittable`, `signal_to_payload`/`SignalEmitter`, `SignalHub.register/unregister(→set)/broadcast`, `app.state.signal_repo`/`signal_db`/`signal_hub`/`subscription_manager` consistent across tasks and with Phase 2. ✅
- **TZ note:** `datetime.now(timezone.utc)` is app code — allowed; the no-`Date.now()` rule applies only to Workflow scripts. ✅
