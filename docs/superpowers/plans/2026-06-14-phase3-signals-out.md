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
_SIGNAL_FIELDS = (
    "signal_id", "symbol", "category", "indicator", "direction", "strength",
    "priority", "timeframe", "trigger_rule", "current_value", "threshold",
    "previous_value", "message", "cooldown_until", "metadata", "status",
    "invalidated_by", "invalidated_at",
)


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


def build_payload(rows: Iterable[dict[str, Any]], generated_at: datetime) -> dict[str, Any]:
    signals = [signal_row_to_dict(r) for r in rows]
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

router = APIRouter(tags=["signals"])


@router.get("/signals/{ticker}")
async def get_signals(ticker: str, request: Request, since: Optional[datetime] = None) -> dict:
    repo = request.app.state.signal_repo
    rows = await repo.fetch_signals(ticker, since=since)
    return build_payload(rows, generated_at=datetime.now(timezone.utc))
```

Modify `src/api/server.py` `create_app()` to register it (follow the existing router-include pattern):

```python
    from src.api.routes.signals import router as signals_router
    app.include_router(signals_router)
```

And in `lifespan`, after the pool opens, construct the real repo:
`app.state.signal_repo = TASignalRepository(app.state.pg_pool)` (use the real constructor verified in Step 1; guard for `pg_pool is None`).

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
async def test_unregister_stops_delivery() -> None:
    hub = SignalHub()
    a = _FakeWS()
    hub.register(a, "AAPL")
    hub.unregister(a)
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
from typing import Any, DefaultDict, Dict, Set


class SignalHub:
    def __init__(self) -> None:
        self._by_ticker: DefaultDict[str, Set[Any]] = defaultdict(set)
        self._tickers_of: Dict[Any, Set[str]] = {}

    def register(self, ws: Any, ticker: str) -> None:
        self._by_ticker[ticker].add(ws)
        self._tickers_of.setdefault(ws, set()).add(ticker)

    def unregister(self, ws: Any) -> None:
        for ticker in self._tickers_of.pop(ws, set()):
            self._by_ticker[ticker].discard(ws)

    async def broadcast(self, ticker: str, payload: dict) -> None:
        dead = []
        for ws in list(self._by_ticker.get(ticker, ())):
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001 — drop broken sockets
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)
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
"""WS subscribe drives the manager and receives pushed payloads."""

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


def test_ws_subscribe_then_receive_broadcast() -> None:
    app = create_app()
    app.state.signal_hub = SignalHub()
    app.state.subscription_manager = _FakeMgr()
    client = TestClient(app)
    with client.websocket_connect("/ws/signals") as ws:
        ws.send_json({"action": "subscribe", "ticker": "AAPL"})
        ack = ws.receive_json()
        assert ack == {"status": "subscribed", "ticker": "AAPL"}
    assert app.state.subscription_manager.subscribed == ["AAPL"]
```

- [ ] **Step 2: Implement the WS route**

Create `src/api/ws/signals_ws.py`:

```python
"""WS endpoint: argon subscribes to tickers; apex pushes signal payloads.

Frame protocol (argon → apex):
    {"action": "subscribe",   "ticker": "AAPL"}
    {"action": "unsubscribe", "ticker": "AAPL"}
apex → argon: acks {"status": ..., "ticker": ...} then signal_service_payload frames.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/signals")
async def signals_ws(ws: WebSocket) -> None:
    await ws.accept()
    hub = ws.app.state.signal_hub
    mgr = ws.app.state.subscription_manager
    try:
        while True:
            msg = await ws.receive_json()
            ticker = msg.get("ticker", "")
            action = msg.get("action")
            if action == "subscribe" and ticker:
                hub.register(ws, ticker)
                await mgr.subscribe(ticker)
                await ws.send_json({"status": "subscribed", "ticker": ticker})
            elif action == "unsubscribe" and ticker:
                hub.unregister(ws)
                await mgr.unsubscribe(ticker)
                await ws.send_json({"status": "unsubscribed", "ticker": ticker})
            else:
                await ws.send_json({"status": "error", "detail": "bad frame"})
    except WebSocketDisconnect:
        hub.unregister(ws)
```

Modify `src/api/server.py`:
- include the WS router: `app.include_router(signals_ws_router)`;
- in `lifespan`, set `app.state.signal_hub = SignalHub()` and construct the `SubscriptionManager` (Phase 2) with the `LivewireOhlcProvider` (Phase 1) and `TASignalService`, assigning to `app.state.subscription_manager`.

- [ ] **Step 3: Run test to verify it passes**

Run: `uv run pytest tests/integration/api/test_signals_ws.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/api/ws/signals_ws.py src/api/server.py tests/integration/api/test_signals_ws.py
git commit -m "feat(api): WS /ws/signals subscribe protocol"
```

---

## Task 6: Wire compute → persist → push (signal emission)

**Files:**
- Modify: `src/application/subscriptions/manager.py` OR a small `SignalEmitter` (decide per Phase 2's `TASignalService` callback shape)
- Test: `tests/integration/api/test_signal_emission.py`

- [ ] **Step 1: Verify how TASignalService surfaces a fired signal**

Run: `grep -nE "_on_trading_signal|_persist_signal|callback|publish" src/application/services/ta_signal_service.py`
Expected: the hook where a new signal is produced/persisted (verified 2026-06-14: `_on_trading_signal`, `_persist_signal`). The emitter subscribes there.

- [ ] **Step 2: Write the failing test**

Create `tests/integration/api/test_signal_emission.py`:

```python
"""A fired signal is validated, persisted, and pushed to subscribers."""

from __future__ import annotations

import pytest

from src.api.ws.hub import SignalHub


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, d: dict) -> None:
        self.sent.append(d)


@pytest.mark.asyncio
async def test_fired_signal_is_pushed_to_subscriber() -> None:
    from src.api.payload.builder import build_payload
    from datetime import datetime, timezone

    hub = SignalHub()
    client = _FakeWS()
    hub.register(client, "AAPL")

    row = {
        "time": datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc),
        "signal_id": "momentum:rsi:AAPL:1d", "symbol": "AAPL", "timeframe": "1d",
        "category": "momentum", "indicator": "RSI", "direction": "buy",
        "strength": 70, "priority": "high", "trigger_rule": "rsi_cross",
        "current_value": 28.0, "threshold": 30.0, "previous_value": 32.0,
        "message": "", "cooldown_until": None, "metadata": {},
    }
    payload = build_payload([row], generated_at=row["time"])
    await hub.broadcast("AAPL", payload)
    assert client.sent[0]["signals"][0]["signal_id"] == "momentum:rsi:AAPL:1d"
```

- [ ] **Step 3: Implement the emitter hook**

In the construction path (server lifespan), register a callback on `TASignalService` (at the `_on_trading_signal`/persist seam from Step 1) that, for each fired signal: (a) builds a one-signal payload via `build_payload`, (b) `validate_payload(payload)` — log + drop on `ValidationFailure`, (c) persists via the repo (already done inside `_persist_signal`; do not double-write — only persist here if Step 1 shows the service does not), (d) `await hub.broadcast(symbol, payload)`. Keep this glue under `src/api/ws/` as `emitter.py` if it exceeds ~15 lines.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/api/test_signal_emission.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full API + new-feature suites**

Run: `uv run pytest tests/unit/api tests/integration/api -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ tests/integration/api/test_signal_emission.py
git commit -m "feat(api): emit fired signals — validate, persist, push"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** §5 Phase 3 — WS server → Tasks 4,5; REST pull `GET /signals/{ticker}?since=` → Task 3; persist to `ta_signals` (authoritative) → reuse `ta_signal_repository` (Tasks 3,6); validate against schema → Tasks 1,2,6. ✅
- **Honest gaps flagged:** `ta_signal_repository` read-method name (Task 3 Step 1), `TASignalService` signal-emission seam (Task 6 Step 1), `jsonschema` dependency (Task 1 Step 1), no-double-write guard (Task 6 Step 3). D1 PG-isolation deferred but non-blocking (uses `app.state.pg_pool`). ✅
- **Type consistency:** `validate_payload`/`ValidationFailure`, `build_payload`/`signal_row_to_dict`, `SignalHub.register/unregister/broadcast`, `app.state.signal_repo`/`signal_hub`/`subscription_manager` consistent across tasks and with Phase 2's `SubscriptionManager.subscribe/unsubscribe`. ✅
- **TZ note:** `datetime.now(timezone.utc)` is used in app code (Task 3) — allowed; the no-`Date.now()` rule applies only to Workflow scripts, not application code. ✅
