# Consuming APEX TA Signals from argon

How argon connects to a running **apex** instance to receive live technical-analysis
signals, with **real, captured** request/response frames (not hand-written examples).

> Audience: argon (the UI). apex is the TA brain; it ingests bars (from livewire) and
> live ticks (from xenon), computes indicators on each closed bar, runs the rule engine,
> and emits `signal_service_payload` frames. argon subscribes per ticker and renders them.
>
> Everything in §7 ("Verified end-to-end") was captured from a running apex wired to the
> **live local xenon (`ws://127.0.0.1:8765`)** and **Postgres (`apex_signals`)** — see that
> section for the exact bytes.

---

## 1. Topology

```
xenon (live IB ticks, :8765) ─┐
                              ├─►  apex  ──(/ws/signals push)──►  argon (this doc)
livewire (OHLC bars) ─────────┤         └─(/signals REST pull)──►
                              │
                     Postgres (apex_signals) ◄── persists fired signals; serves snapshot + backfill
```

apex exposes two consumer surfaces over one HTTP server:

| Surface | Method | Path | Use |
|---|---|---|---|
| **Live push** | WebSocket | `/ws/signals` | Subscribe per ticker; receive signals as they fire. **Primary path.** |
| Backfill pull | HTTP GET | `/signals/{ticker}?since=<iso8601>` | Fetch persisted signals on load/reconnect (requires Postgres). |
| Health | HTTP GET | `/health` | Liveness + `pg_connected`. |

Default listen address: `0.0.0.0:8322` (override with `APEX_API_PORT`).

---

## 2. Running apex (so it actually streams)

apex builds the streaming pipeline when its data roots are configured. The WS endpoint
always *accepts* connections, but **no signals are produced unless `APEX_LIVEWIRE_ROOT`
is set**; live ticks flow from xenon automatically (the URL is baked in — see below).

| Env var | Default | Purpose |
|---|---|---|
| `APEX_LIVEWIRE_ROOT` | *(unset)* | Bronze parquet root for historical bars. **Required** to build the pipeline. Unset → `/ws/signals` connects but emits nothing. |
| `APEX_XENON_WS_URL` | `ws://127.0.0.1:8765` | xenon tick-feed WS URL. **Baked into apex** (matches xenon's `DEFAULT_IB_REALTIME_PORT`); only set this to override. Connects automatically once the pipeline is built. |
| `APEX_TIMEFRAMES` | `1d` | Comma-separated timeframes to compute, e.g. `1m,5m,1d`. Drives the live cadence (§11). |
| `APEX_API_PORT` | `8322` | HTTP/WS listen port. |
| `APEX_PG_URL` | *(unset)* | Postgres DSN. Enables persistence, the WS initial snapshot, and REST backfill (§4, §9). |

Launch (xenon URL omitted — it defaults to the local xenon):

```bash
APEX_LIVEWIRE_ROOT=/data/livewire/bronze \
APEX_TIMEFRAMES=1m,5m,1d \
APEX_PG_URL=postgresql://apex_app:apex@127.0.0.1:5432/apex_signals \
APEX_API_PORT=8322 \
uv run python -m src.api.server
```

---

## 3. Postgres setup (one-time)

apex persists fired signals and serves the snapshot/backfill from `apex_signals`. It uses
a dedicated `apex_app` role (mirrors xenon's `xenon_app` / argon's `argon_app`):

```bash
# as a superuser (e.g. your local admin role)
psql -d postgres -c "CREATE ROLE apex_app LOGIN PASSWORD 'apex';"
psql -d postgres -c "CREATE DATABASE apex_signals OWNER apex_app;"
psql -d apex_signals -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
psql -d apex_signals -c "ALTER SCHEMA public OWNER TO apex_app;"

# create the signal tables (ta_signals + indicator_values + confluence_scores)
psql "postgresql://apex_app:apex@127.0.0.1:5432/apex_signals" -f migrations/005_ta_signals.sql
```

> Only migration **005** is needed for the signal surface; it is self-contained
> (`CREATE TABLE IF NOT EXISTS` + TimescaleDB hypertables, idempotent). The 002–004
> migrations build unrelated futu/IB tables and are not required here.

DSN for `APEX_PG_URL`: `postgresql://apex_app:apex@127.0.0.1:5432/apex_signals`.

---

## 4. WebSocket protocol (`/ws/signals`)

A single text/JSON frame protocol. argon sends control frames; apex replies with an ack
and then streams payload frames.

**argon → apex**

```json
{"action": "subscribe",   "ticker": "AAPL"}
{"action": "unsubscribe", "ticker": "AAPL"}
```

**apex → argon**

| Frame | Shape | When |
|---|---|---|
| Ack | `{"status": "subscribed", "ticker": "AAPL"}` | After each `subscribe`. |
| Ack | `{"status": "unsubscribed", "ticker": "AAPL"}` | After each `unsubscribe`. |
| Error | `{"status": "error", "detail": "bad frame"}` | Unknown action or missing `ticker`. |
| Snapshot | a `signal_service_payload` (§8) | Once, right after a `subscribe` ack, when Postgres is configured (recent persisted signals; empty array if none). |
| Live signal | a `signal_service_payload` (§8) | Each time a signal fires for a subscribed ticker. |

**Discriminating frames:** control frames carry a `status` key; data frames carry a
`signals` key. Branch on which key is present.

Notes:
- One socket may subscribe to **many tickers** — one `subscribe` frame each; each gets its
  own ack. Live frames for any subscribed ticker arrive on that socket.
- `subscribe` lazily opens the upstream xenon subscription for that symbol (ref-counted;
  dropped when the last subscriber for a ticker leaves). No subscribe → no signals.
- Disconnecting cleans up every ticker the socket held — no need to unsubscribe first.
- Keep-alive is handled at the WebSocket protocol level by uvicorn (ping/pong); there is
  **no application-level heartbeat** to implement.

---

## 5. Copy-paste example

### Browser / TypeScript

```ts
const APEX_WS = "ws://localhost:8322/ws/signals";
const TICKERS = ["AAPL", "MSFT"];

const ws = new WebSocket(APEX_WS);

ws.onopen = () => {
  for (const ticker of TICKERS) ws.send(JSON.stringify({ action: "subscribe", ticker }));
};

ws.onmessage = (ev) => {
  const frame = JSON.parse(ev.data);
  if ("status" in frame) {                 // control: subscribed | unsubscribed | error
    if (frame.status === "error") console.warn("apex rejected frame:", frame.detail);
    return;
  }
  for (const sig of frame.signals) {        // data: signal_service_payload
    console.log(`${sig.symbol} ${sig.direction} ${sig.indicator} `
              + `(${sig.strength}/100, ${sig.priority}) — ${sig.message ?? ""}`);
  }
};

ws.onclose = () => { /* reconnect with backoff, re-subscribe, backfill via REST (§9) */ };
// later: ws.send(JSON.stringify({ action: "unsubscribe", ticker: "AAPL" }));
```

### Python (smoke test)

```python
import asyncio, json
from websockets.asyncio.client import connect

async def main() -> None:
    async with connect("ws://localhost:8322/ws/signals") as ws:
        await ws.send(json.dumps({"action": "subscribe", "ticker": "AAPL"}))
        async for raw in ws:
            frame = json.loads(raw)
            if "status" in frame:
                print("control:", frame)
            else:
                for sig in frame["signals"]:
                    print("signal:", sig["signal_id"], sig["direction"], sig["strength"])

asyncio.run(main())
```

---

## 6. When do signals arrive?

Signals are emitted **per closed bar**, not per tick. A live xenon tick is stitched into
the in-progress bar; only when that bar *closes* does apex recompute indicators and run the
rule engine. Max live-frame cadence is one evaluation per timeframe bucket (with
`APEX_TIMEFRAMES=1m`, at most one batch of fired signals per minute per rule). A frame is
sent only when a rule triggers — quiet markets produce no frames. Don't treat the absence of
frames as a disconnect (rely on the WS protocol ping/pong).

---

## 7. Verified end-to-end (real capture)

Captured on **2026-06-14** from a running apex wired to the **live local xenon** and
**Postgres `apex_signals`**. Every frame below is verbatim (the market was closed, so the
signal was injected onto the real bus; it flowed through the real `SignalEmitter`,
`TASignalService` persistence, `fetch_signals`, and `build_payload` — each frame passed
`validate_payload` before the server sent it).

**Startup (all three legs connected):**

```json
{ "pg_connected": true, "signal_repo_built": true, "xenon_client_built": true }
```

**Leg 1 — apex ↔ xenon** (`ws://127.0.0.1:8765`, direct probe; apex already subscribes its
universe). xenon sends `status`, then per-symbol `price`/`batch`. `last` is `null` here
because the market was closed; `close` is the prior session's:

```json
{ "type": "status", "ib_connected": true,
  "subscriptions": ["GLD","QQQ","SPX","SPY","AAPL","MSFT","NVDA","TSLA", "...≈150 symbols..."] }
{ "type": "subscribed", "symbols": ["AAPL"] }
{ "type": "batch", "updates": { "AAPL": {
    "symbol": "AAPL", "last": null, "bid": null, "ask": null, "volume": 0,
    "close": 295.63, "timestamp": "2026-06-14T06:05:25.196Z" } } }
```

**Leg 2 — argon → apex subscribe** on `/ws/signals`. Request, then the ack and the initial
(empty) snapshot:

```json
// argon -> apex
{ "action": "subscribe", "ticker": "AAPL" }
// apex -> argon (ack)
{ "status": "subscribed", "ticker": "AAPL" }
// apex -> argon (initial snapshot; no persisted signals yet)
{ "signals": [], "timestamp": "2026-06-14T06:05:27.053831+00:00", "symbol_count": 0 }
```

**A fired signal arrives live** on the subscribed socket (note `direction: "buy"`):

```json
{
  "signals": [
    {
      "signal_id": "momentum:rsi:AAPL:1m",
      "symbol": "AAPL",
      "category": "momentum",
      "indicator": "rsi",
      "priority": "high",
      "timeframe": "1m",
      "trigger_rule": "rsi_oversold_exit",
      "current_value": 31.4,
      "threshold": 30.0,
      "previous_value": 28.0,
      "message": "RSI exits oversold",
      "direction": "buy",
      "strength": 72,
      "timestamp": "2026-06-14T06:05:27.057312+00:00"
    }
  ],
  "timestamp": "2026-06-14T06:05:27.057312+00:00",
  "symbol_count": 1
}
```

**Leg 3 — Postgres.** The signal was persisted to `apex_signals.ta_signals`. The raw row
holds the **event-native** direction; apex normalises it to the **contract** value on every
read:

```text
-- raw ta_signals row (psql)
time                          | signal_id              | symbol | direction | strength | current_value
2026-06-14 06:05:27.057312+00 | momentum:rsi:AAPL:1m   | AAPL   | LONG      |       72 |          31.4
```

```json
// GET /signals/AAPL  ->  200, read back from Postgres, direction normalised LONG -> "buy"
{
  "signals": [
    { "signal_id": "momentum:rsi:AAPL:1m", "symbol": "AAPL", "category": "momentum",
      "indicator": "rsi", "direction": "buy", "strength": 72, "priority": "high",
      "timeframe": "1m", "trigger_rule": "rsi_oversold_exit", "current_value": 31.4,
      "threshold": 30.0, "previous_value": 28.0, "message": "RSI exits oversold",
      "timestamp": "2026-06-14T06:05:27.057312+00:00" } ],
  "timestamp": "2026-06-14T06:05:28.178410+00:00", "symbol_count": 1
}
```

A **reconnecting** client now gets that signal in its initial snapshot (also `"buy"`),
so argon can render immediately on load and backfill the gap.

---

## 8. The `signal_service_payload` contract

Every payload apex sends — snapshot, live, or REST — is validated against
`config/verification/schemas/signal_service_payload.schema.json` before it leaves the
server (see §12), so argon can trust the shape.

**Envelope**

| Field | Type | Required | Notes |
|---|---|---|---|
| `signals` | array of *trading_signal* | ✓ | May be empty. |
| `timestamp` | string (ISO-8601 date-time) | ✓ | Payload generation time (UTC). |
| `symbol_count` | integer ≥ 0 | – | Distinct symbols in `signals`. Live frames carry one signal. |
| `metadata` | object | – | Optional service metadata. |

**`trading_signal`**

| Field | Type | Required | Notes |
|---|---|---|---|
| `signal_id` | string | ✓ | `"{category}:{indicator}:{symbol}:{timeframe}"`, e.g. `momentum:rsi:AAPL:1m`. Stable identity for dedup/upsert. |
| `symbol` | string | ✓ | |
| `category` | enum | ✓ | `momentum` \| `trend` \| `volatility` \| `volume` \| `pattern` \| `regime` |
| `indicator` | string | ✓ | e.g. `rsi`, `macd`. |
| `direction` | enum | ✓ | `buy` \| `sell` \| `alert` (normalised from the DB's event-native LONG/SHORT/FLAT). |
| `strength` | integer 0–100 | ✓ | |
| `priority` | enum | ✓ | `high` \| `medium` \| `low` |
| `timeframe` | enum | ✓ | `1m` \| `5m` \| `15m` \| `30m` \| `1h` \| `4h` \| `1d` \| `1w` |
| `trigger_rule` | string | ✓ | Rule that fired. |
| `current_value` | number | ✓ | Indicator value at trigger (never null). |
| `timestamp` | string (date-time) | ✓ | Signal generation time. |
| `threshold` | number \| null | – | Trigger threshold if applicable. |
| `previous_value` | number \| null | – | Prior value (state-change detection). |
| `cooldown_until` | string (date-time) \| null | – | Suppression window expiry. |
| `message` | string | – | Human-readable description. |
| `metadata` | object | – | Decoded from JSONB on read. |
| `status` | enum (`active`\|`invalidated`) | – | Defaults to `active`. Currently omitted by apex (lifecycle not yet persisted — treat as `active`). |
| `invalidated_by`, `invalidated_at` | string \| null | – | Deferred (omitted today). |

---

## 9. REST backfill (`GET /signals/{ticker}`)

```text
GET /signals/AAPL                              # recent persisted signals for AAPL
GET /signals/AAPL?since=2026-06-14T06:00:00Z   # only since a timestamp
```

Returns a `signal_service_payload` (§8) of persisted signals (verified live in §7).
Requires `APEX_PG_URL` (§3). If Postgres is not configured the endpoint is unavailable and
the live WS push is the source of truth.

**Reconnect pattern:** track the `timestamp` of the last signal you rendered; on reconnect,
`GET /signals/{ticker}?since=<last_seen>` to fill the gap, then resume the live stream.
`signal_id` is stable — de-duplicate the overlap by `signal_id`.

---

## 10. Contract verification

- **Single source of truth:** `config/verification/schemas/signal_service_payload.schema.json`
  (JSON Schema 2020-12).
- **Enforced on every egress:** `validate_payload()` runs before the server sends any frame —
  REST (`routes/signals.py`), the WS snapshot (`ws/signals_ws.py`), and live pushes
  (`ws/emitter.py`). An invalid live signal is dropped (logged), never sent malformed.
- **No cross-path drift:** WS and REST/snapshot share one normalization module
  (`api/payload/contract.py`) that maps the DB's event-native `LONG/SHORT/FLAT` → contract
  `buy/sell/alert`, coerces `strength` to an int 0–100, and decodes JSONB `metadata`. §7
  shows the same `momentum:rsi:AAPL:1m` signal emitting `"buy"` over the live socket, the
  REST pull, and the reconnect snapshot, while Postgres stores `"LONG"`.

---

## 11. Current limitations

- **Signal lifecycle** (`status`/`invalidated_*`) is not yet persisted; signals are
  effectively append-only and `active`.
- **Per-bar cadence**, not per-tick (§6).
- apex must be started with `APEX_LIVEWIRE_ROOT` or `/ws/signals` connects but stays silent.
- The snapshot/REST backfill require `APEX_PG_URL` (§3); without it, only the live WS push
  is available.
