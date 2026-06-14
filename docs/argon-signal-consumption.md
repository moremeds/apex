# Consuming APEX TA Signals from argon

How argon connects to a running **apex** instance to receive live technical-analysis
signals, plus a copy-paste WebSocket example to get going.

> Audience: argon (the UI). apex is the TA brain; it ingests bars (from livewire) and
> live ticks (from xenon), computes indicators on each closed bar, runs the rule engine,
> and emits `signal_service_payload` frames. argon subscribes per ticker and renders them.

---

## 1. Topology

```
xenon (live IB ticks) ─┐
                       ├─►  apex  ──(/ws/signals push)──►  argon (this doc)
livewire (OHLC bars) ──┘         └─(/signals REST pull)──►
```

apex exposes two surfaces over the same HTTP server:

| Surface | Method | Path | Use |
|---|---|---|---|
| **Live push** | WebSocket | `/ws/signals` | Subscribe per ticker; receive signals as they fire. **Primary path.** |
| Backfill pull | HTTP GET | `/signals/{ticker}?since=<iso8601>` | Fetch persisted signals on load/reconnect. *Requires Postgres — see §7.* |
| Health | HTTP GET | `/health` | Liveness + `pg_connected` flag. |

Default listen address: `0.0.0.0:8322` (override with `APEX_API_PORT`).

---

## 2. Running apex (so it actually streams)

apex only builds the streaming pipeline when its data roots are configured. The WS
endpoint always *accepts* connections, but **no signals are produced unless
`APEX_LIVEWIRE_ROOT` is set**, and **live ticks only flow when `APEX_XENON_WS_URL`
is also set**.

| Env var | Default | Purpose |
|---|---|---|
| `APEX_LIVEWIRE_ROOT` | *(unset)* | Bronze parquet root for historical bars. **Required** to build the pipeline (provider + TA service + emitter + subscription manager). Unset → `/ws/signals` connects but emits nothing. |
| `APEX_XENON_WS_URL` | *(unset)* | xenon tick-feed WS URL, e.g. `ws://127.0.0.1:8765`. Enables live-in. Requires `APEX_LIVEWIRE_ROOT`. |
| `APEX_TIMEFRAMES` | `1d` | Comma-separated timeframes to compute, e.g. `1m,5m,1d`. Drives the live-frame cadence (see §5). |
| `APEX_API_PORT` | `8322` | HTTP/WS listen port. |
| `APEX_API_WORKERS` | `1` | uvicorn worker count. |
| `APEX_PG_URL` | *(unset)* | Postgres DSN. Enables persistence, the WS initial snapshot, and the REST backfill (§7). |

Launch:

```bash
APEX_LIVEWIRE_ROOT=/data/livewire/bronze \
APEX_XENON_WS_URL=ws://127.0.0.1:8765 \
APEX_TIMEFRAMES=1m,5m,1d \
APEX_API_PORT=8322 \
uv run python -m src.api.server
```

---

## 3. WebSocket protocol (`/ws/signals`)

A single text/JSON frame protocol. argon sends control frames; apex replies with an
ack and then streams payload frames.

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
| Snapshot | a `signal_service_payload` (§6) | Once, right after a `subscribe` ack — **only if Postgres is configured** (§7). Skipped otherwise. |
| Live signal | a `signal_service_payload` (§6) | Each time a signal fires for a subscribed ticker. |

**Discriminating frames:** control frames carry a `status` key; data frames carry a
`signals` key. Branch on which key is present.

Notes:
- One socket may subscribe to **many tickers** — send one `subscribe` frame each; each
  gets its own ack. Live frames for any subscribed ticker arrive on that socket.
- `subscribe` is what *lazily opens the upstream xenon subscription* for that symbol.
  No subscribe → apex never asks xenon for that symbol → no signals. The subscription
  is reference-counted: the upstream feed is dropped only when the last argon client
  for a ticker unsubscribes or disconnects.
- Disconnecting cleans up every ticker the socket held — no need to unsubscribe first.
- Keep-alive is handled at the WebSocket protocol level by uvicorn (ping/pong control
  frames); there is **no application-level heartbeat** to implement on `/ws/signals`.

---

## 4. Copy-paste example

### Browser / TypeScript (argon's likely runtime)

```ts
const APEX_WS = "ws://localhost:8322/ws/signals";
const TICKERS = ["AAPL", "MSFT"];

const ws = new WebSocket(APEX_WS);

ws.onopen = () => {
  for (const ticker of TICKERS) {
    ws.send(JSON.stringify({ action: "subscribe", ticker }));
  }
};

ws.onmessage = (ev) => {
  const frame = JSON.parse(ev.data);

  if ("status" in frame) {
    // control frame: { status: "subscribed" | "unsubscribed" | "error", ticker?, detail? }
    if (frame.status === "error") console.warn("apex rejected frame:", frame.detail);
    return;
  }

  // data frame: signal_service_payload
  for (const sig of frame.signals) {
    console.log(`${sig.symbol} ${sig.direction} ${sig.indicator} `
              + `(${sig.strength}/100, ${sig.priority}) — ${sig.message ?? ""}`);
    // render sig …
  }
};

ws.onclose = () => {
  // reconnect with backoff, then re-subscribe; backfill the gap via REST (§7)
};

// later: ws.send(JSON.stringify({ action: "unsubscribe", ticker: "AAPL" }));
```

### Python (handy for smoke-testing)

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

### One-liner with `wscat`

```text
$ wscat -c ws://localhost:8322/ws/signals
> {"action":"subscribe","ticker":"AAPL"}
< {"status":"subscribed","ticker":"AAPL"}
< {"signals":[{"signal_id":"momentum:rsi:AAPL:1m", ...}],"timestamp":"...","symbol_count":1}
```

---

## 5. When do signals arrive?

Signals are emitted **per closed bar**, not per tick. A live xenon tick is stitched into
the in-progress bar; only when that bar *closes* does apex recompute indicators and run
the rule engine. So the maximum live-frame cadence is one evaluation per timeframe bucket:
with `APEX_TIMEFRAMES=1m`, expect at most one batch of fired signals per minute per rule;
with `1d`, once per session. A frame is sent only when a rule actually triggers — quiet
markets produce no frames. Do not treat absence of frames as a connection failure (rely
on the WS protocol ping/pong instead).

---

## 6. The `signal_service_payload` contract

Every payload apex sends — snapshot, live, or REST — is validated against
`config/verification/schemas/signal_service_payload.schema.json` before it leaves the
server, so argon can trust the shape.

**Envelope**

| Field | Type | Required | Notes |
|---|---|---|---|
| `signals` | array of *trading_signal* | ✓ | May be empty. |
| `timestamp` | string (ISO-8601 date-time) | ✓ | Payload generation time (UTC). |
| `symbol_count` | integer ≥ 0 | – | Distinct symbols in `signals`. Live frames carry exactly one signal (`symbol_count: 1`). |
| `metadata` | object | – | Optional service metadata. |

**`trading_signal`**

| Field | Type | Required | Notes |
|---|---|---|---|
| `signal_id` | string | ✓ | `"{category}:{indicator}:{symbol}:{timeframe}"`, e.g. `momentum:rsi:AAPL:1m`. Stable identity for dedup/upsert. |
| `symbol` | string | ✓ | |
| `category` | enum | ✓ | `momentum` \| `trend` \| `volatility` \| `volume` \| `pattern` \| `regime` |
| `indicator` | string | ✓ | e.g. `rsi`, `macd`. |
| `direction` | enum | ✓ | `buy` \| `sell` \| `alert` |
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
| `metadata` | object | – | |
| `status` | enum (`active`\|`invalidated`) | – | Defaults to `active`. **Currently omitted by apex** (lifecycle not yet persisted — treat as `active`). |
| `invalidated_by`, `invalidated_at` | string \| null | – | Deferred (omitted today). |

Example live frame:

```json
{
  "signals": [
    {
      "signal_id": "momentum:rsi:AAPL:1m",
      "symbol": "AAPL",
      "category": "momentum",
      "indicator": "rsi",
      "direction": "buy",
      "strength": 72,
      "priority": "high",
      "timeframe": "1m",
      "trigger_rule": "rsi_oversold_exit",
      "current_value": 31.4,
      "threshold": 30.0,
      "previous_value": 28.0,
      "message": "RSI exits oversold",
      "timestamp": "2026-06-14T15:01:10+00:00"
    }
  ],
  "timestamp": "2026-06-14T15:01:10+00:00",
  "symbol_count": 1
}
```

---

## 7. REST backfill (`GET /signals/{ticker}`) — requires Postgres

```text
GET /signals/AAPL                                  # all persisted signals for AAPL
GET /signals/AAPL?since=2026-06-14T15:00:00Z       # only since a timestamp
```

Returns a `signal_service_payload` (§6) of *persisted* signals.

> **Status:** this endpoint and the WS initial snapshot read from a persisted signal
> repository, which is only present when apex runs with `APEX_PG_URL` configured **and**
> the repository is wired. In the current build the repository is not yet wired, so:
> - the WS `subscribe` flow returns *ack → live frames* (snapshot is skipped), and
> - the REST endpoint is **not operational yet**.
>
> Until then, treat the **live WS push as the source of truth** and hold recent signals
> client-side. Once persistence lands, the reconnect pattern below applies unchanged.

**Intended reconnect pattern (once persistence is live):** track the `timestamp` of the
last signal you rendered; on reconnect, `GET /signals/{ticker}?since=<last_seen>` to fill
the gap, then resume the live stream. `signal_id` is stable, so de-duplicate the overlap
by `signal_id`.

---

## 8. Current limitations

- **Live WS push is the fully-wired path.** Snapshot + REST backfill await PG wiring (§7).
- **Per-bar cadence**, not per-tick (§5).
- **Signal lifecycle** (`status`/`invalidated_*`) is not yet persisted; signals are
  effectively append-only and `active`.
- apex must be started with `APEX_LIVEWIRE_ROOT` (+ `APEX_XENON_WS_URL` for live ticks),
  or `/ws/signals` connects but stays silent.

---

## 9. Quick local smoke test

```bash
# 1) start apex pointed at a bars root + xenon (see §2)
uv run python -m src.api.server

# 2) in another shell, subscribe and watch frames
wscat -c ws://localhost:8322/ws/signals
> {"action":"subscribe","ticker":"AAPL"}
```

You should see a `{"status":"subscribed","ticker":"AAPL"}` ack immediately, then a
`signal_service_payload` frame whenever a rule fires on a closed bar for AAPL.
