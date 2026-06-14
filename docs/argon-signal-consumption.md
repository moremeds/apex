# Consuming APEX TA Signals from argon

How argon connects to a running **apex** instance to receive live technical-analysis
signals **and to fetch everything needed to render a full chart** (candles, indicator
overlays/oscillators, confluence) — with **real, captured** request/response frames
(not hand-written examples).

> Audience: argon (the UI). apex is the TA brain; it ingests bars (from livewire) and
> live ticks (from xenon), computes indicators on each closed bar, runs the rule engine,
> and emits `signal_service_payload` frames. argon subscribes per ticker and renders them.
>
> **argon stores nothing** — it pulls signals (§4–§9) *and* all chart data (§10) from apex
> on demand. apex is the single source: bars from livewire, indicator lines recomputed on
> read, confluence + signals from Postgres.
>
> Everything in §7 and §10 ("Verified end-to-end") was captured from a running apex wired to
> the **live local xenon (`ws://127.0.0.1:8765`)** and **Postgres (`apex_signals`)** — see
> those sections for the exact bytes.

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
| Chart: candles | HTTP GET | `/bars/{ticker}` | OHLCV bars for the chart body (from livewire). **See §10.** |
| Chart: indicators | HTTP GET | `/indicators/{ticker}` | Per-bar indicator series (compute-on-read) for overlays/oscillators. **See §10.** |
| Chart: confluence | HTTP GET | `/confluence/{ticker}` | Multi-timeframe confluence scores (from Postgres). **See §10.** |
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
| `APEX_TIMEFRAMES` | `1d` | Comma-separated timeframes to compute, e.g. `1m,5m,1d`. Drives the live cadence (§6). |
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
server (see §11), so argon can trust the shape.

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
Requires `APEX_PG_URL` (§3); when Postgres is not configured the endpoint returns
**`503 Service Unavailable`** and the live WS push is the source of truth.

**Ordering:** `?since=` returns oldest-first from the cursor (so paging past the row limit
is contiguous); without `since` it returns the most recent signals (newest-first).

**Reconnect pattern:** track the `timestamp` of the last signal you rendered; on reconnect,
`GET /signals/{ticker}?since=<last_seen>` to fill the gap, then resume the live stream.
`signal_id` is stable — de-duplicate the overlap by `signal_id`.

---

## 10. Chart read surface (bars + indicators + confluence)

So argon can render a **full chart** — candles, indicator overlays/oscillators, and the
multi-timeframe confluence blocks — while **storing nothing itself**, apex exposes three REST
read endpoints. argon GETs what it needs on load / scroll-back / timeframe-switch; apex is
the single source.

| Surface | Method | Path | Backed by |
|---|---|---|---|
| Candles | GET | `/bars/{ticker}?timeframe=&start=&end=` | livewire bronze parquet (read on the fly) |
| Indicator series | GET | `/indicators/{ticker}?timeframe=&indicator=&start=&end=` | **compute-on-read** (recomputed from bars) |
| Confluence | GET | `/confluence/{ticker}?timeframe=&start=&end=&limit=` | Postgres `confluence_scores` (persisted) |

**Storage model.** Nothing is cached on argon's side. On apex's side it is a deliberate mix:

| Layer | apex source | How |
|---|---|---|
| bars | livewire parquet | read on the fly — **no apex storage** |
| indicators | recomputed per request | **compute-on-read** — full depth, gap-free, always matches the candles |
| confluence | `confluence_scores` PG table | served from storage (depth = persisted history) |

- `start`/`end` are optional ISO-8601; **omit them for the most recent 500 bars** (apex
  over-fetches the calendar lookback and tail-slices to 500, so closures don't shrink it).
- **Supported timeframes for `/bars` + `/indicators`: `1m`, `5m`, `30m`, `1h`, `1d`** (what
  livewire warehouses). `15m`/`4h`/`1w` → **`400`**. `/confluence` accepts any timeframe that
  has persisted rows.
- `/confluence` takes an optional `limit` (default `500`) so you can pull more than the
  repository default — it is **not** silently capped.
- All timestamps are emitted in **UTC** (matching the signal contract).
- Indicators are computed with apex's **default parameters — identical to the live engine**,
  so chart lines line up with the fired signals.
- `/bars` and `/indicators` need `APEX_LIVEWIRE_ROOT` (the bar source) → **`503`** without it.
  `/confluence` needs `APEX_PG_URL` → **`503`** without it.

**Indicator catalog.** Any of apex's **48 registered indicators** by name — `rsi`, `macd`,
`bollinger`, `supertrend`, `obv`, `atr`, `ema`, `sma`, `ichimoku`, `adx`, `vwap`, `cci`,
`stochastic`, … The `state` object's shape varies per indicator (see frames below). An
unknown name → **`404`**.

```text
GET /bars/AAPL?timeframe=1d
GET /bars/AAPL?timeframe=1d&start=2026-01-01T00:00:00Z&end=2026-06-12T00:00:00Z
GET /indicators/AAPL?timeframe=1d&indicator=rsi
GET /indicators/AAPL?timeframe=1d&indicator=macd&start=2026-05-01T00:00:00Z&end=2026-06-12T00:00:00Z
GET /confluence/AAPL?timeframe=1d
```

### Verified end-to-end (real capture)

Captured on **2026-06-14** from a running apex (real FastAPI lifespan → `LivewireOhlcProvider`
reads the **real livewire bronze data lake** via DuckDB → compute-on-read → `validate_payload`
→ JSON; confluence is a real Postgres round-trip on `apex_signals`). The bars are **real AAPL
daily bars** from the data lake, the indicator lines are recomputed from them, and the
confluence row was written via apex's persistence API and read back — every frame passed
`validate_payload` before it was sent. (Reading the real lake relies on the bronze schema fix
in PR #135, which this branch builds on.) Arrays are trimmed to first-2 + last for readability;
floats are verbatim.

**Startup:**

```json
{ "pg_connected": true, "ohlc_provider_built": true, "signal_repo_built": true, "indicator_registry_size": 48 }
```

**`GET /bars/AAPL?timeframe=1d`** → `200`

```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "bars": [
    { "time": "2024-06-14T00:00:00+00:00", "open": 213.81, "high": 215.17, "low": 211.3,  "close": 212.49, "volume": 45295827, "vwap": null },
    { "time": "2024-06-17T00:00:00+00:00", "open": 213.36, "high": 218.95, "low": 212.72, "close": 216.67, "volume": 63750025, "vwap": null },
    "… (500 total) …",
    { "time": "2026-06-12T00:00:00+00:00", "open": 296.03, "high": 297.14, "low": 289.62, "close": 291.13, "volume": 38784790, "vwap": null }
  ],
  "count": 500,
  "generated_at": "2026-06-14T12:52:21.266353+00:00"
}
```

**`GET /indicators/AAPL?timeframe=1d&indicator=rsi`** → `200`

```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "indicator": "rsi",
  "points": [
    { "time": "2024-06-14T00:00:00+00:00", "state": { "value": 75.79481014475162, "zone": "overbought" }, "bar_close": 212.49 },
    { "time": "2024-06-17T00:00:00+00:00", "state": { "value": 78.37764571243576, "zone": "overbought" }, "bar_close": 216.67 },
    "… (500 total) …",
    { "time": "2026-06-12T00:00:00+00:00", "state": { "value": 42.69399117765261, "zone": "neutral" }, "bar_close": 291.13 }
  ],
  "count": 500,
  "generated_at": "2026-06-14T12:52:21.349853+00:00"
}
```

**`GET /indicators/AAPL?timeframe=1d&indicator=macd`** → `200` (multi-field `state`)

```json
{
  "symbol": "AAPL", "timeframe": "1d", "indicator": "macd",
  "points": [
    { "time": "2024-06-14T00:00:00+00:00", "state": { "macd": 7.344795778178678, "signal": 5.59932029100524, "histogram": 3.4909509743468767, "direction": "bullish" }, "bar_close": 212.49 },
    "… (500 total) …",
    { "time": "2026-06-12T00:00:00+00:00", "state": { "macd": 2.1210869267071644, "signal": 5.866075222112672, "histogram": -7.489976590811015, "direction": "bearish" }, "bar_close": 291.13 }
  ],
  "count": 500,
  "generated_at": "2026-06-14T12:52:21.433597+00:00"
}
```

**`GET /indicators/AAPL?timeframe=1d&indicator=bollinger`** → `200`

```json
{
  "symbol": "AAPL", "timeframe": "1d", "indicator": "bollinger",
  "points": [
    { "time": "2024-06-14T00:00:00+00:00", "state": { "upper": 212.54451958471378, "middle": 196.02649999999957, "lower": 179.50848041528536, "bandwidth": 16.852843451996794, "percent_b": 99.83496936653282, "zone": "neutral", "squeeze": false }, "bar_close": 212.49 },
    "… (500 total) …",
    { "time": "2026-06-12T00:00:00+00:00", "state": { "upper": 319.7833280779816, "middle": 304.10824999999966, "lower": 288.4331719220177, "bandwidth": 10.308880523946296, "percent_b": 8.602279569408926, "zone": "neutral", "squeeze": false }, "bar_close": 291.13 }
  ],
  "count": 500,
  "generated_at": "2026-06-14T12:52:21.522975+00:00"
}
```

> Here the default window includes a warmup lead, so **every visible bar carries a populated
> `state`** (no leading zeros) — note the very first bar already has a real RSI/MACD/Bollinger
> value. Only when the window starts at the very beginning of livewire's history (no lead
> available) do the first ~warmup bars read neutral/zero. See "Warmup & history depth" below.

**`GET /confluence/AAPL?timeframe=1d`** → `200` (one row written via apex's persistence API,
then read back through the endpoint — a real PG round-trip)

```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "points": [
    { "time": "2026-06-12T00:00:00+00:00", "alignment_score": 0.42, "bullish_count": 4,
      "bearish_count": 1, "neutral_count": 2, "total_indicators": 7, "dominant_direction": "bullish" }
  ],
  "count": 1,
  "generated_at": "2026-06-14T12:52:21.535493+00:00"
}
```

**Unknown indicator** → `404`

```json
{ "detail": "unknown indicator: nope" }
```

### Chart payload shapes

All three payloads share an envelope (`symbol`, `timeframe`, `count`, `generated_at` UTC) and
are validated on egress against their schemas under `config/verification/schemas/`
(`bars_payload`, `indicator_series_payload`, `confluence_payload`).

**bar** (`bars[]`): `time` (date-time), `open`/`high`/`low`/`close` (number, required),
`volume` (integer\|null), `vwap` (number\|null).

**indicator point** (`points[]`): `time` (date-time), `state` (object — **shape per
indicator**, e.g. `{value, zone}` for rsi, `{macd, signal, histogram, direction}` for macd),
`bar_close` (number\|null — the close at that bar, for aligning an oscillator to price).

**confluence point** (`points[]`): `time`, `alignment_score` (number, −1..+1), `bullish_count`,
`bearish_count`, `neutral_count`, `total_indicators` (integers), `dominant_direction`
(`bullish`\|`bearish`\|`neutral`\|null).

### Warmup & history depth

- **Compute-on-read uses a warmup lead.** apex fetches extra bars *before* your requested
  `start` (≈ the indicator's warmup × 3) so the visible window has valid values. When the
  window starts at the very beginning of livewire's data (no lead available), the first
  ~warmup bars carry neutral/zero `state` — visible in the bollinger frame above. A request
  *inside* a longer history returns fully-populated lines.
- **Bars have full history** (livewire); **indicator lines have full depth too** because they
  are recomputed from those bars on every request. **Confluence depth = persisted history**
  (`confluence_scores`), i.e. only from when apex began computing it for that symbol/timeframe.

---

## 11. Contract verification

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

## 12. Current limitations

- **Signal lifecycle** (`status`/`invalidated_*`) is not yet persisted; signals are
  effectively append-only and `active`.
- **Per-bar cadence**, not per-tick (§6).
- apex must be started with `APEX_LIVEWIRE_ROOT` or `/ws/signals` connects but stays silent.
- The snapshot/REST backfill require `APEX_PG_URL` (§3); without it, only the live WS push
  is available.
- **Chart data is REST-only (poll).** There is no live WS push for bars/indicators/confluence
  yet — argon polls `/bars` + `/indicators` (and gets live *signals* over `/ws/signals`).
- **Chart indicators are compute-on-read** (recomputed per request, uncached — run off the
  event loop in a worker thread so they don't block the signal stream); confluence history is
  limited to what is persisted in `confluence_scores`.
- **§10 sample values are a point-in-time capture** (2026-06-14) from the **real** livewire
  bronze data lake — the schema is matched to livewire's writers and smoke-tested against real
  bytes (PR #135; intraday `TIMESTAMPTZ` is normalized to UTC on read). The payload *shapes*
  are stable; the *numbers* move as livewire ingests new bars.
