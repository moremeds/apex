# apex API reference (for argon)

Everything argon needs to consume apex: the WebSocket signal stream and the REST read surface
(signals backfill + chart bars/indicators/confluence). **argon stores nothing** — it pulls all
state from apex on demand.

> **Doc roles.** This file is the **API reference** (endpoints, params, status codes, shapes).
> For the narrative guide with **real captured frames**, runnable copy-paste examples, and the
> contract deep-dive, see [argon-signal-consumption.md](argon-signal-consumption.md). The
> **contract source of truth** is the JSON Schemas in `config/verification/schemas/` — apex runs
> `validate_payload()` on **every** frame before sending, so argon can trust the shape.

---

## 1. Connect

| | |
|---|---|
| Base URL | `http://<host>:8322` (HTTP) / `ws://<host>:8322` (WS). Port = `APEX_API_PORT`, default `8322`. |
| Run apex | `uv run python -m src.api.server` |
| Health | `GET /health` — see the real shape below |

```json
{
  "status": "ok",
  "version": "0.1.4",
  "uptime": 102407.2,
  "service": "apex-signal-server",
  "pg_connected": true,
  "livewire": {
    "configured": true,
    "configured_price_mode": "adjusted",
    "effective_price_mode": "adjusted",
    "recency": {
      "bronze_last_trade_date": "2026-08-21",
      "silver_last_trade_date": "2026-08-21",
      "lag_days": 0
    }
  },
  "silver_revision": {
    "enabled": true, "running": true,
    "observed_revision": 33, "last_fully_applied_revision": 33,
    "per_symbol_revision": {}, "failed": {}, "pending": [],
    "consecutive_failures": 0, "last_error": null,
    "last_success_age_seconds": 53594.0
  }
}
```

`recency` is read from the parquet artifacts, **not** from livewire's 11:00 UTC coverage
snapshot — that snapshot under-reports and would show a lag that does not exist.
`lag_days` is **calendar** days, not trading sessions.

apex's data sources are env-gated, which determines what's available:

| Feature | Requires | If unset |
|---|---|---|
| Live signal push + chart bars/indicators warmup | `APEX_LIVEWIRE_ROOT` (see [livewire doc](livewire-apex-integration.md)) | `/ws/signals` connects but stays silent; `/bars`,`/indicators` → `503` |
| Signal snapshot/backfill + confluence | `APEX_PG_URL` (Postgres) | `/signals`,`/confluence` → `503`; live WS push still works |
| Live ticks | `APEX_XENON_WS_URL` (default `ws://127.0.0.1:8765`) | no live frames; snapshot/REST still work |

All timestamps in every response are **UTC** ISO-8601.

---

## 2. WebSocket — `GET /ws/signals` (live signals)

Bidirectional. argon sends action frames; apex replies with acks, an initial snapshot, then live
signal frames as rules fire.

**argon → apex**

```json
{ "action": "subscribe",   "ticker": "AAPL" }
{ "action": "unsubscribe", "ticker": "AAPL" }
```

**apex → argon**

1. **Ack** — `{ "status": "subscribed", "ticker": "AAPL" }` (or `"unsubscribed"`).
2. **Initial snapshot** (only on subscribe, only if `APEX_PG_URL` set) — a `signal_service_payload`
   of recent persisted signals for that ticker, so argon can render immediately on load/reconnect.
   Empty `signals: []` if none yet.
3. **Live frames** — a one-signal `signal_service_payload` each time a rule fires for a subscribed
   ticker.
4. Bad frame → `{ "status": "error", "detail": "bad frame" }`.

**Semantics**

- Signals fire **per closed bar**, not per tick (a live tick is stitched into the in-progress
  bar; indicators + rules run only when the bar closes). Quiet markets produce **no frames** —
  don't treat silence as a disconnect; rely on the WS ping/pong.
- Many argon clients can subscribe to the same ticker — apex computes once and fans out
  (ref-counted). Unsubscribe/disconnect decrements; compute stops at zero.

---

## 3. REST endpoints

| Method · Path | Purpose | Backed by | Key errors |
|---|---|---|---|
| `GET /signals/{ticker}` | Signal backfill (load / reconnect / `?since=`) | PG `ta_signals` | `503` no PG |
| `GET /bars/{ticker}` | OHLCV candles | livewire bronze (DuckDB) | `400` bad tf · `503` no provider |
| `GET /indicators/{ticker}` | Per-bar indicator series (compute-on-read) | `indicator.calculate()` | `404` unknown indicator · `400` bad tf · `503` no provider |
| `GET /confluence/{ticker}` | Multi-timeframe confluence | PG `confluence_scores` | `503` no PG |

### `GET /signals/{ticker}`
Query: `since` (optional ISO-8601 — only signals at/after it). → `signal_service_payload`.

```
GET /signals/AAPL
GET /signals/AAPL?since=2026-06-14T00:00:00Z
```

### `GET /bars/{ticker}`
Query: `timeframe` (default `1d`), `start`, `end` (optional ISO-8601). → `bars_payload`.

- **Omit `start`/`end`** → the **most recent 500 bars** (apex over-fetches calendar lookback and
  tail-slices to 500, so market closures don't shrink the result).
- Provide `start` (± `end`) → that exact range, **uncapped** (trusted-network service).
- Timeframes: `1m`,`5m`,`30m`,`1h`,`1d`. Others (`15m`,`4h`,`1w`) → `400`.

```
GET /bars/AAPL?timeframe=1d
GET /bars/AAPL?timeframe=1h&start=2026-06-01T00:00:00Z&end=2026-06-12T00:00:00Z
```

### `GET /indicators/{ticker}`
Query: `indicator` (**required**), `timeframe` (default `1d`), `start`, `end`. → `indicator_series_payload`.

- **Compute-on-read**: recomputed from livewire bars on every request → full depth, gap-free,
  always aligned to the candles. Uses apex's **default params, identical to the live engine**, so
  chart lines match fired signals.
- Same window rules as `/bars` (no-arg → last 500).
- `indicator` is any of apex's **48 registered indicators** by name (`rsi`, `macd`, `bollinger`,
  `supertrend`, `obv`, `atr`, `ema`, `sma`, `ichimoku`, `adx`, `vwap`, `cci`, `stochastic`, …).
  Unknown name → `404`. The `state` object's shape **varies per indicator**.

```
GET /indicators/AAPL?timeframe=1d&indicator=rsi
GET /indicators/AAPL?timeframe=1d&indicator=macd&start=2026-05-01T00:00:00Z
```

### `GET /confluence/{ticker}`
Query: `timeframe` (default `1d`), `start`, `end`, `limit` (default `500`, `1..5000`).
→ `confluence_payload`.

- PG-backed, so it accepts **any** timeframe that has persisted rows.
- Depth = persisted history (only from when apex began computing it for that symbol/timeframe).
- `limit` is explicit and **not** silently capped at the repo default.

```
GET /confluence/AAPL?timeframe=1d
GET /confluence/AAPL?timeframe=1d&limit=2000
```

---

## 3a. The `/v1` surface (asset-class aware)

Every route is `/v1/{asset_class}/{symbol}/...`. The flat routes in §3 still work and are
**deprecated aliases** — see the migration table below.

### Asset classes

| Class | Timeframes | Payload | Adjustment | Notes |
|---|---|---|---|---|
| `equity` | `1m 5m 30m 1h 1d` | `bars_payload` | raw **or adjusted** | The only class with a Silver tree |
| `fx` | `1m 5m 30m 1h 1d` | `bars_payload` | raw only | All 21 pairs carry the full ladder |
| `volatility` | `5m 30m 1h 1d` | `bars_payload` | raw only | **No `1m`** anywhere; 30 of 44 symbols are `1d`-only |
| `cmdty` | `1d` | `bars_payload` | raw only | |
| `futures` | `1d` | `bars_payload` | raw only | Adds `settlement`, `open_interest`, `contract` |
| `rates` | `1d` | `rates_series_payload` | n/a | A **yield**, not a price — no OHLC |

`timeframes` is a per-**class** ceiling, not a per-symbol guarantee. A symbol that lacks a
given timeframe returns `404 unknown_symbol` — measured, not theoretical: `AACT` carries
`1m/5m/30m/1h` in the live tree and no `1d` at all. Ask
`GET /v1/{asset_class}/{symbol}` for the ladder a specific symbol actually has.

### Routes

| Method · Path | Purpose | Key errors |
|---|---|---|
| `GET /v1/{asset_class}/{symbol}/bars` | OHLCV candles | `400` class/tf/mode · `404` no artifact · `409` ambiguous · `501` delisted · `503` no Silver |
| `GET /v1/rates/{symbol}/series` | Treasury yield series | `404` no artifact · `503` no provider |
| `GET /v1/{asset_class}/{symbol}/indicators` | Per-bar indicator series | `400` bad class/tf/indicator · `404` no artifact · `503` |
| `GET /v1/equity/{symbol}/confluence` | Multi-timeframe confluence (PG) | `503` no PG |
| `GET /v1/equity/{symbol}/signals` | Signal backfill (PG) | `503` no PG |
| `GET /v1/instruments` | Discovery across all classes | `400` bad class · `501` delisted · `503` no catalog |
| `GET /v1/{asset_class}/{symbol}` | One instrument's metadata | `400` bad class · `404` no artifact · `503` |

`GET /v1/{asset_class}/{symbol}` also returns `coverage_source`: `livewire_coverage_snapshot`
when the catalog answered, `not_configured` or `unavailable` when it did not. Without it a
null `first_date` would be ambiguous between "this symbol has no recorded coverage" and
"apex could not read the catalog".
| `GET /v1/equity/{symbol}/actions` | Corporate actions | **`501` always** — blocked on livewire |
| `GET /v1/equity/{symbol}/delisting` | Delisting terminal state | **`501` always** — blocked on livewire |

### Query parameters

| Param | Routes | Default | Meaning |
|---|---|---|---|
| `timeframe` | bars, indicators | `1d` | Must be in the class's ladder |
| `start` / `end` | all series | none | ISO-8601. Omit both → most recent `limit` bars |
| `limit` | bars, indicators, confluence, instruments | `2000` (bars) | Tail-slice; `<=0` → full history |
| `price_mode` | bars | provider default | `raw` \| `adjusted`. A **request**, not a hint |
| `listing` | bars, instruments | `listed` | `listed` \| `delisted` \| `any` |
| `indicator` | indicators | **required** | Any of apex's registered indicators |
| `asset_class` | instruments | all | Filter |
| `q` | instruments | none | Symbol **prefix** filter (`_`/`%` are escaped) |

### Error envelope

Every failure returns `{"error": {"code", "message", "symbol"?, "asset_class"?}}`.

| Code | Status | Meaning |
|---|---|---|
| `invalid_parameter` | 400 | A query value is malformed: bad `listing`, bad `price_mode`, unknown `indicator`, `start` after `end` |
| `unsupported_timeframe` | 400 | Timeframe not in this class's ladder |
| `unsupported_asset_class` | 400 | Unknown class, or a class whose payload is not bars |
| `adjusted_not_supported` | 400 | `price_mode=adjusted` on a class with no Silver |
| `unknown_symbol` | 404 | No artifact under that partition, in any tree the read would use |
| `ambiguous_symbol` | 409 | `listing=any` on a ticker that is both live and delisted |
| `not_yet_available` | 501 | Specified but blocked on upstream livewire work |
| `provider_not_configured` | 503 | Provider / PG / coverage catalog unavailable |
| `adjusted_unavailable` | 503 | Silver artifact missing or quarantined — retry later |
| `internal_error` | 500 | Unanticipated failure (e.g. the lake volume went away) |

Framework-level request validation (a non-integer `limit`, an unparseable date) keeps
FastAPI's **422** status but uses this same envelope with `invalid_parameter`, so there is
exactly one error shape on the surface rather than two.

A code always names the thing that was wrong. A malformed query value is
`invalid_parameter`, never `unknown_symbol` or `not_yet_available` — those would send
you debugging a symbol or an upstream outage when the fault is a typo in the request.

`adjusted_unavailable` is a **503, not a 4xx**: a quarantined Silver artifact is an upstream
condition livewire may repair, so the request was not wrong. 243 equity symbols are in this
state, including HON, MMM, CMCSA, AIG, ECL, MSI, WY and LEN.

### Deprecation

The flat routes keep working and now emit `Deprecation: true`, `Sunset: Wed, 31 Dec 2026
23:59:59 GMT`, and a `Link` header naming the successor.

| Old | New |
|---|---|
| `GET /bars/AAPL` | `GET /v1/equity/AAPL/bars` |
| `GET /indicators/AAPL?indicator=rsi` | `GET /v1/equity/AAPL/indicators?indicator=rsi` |
| `GET /confluence/AAPL` | `GET /v1/equity/AAPL/confluence` |
| `GET /signals/AAPL` | `GET /v1/equity/AAPL/signals` |

### Two contract narrowings

- **`vwap` is gone** from bar objects. No parquet in the lake carries that column, so it was
  always `null`.
- **The `timeframe` enum no longer accepts `15m`, `4h` or `1w`.** livewire warehouses none of
  them, so those values could only ever `400`.

---

## 4. Payload shapes

All chart payloads share an envelope: `symbol`, `timeframe`, `count`, `generated_at` (UTC), plus
the data array. Validated on egress against `config/verification/schemas/`.

**`signal_service_payload`** (`/signals`, WS snapshot + live) — `signals[]`, `timestamp`,
`symbol_count`. Each signal: `signal_id` (`{category}:{indicator}:{symbol}:{timeframe}`),
`symbol`, `category` (`momentum|trend|volatility|volume|pattern|regime`), `indicator`,
`direction` (**`buy|sell|alert`**, normalized from the DB's `LONG/SHORT/FLAT`), `strength`
(0–100), `priority` (`high|medium|low`), `timeframe`, `trigger_rule`, `current_value`,
`timestamp`, optional `threshold`/`previous_value`/`message`/`metadata`. Full field table: §8 of
the [consumption guide](argon-signal-consumption.md).

**`bars_payload`** — `bars[]` of `{ time, open, high, low, close, volume|null, vwap|null }`.

**`indicator_series_payload`** — `points[]` of `{ time, state (object, shape per indicator),
bar_close (number|null — close at that bar, to align an oscillator to price) }`.

**`confluence_payload`** — `points[]` of `{ time, alignment_score (−1..+1), bullish_count,
bearish_count, neutral_count, total_indicators, dominant_direction (bullish|bearish|neutral|null) }`,
oldest-first.

---

## 5. Status codes

| Code | When |
|---|---|
| `200` | OK |
| `400` | Unsupported timeframe on `/bars` or `/indicators` (livewire warehouses `1m/5m/30m/1h/1d`) |
| `404` | Unknown indicator name on `/indicators` |
| `503` | Required source not configured: no `APEX_LIVEWIRE_ROOT` (`/bars`,`/indicators`) or no `APEX_PG_URL` (`/signals`,`/confluence`) |

---

## 6. Minimal examples

**Browser / TypeScript — live + chart**

```ts
// live signals
const ws = new WebSocket("ws://localhost:8322/ws/signals");
ws.onopen = () => ws.send(JSON.stringify({ action: "subscribe", ticker: "AAPL" }));
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.status) return;                 // ack
  for (const s of msg.signals ?? []) renderSignal(s);   // snapshot + live frames
};

// chart on load (stateless: re-fetch whatever you need to draw)
const tf = "1d";
const [bars, rsi, conf] = await Promise.all([
  fetch(`http://localhost:8322/bars/AAPL?timeframe=${tf}`).then(r => r.json()),
  fetch(`http://localhost:8322/indicators/AAPL?timeframe=${tf}&indicator=rsi`).then(r => r.json()),
  fetch(`http://localhost:8322/confluence/AAPL?timeframe=${tf}`).then(r => r.json()),
]);
drawCandles(bars.bars); drawOscillator(rsi.points); drawConfluence(conf.points);
```

**Python smoke test**

```python
import httpx
b = httpx.get("http://localhost:8322/bars/AAPL", params={"timeframe": "1d"}).json()
print(b["count"], b["bars"][-1])
```

---

## 7. Notes & current limitations

- Chart data is **REST-only (poll)** today — there is no live WS push for bars/indicators/
  confluence. Live **signals** stream over `/ws/signals`; re-pull `/bars` + `/indicators` on
  timeframe-switch / scroll-back.
- Indicators are **compute-on-read** (uncached, recomputed per request in a worker thread so they
  never stall the signal stream).
- Signal **lifecycle** (`status`/`invalidated_*`) is not yet persisted — treat signals as
  append-only and `active`.
- Per-bar cadence, not per-tick (§2).
