# Phase 4 — Live-in — Design Spec

Status: Approved (design); pending spec review
Date: 2026-06-14
Branch: `feat/apex-adaptation`
Roadmap source: `docs/superpowers/specs/2026-06-14-apex-adaptation-design.md` §3 (Phase 4)
Goal source of truth: `~/projects/livewire/docs/architecture/apex-adaptation.md` §3, §3.1
xenon contract source: `~/projects/xenon/scripts/infra/ib_realtime/ib_realtime_server.js` (read 2026-06-14)

---

## 1. Goal

Connect apex to xenon's live IB tick feed so a **subscribed** ticker streams
real ticks into the existing compute pipeline, producing live TA signals over the
Phase 3 WS/REST surface to argon. This closes the critical path `0 → 1 → 2 → 3
→ 4` to a live signal.

```
xenon ib_realtime_server.js ──ws──▶ XenonTickClient ──MARKET_DATA_TICK(dict)──▶ EventBus
                                          │                                          │
                              (subscribe/unsubscribe per ticker)        BarAggregator → IndicatorEngine
                                          ▲                              → RuleEngine → TRADING_SIGNAL
                              SubscriptionManager.subscribe(ticker) ──────────────────┘ → SignalEmitter → argon
```

## 2. Key reframing — Phase 4 is an adapter, not a bar engine

The roadmap phrases Phase 4 as "WS client to xenon; **tick→bar stitch**; service
auth." Investigation of the existing code (2026-06-14) shows the tick→bar stitch
**already exists** in the domain and runs in the Phase 2/3 pipeline:

- `EventType.MARKET_DATA_TICK` → `TASignalService._on_market_data_tick`
  (`src/application/services/ta_signal_service.py:331`) → fans out to one
  `BarAggregator` per timeframe.
- `BarAggregator.on_tick` (`src/domain/signals/data/bar_aggregator.py:93`) folds
  ticks into the current `BarBuilder`, emits `BAR_CLOSE` on bar rollover →
  `IndicatorEngine` → `RuleEngine` → `TRADING_SIGNAL` → `SignalEmitter` (Phase 3).

So Phase 4 **must not** build a new aggregator. It supplies ticks to the bus and
wires subscribe/unsubscribe. Everything downstream is already proven.

## 3. The xenon contract (verified, not assumed)

From `ib_realtime_server.js` + `ib_tick_handler.js` (read 2026-06-14):

### 3.1 Connection & auth (resolves D2 for now)
- Endpoint: `ws://<host>:<port>`; default port **8765**, with a fallback port
  scan. The live port + pid are written to a runtime file
  (`$IB_REALTIME_RUNTIME_FILE` or `<tmpdir>/xenon-ib-realtime.json`) as
  `{port, pid, started_at}`.
- **Ticket auth is skipped** by the upgrade handler when `CLERK_JWKS_URL` is
  unset **or** the peer is loopback (`127.0.0.1`/`::1`/`::ffff:127.0.0.1`)
  (`ib_realtime_server.js:362-373`). Otherwise it requires `?ticket=<UUID>`
  validated via `POST $TICKET_VALIDATE_URL`.
- **Decision D2 (this phase):** use the trusted-network / loopback path that
  needs **no ticket**, behind a pluggable `AuthProvider`. xenon's service-JWT
  ticket minting is not built yet (open xenon decision), so we do not code
  against it now; `TicketAuthProvider` drops in later with no client change.

### 3.2 Client → server messages (JSON)
- `{"action":"subscribe","symbols":["AAPL",...]}`
- `{"action":"unsubscribe","symbols":[...]}`
- `{"action":"pong"}` — reply to the server's keep-alive ping
- (also `snapshot`, `ping`, `search`, option `contracts`, `indexes` — unused by apex)

### 3.3 Server → client messages (JSON)
- `{"type":"price","symbol":SYM,"data":PriceData}` — sent on subscribe
- `{"type":"batch","updates":{SYM:PriceData,...}}` — **steady-state ticks**,
  batched every 100ms (last-write-wins per symbol)
- `{"type":"status",...}`, `{"type":"subscribed","symbols":[...]}`,
  `{"type":"unsubscribed","symbols":[...]}`, `{"type":"error","message":...}`
- `{"type":"ping"}` — server keep-alive; **client must reply** `{"action":"pong"}`
  or be closed after 65s of silence (`PING_INTERVAL_MS=30s`, `PONG_TIMEOUT_MS=65s`).
- (`snapshot`, `fundamentals`, `searchResults` — ignored by apex)

### 3.4 `PriceData` shape (`ib_tick_handler.js:createPriceData`)
`{symbol, last, lastIsCalculated, bid, ask, bidSize, askSize, volume, high,
low, open, close, ..., timestamp}` where `timestamp` is an **ISO-8601 string**
(`new Date().toISOString()`). `last` may be `null` (pre-first-tick / after hours).

## 4. apex ingest seam (verified)

`BarAggregator.on_tick` accepts a plain dict (`bar_aggregator.py:221-264`):
- `_extract_symbol`: dict key `symbol`.
- `_extract_price`: priority `last → mid → price → (bid+ask)/2`.
- `_extract_volume`: dict key `volume`.
- `_extract_timestamp`: dict key `timestamp`, returned **as-is** (so it must
  already be a `datetime` — the raw value is later compared with
  `builder.bar_start`/`bar_end`).

**Consequence — the only semantic conversion:** xenon's ISO-string `timestamp`
→ tz-aware `datetime`. Everything else is a key copy.

## 5. Components (each independently testable)

| Unit | Responsibility | Depends on | Tested by |
|---|---|---|---|
| `LiveFeedPort` (Protocol) | `connect/subscribe/unsubscribe/close` — lets `SubscriptionManager` drive a live feed without knowing the feed's identity | — | fake impl |
| `AuthProvider` (Protocol) + `NoAuthProvider` | `async ticket() -> str \| None`; default returns `None` (loopback path) | — | unit |
| `XenonTickTranslator` | pure: `PriceData` → tick dict `{symbol,last,bid,ask,volume,timestamp:datetime}`; ISO→datetime; drop no-price ticks | — | pure unit |
| `XenonTickClient` (impl `LiveFeedPort`) | own the `websockets` conn: subscribe framing, receive `batch`/`price`, **ping/pong**, **reconnect+resubscribe**; publish translated ticks on `MARKET_DATA_TICK` | `websockets`, `AuthProvider`, translator, event bus | fake WS server |
| Wiring | `SubscriptionManager` optional `live_feed`; `server.py` lifespan env-gated `XenonTickClient` | above | integration |

### 5.1 Why a dict tick (not `MarketDataTickEvent`)
We publish a plain dict on `MARKET_DATA_TICK`, not a `MarketDataTickEvent`,
because the dict path (a) carries the tick's **real event-time** (the event
class's timestamp is construction-time) and **volume** (the event class has no
`volume` field, `domain_events.py:656`), and (b) keeps the xenon adapter
decoupled from the domain event class. The only `MARKET_DATA_TICK` consumer in
the TA path is `BarAggregator`, which fully supports dicts; `delta_publisher`
(risk monitor) is drop-set and does not run in the streaming-TA service.

## 6. Data flow

Steady state: xenon `{type:"batch",updates:{SYM:PriceData}}` (≈10/s) → client
emits one tick dict per symbol → bus → aggregator folds into the forming bar →
on bar close the existing indicator/rule/emitter chain fires → argon receives a
live `signal_service_payload`. The first `{type:"price"}` on subscribe is
translated the same way (one tick).

## 7. Subscription wiring

`SubscriptionManager` (`src/application/subscriptions/manager.py`) gains an
**optional** `live_feed: LiveFeedPort | None`:
- `subscribe(symbol)`: after seeding history (existing), if `live_feed` is set,
  `await live_feed.subscribe(symbol)` — opens the xenon sub so live ticks flow.
- `unsubscribe(symbol)` at refcount→0: if `live_feed` is set,
  `await live_feed.unsubscribe(symbol)` — drops the xenon sub.
- Ordering mirrors the existing seed-before-acquire discipline: a live-feed
  failure must not poison the in-memory sub entry.

## 8. Lifespan wiring (env-gated, default OFF)

In `src/api/server.py` `lifespan`, env-gated on **`APEX_XENON_WS_URL`**:
- If set, construct `XenonTickClient(url, auth=NoAuthProvider(), event_bus=<the
  pipeline bus>, translator=XenonTickTranslator())`, `connect()` it, and pass it
  as `live_feed` to the `SubscriptionManager`. Start its receive loop as a
  tracked task; `close()` on shutdown.
- If unset (default, incl. all local/CI runs without xenon), nothing new runs —
  the PR is dormant. The client shares the **same event bus** the
  `TASignalService`/`SubscriptionManager` use (so ticks reach the aggregators).

## 9. Error handling & resilience

- **Reconnect:** capped exponential backoff; on reconnect, re-send `subscribe`
  for every active symbol (mirrors xenon's own `restoreSubscriptions`).
- **Keep-alive:** on server `{type:"ping"}` reply `{action:"pong"}`; if the read
  loop sees no traffic within a timeout, drop and reconnect.
- **Bad/null ticks:** translator skips ticks with no usable price (no bus spam);
  malformed JSON frames are logged and skipped, not fatal.
- **Reconnect gap (documented limitation, deferred to Phase 4.1):** bars forming
  during an outage resume on the next tick (a wide time-jump closes the stale
  bar). **Re-seeding missed bars from livewire on reconnect is NOT in Phase 4**
  — simplicity first; noted here as the known gap.

## 10. Testing strategy

No live xenon in CI (needs IB Gateway + market hours). Verify against a **fake
xenon WS server fixture** — a real in-process `websockets` server speaking the
actual protocol (`batch`/`price`/`ping`/`subscribed` frames). TDD throughout:

- **Translator** (pure): ISO→datetime, price priority, volume passthrough, drop
  on null price.
- **Client** (vs fake server): connect→subscribe(send frame)→receive
  `batch`→publish ticks; server `ping`→client `pong`; kill server→reconnect→
  re-subscribe active set.
- **End-to-end:** fake xenon tick → bus → `BarAggregator` → a `TRADING_SIGNAL`
  reaches a `SignalEmitter` (Phase 3 smoke pattern, now driven by a live tick).
- **Wiring:** `SubscriptionManager.subscribe` opens a feed sub and `unsubscribe`
  drops it (fake `LiveFeedPort`); live-feed failure leaves no poisoned entry.

## 11. Scope

**In:** `LiveFeedPort`, `AuthProvider`/`NoAuthProvider`, `XenonTickTranslator`,
`XenonTickClient` (reconnect + keep-alive + resubscribe), `SubscriptionManager`
live-feed wiring, env-gated lifespan construction, all tests above, `websockets`
declared as a direct dependency.

**Out (deferred):** service-JWT `TicketAuthProvider` (D2 full — blocked on xenon);
reconnect re-seed from livewire (Phase 4.1); options/index/snapshot/search/
fundamentals xenon message types; any live integration test against real xenon;
HK/Asia feeds. Backtest untouched (`src/backtest`, `domain/backtest`).

## 12. Success criteria

- (a) A tick delivered by the fake xenon server reaches `BarAggregator` via
  `MARKET_DATA_TICK` and, across a bar boundary, yields a `TRADING_SIGNAL`
  emitted to a subscribed socket — proven by an end-to-end test.
- (b) `SubscriptionManager.subscribe`/`unsubscribe` open/drop the live feed sub
  (refcounted), with no poisoned state on failure.
- (c) Client survives a server drop: reconnects and re-subscribes the active set.
- (d) With `APEX_XENON_WS_URL` unset, the service behaves exactly as Phase 3
  (no new runtime behavior) — full suite green.
- (e) No fabrication: every xenon frame/field the client handles is one verified
  in `ib_realtime_server.js` / `ib_tick_handler.js`.
