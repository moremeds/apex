# src/api/ — FastAPI server + WS hub

Root `CLAUDE.md` is authoritative for policy.

## Entry point

`src/api/server.py` — app factory with lifespan. Port **8322** (`APEX_API_PORT`). Consumed by argon.

## Routes

**Do not maintain a route table here** — it drifts on every route commit. The authoritative reference is `docs/argon-apex-api.md`: §3 for the flat routes, **§3a for the `/v1` asset-class surface** (query params, error envelope, and the flat→`/v1` deprecation mapping). Routers live in `src/api/routes/` (`chart`, `instruments`, `signals`, `regime`, `screener`, `strategy`, `backtest`, `health`) plus `src/api/ws/signals_ws.py`; registration order is in `server.py`.

Two facts about that surface that the code alone does not explain:

- **`GET /v1/equity/{symbol}/actions` and `GET /v1/equity/{symbol}/delisting` return `501` unconditionally.** They are not stubs to fill in — the data does not exist upstream; they are blocked on livewire. Keep the `501` until livewire ships them.
- **`GET /health` is not a bare liveness check.** It returns version, uptime, `pg_connected`, a `livewire` block (configured vs effective price mode, plus **artifact-derived** recency from `provider.fetch_recency()` — deliberately _not_ livewire's 11:00 UTC coverage snapshot, which under-reports and would show a lag that does not exist), and a `silver_revision` block. Anything added here must stay non-blocking.

## Lifespan startup order

`server.py`, in order: PG pool (`APEX_PG_URL`) → `SignalHub` + signal repo → `LivewireOhlcProvider` → `CoverageCatalog` (gates `/v1/instruments`; `503` when `APEX_LIVEWIRE_COVERAGE_DB` is unset) → indicator registry → event bus → TA service → `SignalEmitter` → `SubscriptionManager` → xenon WS client.

Every stage is optional and degrades to `503` rather than failing the boot. Everything is torn down in `finally` so a half-built pipeline never leaks the pool.

The Silver revision watcher (`src/application/subscriptions/revision_watcher.py`) is stored on app state and surfaces through `/health`; on a new valid revision it reseeds only affected active subscriptions, buffering that symbol's xenon ticks across the swap.

## WS hub (`src/api/ws/`)

`SignalHub` (`hub.py`) manages client connections; `SignalEmitter` (`emitter.py`) fans out `TRADING_SIGNAL` events from the domain bus. Route: `ws/signals_ws.py` (`WS /ws/signals`).

## Rules

- **Routers are thin.** No business logic — resolve params → call `src/application/` or `src/domain/` → return model.
- **Chart routes are compute-on-read.** Bars and indicators recompute from the livewire lake per request; there is no pre-computed cache to invalidate.
- **`_chart_guards.py` owns pre-read validation.** Everything answering "is this request coherent, and which artifact would it read?" goes there; `chart.py` keeps routes and response assembly. Split out when `chart.py` crossed the 500-line budget — keep the seam.
- **`src/api/payload/`** builds and validates response payloads; routes should not hand-assemble dicts.
- **`JobManager`** (`src/api/jobs/`) runs long compute (backtest, screener) as background tasks so routes stay non-blocking.
- **CORS is a single origin**: `allow_origins=["http://localhost:3000"]`. Do not widen without an architectural reason.
- **Errors go through `src/api/errors.py`** so the `/v1` error envelope stays uniform.
