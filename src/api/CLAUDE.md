# src/api/ — FastAPI server + WS hub

Root `CLAUDE.md` is authoritative for policy.

## Entry point

`src/api/server.py` — app factory with lifespan. Port **8322** (`APEX_API_PORT`). Consumed by argon.

## Routes

**Do not maintain a route table here** — it drifts on every route commit. The authoritative reference is `docs/argon-apex-api.md`: §3 for the flat routes, **§3a for the `/v1` asset-class surface** (query params, error envelope, and the flat→`/v1` deprecation mapping). Routers live in `src/api/routes/` (`chart`, `instruments`, `signals`, `regime`, `screener`, `strategy`, `backtest`, `health`) plus `src/api/ws/signals_ws.py`; registration order is in `server.py`.

Two facts about that surface that the code alone does not explain:

- **`GET /v1/equity/{symbol}/actions` and `GET /v1/equity/{symbol}/delisting` are served from the lake** (they returned `501` until 0.1.9). Both are **ticker-keyed, not security-keyed**, and say so in the payload (`identity: "ticker"`) — the corporate-action log and the security master are both stored per ticker, so for a reused ticker the rows may belong to a different, living company. `/delisting` is **not** a terminal-state record: the security master carries no delisting reason and no final consideration (measured 2026-09-21 — `relationship_type` and `related_security_id` are null across the file), so it returns the `[effective_from, effective_to)` identity intervals and says `delisting_reason_available: false`. Do not invent a reason field on top of it.
- **`GET /v1/equity/bars` is the bulk read and must be registered before `instruments`.** `/v1/{asset_class}/{symbol}` would otherwise match it as `symbol="bars"` — the same trap `/v1/equity/returns` and `/v1/membership/*` sit in. `src/api/routes/bulk_bars.py`; `tests/unit/api/test_bulk_bars.py::test_bars_is_not_swallowed_as_a_symbol` is the guard.
- **`listing=delisted` and `listing=any` serve bars** from `bronze-delisted/`, raw only. `price_mode=adjusted` over that tree is a `400 adjusted_not_supported`, never a silent raw fallback. A dual-resident ticker returns the union with the live tree winning every shared America/New_York trading date, labelled `listing_status: "dual"`. Presence in both trees does not by itself mean two issuers — check `GET /v1/equity/{symbol}/delisting` (security-master intervals) before computing a return across the seam. `_chart_guards.py::_check_listing` owns the resolution.
- **`GET /health` is not a bare liveness check.** It returns version, uptime, `pg_connected`, a `livewire` block (configured vs effective price mode, plus **artifact-derived** recency from `provider.fetch_recency()` — deliberately _not_ livewire's 11:00 UTC coverage snapshot, which under-reports and would show a lag that does not exist), and a `silver_revision` block. Anything added here must stay non-blocking.

## Lifespan startup order

`server.py`, in order: PG pool (`APEX_PG_URL`) → PG read pools (`app.state.pg_read_pools` from `APEX_PG_READ_URLS`; empty map when unset, so `/v1/db/*` and `/v1/uw/*` degrade to `503`) → `SignalHub` + signal repo → `LivewireOhlcProvider` → `CoverageCatalog` (gates `/v1/instruments`; `503` when `APEX_LIVEWIRE_COVERAGE_DB` is unset) → indicator registry → event bus → TA service → `SignalEmitter` → `SubscriptionManager` → xenon WS client.

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
