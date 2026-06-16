# src/api/ — FastAPI server + WS hub

Root `CLAUDE.md` is authoritative for policy.

## Entry point

`src/api/server.py` — app factory with lifespan. Port **8322**. Consumed by argon.

## Routes

| Route | Purpose |
|-------|---------|
| `GET /bars` | OHLCV bars from livewire lake or PG |
| `GET /indicators` | Compute-on-read indicator series |
| `GET /confluence` | Confluence points (oldest-first ASC) |
| `WS /ws/signals` | Streaming signal events via SignalHub |
| `GET /regime` | Current regime classification (R0–R3) |
| `GET /screener` | Screener results (momentum, PEAD) |
| `POST /backtest` | Trigger strategy backtest |
| `GET /health` | Liveness check |

## Lifespan startup order

1. PG pool (`APEX_PG_URL`) — optional; routes degrade gracefully when absent
2. `SignalHub` (`src/api/ws/hub.py`) — WS broadcast hub for signal events
3. xenon WS client (`XENON_WS_URL`, default `ws://127.0.0.1:8765`) — live tick feed; bootstraps livewire history on connect

Everything is torn down in `finally` so a half-built pipeline never leaks the pool.

## WS hub (`src/api/ws/`)

`SignalHub` manages client connections and fans out `TRADING_SIGNAL` events from the domain bus to all subscribed argon clients. Route: `src/api/routes/signals.py`.

## Rules

- **Routers are thin.** No business logic — call into `src/application/` or `src/domain/`. A route resolves params → calls service → returns model.
- **chart.py is compute-on-read.** `GET /bars` and `GET /indicators` recompute from the livewire lake per request; there is no pre-computed cache to invalidate.
- **CORS** — loopback (`127.0.0.1` / `localhost`) on ports `3000–3003` for local argon dev. Do not widen without an architectural reason.
- **JobManager** (`src/api/jobs/`) handles long-running compute (backtest, screener) as background tasks so routes stay non-blocking.
