# APEX — streaming technical-analysis signal service

**APEX reads market bars and live ticks, computes technical-analysis indicators +
rule-engine signals + market regime, and serves signal payloads over WebSocket/REST.**

It is one of four decoupled systems:

```
livewire ──bars──▶  APEX  ──signals (WS/REST)──▶  argon
(parquet           (TA brain)                     (stateless UI)
 data lake)            ▲
                       └──live ticks── xenon (WebSocket)
```

- **livewire** — the bar warehouse (Hive-partitioned Parquet "bronze" lake, read on demand via DuckDB).
- **xenon** — the live tick feed (WebSocket).
- **APEX** — *this repo*: indicators (TA-Lib), a rule engine, regime detection. Stateless compute; persists fired signals to Postgres.
- **argon** — a stateless UI that pulls everything from APEX on demand (stores nothing).

> **Project status (2026-06):** APEX was pivoted from a risk-monitoring + backtesting
> monolith into this streaming service. The streaming path (bars-in → subscription →
> signals-out → live-in) and the chart read surface are implemented and merged. A Phase 6
> **strip-down** of the legacy backtest/risk code is still pending, so some of that code is
> still present under `src/backtest/` and `src/domain/backtest/`. Pre-pivot documentation
> has moved to [`docs/legacy/`](docs/legacy/).

---

## Run it

APEX is a FastAPI app served by uvicorn on port **8322** (override with `APEX_API_PORT`).

```bash
# 1. point APEX at its data sources via .env (gitignored)
cat > .env <<'EOF'
APEX_LIVEWIRE_ROOT=/absolute/path/to/livewire/data-lake/bronze
APEX_PG_URL=postgresql://apex_app:apex@127.0.0.1:5432/apex_signals
EOF

# 2. launch (the script loads .env, runs preflight checks, then execs the server)
scripts/serve.sh
#   …or directly, loading the env file yourself:
uv run --env-file .env python -m src.api.server
```

Health check: `curl -s http://127.0.0.1:8322/health` → `{"status":"ok", "pg_connected":…}`.

### Configuration (environment variables)

APEX's data sources are **env-gated** — it boots regardless, and each unset source just
makes its endpoints return `503` (degrades, never crashes).

| Variable | Default | Enables |
|---|---|---|
| `APEX_LIVEWIRE_ROOT` | unset | `/bars`, `/indicators`, and the streaming warmup seed |
| `APEX_PG_URL` | unset | `/signals` snapshot/backfill, `/confluence`, and signal persistence |
| `APEX_XENON_WS_URL` | `ws://127.0.0.1:8765` | live ticks → live WS signal frames |
| `APEX_TIMEFRAMES` | `1d` | timeframes the streaming pipeline subscribes/warms |
| `APEX_API_PORT` | `8322` | listen port |
| `APEX_API_WORKERS` | `1` | uvicorn workers |

---

## API surface

| Method · Path | Purpose |
|---|---|
| `GET /health` | liveness + `pg_connected` |
| `WS /ws/signals` | subscribe to a ticker → ack + PG snapshot + live signal frames |
| `GET /signals/{ticker}` | signal backfill (load / reconnect / `?since=`) |
| `GET /bars/{ticker}` | OHLCV candles from livewire |
| `GET /indicators/{ticker}?indicator=` | per-bar indicator series (compute-on-read) |
| `GET /confluence/{ticker}` | multi-timeframe confluence |

Full reference and payload shapes: **[`docs/argon-apex-api.md`](docs/argon-apex-api.md)**.

> **Lifecycle:** APEX is idle at rest. A WS subscribe triggers a one-time **warmup seed**
> (preloads history into the indicator buffer) and attaches the live feed; signals fire
> per **closed bar**, ref-counted across subscribers. The REST chart surface
> (`/bars`, `/indicators`, `/confluence`) is independent and **compute-on-read**.

---

## Documentation

- **[`docs/argon-apex-api.md`](docs/argon-apex-api.md)** — API reference for argon (WS + REST, status codes, payloads).
- **[`docs/argon-signal-consumption.md`](docs/argon-signal-consumption.md)** — consumption guide with real captured frames + examples.
- **[`docs/livewire-apex-integration.md`](docs/livewire-apex-integration.md)** — the livewire bronze data contract APEX reads.
- **[`docs/superpowers/specs/2026-06-14-apex-adaptation-design.md`](docs/superpowers/specs/2026-06-14-apex-adaptation-design.md)** — the adaptation roadmap (the master design; Phase 6 strip-down still governs here).
- **[`docs/superpowers/archive/`](docs/superpowers/archive/)** — completed Phase 0–4 + chart-surface specs/plans (delivered, kept for reference).
- **[`docs/legacy/`](docs/legacy/)** — pre-pivot monolith docs (risk monitor, backtesting). Historical only.

---

## Development

```bash
make install                                   # install deps (uv)
uv run python -m pytest tests/unit --no-cov    # unit tests
uv run mypy src/ --ignore-missing-imports      # type check
make format                                    # black + isort
```

Signal persistence schema: `migrations/005_ta_signals.sql` (`ta_signals` / `indicator_values` / `confluence_scores`).
