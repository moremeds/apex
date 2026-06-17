# src/infrastructure/ — Adapters, persistence, stores

Root `CLAUDE.md` is authoritative for policy.

## Adapters (`adapters/`)

| Adapter | Role |
|---------|------|
| `livewire/` | Reads OHLCV Parquet from the **local-filesystem** bronze lake (`APEX_LIVEWIRE_ROOT`) via DuckDB (in-memory, not a datastore) |
| `xenon/` | WebSocket client for xenon's IB realtime feed — live tick source |
| `r2/` | Cloudflare R2 S3-compatible store (boto3); Parquet lake read/write |
| `fmp/` | Financial Modeling Prep API — daily OHLCV deltas, screener data |
| `ib/` | IB Gateway adapter (secondary; live ticks come via xenon WS in normal operation) |
| `futu/` | Futu OpenD read-only adapter |
| `yahoo/` | Yahoo Finance — R2 bulk backfill **only**, never live or incremental |

## DuckDB (livewire)

`adapters/livewire/ohlc_provider.py` creates an **in-memory** DuckDB session per request to `read_parquet()` over the **local** bronze lake at `APEX_LIVEWIRE_ROOT` (a Hive `asset_class=…/symbol=…/<tf>.parquet` tree — livewire writes it; apex only reads). On the macmini it lives on an external disk, bind-mounted read-only into the container (see `docker-compose.yml`). **Not a persistent database** — any `*.duckdb` file on disk is a stale artifact (gitignored). Never treat it as a source of truth. (R2 — the `r2/` adapter below — is a separate backfill pipeline, not this read path.)

## FMP Intraday Limits

FMP caps intraday at ~410 rows/request. Pagination required for full history:

| Timeframe | Cap | Chunk size |
|-----------|-----|------------|
| 1h | ~410 bars (~3 months) | 90-day windows |
| 4h | ~245 bars (~6 months) | 180-day windows |
| 1d | 2,500+ bars | No pagination needed |

Strategy: Yahoo for initial bulk 1h/4h R2 fill; FMP for daily deltas.

## Persistence (`persistence/`)

- `pg_schema.py` — DDL for 6 tables: bars, signals, summary, score_history, + 2 more. CLI for init/reset.
- `pg_repositories.py` — asyncpg writes; thin repository over the pool.
- `signal_listener.py` — PostgreSQL LISTEN/NOTIFY for real-time signal fan-out.

Schema is managed via `make db-init` / `make db-reset` — no migration framework; schema is recreated from DDL.

## Stores (`stores/`)

RCU (Read-Copy-Update) pattern: readers get lock-free snapshots; writers swap in a new copy atomically. Used for market data and position state that's read frequently from the signal pipeline.

- `rcu_store.py` — base RCU implementation
- `market_data_store.py` — live quotes
- `position_store.py` — live positions
- `parquet_historical_store.py` — cached Parquet data
- `duckdb_coverage_store.py` — data coverage metadata

## Anti-patterns (DO NOT)

- Do NOT call `prune_stale_subscriptions()` on fetch cycles — causes subscription churn
- Do NOT filter positions before `fetch_market_data()` when pruning is involved
- Do NOT forget `MarketDataFetcher.start_dispatch()` — processes IB callback thread
- Do NOT merge tick data without updating `MarketData.timestamp`
- Do NOT treat the livewire DuckDB as a persistent store — it is in-memory per request
