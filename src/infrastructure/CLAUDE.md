# src/infrastructure/ — Adapters, persistence, stores

Root `CLAUDE.md` is authoritative for policy.

## livewire (`adapters/livewire/`) — the only live read path

`adapters/livewire/` reads two lakes. **Bronze** (`APEX_LIVEWIRE_ROOT`) is livewire's raw per-ticker Hive tree, `asset_class=<class>/symbol=<encode_symbol(SYM)>/<tf>.parquet`; `paths.py` owns that contract (including livewire's exact symbol percent-encoding) and `asset_classes.py` the six classes (`equity`, `volatility`, `fx`, `cmdty`, `futures`, `rates`) with their differing timeframe ladders — cmdty/futures/rates are daily-only. **Silver** (`APEX_LIVEWIRE_SILVER_ROOT`) is equity-only, split-and-dividend adjusted, and published _atomically as numbered revisions_: `revisions.py` reads `revisions/current.json`, validates `schema_version`, and SHA-256-verifies every referenced artifact before a revision is accepted (`RevisionManifestError` on anything malformed). `ohlc_provider.py` serves both, per-request in-memory DuckDB, under `APEX_LIVEWIRE_PRICE_MODE` (`raw` default; `adjusted` reads materialized Silver for daily and joins Silver factor intervals onto Bronze for intraday) — missing Silver data raises `AdjustedDataUnavailable`, never a silent fallback to raw. `coverage.py` reads livewire's `analytics.duckdb` catalog (`APEX_LIVEWIRE_COVERAGE_DB`) for `/v1/instruments` discovery, because scanning the lake costs ~19 minutes; those first/last dates are a snapshot and lag by up to a day. `src/application/subscriptions/revision_watcher.py` polls every `APEX_LIVEWIRE_REVISION_POLL_SECONDS` (30) and, on a new valid revision, reseeds only affected subscriptions while buffering that symbol's xenon ticks.

`membership.py` reads a third tree, rooted at `APEX_LIVEWIRE_LAKE_ROOT` — the lake root whose siblings are `bronze/` (what `APEX_LIVEWIRE_ROOT` points at), `index_membership/` and `security_master/`. `index_membership/<index_id>/events.parquet` is an append-only **event log**, not a member list: membership as of a date is a replay of `add`/`remove` ordered by `(effective_at, known_at, revision, event_id)`, after dropping every event some later-known event `supersedes`. Two reading modes: with no `known_at` the replay uses everything on disk — today's best reconstruction of that date; with a `known_at` cutoff only rows known by then are visible, and the superseded set is derived from those same rows, so a correction nobody had yet cannot retract history — the true point-in-time read. Only `status='verified'` counts by default; the looser reading (everything but `rejected`) exists because `r2k-proxy` is never verified. `security_master/events.parquet` holds `[effective_from, effective_to)` intervals — use only verified rows no other row supersedes — and labels a replayed member with its as-of ticker; a ticker matching two securities is reported as ambiguous, never guessed. Both files are replaced atomically by livewire with no manifest, so reads are per-request and uncached. An index whose log holds no row at all under the requested status reading fails closed with a 503 — upstream backfill is incomplete, and an empty list would read as "this index has no members". An empty replay over a log that *does* hold such rows is the real answer (the date predates the first constituent, or the `known_at` cutoff predates every event) and returns 200 with an empty list. `/history` is ungated — every status, superseded rows included, each carrying its `supersedes` — and falls back to livewire's `unresolved:<TICKER>` placeholder when the master does not know the ticker.

DuckDB here is **in-memory per request** (`duckdb.connect(database=":memory:")`), not a datastore. Any `*.duckdb` file apex writes is a stale artifact (gitignored) — never a source of truth. The one exception is livewire's own `analytics.duckdb`, which apex opens **read-only** for coverage.

Two different things are called "coverage" — keep them apart: `adapters/livewire/coverage.py` reads livewire's catalog (discovery, `/v1/instruments`), while `stores/duckdb_coverage_store.py` is legacy local bar-coverage bookkeeping.

## Other adapters

`adapters/` also holds `xenon/` (WS client for the live tick feed — the only live IB path), plus the frozen-subsystem adapters `fmp/`, `ib/`, `futu/`, `yahoo/`, `r2/`, `earnings/` and the loose `market_data_fetcher.py` / `market_data_manager.py` / `broker_manager.py` modules that the anti-patterns below refer to.

FMP caps intraday at ~410 rows/request, so full history needs pagination: 1h → 90-day windows (~410 bars), 4h → 180-day windows (~245 bars), 1d → 2,500+ bars in one call. Yahoo for the initial bulk 1h/4h fill, FMP for daily deltas — backfill only, never live.

## Persistence (`persistence/`)

- `pg_schema.py` — DDL for six tables: `bars`, `signals`, `summary`, `score_history`, `screener_results`, `backtest_results`. CLI for init/reset (`make db-init` / `make db-reset`); no migration framework, the schema is recreated from DDL.
- `pg_repositories.py` + `repositories/` — asyncpg writes over the shared pool.
- `signal_listener.py` — PostgreSQL LISTEN/NOTIFY for real-time signal fan-out.
- `database.py` — pool wrapper the repositories take.

**There is a second schema path.** `migrations/005_ta_signals.sql` creates `ta_signals` / `indicator_values` / `confluence_scores` — the tables the streaming signal surface actually reads — and `pg_schema.py` does **not** create them. `make db-init` alone is not enough to stand up the signal service; apply the migration too.

## Stores (`stores/`)

RCU (Read-Copy-Update): readers get lock-free snapshots, writers swap in a new copy atomically. Used for market data and position state read hot from the signal pipeline. `rcu_store.py` is the base; never mutate a snapshot in place.

## Anti-patterns (DO NOT)

- Do NOT call `prune_stale_subscriptions()` on fetch cycles — causes subscription churn
- Do NOT filter positions before `fetch_market_data()` when pruning is involved
- Do NOT forget `MarketDataFetcher.start_dispatch()` — processes the IB callback thread
- Do NOT merge tick data without updating `MarketData.timestamp`
- Do NOT treat the livewire DuckDB session as persistent — it is in-memory per request
- Do NOT add a raw fallback when adjusted data is missing — let `AdjustedDataUnavailable` propagate
