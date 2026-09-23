# UW read API — design

Status: reviewed and implemented; activation pending (`APEX_PG_READ_URLS`/`APEX_PG_READ_TOKEN` provisioning plus the DBA grant step below). Branch `feat/uw-read-api`, worktree `.worktrees/feat-uw-read-api`.
Evidence: macmini Postgres inventory and join verification, 2026-09-22 (scratchpad files
`pg_join_map.md`, `uw_scan_predefined_joins.md`, `uw_scan_joins_verified.md`).

## Goal

Expose the macmini Postgres instance over the apex REST API, read-only, so a client
(argon UI, signal-lab, notebooks) can read any table and run the predefined joins
without opening a DB connection. Scope of joins: inside `option_wizard.uw_scan` only.

## Non-goals

- No writes. apex stays a reader; `uw_scan` DDL belongs to argon (alembic).
- No cross-database joins. `core`, `option_chain`, `apex_signals` get table reads only.
- No free-form SQL from clients in v1 (see "Later").

## What exists today

- apex: one asyncpg pool on `APEX_PG_URL` (`apex_signals`, role `apex_app`), three routes
  over three tables, per-payload JSON schema + `validate_payload`, `ApiError` codes.
- `apex_app` cannot read `uw_scan` (owner `argon_app`), `archive` or `core.marketdata`.
- `uw_scan`: 180 tables. argon runs its joins as inline SQL; the schema has two views,
  neither is a core join.

## Design

### 1. Access (DBA step, no code)

Role `apex_reader`, LOGIN, `NOSUPERUSER`, `default_transaction_read_only = on`,
`statement_timeout = '30s'`. Per database: `GRANT CONNECT`, `GRANT USAGE ON SCHEMA`,
`GRANT SELECT ON ALL TABLES IN SCHEMA`, matching `ALTER DEFAULT PRIVILEGES` so new
argon tables are readable without a re-grant. New env `APEX_PG_READ_URLS`, a
comma-separated list of DSNs; each DSN's database name is the API's `{database}` segment.
Unset means every `/v1/db/*` and `/v1/uw/*` route answers 503
`provider_not_configured`, like every other source.

The blanket `GRANT SELECT ON ALL TABLES` also covers the ops/audit tables listed
below, so it must be followed by an explicit `REVOKE SELECT` on each of them —
the API filters the same relations, and the database-level revoke is the
defense-in-depth layer. The exclusion applies globally by relation name: the
catalog drops those tables in every schema of every configured database, so the
revoke list is by table name, not by schema-qualified name.

Default privileges must target each actual object-creator role, not the DBA's
session role. For `uw_scan`, the required statement is
`ALTER DEFAULT PRIVILEGES FOR ROLE argon_app IN SCHEMA uw_scan GRANT SELECT ON TABLES TO apex_reader`.
Repeat for the verified creator/schema pairs in each configured database. Provision
the login password outside source control. Verify role flags, existing SELECT and
`pg_default_acl` before enabling DSNs. This implementation does not execute DBA grants.

### 1a. Caller authentication

All three routers (`/v1/db/catalog`, `/v1/db/{database}/{schema}/{table}`,
`/v1/uw/{join}`) require `Authorization: Bearer <APEX_PG_READ_TOKEN>`, enforced by
a shared router-level dependency (`src/api/routes/_db_auth.py`,
`secrets.compare_digest` on UTF-8 bytes, scheme matched case-insensitively).
`APEX_PG_READ_TOKEN` unset means the surface answers 503 `provider_not_configured`
even to well-formed requests; a configured token with a missing, wrong,
non-Bearer, or malformed `Authorization` header is 401 `unauthorized` with a
`WWW-Authenticate: Bearer` challenge. Auth runs before any pool acquisition or
catalog lookup, and the token never appears in responses or logs. Provision the
token outside source control, same as the DSN passwords.

### 2. Catalog: `GET /v1/db/catalog[?database=]`

Reads `pg_catalog` relation/attribute/type metadata and `pg_constraint` once per pool
at first use (rows filtered to relations the pool role can actually `SELECT`), caches
in memory with a 10-minute TTL. Returns databases, schemas, tables, columns with types,
PK, UNIQUE and FK edges. Excludes `pg_*`, `_timescaledb*`, `timescaledb_*`,
`information_schema`, and the ops/audit tables listed below. The catalog is also the
allowlist: a table or column absent from it is a 400 `invalid_parameter`, never
interpolated into SQL.

### 3. Table read: `GET /v1/db/{database}/{schema}/{table}`

Query params: `columns=a,b`, `where=col:op:value` (repeatable; ops `eq ne lt le gt ge
in like isnull`), `order=col[:desc]`, `limit` (default 500, max 5000), `offset`.
Identifiers come from the catalog; values are bound parameters. Response is one generic
`tabular_payload` schema: `{database, schema, table, columns:[{name,type}], rows:[[...]],
count, truncated, generated_at}`. Rows as arrays, not objects, to keep 5000-row
payloads small. Dates serialise ISO; `numeric` as string to avoid float drift.

Split `where` on the first two colons only. Repeated predicates combine with AND.
`in` accepts comma-separated values (values containing commas are unsupported in
v1); `isnull:true|false` tests SQL NULL. `count` means rows in this response.
Default order uses the primary key. Without a PK, offset pagination requires an
explicit order; a non-unique order does not guarantee stable pages under ties or
concurrent writes. Callers needing a snapshot must use a frozen data product.

### 4. Predefined joins: `GET /v1/uw/{join}`

One registry module maps a join name to a SQL template plus its allowed filters
(`ticker`, `start`, `end`, `run_id`, `expiry`). The SQL lives in apex, not as DB views,
because apex is the only reader that needs them today and a view migration would have
to land in argon first. Same `tabular_payload`. Every join also reports `coverage`:
which right-hand tables matched, so partial joins are visible, not silent.

In v1 (original sample-match claims are not universal coverage guarantees):

| join | tables | key |
|---|---|---|
| watchlist_active_card | watchlist, watchlist_card, intraday_quote, scan_runs | ticker, run_id |
| scan_run_signal_bundle | signal_gates, signal_hits, signal_context_flags | run_id, ticker |
| oi_change_with_quote | oi_change_events, option_contract_snapshots | run_id, option_symbol |
| strike_grid | greeks_by_expiry_strike, exposures_by_expiry_strike, option_surface_grid_daily | run_id / ticker, expiry, strike, market_date |
| trade_insight_thread | trade_insight_snapshots, candidates, ai_analyses, outcomes | snapshot_id, analysis_id (no FK) |
| chain_exposure | research_chains, chain_membership, company_exposure | taxonomy_version, chain, ticker, valid_to is null |
| universe_identity_sector | fundamental_universe, company_identity, company_sector | ticker (+ tier filter) |
| daily_ohlc_technical | daily_ohlc, technical_daily | ticker, date = as_of |
| macro_evidence_chain | macro_domain_states, evidence, observations, source_artifacts | FK chain |
| fundamental_evidence_chain | fundamental_scores, result_provenance, statement_obs | FK chain |

All joins preserve their driving rows with LEFT JOIN and expose each right table's
match flag. Coverage counts describe returned rows, not a whole-database statistic.
One-to-many grains must be documented; signal hits and flags aggregate separately
to avoid multiplying each other. Additional daily panel:

- daily_signal_panel: vrp_daily + uw_gex_levels_daily (77%) + skew_analytics_snapshot
  `basis='eod'` + iv_rank_history. matrix_state_snapshots excluded (4 tickers only).

Not in v1 (tier C): fundamental_scores ↔ valuation_anchors (duplicate keys both sides),
earnings_calendar ↔ earnings_reactions (5% populated).

### 5. Resolving "latest run"

`scan_runs` has runs with no fact rows (the latest SPY run today has none). `run_id`
defaults to "latest run that has rows in the join's driving table", computed per request
with `max(run_id) ... where ticker = $1`. Clients can pass `run_id` explicitly.

This applies to signal bundle, OI change, strike grid and trade insight. The active
watchlist/card/quote tables are ticker-keyed current snapshots, so that endpoint
accepts only ticker and does not claim historical run alignment.

| join | filters (plus limit/offset) | output grain |
|---|---|---|
| watchlist_active_card | ticker | active watchlist ticker |
| scan_run_signal_bundle | ticker required, run_id | gate row; hits/flags are separate arrays |
| oi_change_with_quote | ticker required, run_id | OI event/contract in selected run |
| strike_grid | ticker required, run_id, start, end, expiry | selected run, ticker, expiry, strike |
| trade_insight_thread | ticker required, run_id, start, end | snapshot; candidates/analyses are separate arrays |
| chain_exposure | ticker required | current membership and matched exposure |
| universe_identity_sector | ticker, tier | active universe tier/ticker |
| daily_ohlc_technical | ticker required, start, end | ticker/date |
| macro_evidence_chain | start, end | published state/evidence |
| fundamental_evidence_chain | ticker required, start, end | score/provenance |
| daily_signal_panel | ticker required, start, end | ticker/market_date |

Date ranges are inclusive. The two joins whose driving column is a `timestamptz`
(`trade_insight_thread`, `macro_evidence_chain`) compare `(as_of AT TIME ZONE 'UTC')::date`
to the bounds — UTC calendar-date semantics, session-zone independent; the `date`-keyed
joins compare the `date` column directly. Coverage is real partial data, not a
guarantee: the 2026-09-22 fixtures retain rows with no provenance, analysis,
outcome, or IV-rank match, and `daily_signal_panel` joins `uw_gex_levels_daily`
at roughly three-quarter coverage by design.

### 6. Option-chain contract

SPX and equities share the same tables and OCC symbols. `oi_change_with_quote`
adds parsed `root`. `strike_grid` has independent call and put symbols, so it
returns `call_root`, `put_root`, and `root` only when both parsed roots agree
(otherwise null). These fields identify the observed series; they do not provide
settlement metadata. Multiplier and settlement are not stored in `uw_scan`;
v1 does not invent them.

### 7. Errors and limits

Reuse existing `INVALID_PARAMETER` (400) for unknown databases,
tables, columns and invalid filters. New `QUERY_TIMEOUT` (504) covers query timeouts
and pool-acquisition timeouts; new `UNAUTHORIZED` (401, with `WWW-Authenticate:
Bearer`) covers the credential boundary in §1a. Per-database pool `max_size=3`.
Fetch one extra row and set `truncated=true` only when rows beyond the requested
page actually exist. `limit` is clamped to 5000; `offset` must fit signed int64 —
the generic route binds it as a bigint parameter, while the join templates cast
`OFFSET` to int and bound it at int32 max. Column `type` metadata is honest but
not uniform: generic reads report `format_type` (typmods included, e.g.
`numeric(10,2)`) while the curated joins report the native type name from the
prepared statement (no typmods).

### 8. Implementation acceptance clarifications

- Enforce an explicit read-only transaction and 30-second statement timeout for every
  query. These pools are separate from the signal writer pool. Do not log DSNs.
- Catalog entries require table-level SELECT privilege. Filter excluded relations
  from FK metadata too; pair composite constraint columns by ordinality.
- Register literal `/v1/db` and `/v1/uw` namespaces before asset-class catch-all routes.
- Obtain column metadata even for empty results. Shared payload serialization belongs
  under `src/api/payload/`; server lifecycle and error wiring are also in scope.
- Verify each join's grain/cardinality against current mini data, not only a match
  percentage. Define coverage on the returned page, with explicit matched row counts.
- Verify OCC availability for strike_grid before implementing root. An aggregated
  strike row cannot claim a single root if multiple contract series contributed.
- Unit tests cannot establish live DB permissions or SQL correctness. Execute all
  registered templates on mini and preserve verification evidence separately.

## Files

```
src/api/routes/_db_auth.py        shared Bearer dependency for all three routers
src/api/routes/db_catalog.py      catalog route + allowlist cache
src/api/routes/db_table.py        generic table read, filter parser
src/api/routes/uw_joins.py        /v1/uw/{join} route
src/api/uw_join_registry.py       join name -> SQL template + filters + coverage columns
src/api/payload/tabular.py        shared tabular_payload builder + lossless JSON values
src/api/errors.py                 QUERY_TIMEOUT + UNAUTHORIZED codes, Bearer challenge header
src/api/server.py                 router registration order + read-pool lifespan wiring
src/infrastructure/persistence/read_pools.py   APEX_PG_READ_URLS -> {database: pool}
config/verification/schemas/tabular_payload.schema.json
config/verification/schemas/db_catalog_payload.schema.json
scripts/check_pg_read_api.py      authenticated read-only verification (runs on the DB host)
tests/fixtures/pg_read_api/       frozen real rows (2026-09-22) + README
tests/unit/api/test_db_table.py, test_uw_joins.py, test_db_catalog.py,
    test_server_lifespan.py, test_errors.py, test_tabular_payload.py
```

Each route file stays under the 500-line budget; the filter parser is its own seam.

## Tests

- Filter parser: every op, rejected identifiers, bound values never interpolated.
- Catalog allowlist: excluded schemas and ops tables never appear.
- Join registry: each template compiles against a fixture catalog; coverage columns exist.
- Route tests mock the pool (asyncpg `fetch`) with frozen rows taken from the mini on
  2026-09-22 (real tickers, real values, as-of dated).

## Ops tables excluded from catalog and reads

api_request_audit, raw_payloads, external_api_requests, jobs, job_failures,
worker_heartbeat, pipeline_benchmark_snapshots, data_gap_*, data_freshness_snapshots,
volatility_backfill_status, ws_consumer_state, macro_source_status, uw_fetch_memo,
pg_stat_statements*.

## Later

- `POST /v1/db/{database}/query` with a SELECT-only parser, once there is a caller.
- Move tier-A joins into `uw_scan` views via an argon migration so psql and Grafana share
  them; apex then reads the views through the generic table route and the registry shrinks.
- postgres_fdw analytics database if cross-database joins become a real need.
