# Changelog

All notable changes to apex are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`VERSION` (repo root) is the source of truth and must match
`pyproject.toml [project].version` (enforced by `scripts/release/version_sync_check.py`).

## [Unreleased]

### Fixed

- **`uv.lock` now carries the released version.** `scripts/release/cut.sh` bumped `VERSION` and
  `pyproject.toml` but not `uv.lock`, which also records the project version, so every release
  left the lock one version behind (master's lock said `0.1.10` after `v0.1.11`). The script now
  runs `uv lock` after the rewrites and commits the lock with the release; this change also
  brings master's lock to `0.1.11`.

## [0.1.11] — 2026-09-23


### Fixed

- **A dropped table or column no longer returns `400` for up to 10 minutes.** The PostgreSQL
  read API caches each database's catalog for 10 minutes. When a query hits a table or column
  the database no longer has, the route still answers `400`, and now also drops that
  database's cached catalog, so the next request rebuilds it from the live database. A
  revoked grant or a connection failure leaves the cache alone.
## [0.1.10] — 2026-09-23


### Added

- **Read-only PostgreSQL API behind a Bearer token.** `GET /v1/db/catalog`,
  `GET /v1/db/{database}/{schema}/{table}` and `GET /v1/uw/{join_name}` read the macmini
  databases named in `APEX_PG_READ_URLS` (apex_signals, core, option_chain, option_wizard).
  Every request needs `Authorization: Bearer <APEX_PG_READ_TOKEN>`: a missing or wrong token
  is `401`, and an unset token makes the routes `503`. The token applies only to these routes,
  not to the existing chart and bars routes. Each database gets its own small pool: read-only,
  30s statement timeout, at most 3 connections.
- **The catalog is the allowlist.** Tables and columns come from the database's own catalog;
  system schemas and 16 option_wizard ops tables (request audit, raw payloads, jobs, heartbeats,
  data-gap bookkeeping) are excluded and answer `400` as unknown. Identifiers are quoted from
  the catalog and every value is a bound parameter. Without `?database=`, the catalog lists the
  databases it can reach and names the rest in `unavailable` instead of failing the whole call.
- **11 predefined joins over option_wizard.uw_scan**, each verified against the live database
  before release. A response reports page-level `coverage`: for each right-hand table, how many
  of the returned rows found a match. It is a per-page count, not a guarantee of completeness.
- Driver failures have typed codes and never carry SQL: a table or column dropped since the
  catalog was cached is `400`, a revoked grant is `403 forbidden`, and a database that is down,
  saturated or rejecting the credentials is `503`.
## [0.1.9] — 2026-09-21


### Fixed

- **A bare date or naive timestamp in `start`/`end` is a `400`, not a `500`.** FastAPI coerces
  both `2024-01-02` and `2024-01-02T00:00:00` to a naive datetime without complaint, and the
  first tz-aware comparison in the window resolver then raised `TypeError`, surfacing a malformed
  query as an opaque `internal_error`. Every bars, indicator, confluence, rates-series and bulk
  route shares that resolver, so all of them were affected. The error now names the parameter and
  shows the accepted form. The offset is required rather than defaulted to UTC on purpose: a
  caller who meant an America/New_York boundary would otherwise be shifted silently.

### Documentation

- **Documented what `missing` means on the bulk bars route.** Measured across the whole equity
  tree 2026-09-21: 1,287 of 14,942 symbol directories (8.6%) have no `1d.parquet` in `bronze/`
  and every one of them has one in `bronze-delisted/`, with only 38 carrying a `.WS`/`.U` suffix.
  Reading `missing` as "absent, safe to skip" under the default `listing=listed` therefore drops
  the delisted cohort and reintroduces survivorship bias while appearing to succeed. Documented
  `listing=any` as the survivorship-free pull, and that it is raw-only because no Silver exists
  over the archive. Corrects an inaccuracy in that same note: `listing=any` with
  `price_mode=adjusted` is a `400` only on the per-symbol route; the bulk route returns `200`,
  serves the listed names, and reports each delisted one in `missing` with its reason.

### Added

- **`basis` on every bars payload.** `price_mode: "adjusted"` now also reports
  `basis: "split+dividend"`, and `raw` reports `basis: "unadjusted"`, so a consumer never has
  to infer the adjustment from a mode label. `split+dividend` is measured, not assumed: Silver
  revision 76 gives SPY a `price_adjustment_factor` of `0.9975231654864936` across
  `2026-06-18..09-17`, an interval containing no split, so "split-adjusted" would understate
  what the numbers are. `/v1/equity/returns` carries the same field.
- **Delisted bars.** `listing=delisted` serves raw bars from `bronze-delisted/` (8,620 equity
  symbols, `1d/1h/5m/1m`) instead of the old `501`. `listing=any` resolves to whichever tree
  holds the symbol, and for a dual-resident ticker returns the **union** of both with the live
  tree winning every shared America/New_York trading date and `listing_status: "dual"` —
  replacing the previous `409 ambiguous_symbol`. `price_mode=adjusted` over the archived tree is a
  `400 adjusted_not_supported` (`no Silver for delisted names; use price_mode=raw`), never a
  silent raw fallback: livewire publishes no Silver there, and splicing an adjusted segment
  onto a raw one would put two definitions in one series.
- **`GET /v1/equity/bars`** — bulk OHLCV for up to 200 tickers in one request, mirroring
  `/v1/equity/returns` conventions (comma list, upper-cased, de-duplicated). In adjusted mode
  the whole table reads under **one pinned Silver revision**, so every series is adjusted on
  the same corporate-action set; 200 single-symbol calls pin 200 revisions independently. A
  symbol that cannot be served lands in `missing: {SYM: reason}` rather than failing the
  request. Registered before `instruments` — `/v1/{asset_class}/{symbol}` would otherwise match
  it as `symbol="bars"`.
- **`GET /v1/equity/{symbol}/actions`** — corporate actions read from livewire bronze
  (`asset_class=corporate_action`, 15,015 symbols) instead of the old `501`. Only
  `status='active'` rows (a correction is a new `action_id` superseding the old row, which is
  re-marked `corrected`; listing both would double-count a dividend), ordered by `ex_date`; optional `type=split|cash_dividend` and `start`/`end` on `ex_date`.
- **`GET /v1/equity/{symbol}/delisting`** — security-master identity intervals instead of the
  old `501`. It is **not** a terminal-state record and does not pretend to be: measured
  2026-09-21 the master carries no delisting reason and no final consideration
  (`relationship_type` and `related_security_id` are null across the whole file), so the
  response is the `[effective_from, effective_to)` intervals with `issuer_name`,
  `exchange_mic` and `delisting_reason_available: false`.

Both reference endpoints are **ticker-keyed, not security-keyed**, and say so in the payload
(`identity: "ticker"`): for a reused ticker the rows may belong to a different, living company.

### Changed

- `bars_payload.schema.json` requires `basis` and accepts `listing_status: "dual"`.
- `ambiguous_symbol` (409) is now reserved — no route emits it.
## [0.1.8] — 2026-09-16


### Fixed

- `GET /v1/membership/history` no longer hides the part of a ticker's log that predates
  livewire's identity floor. Events before the floor stay logged under the placeholder
  `unresolved:<TICKER>` even after the backfill resolves the ticker, so the route queried
  only the resolved `security_id` and returned an empty timeline (`symbol=INTC&index_id=djia`
  returned `events: []` while the log held an add in 1999 and a remove in 2024). The route now
  queries both the resolved id and the placeholder and each event carries its own `security_id`.
  `/history` returns the **effective timeline**, not the raw audit rows: rows another row
  supersedes and rows with status `rejected` are dropped across the union of both ids, so the
  backfill's three-row pattern (original placeholder event, its rejected revision, the resolved
  replacement) collapses to the one live event instead of showing a duplicate add. Top-level
  `security_id` is still the resolved id when the master resolves the ticker, the placeholder
  when only it has events, and an unknown ticker with no events is still a 404.
## [0.1.7] — 2026-09-15


### Added

- Point-in-time index membership over REST, read from livewire's lake:
  `GET /v1/membership/indices`, `GET /v1/membership/{index_id}?as_of=&known_at=&include_candidates=`
  and `GET /v1/membership/history?symbol=&index_id=&as_of=`. Members are replayed from the
  append-only event log rather than read from a materialised list: events some later-known event
  `supersedes` are dropped, `known_at` gates both the replay and the security master to what was
  known on a date, and surviving members are kept only while the master still calls their identity
  verified. Tickers are resolved through the master at the same as-of date. Gated on the new
  `APEX_LIVEWIRE_LAKE_ROOT` (503 when unset). An index with no published events under the requested
  status reading fails closed with a 503; an empty replay over a populated log is a real
  point-in-time answer and returns 200 with an empty member list. `/history` is ungated (every
  status, superseded rows included, each carrying its `supersedes`) and falls back to livewire's
  `unresolved:<TICKER>` placeholder when the master does not know the ticker yet.
## [0.1.6] — 2026-09-08

### Added

- `GET /v1/equity/returns?symbols=A,B,C&start=&end=` — one call returns, per symbol, the
  window's daily closes and returns plus window return, YTD, distance from the 52-week high
  and excess vs SPY and QQQ. Every derived number is computed server-side on one price basis
  so no consumer does the arithmetic; symbols that cannot be served are listed in `missing`
  with a reason instead of being dropped. At most 200 symbols per call. (#160)
- Documented that `start`/`end` bound a bars window inclusively at **both** ends, intraday
  included — a `1m` request over `12:25:00Z..12:35:00Z` returns 11 bars, not 10. (#160)

## [0.1.5] — 2026-09-08


### Fixed

- Livewire Silver reads pin one manifest per logical request. The adapter validates
  `revisions/current.json` (schema 1, immutable pointer identity, complete membership, exactly one
  daily/factor artifact per member, safe encoded paths, declared SHA-256) and hashes only the
  artifact it reads. Missing, corrupt, withdrawn or uncovered data keeps the
  `503 adjusted_unavailable` contract. Chart, instrument-detail and canary responses report the
  revision they actually pinned. Live signal recomputation is still not atomic across a Silver
  revision; that consumer redesign is deferred to the Apex rewrite.

### Build

- `plotly>=6.0,<7`: plotly 7.0.0 dropped the scattermapbox traces vectorbt 0.28.2 registers, which
  broke the vectorbt engine tests on CI (CI installs from pyproject, not the lockfile).
- isort ordering in `backtest/execution` and a `Literal` direction for `optuna.create_study`, so
  CI's unpinned isort 9 and mypy accept the tree. `uv.lock` refreshed alongside.
## [0.1.4] — 2026-08-23

### Added

- `/v1/{asset_class}/{symbol}/bars` covering equity, volatility, fx, cmdty and futures.
- `/v1/rates/{symbol}/series` for FRED Treasury yields (a yield has no OHLC, so it gets its
  own payload shape).
- `/v1/instruments` and `/v1/{asset_class}/{symbol}` discovery.
- `/v1/equity/{symbol}/actions` and `/delisting` specified, returning a typed 501 pending livewire.
- Typed error envelope with machine-readable codes on every failure.
- `price_mode`, `listing_status`, `asset_class` and `adjustment_revision` on every bars payload.
- Futures bars carry `settlement`, `open_interest` and contract identity.
- `/health` now reports real bronze/silver recency read from the artifacts.
### Changed

- Flat routes (`/bars/{ticker}` etc.) are deprecated aliases; they emit `Deprecation` and `Sunset`.
- **Breaking for `/indicators` callers:** an unknown `indicator` now returns `400 invalid_parameter`
  instead of `404`. The symbol was never the problem; the old code sent callers to check their
  ticker.
- FastAPI's own request-validation failures (bad `limit`, unparseable date) keep their 422 status
  but now use the error envelope instead of `{"detail": [...]}`, so the surface has one error shape.
- `bars_payload` no longer emits `vwap` (always null -- no lake parquet carries the column).
- `bars_payload` timeframe enum narrowed to `1m/5m/30m/1h/1d`.
### Fixed

- Missing Silver artifacts return `503 adjusted_unavailable` instead of a bare 500 (243 symbols
  including HON, MMM, CMCSA).
- Non-equity symbols no longer resolve into `asset_class=equity` and read an absent file.
- fx and volatility intraday are reachable: both publish an intraday ladder that a
  daily-only assumption would have hidden (126 parquet files).
- Unknown symbols on `/indicators` and `/rates/.../series` return `404` instead of an empty
  `200`, which read as "no signal fired" / "this yield had no observations".
- `listing=any` probes the requested asset class, not always equity (`bronze-delisted` holds
  `asset_class=fx` too), so fx ticker reuse is no longer reported as unambiguous.
- A `start` after `end` is a `400`, not a `200` with zero rows.
- Indicators reject `rates`: computing over a yield's null OHLC returned a number-shaped
  answer to a question that has none.
- Adjusted daily reads probe Silver as well as Bronze, so a Silver-only symbol with an empty
  window is no longer a false `404`.
- Coverage-catalog reads run off the event loop; a synchronous DuckDB read was stalling every
  other request on the worker.
- Unanticipated failures (the lake volume going away, a truncated parquet) return a typed
  `500 internal_error` in the envelope rather than an unparseable bare 500, and no longer echo
  absolute lake paths to the client.
- Deploy config now matches production: silver mount plus `APEX_LIVEWIRE_SILVER_ROOT`,
  `APEX_LIVEWIRE_PRICE_MODE`, `APEX_LIVEWIRE_REVISION_POLL_SECONDS`. The published image had
  been pre-Silver since 0.1.3 because no release tag was cut after #150-#152 merged.
- CI now runs on pull requests to any base branch. It was filtered to `master`/`main`, so a
  stacked PR (base = another feature branch) got no CI at all — and GitHub's auto-retarget when
  the base merges fires `edited`, which is not in the default trigger set, so it would have
  stayed untested right up to the merge button.
- Delisted-tree mount is declared ahead of the `/v1` API work so that rollout is an image
  bump, not a host reconfiguration. The coverage catalog is intentionally left unmounted:
  it lives outside colima's VM mount set, so binding it would silently yield an empty
  directory instead of the database.
## [0.1.3] — 2026-07-08

### Fixed
- `/health` now reports the real running version (new `version` field) instead of a
  hardcoded `0.1.0`. Resolved from installed dist metadata so it can't drift from the
  shipped image — makes "is it live?" a one-curl check (`curl .../health | jq .version`).
## [0.1.2] — 2026-07-08

### Changed
- `GET /bars` and `GET /indicators` now accept a `limit` query param (default **2000**,
  up from a hardcoded 500) and no longer cap it — `limit<=0` returns full history. The
  param was previously undeclared, so callers passing `?limit=N` were silently ignored.
- R2 daily pipeline schedule disabled; the workflow is now manual-dispatch only.
### Fixed
- R2 daily pipeline failures and a TA-Lib version mismatch (#141).
## [0.1.1] — 2026-06-17

### Added
- Release pipeline: `VERSION` + `CHANGELOG.md` + `scripts/release/cut.sh` (interactive
  bump → tag) and tag-triggered `release.yml` (verify → GitHub Release → GHCR arm64 image).
- Docker deploy: `docker/api.Dockerfile` + `docker-compose.yml` for the macmini, with the
  livewire bronze lake bind-mounted read-only and Watchtower auto-deploy on `:latest`.
### Fixed
- Docs: corrected the "R2-backed livewire" claim — the livewire read path is a local-FS
  Parquet lake (`APEX_LIVEWIRE_ROOT`), not R2. R2 is only the `make r2-backfill` pipeline.