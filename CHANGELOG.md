# Changelog

All notable changes to apex are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`VERSION` (repo root) is the source of truth and must match
`pyproject.toml [project].version` (enforced by `scripts/release/version_sync_check.py`).

## [Unreleased]

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