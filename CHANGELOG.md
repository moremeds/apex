# Changelog

All notable changes to apex are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`VERSION` (repo root) is the source of truth and must match
`pyproject.toml [project].version` (enforced by `scripts/release/version_sync_check.py`).

## [Unreleased]

## [0.1.4] — 2026-08-22

### Fixed

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
