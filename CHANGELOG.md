# Changelog

All notable changes to apex are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`VERSION` (repo root) is the source of truth and must match
`pyproject.toml [project].version` (enforced by `scripts/release/version_sync_check.py`).

## [Unreleased]

## [0.1.1] — 2026-06-17


### Added
- Release pipeline: `VERSION` + `CHANGELOG.md` + `scripts/release/cut.sh` (interactive
  bump → tag) and tag-triggered `release.yml` (verify → GitHub Release → GHCR arm64 image).
- Docker deploy: `docker/api.Dockerfile` + `docker-compose.yml` for the macmini, with the
  livewire bronze lake bind-mounted read-only and Watchtower auto-deploy on `:latest`.

### Fixed
- Docs: corrected the "R2-backed livewire" claim — the livewire read path is a local-FS
  Parquet lake (`APEX_LIVEWIRE_ROOT`), not R2. R2 is only the `make r2-backfill` pipeline.
