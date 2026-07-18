# Plan: apex release procedure (xenon-style) + Dockerized deploy

> **STATUS (2026-06-17):** Phases 1–4 + doc fixes **built & validated locally** on
> branch `feat/release-pipeline` (YAML parses, black/isort/flake8/mypy clean,
> version-sync green, cut.sh mutations dry-run-proven). **Phase 0 (colima mount +
> restart)** and **Phase 6 (cutover)** are parked for a maintenance window — they
> touch the live macmini stack. Next: `/codex-review` → PR.


Goal: mirror xenon's release flow for apex —
**merge to master → `cut.sh` bumps VERSION + CHANGELOG + tags → push tag fires `release.yml` → builds arm64 image → pushes GHCR `:latest` → Watchtower on macmini pulls + restarts.**

## Evidence (verified on macmini 2026-06-17)

- Prod host = macmini (arm64), Docker via **colima** (Linux VM, virtiofs, `mounts: []`).
- xenon stack runs `ghcr.io/moremeds/xenon-*:latest` + **Watchtower** (auto-deploy on `:latest`).
- apex today = **bare unsupervised process** (`uv run python -m src.api.server`); only api-server runs in prod (no signal-service). Dies on reboot.
- Postgres on host (:5432), DB `apex_signals`, user `apex_app`.
- Livewire lake = symlink → `/Volumes/DATA_LAKE/livewire/data-lake` — a **13TB exFAT external disk** (5.9TB, 20,395 symbols). **colima VM cannot see `/Volumes`** → naive bind-mount fails. **Root cause of the "can't read livewire in Docker" concern.**

## Decision (chosen): Option A — colima mount + read-only bind-mount

Add the external volume to colima's mount list, bind-mount the real path read-only into the apex container. Keeps the exact xenon Docker model.

---

## Phase 0 — macmini infra prep (one-time, maintenance window)

- [ ] Edit `~/.colima/default/colima.yaml`: add
      `mounts: [{location: /Volumes/DATA_LAKE/livewire/data-lake, writable: false}]`
- [ ] Ensure external disk auto-mounts **before** colima starts (boot ordering) — else VM mount is empty after reboot
- [ ] `colima restart` (⚠ cycles the whole Docker VM — xenon api/web/realtime blip ~1 min, recover via `restart:unless-stopped` + Watchtower). Coordinate timing.
- [ ] Verify: `colima ssh -- ls /Volumes/DATA_LAKE/livewire/data-lake/bronze/asset_class=equity | head`
- [ ] Re-run the DuckDB read proof in a throwaway container (already drafted) → confirm AAPL 1d rows read back

## Phase 1 — Versioning + CHANGELOG discipline (Python-only; simpler than xenon's triad)

- [ ] Add `VERSION` (seed `0.1.0`, matches pyproject)
- [ ] Add `CHANGELOG.md` (Keep-a-Changelog, with `## [Unreleased]`)
- [ ] `scripts/release/_lib.sh` — `bump_semver`, `extract_changelog_section` (port from xenon)
- [ ] `scripts/release/version_sync_check.py` — assert `VERSION` == `pyproject.toml [project].version`
- [ ] `scripts/release/cut.sh` — preflight (on master, clean, synced w/ origin, **CI green for HEAD SHA**, CHANGELOG `[Unreleased]` non-empty) → interactive bump → rewrite VERSION + pyproject + CHANGELOG → commit `release: vX.Y.Z` → annotated tag. Does NOT push.
- [ ] Wire `version_sync_check.py` into `ci.yml` as a gate

## Phase 2 — Docker image

- [ ] `docker/api.Dockerfile` — `python:3.13-slim` → install TA-Lib C lib (0.6.4 from source, cached layer) → `uv pip install -e ".[server,observability]"` → `CMD ["python","main.py","--service","api"]`
- [ ] `.dockerignore` (exclude .venv, tests fixtures, data, output, *.png, .git)
- [ ] Single `apex-api` image. signal-service deferred (not in prod) — add `--service` override later if needed.

## Phase 3 — docker-compose.yml (macmini deploy)

- [ ] `apex-api` service:
  - `image: ghcr.io/moremeds/apex-api:latest`
  - `ports: ["8322:8322"]`
  - `volumes: ["/Volumes/DATA_LAKE/livewire/data-lake/bronze:/data/livewire:ro"]`
  - env: `APEX_LIVEWIRE_ROOT=/data/livewire`, `APEX_PG_URL=postgresql://apex_app:***@host.docker.internal:5432/apex_signals`, `APEX_XENON_WS_URL=ws://host.docker.internal:8765`
  - `extra_hosts: ["host.docker.internal:host-gateway"]`
  - healthcheck on `/health`; `restart: unless-stopped`
  - **REQUIRED** `labels: ["com.centurylinklabs.watchtower.enable=true"]` — Watchtower runs `WATCHTOWER_LABEL_ENABLE=true` (verified), so it watches ONLY labelled containers. No label → no auto-deploy.
- [ ] PG reachability — VERIFIED: host PG listens `*:5432`; `pg_hba` accepts colima bridge `192.168.50.0/24` (scram-sha-256); `xenon-api` already connects via `host.docker.internal`. apex change: `127.0.0.1` → `host.docker.internal` (keep `apex_app` scram password).

## Phase 4 — release.yml (tag-triggered, mirrors xenon)

- [ ] `on: push: tags: ["v*"]`
- [ ] `verify` job — ubuntu + Postgres service + TA-Lib + `uv sync` + `pytest` (mirror ci.yml) + `version_sync_check.py`
- [ ] `publish` job — extract CHANGELOG section → `softprops/action-gh-release` (this ALSO fires the existing `release-holdout-gate.yml` on `release:published` — chains for free)
- [ ] `ghcr-push` job — `ubuntu-24.04-arm` (native arm64) → build `docker/api.Dockerfile` → push `ghcr.io/moremeds/apex-api:<version>` + `:latest` (latest only for non-prerelease tags)

## Phase 5 — Watchtower auto-deploy

- VERIFIED: `xenon-watchtower-1` runs `WATCHTOWER_LABEL_ENABLE=true`, `POLL_INTERVAL=60`, `CLEANUP=true`, ntfy notifications. It is **label-scoped** — does NOT auto-watch all containers.
- [ ] apex-api compose carries `com.centurylinklabs.watchtower.enable=true` (Phase 3) → on new `:latest` it pulls + recreates within ~60s, preserving the read-only lake mount + restart policy.
- [ ] (optional) point apex deploy notifications at its own ntfy topic.

## Phase 6 — cutover + verification

- [ ] Stop the bare `src.api.server` process on macmini
- [ ] First manual `docker compose up -d apex-api`; confirm `/health`
- [ ] `curl /bars?symbol=AAPL&timeframe=1d` → real data through the mounted lake (proves the whole concern is solved end-to-end)
- [ ] Confirm argon still reaches apex on :8322
- [ ] Add release runbook to CLAUDE.md (`scripts/release/cut.sh` → `git push origin master --follow-tags`)
- [ ] **Doc fix** — correct the "R2-backed livewire" error in 5 places (`CLAUDE.md:20,82,87`, `src/infrastructure/CLAUDE.md:9,19`): the livewire READ path is local-FS (`APEX_LIVEWIRE_ROOT`), via `read_parquet('{local path}')`. R2 is only apex's separate `make r2-backfill` pipeline, not the live read path.

## Open risks to watch

- colima restart = brief xenon prod blip (Phase 0) — needs a window
- external disk must mount before colima on reboot
- exFAT-over-virtiofs read validation (read-only, should be fine)
- host-PG reachability from container (Phase 3 verify)
- TA-Lib C-lib build adds image build time (cache the layer)
