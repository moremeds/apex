# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Master policy file. Subsystem-specific rules live in subdirectory `CLAUDE.md` files:

| Area                                            | File                           |
| ----------------------------------------------- | ------------------------------ |
| FastAPI server, routes, WS hub                  | `src/api/CLAUDE.md`            |
| Domain layer: signals, regime, strategy, events | `src/domain/CLAUDE.md`         |
| Adapters, persistence, stores                   | `src/infrastructure/CLAUDE.md` |
| Backtest engine, optimization — **FROZEN**      | `src/backtest/CLAUDE.md`       |
| Test suite layout and rules                     | `tests/CLAUDE.md`              |
| Config YAML management                          | `config/CLAUDE.md`             |

## Identity

**Apex** — a streaming technical-analysis signal service and livewire data-lake API. It reads bars from the livewire Parquet lake and live ticks from xenon, computes TA-Lib indicators + rule-engine signals + market regime, and serves them over REST + WebSocket to argon (stateless UI). Not a broker terminal — no order placement, and apex never connects to IB directly.

Apex was pivoted out of a risk-monitoring + backtesting monolith. The streaming path and the chart/data-lake read surface are the live product. The legacy backtest, strategy-playbook, screener and R2 code is still on disk pending the Phase 6 strip-down — treat it as frozen (rule 10).

## Commands

```bash
# Install
uv pip install -e ".[dev,observability,api,cloudflare]"

# Run — scripts/serve.sh loads .env, runs preflight, then execs the server
scripts/serve.sh
uv run --env-file .env python -m src.api.server   # equivalent, env loaded by you
make api-server        # REST + WS API on :8322
make mcp-server        # read-only lake MCP on :8333 (needs APEX_MCP_API_KEY)
make dev               # api + signal service via main.py --service all

# Test
make test                                   # pytest tests/unit/ -v
uv run pytest tests/unit/ --no-cov          # faster, no coverage gate
uv run pytest -k "test_rule"                # pattern match
make validate-fast                          # PR gate (validation_runner, 10 symbols)
make coverage                               # full suite + HTML coverage

# Code quality (run before every commit)
make format            # black + isort
make lint              # black + isort + flake8
make type-check        # mypy src/ tests/
make quality           # lint + type-check + dead-code + complexity

# Adjusted-mode preflight — read-only raw-vs-adjusted Livewire canary. No make target.
uv run python scripts/check_silver_canary.py

# Release — bumps VERSION + pyproject in lockstep; never edit either by hand
scripts/release/cut.sh

# Database
make db-init           # Create PG schema
make db-reset          # Drop + recreate PG schema
```

Legacy make targets (`momentum`, `pead`, `strategy-compare`, `strategy-verify`, `r2-*`, `quantitative-moment*`, `jobs-*`) belong to the frozen subsystems — see rule 10 before running or changing them.

## Architecture

```
src/api/            FastAPI app factory, routes, WS hub, JobManager, payload builders
src/mcp_server/     read-only lake MCP (20 tools over src/application/lake); separate process, no PG/streaming
src/application/    bootstrap, orchestrator, chart service, subscriptions, revision watcher
src/domain/         signals, indicators, regime, events, strategy, interfaces
src/infrastructure/ adapters (livewire, xenon, fmp, ib, futu, yahoo, r2, earnings), persistence, stores, observability
src/services/       long-running services — signal_service.py is what `make signal-service` runs
src/runners/        CLI runners (momentum, pead, strategy_compare, optimize, validation, trading)
src/verification/   signal + regime verifiers; run as their own CI job
src/models/         shared dataclasses (position, order, account, risk)
src/utils/          helpers
```

The repo-root `services/` directory is **not** `src/services/` — it is a separate tree (`compute/ market_data/ shared/ web/`); do not confuse them.

Signal pipeline: tick (xenon WS) → `BarAggregator` → `IndicatorEngine` → `RuleEngine` → PostgreSQL → argon via REST + WS.

### The livewire lake

Apex reads two lakes. **Bronze** (`APEX_LIVEWIRE_ROOT`) is livewire's raw per-ticker Hive tree, `asset_class=<class>/symbol=<encode_symbol(SYM)>/<tf>.parquet`, covering six asset classes with differing timeframe ladders. **Silver** (`APEX_LIVEWIRE_SILVER_ROOT`) is equity-only, split/dividend adjusted, and published atomically as numbered revisions — apex validates `revisions/current.json` and SHA-256-verifies every referenced artifact before accepting one. `APEX_LIVEWIRE_PRICE_MODE` selects `raw` (default) or `adjusted`; a long-running service polls for new revisions and reseeds only the affected subscriptions. Reads go through per-request in-memory DuckDB — apex only reads, livewire writes. Details: `src/infrastructure/CLAUDE.md`.

## Environment

Every data source is env-gated: apex boots regardless, and each unset source makes its endpoints return `503`. It degrades, never crashes.

| Variable                              | Default               | Enables                                                  |
| ------------------------------------- | --------------------- | -------------------------------------------------------- |
| `APEX_LIVEWIRE_ROOT`                  | unset                 | bars/indicators reads and the streaming warmup seed      |
| `APEX_LIVEWIRE_SILVER_ROOT`           | unset                 | Silver revision watcher, adjusted daily/factor artifacts |
| `APEX_LIVEWIRE_PRICE_MODE`            | `raw`                 | `raw` Bronze or `adjusted` Silver/factor-joined Bronze   |
| `APEX_LIVEWIRE_REVISION_POLL_SECONDS` | `30`                  | Silver revision poll interval                            |
| `APEX_LIVEWIRE_COVERAGE_DB`           | unset                 | the coverage catalog behind `/v1/instruments`, `/v1/lake/coverage` |
| `APEX_LIVEWIRE_LAKE_ROOT`             | unset                 | index membership, security master — `/v1/membership/*`, `/v1/equity/{symbol}/delisting`, `/v1/security/{symbol}` |
| `APEX_LIVEWIRE_DELISTED_ROOT`         | unset                 | `listing=delisted`/`any` bars from bronze-delisted (raw) |
| `APEX_LIVEWIRE_REPAIRS_ROOT`          | unset                 | gap-engine repair-report evidence on `/v1/{asset_class}/{symbol}/gaps`; unset leaves `repairs.state=not_configured` and gaps still work |
| `APEX_LAKE_QUERY_TIMEOUT_SECONDS`     | `30`                  | per-query deadline on every parquet read (`LakeDb`); expiry interrupts only that query and answers `504 query_timeout` (bulk bars: that symbol goes to `missing`) |
| `APEX_PG_URL`                         | unset                 | signal backfill, confluence, signal persistence          |
| `APEX_XENON_WS_URL`                   | `ws://127.0.0.1:8765` | live ticks → live WS signal frames                       |
| `APEX_TIMEFRAMES`                     | `1d`                  | timeframes the streaming pipeline subscribes/warms       |
| `APEX_API_PORT`                       | `8322`                | listen port                                              |
| `APEX_MCP_API_KEY`                    | unset                 | **required** by the MCP server (Bearer; it refuses to boot without one) |
| `APEX_MCP_HOST` / `APEX_MCP_PORT`     | `127.0.0.1` / `8333`  | MCP listen address                                       |
| `APEX_MCP_ALLOWED_HOSTS`              | loopback:port         | Host header values MCP clients may send (DNS-rebinding guard) |

FMP (`FMP_API_KEY` or `config/secrets.yaml`) and R2 (`R2_*` in `config/secrets.yaml`) serve the frozen screener and backfill pipelines only, never the live read path.

## Mandatory Rules

1. **uv only** — `uv run pytest`, never bare `python`, `pip`, or activated venvs
2. **Never Yahoo Finance as a live source** — bulk backfill only
3. **No naked shorts** — defined-risk only
4. **Never commit without explicit user request** — draft first, wait
5. **Always open a PR before merging to master** — never `git push origin master` directly
6. **`/review-cycle` before finalizing any substantial change** — it runs `/tribunal-review` as its cross-model pass. (Supersedes the old `/codex-review`.)
7. **Module size budget** — target <500 lines; split on a responsibility seam, not a layer. `src/application/lake/guards.py` is the precedent: it carries "is this request coherent, and which artifact would it read?" out of `chart.py` (originally split out as `src/api/routes/_chart_guards.py`, then moved into `src/application/lake/` so REST and MCP share it), leaving routes and response assembly behind.
8. **Fix every issue you spot** — no "pre-existing" dismissals
9. **Wire all features** — never leave code dead/unconnected; no accumulating dead code behind flags
10. **Backtest is frozen** — do not modify `src/backtest/` or `src/domain/backtest/` unless explicitly asked; they are frozen for the Phase 6 strip-down
11. **`VERSION` == `pyproject.toml [project].version`** — enforced by the CI _Version Sync_ job (`scripts/release/version_sync_check.py`). Bump only via `scripts/release/cut.sh`; never hand-edit either file.
12. **Adjusted mode never silently falls back to raw** — missing or incomplete Silver raises `AdjustedDataUnavailable`. Do not add a raw fallback.

CI installs `black`/`isort`/`flake8`/`bandit` **unpinned** (`.github/workflows/ci.yml`), so a locally-clean `make lint` can still fail CI after an upstream release. Re-check against CI output, not local exit codes.

## Python Standards

- Python 3.13, 4-space indent, 100-char line length
- Type hints required on all functions (mypy strict)
- Use `ib_async` (not `ib_insync`), TA-Lib for indicators
- `try/except` always logs the error
- Typed dataclasses for return values (not `Dict[str, Any]`)
- **Naming**: `get_*` (from cache), `fetch_*` (network/disk), `load_*` (deserialize from file/DB)
