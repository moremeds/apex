# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Master policy file. Subsystem-specific rules live in subdirectory `CLAUDE.md` files:

| Area | File |
|------|------|
| FastAPI server, routes, WS hub | `src/api/CLAUDE.md` |
| Domain layer: signals, regime, screeners, strategy, events | `src/domain/CLAUDE.md` |
| Adapters, persistence, stores | `src/infrastructure/CLAUDE.md` |
| Backtest engine, optimization | `src/backtest/CLAUDE.md` |
| Test suite layout and rules | `tests/CLAUDE.md` |
| Config YAML management | `config/CLAUDE.md` |

## Identity

**Apex** — signal computation engine and chart data API. Reads OHLCV from the livewire Parquet lake, runs the full indicator + signal pipeline, and exposes results via REST + WebSocket consumed by argon (Options Analytics Cockpit).

Not a broker terminal — no direct order placement. Live IB ticks arrive via xenon's WS feed (`APEX_XENON_WS_URL`). Historical bars come from the livewire bronze lake — a **local-filesystem** Parquet tree read on demand via DuckDB (`APEX_LIVEWIRE_ROOT`).

## Commands

```bash
# Install
uv pip install -e ".[dev,observability,api]"

# Run
make api-server        # REST + WS API (:8322) — primary service for argon
make signal-service    # Signal daemon (ticks → signals → PG)
make dev               # Both together

# Test
uv run pytest tests/unit/             # unit only
uv run pytest                         # all tests
uv run pytest -k "test_rule"          # pattern match

# Code quality (run before every commit)
make format            # black + isort
make lint              # black + isort + flake8
make type-check        # mypy src/ tests/
make quality           # all: lint + type-check + dead-code + complexity

# Screeners
make momentum          # Momentum: universe + OHLCV + screen
make pead              # PEAD: full pipeline

# Strategy verification (after any strategy change)
make strategy-verify   # unit tests → parity → mypy → format → compare

# Database
make db-init           # Create PG schema
make db-reset          # Drop + recreate PG schema
```

## Architecture

Hexagonal — five layers:

```
API             src/api/          FastAPI: /bars /indicators /confluence /ws/signals /regime /screener
CLI             src/runners/      momentum, pead, strategy_compare, optimize, validation
APPLICATION     src/application/  bootstrap, orchestrators, chart service, subscription manager
DOMAIN          src/domain/       signals, strategy, screeners, events, interfaces (no infra imports)
INFRASTRUCTURE  src/infrastructure/ adapters (livewire, xenon_ws, fmp, ib), persistence, stores
```

Signal pipeline: tick (xenon WS) → `BarAggregator` → `IndicatorEngine` → `RuleEngine` → PostgreSQL → argon via REST + WS.

## External Dependencies

| Dependency | Default | Env var |
|---|---|---|
| livewire bronze lake (local Parquet, read via DuckDB) | — | `APEX_LIVEWIRE_ROOT` |
| xenon WS live ticks | `ws://127.0.0.1:8765` | `APEX_XENON_WS_URL` |
| PostgreSQL | — | `APEX_PG_URL` |
| FMP API (screeners) | — | `FMP_API_KEY` or `config/secrets.yaml` |

## Data Source Priority

livewire bronze lake (local Parquet) → FMP (paid, daily deltas) → Yahoo (bulk backfill only, **never live**)

IB ticks arrive via xenon WS — apex never connects to IB directly.

## Mandatory Rules

1. **uv only** — `uv run pytest`, never bare `python`, `pip`, or activated venvs
2. **Never Yahoo Finance as a live source** — historical bulk backfill only
3. **No naked shorts** — defined-risk only
4. **Never commit without explicit user request** — draft first, wait
5. **Always open a PR before merging to master** — never `git push origin master` directly
6. **`/codex-review` before finalizing any substantial change**
7. **Module size budget** — target <500 lines; 1000+ → stop and propose a domain-seam split
8. **Fix every issue you spot** — no "pre-existing" dismissals
9. **Wire all features** — never leave code dead/unconnected; no accumulating dead code behind flags

## Python Standards

- Python 3.13, 4-space indent, 100-char line length
- Type hints required on all functions (mypy strict)
- Use `ib_async` (not `ib_insync`), TA-Lib for indicators
- `try/except` always logs the error
- Typed dataclasses for return values (not `Dict[str, Any]`)
- **Naming**: `get_*` (from cache), `fetch_*` (network/disk), `load_*` (deserialize from file/DB)

## Strategy Development SOP

Every strategy change must complete **all** steps:

1. `config/strategy/{name}.yaml` — add/update `params:`, push old to `history:`
2. `src/domain/strategy/signals/{name}.py` — signal generator; `warmup_bars` = longest lookback
3. `src/domain/strategy/playbook/{name}.py` — `@register_strategy("name")`
4. `src/backtest/optimization/strategy_objective.py` — add in `_suggest_params()`, ≤7 tunable
5. `tests/integration/test_strategy_parity.py` — smoke, warmup, entry/exit no-overlap
6. `make strategy-verify`
