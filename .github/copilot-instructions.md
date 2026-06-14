# Copilot / AI Agent Instructions

APEX is a **streaming technical-analysis signal service**: it reads bars from livewire
(Parquet/DuckDB) + live ticks from xenon (WebSocket), computes TA-Lib indicators +
rule-engine signals + regime, and serves signal payloads over WebSocket/REST to argon.
See [README.md](../../README.md) for the overview and [`docs/`](../../docs/) for the API
and data contracts. (Legacy risk-monitor/backtest docs live in `docs/legacy/`; backtest is
out of scope and slated for the Phase 6 strip-down — do not modify `src/backtest/` or
`src/domain/backtest/`.)

## Quick Reference

- **Language:** Python 3.13, mypy, black + isort (line length 100)
- **Build:** `make install` (uses uv)
- **Test:** `uv run python -m pytest tests/unit --no-cov` (targeted: add `-k "pattern"`)
- **Type check:** `uv run mypy src/ --ignore-missing-imports`
- **Quality:** `make format` before commits
- **Run the service:** `scripts/serve.sh` (loads `.env`, preflight, then uvicorn on :8322),
  or `uv run --env-file .env python -m src.api.server`
- **Key dirs:** `src/` (source), `config/` (YAML configs), `migrations/` (PG schema), `tests/` (pytest)
- **Entry point:** `src/api/server.py` (`python -m src.api.server`)
