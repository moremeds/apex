# Copilot / AI Agent Instructions

See [CLAUDE.md](../../CLAUDE.md) for comprehensive project guidance including:
- Build/test/run commands
- Architecture overview (hexagonal, event-driven)
- Coding standards and guardrails
- Strategy development SOP

## Quick Reference

- **Language:** Python 3.13, strict mypy, black + isort formatting
- **Build:** `make install` (uses uv)
- **Test:** `make test` (unit), `make test-all` (all), `pytest -k "pattern"` (targeted)
- **Quality:** `make format && make quality` before commits
- **Run:** `make run` (dev TUI), `make run-demo` (offline)
- **Key dirs:** `src/` (source), `config/` (YAML configs), `tests/` (pytest)
- **Entry point:** `main.py`
