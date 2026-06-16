# tests/ — pytest suite

Root `CLAUDE.md` is authoritative for policy.

## Layout

```
tests/
├── unit/         # pure-function tests, no DB, no network (CI: hard gate)
├── integration/  # real services (PG, R2, xenon WS)
├── partial/      # excluded from CI (long-running, external deps)
└── support/      # shared fixtures and helpers
```

Standalone test files at `tests/` root (`test_vix_alert.py`, etc.) are legacy — new tests go inside the tree.

## Rules

- **`uv run pytest`** — never bare `pytest`
- **Coverage threshold: 40%** (`--cov-fail-under=40`); enforced in CI
- **Read `tests/unit/signals/conftest.py`** before writing signal tests — existing fixtures must be reused
- Run targeted tests during dev (`-k "pattern"`), full suite before commit
- After generating >20 tests, run immediately — don't batch failures

## CI exclusions

CI (`pytest tests/unit/ tests/integration/`) excludes:
- `tests/partial/`
- `tests/integration/test_futu_adapter.py`
- `tests/integration/test_multi_broker.py`

## Integration test notes

- Integration tests may require `APEX_PG_URL` (PG) and `R2_*` env vars (livewire lake)
- Xenon WS e2e tests (`test_xenon_live_e2e.py`, `test_xenon_to_ws_e2e.py`) require a running xenon instance — excluded from default CI run
- Strategy parity tests (`test_strategy_parity.py`) are the canonical regression gate for strategy changes
