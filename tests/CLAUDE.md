# tests/ — pytest suite

Root `CLAUDE.md` is authoritative for policy.

## Layout

```
tests/
├── unit/         # pure functions, no DB, no network (CI: hard gate)
├── integration/  # real services (PG, xenon WS, a real livewire tree)
├── carve/        # architectural guards — import graph, keepset isolation, stubs/
├── support/      # shared fixtures and helpers (importable as `tests.support.*`)
├── fixtures/     # static test data
└── partial/      # excluded from CI (long-running, external deps)
```

`tests/carve/` is where architecture is enforced, not documented: `test_import_graph.py` and `test_keepset_imports_isolated.py`. Tighten the domain→infrastructure rule there (see `src/domain/CLAUDE.md`).

Loose files at `tests/` root (`test_vix_alert.py`, `diagnose_live_greeks.py`) are legacy — new tests go inside the tree. `tests/conftest.py` at root is not legacy; it is load-bearing.

## pytest configuration (`pyproject.toml [tool.pytest.ini_options]`)

- `asyncio_mode = "auto"` — do **not** decorate async tests with `@pytest.mark.asyncio`
- `pythonpath = ["."]` — so `tests.support.*` imports work under bare `pytest`, not just `python -m pytest`
- **No custom markers are registered.** Do not invent `@pytest.mark.<name>`; it will warn and filter nothing.
- Coverage gate: `--cov-fail-under=40`, applied by default to any run that does not pass `--no-cov`

## Rules

- **`uv run pytest`** — never bare `pytest`
- **Read `tests/unit/signals/conftest.py`** before writing signal tests — reuse the existing fixtures
- Run targeted tests during dev (`-k "pattern"`, `--no-cov`), full suite before commit
- After generating >20 tests, run them immediately — don't batch failures

## CI

The unit job is a hard gate (`tests/unit/` with the coverage gate); the integration job runs `tests/integration/` with `--no-cov` and a set of `--ignore` paths. **Read the integration step in `.github/workflows/ci.yml` for the current ignore list** rather than duplicating it here — it drifts.

## Integration test notes

- Needs `APEX_PG_URL` for PG-backed tests and the livewire vars (`APEX_LIVEWIRE_ROOT`, `APEX_LIVEWIRE_SILVER_ROOT`, `APEX_LIVEWIRE_COVERAGE_DB`) for lake-backed ones. R2 is _not_ the livewire lake.
- Silver revisions: `test_silver_revision_e2e.py` end-to-end; build manifests with `tests/support/silver_manifest.py` rather than hand-writing `revisions/current.json`.
- Xenon e2e (`test_xenon_live_e2e.py`, `test_xenon_to_ws_e2e.py`) needs a running xenon.
- `test_strategy_parity.py` is the regression gate for the frozen strategy code.
