# src/backtest/ — Backtest engine and optimization

Root `CLAUDE.md` is authoritative for policy.

## Two execution paths

| Path | Entry | When to use |
|------|-------|-------------|
| **ApexEngine** (`core/`) | `runner.py` + `--strategy` flag | Event-driven; same strategy code as live trading. Use for strategy verification and parity tests |
| **BehavioralRunner** (`behavioral_runner.py`) | CLI args | Walk-forward analysis, ablation, behavioral experiments |

The same `Clock` abstraction lets strategy code run identically in live and backtest — never branch on "am I in backtest?" in strategy code.

## Optimization (`optimization/`)

- `strategy_objective.py` — Optuna objective; add new strategies in `_suggest_params()`. Keep ≤ 7 optimizable params per strategy; freeze non-tunable ones.
- `bayesian.py` — Optuna sampler wiring (TPE with warm-start)
- `stress_validator.py` — `_calc_max_drawdown()` returns **negative** values (drawdown as loss)
- `grid.py` — grid search for coarse sweeps before Optuna

**Nested CV rule:** Optuna sees inner CV only. Outer test fold is held out for evaluation — never touch it during tuning.

## Analysis (`analysis/`)

Post-run analysis: attribution, trade-level stats, drawdown, Sharpe. These are pure functions over `RunResult` objects — no DB access.

## Spec files (`config/`)

`config/backtest/` holds Optuna search spaces and spec YAMLs. NOT strategy param defaults — those live in `config/strategy/{name}.yaml`.

## Runner entry

```bash
uv run python -m src.backtest.runner --strategy trend_pulse --symbols AAPL --start 2024-01-01 --end 2024-06-30
uv run python -m src.backtest.runner --list-strategies
uv run python -m src.backtest.runner --spec config/backtest/playbook/trend_pulse_validate.yaml
```

## Constructor kwarg filtering

`runner.py` uses `inspect.signature()` to filter YAML params before passing to `BacktestEngine` — never pass raw YAML dicts directly to constructors.
