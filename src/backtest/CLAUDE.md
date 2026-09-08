# src/backtest/ — FROZEN

> **Do not modify `src/backtest/` or `src/domain/backtest/` unless explicitly asked.** This is pre-pivot code kept on disk pending the Phase 6 strip-down (root rule 10). Read it, cite it, run it — but a change here needs an explicit request, and it is not part of the streaming service apex now ships.

Root `CLAUDE.md` is authoritative for policy.

## Entry point

`src/backtest/runner.py` is a **facade** — it re-exports `cli.main` plus the config loaders, runners and engine workers. The real CLI is `cli/parser.py` (arguments) + `cli/commands.py` (dispatch).

```bash
uv run python -m src.backtest.runner --list-strategies
uv run python -m src.backtest.runner --strategy trend_pulse --symbols AAPL --start 2024-01-01 --end 2024-06-30
uv run python -m src.backtest.runner --spec config/backtest/playbook/trend_pulse_validate.yaml
```

`--behavioral` / `--behavioral-cases` dispatch into `behavioral_runner.run_behavioral_validation` (DualMACD gate validation, walk-forward/ablation). `behavioral_runner.py` is a module of functions, not a `BehavioralRunner` class, and has no `__main__` — reach it through the CLI flags.

## Execution paths (`execution/`)

`execution/engines/` holds the engines: `apex_engine.py` (`ApexEngine`, event-driven, same strategy code as live), `vectorbt_engine.py` (vectorized), `backtest_engine.py` (`BacktestEngine`), behind the `interface.py` protocol with `apex_worker.py` / `vectorbt_worker.py` as the process-pool entry points. Drivers: `single_backtest.py`, `systematic_experiment.py`, `backtrader_runner.py`, `parallel.py`, `order_matching.py`. `core/` is the result plumbing (`run`, `trial`, `experiment`, `manifest`, `hashing`), not an engine.

The shared `Clock` abstraction lets strategy code run identically live and in backtest — never branch on "am I in backtest?" in strategy code.

## Optimization (`optimization/`)

- `strategy_objective.py` — Optuna objective; new strategies go in `_suggest_params()`. Keep ≤7 tunable params; freeze the rest. It filters unknown params via `inspect.signature(self._strategy_class.__init__)` so Optuna cannot "optimize" a kwarg the strategy ignores — do not pass raw YAML dicts to constructors.
- `bayesian.py` — Optuna sampler wiring (TPE with warm-start)
- `stress_validator.py` — `_calc_max_drawdown()` returns **negative** values
- `grid.py` — coarse sweeps before Optuna

**Nested CV rule:** Optuna sees the inner CV only. The outer test fold is held out — never touch it during tuning.

## Analysis (`analysis/`)

Attribution, trade stats, drawdown, Sharpe — pure functions over result objects, no DB access.

## Spec files

`config/backtest/` holds Optuna search spaces and experiment specs (`playbook/`, `regime/`, `dual_macd_behavioral.yaml`). **Not** strategy param defaults — those live in `config/strategy/{name}.yaml`.

## Strategy SOP (condensed, frozen)

If a strategy change is explicitly requested, all six steps apply:

1. `config/strategy/{name}.yaml` — update `params:`, push old values to `history:`
2. `src/domain/strategy/signals/{name}.py` — signal generator; `warmup_bars` = longest lookback
3. `src/domain/strategy/playbook/{name}.py` — `@register_strategy("name")`
4. `src/backtest/optimization/strategy_objective.py` — add in `_suggest_params()`, ≤7 tunable
5. `tests/integration/test_strategy_parity.py` — smoke, warmup, entry/exit no-overlap
6. `make strategy-verify`

Registration comes from the `@register_strategy` decorator, **not** from the YAML file. `src/runners/strategy_compare_runner.py` builds `STRATEGY_REGISTRY` from `list_strategies()` (the decorator registry) and only then reads YAML metadata.

**Known defect:** `pead` has `config/strategy/pead.yaml` and `src/domain/strategy/signals/pead.py` but **no `playbook/pead.py` and no `@register_strategy`**, so it is invisible to `--list-strategies`, `/strategy/list` and `strategy_compare`. Five strategies are actually registered: `buy_and_hold`, `regime_flex`, `rsi_mean_reversion`, `sector_pulse`, `trend_pulse`.
