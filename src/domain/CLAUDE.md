# src/domain/ — Domain layer

Root `CLAUDE.md` is authoritative for policy.

## Invariant (aspiration, not yet enforced)

Domain should not import infrastructure; dependencies flow in via `src/domain/interfaces/` and constructor injection. Three runtime violations exist today — do not add a fourth, and prefer removing one when you touch the file:

- `signals/confluence_calculator.py` → `...infrastructure.observability`
- `services/data_validator.py` → `...infrastructure.stores.parquet_historical_store`
- `services/bar_replay_service.py` → `...infrastructure.stores.parquet_historical_store`

Several more (`indicator_engine.py`, `signal_engine.py`, `rule_engine.py`, `signals/data/bar_aggregator.py`) are `TYPE_CHECKING`-guarded and therefore fine. `tests/carve/test_import_graph.py` is the enforcement point — tighten it there, not by convention.

## Signal Pipeline

```
Tick (xenon WS) → BarAggregator (per-timeframe, publishes BAR_CLOSE)
  → IndicatorEngine (ThreadPool, per-symbol RLocks, publishes INDICATOR_UPDATE)
  → RuleEngine (threshold_cross / state_change / range, cooldowns, publishes TRADING_SIGNAL)
  → ConfluenceCalculator (optional cross-indicator / multi-timeframe)
  → PostgreSQL → argon via REST + WS
```

Top-level wiring: `signals/signal_engine.py`. Note there are **two** `rule_engine.py` files — `signals/rule_engine.py` is the signal one; `services/risk/rule_engine.py` belongs to the legacy risk path.

Performance invariants:

- DataFrame created once per bar, shared across threads (40× memory reduction)
- Per-(symbol, timeframe) RLocks eliminate cross-symbol contention
- `detect_initial` flag: threshold rules fire on first evaluation after restart

## Indicators (`signals/indicators/`)

`indicator_engine.py` holds no indicators — it only schedules them. ~50 `Indicator` subclasses live in the category packages (`trend/`, `momentum/`, `volatility/`, `volume/`, `pattern/`) and are **auto-discovered** by `indicators/registry.py` (`get_indicator_registry()`). Adding an indicator means adding a class in the right category package; there is no list to edit.

## Regime Detector (`signals/indicators/regime/`)

| Regime | Name            | Trading implication     |
| ------ | --------------- | ----------------------- |
| R0     | Healthy Uptrend | Full trading            |
| R1     | Choppy/Extended | Reduced frequency       |
| R2     | Risk-Off        | No new positions        |
| R3     | Rebound Window  | Small defined-risk only |

Pipeline: component states (Trend/Vol/Chop/Extension/IV) → decision tree → hysteresis → composite score (0–100).

`MarketRegime` members are `R0_HEALTHY_UPTREND`, `R1_CHOPPY_EXTENDED`, `R2_RISK_OFF`, `R3_REBOUND_WINDOW`; `.value` returns the short form (`"R0"`, …).

## Priority Event Bus (`events/priority_event_bus.py`)

| Lane     | Events                     | Behavior                                     |
| -------- | -------------------------- | -------------------------------------------- |
| **Fast** | Risk, Trading, Market Data | Priority queue, 500 events or 50ms yield     |
| **Slow** | Snapshot, UI, Diagnostics  | Debounced 100ms, coalesced by (type, symbol) |

`register_heavy_callback()` offloads to a thread pool (max 4). Never register blocking callbacks on the fast lane.

## Legacy subtrees

`strategy/`, `screeners/`, `backtest/`, `services/risk/`, `reality/` belong to the pre-pivot monolith. `domain/backtest/` is frozen (root rule 10). The rest is dormant but still imported; four non-obvious behaviours worth knowing before touching `strategy/`:

| File                         | Gotcha                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `strategy/param_loader.py`   | `get_strategy_params()` is the **only** sanctioned way to read strategy YAML — never hardcode defaults |
| `strategy/regime_gate.py`    | `evaluate()` takes `bar_count`, not `bar_idx`                                                          |
| `strategy/exit_manager.py`   | 5-level priority; skips the ATR trail when `atr <= 0`                                                  |
| `strategy/position_sizer.py` | `portfolio_value` goes in the constructor; `max_position_pct` caps shares                              |

Screener data waterfall (FMP paid → Yahoo batch fallback) applies to `screeners/` only; the live service reads livewire, never FMP or Yahoo.
