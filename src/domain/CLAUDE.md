# src/domain/ — Domain layer

Root `CLAUDE.md` is authoritative for policy.

## Invariant

Domain never imports infrastructure. All external dependencies flow in via interfaces (`src/domain/interfaces/`). Dependency injection is via constructor.

## Signal Pipeline

```
Tick (xenon WS) → BarAggregator (per-timeframe, publishes BAR_CLOSE)
  → IndicatorEngine (ThreadPool, per-symbol RLocks, publishes INDICATOR_UPDATE)
  → RuleEngine (threshold_cross / state_change / range, cooldowns, publishes TRADING_SIGNAL)
  → ConfluenceCalculator (optional cross-indicator / multi-timeframe)
  → PostgreSQL (bars, signals, summary, score_history)
  → argon via REST + WS
```

Key files: `signal_engine.py` (top-level wiring), `indicator_engine.py` (40+ indicators), `rule_engine.py` (threshold/state/range rules), `confluence_calculator.py`.

Performance invariants:
- DataFrame created once per bar, shared across threads (40× memory reduction)
- Per-(symbol, timeframe) RLocks eliminate cross-symbol contention
- `detect_initial` flag: threshold rules fire on first evaluation after restart

## Regime Detector (`signals/indicators/regime/`)

4-regime classification that gates position sizing:

| Regime | Name | Trading implication |
|--------|------|---------------------|
| R0 | Healthy Uptrend | Full trading |
| R1 | Choppy/Extended | Reduced frequency |
| R2 | Risk-Off | No new positions |
| R3 | Rebound Window | Small defined-risk only |

Pipeline: Component states (Trend/Vol/Chop/Extension/IV) → Decision tree → Hysteresis → Composite score (0–100).

`MarketRegime` enum — full member names: `R0_HEALTHY_UPTREND`, `R1_CHOPPY_EXTENDED`, `R2_RISK_OFF`, `R3_REBOUND_WINDOW`. `.value` returns short form ("R0", "R1", ...).

## Screeners (`screeners/`)

- **Momentum** (`screeners/momentum/`) — 12-1 momentum / FIP; universe + OHLCV + screen
- **PEAD** (`screeners/pead/`) — earnings drift; earnings-driven entries

Both use the waterfall: FMP (paid) → Yahoo (batch fallback) — never Yahoo as primary.

## Strategy (`strategy/`)

| File | Role |
|------|------|
| `playbook/{name}.py` | `@register_strategy("name")` classes, one per strategy |
| `signals/{name}.py` | Event-driven signal generators |
| `param_loader.py` | `get_strategy_params()` — only way to read strategy YAML |
| `regime_gate.py` | Per-symbol regime state; `evaluate()` takes `bar_count` not `bar_idx` |
| `exit_manager.py` | 5-level priority exit; skips ATR trail when `atr <= 0` |
| `position_sizer.py` | `portfolio_value` in constructor; `max_position_pct` caps shares |
| `risk_gate.py` | Runtime risk checks |

Entry shift: signal generators always shift entry +1 bar. Same-bar conflict: exit wins.

## Priority Event Bus (`events/priority_event_bus.py`)

| Lane | Events | Behavior |
|------|--------|----------|
| **Fast** | Risk, Trading, Market Data | Priority queue, 500 events or 50ms yield |
| **Slow** | Snapshot, UI, Diagnostics | Debounced 100ms, coalesced by (type, symbol) |

`register_heavy_callback()` offloads to thread pool (max 4). Never register blocking callbacks on the fast lane.
