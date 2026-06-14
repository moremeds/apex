# Legacy docs (pre-pivot risk/backtest monolith)

These documents describe **APEX as it was before the pivot** — a risk-monitoring +
backtesting monolith (TUI dashboard, IB/Futu broker adapters, portfolio/position risk,
strategy backtesting). APEX is now a **streaming technical-analysis signal service**
(see the top-level [README](../../README.md)), so the guides below no longer describe how
the project works.

They are kept for historical reference only. The functionality they document is either
out of scope (backtesting — slated for removal in the Phase 6 strip-down) or already
removed/superseded. **Do not follow these for the current service.**

| File | Described (now-legacy) function |
|---|---|
| `QUICKSTART.md` | IB connection + TUI risk-monitor setup |
| `QUICK_START_GUIDE.md` | RiskEngine / market-data quality / Greeks / breach-detection skeleton |
| `USER_MANUAL.md` | Full manual: brokers, TUI dashboard, history loader, backtest runner, risk limits |
| `STRATEGY_GUIDE.md` | Backtest framework (VectorBT/Backtrader), strategy lifecycle, runners |
| `PERSISTENCE_LAYER.md` | Broker order history, position snapshots, warm-start persistence |
| `OBSERVABILITY_SETUP.md` | Prometheus/Grafana for the old `main.py` risk monitor |

## Where the current docs live

- **What APEX is / how to run it** → [`../../README.md`](../../README.md)
- **API reference for argon** (WS + REST) → [`../argon-apex-api.md`](../argon-apex-api.md)
- **Argon consumption guide** (real frames, examples) → [`../argon-signal-consumption.md`](../argon-signal-consumption.md)
- **livewire bronze data contract** → [`../livewire-apex-integration.md`](../livewire-apex-integration.md)
- **Adaptation roadmap** → [`../superpowers/specs/2026-06-14-apex-adaptation-design.md`](../superpowers/specs/2026-06-14-apex-adaptation-design.md)
- **Signal persistence (current):** the streaming service persists signals via
  `TASignalRepository` → the `ta_signals` / `indicator_values` / `confluence_scores`
  tables (migration `migrations/005_ta_signals.sql`); consumption is documented in
  `argon-signal-consumption.md`.
