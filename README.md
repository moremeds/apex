# APEX - Live Risk Management & Backtesting System

**Production-grade portfolio risk monitoring and strategy backtesting for options and derivatives trading**

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Quick Reference (Make Commands)

```bash
# Setup
make install              # Install all dependencies

# Run TUI Dashboard
make run                  # Start TUI (dev mode, verbose)
make run-prod             # Start TUI (production mode)
make run-demo             # Start TUI (demo/offline, no broker needed)
make run-headless         # Run without TUI

# Quality Checks
make lint                 # Check formatting (black, isort, flake8)
make format               # Auto-fix formatting
make type-check           # Run mypy
make quality              # All checks (lint + type-check + dead-code + complexity)

# Testing
make test                 # Unit tests
make test-all             # All tests
make coverage             # Tests with HTML coverage

# Signal Pipeline
make signals-test         # Quick test (20 symbols) + HTTP server
make signals              # Full pipeline (caps, retrain, report)
make signals-deploy       # Deploy to GitHub Pages

# Backtesting
make behavioral           # Behavioral gate test + serve
make behavioral-full      # Optuna optimization + walk-forward + serve
make behavioral-cases     # Predefined case studies

# TrendPulse Validation
make tp-validate          # Full 3-stage validation (36 symbols)
make tp-holdout           # Holdout only (faster)
make tp-optimize          # Phase 1 Optuna optimization
make tp-universe          # Full universe backtest + HTML report

# Validation
make validate-fast        # PR gate validation
make validate             # Full validation suite

# Help
make help                 # Show all available commands
```

---

## Overview

APEX is a comprehensive risk management and backtesting platform designed for active traders managing options and derivatives portfolios. Built with a hexagonal architecture and event-driven patterns, it provides:

### Core Features

| Feature | Description |
|---------|-------------|
| **Real-time Risk Monitoring** | P&L (unrealized/daily), Greeks aggregation, concentration limits |
| **Multi-Broker Support** | Interactive Brokers, Futu OpenD with auto-reconnect |
| **Triple-Engine Backtesting** | ApexEngine (event-driven) + VectorBT (vectorized, 100x faster) + Backtrader |
| **47+ Technical Indicators** | Full TA-Lib integration with custom rule engine across 5 categories |
| **TrendPulse Indicator** | Hybrid multi-factor trend scoring with validation pipeline |
| **Behavioral Gate Validation** | Regime-aware strategy validation with case studies |
| **Terminal Dashboard** | Textual TUI with 6 views, keyboard navigation |
| **Signal Report Pipeline** | Automated HTML reports with heatmaps, deployed to GitHub Pages |
| **Email Summaries** | Automated email summary generation with market highlights |
| **Event-Driven Architecture** | Priority event bus with fast/slow lanes |
| **Persistence Layer** | DuckDB/PostgreSQL with TimescaleDB support |
| **Observability** | OpenTelemetry, Prometheus metrics, Grafana dashboards |

### Technical Highlights

- **~148,000 lines** of production Python code across **537 modules**
- **Async-first** design using `asyncio` and `ib_async`
- **Thread-safe** RCU (Read-Copy-Update) stores for lock-free reads
- **Hexagonal architecture** with clear domain/infrastructure separation
- **118 test files** across unit, integration, and partial test suites
- **8 registered strategies** with live/backtest parity via clock abstraction
- **4-Regime market classification** (R0-R3) with ML turning point detection

---

## Architecture

```
+-------------------------------------------------------------------+
|                      PRESENTATION LAYER                            |
|  Terminal Dashboard (Textual)  |  CLI Runners  |  Metrics Endpoint |
+-------------------------------------------------------------------+
                              |
+-------------------------------------------------------------------+
|                     APPLICATION LAYER                              |
|  Orchestrator  |  Coordinators  |  Services  |  PriorityEventBus  |
+-------------------------------------------------------------------+
                              |
+-------------------------------------------------------------------+
|                       DOMAIN LAYER                                 |
|  RiskEngine  |  IndicatorEngine  |  Strategy Framework  |  MDQC   |
|  RegimeDetector  |  RuleEngine  |  TrendPulse  |  DualMACD        |
+-------------------------------------------------------------------+
                              |
+-------------------------------------------------------------------+
|                   INFRASTRUCTURE LAYER                             |
|  IB/Futu/Yahoo Adapters  |  Backtest Engines  |  Stores  |  Reports|
+-------------------------------------------------------------------+
```

### Architecture Diagrams

Auto-generated diagrams are available in [`docs/diagrams/`](docs/diagrams/README.md):

| Type | Tool | Description |
|------|------|-------------|
| **Class Diagrams** | pyreverse | UML-style class relationships and inheritance |
| **Dependency Graphs** | pydeps | Module import relationships |
| **Call Flow Diagrams** | code2flow | Function call graphs and execution flow |

Generate locally with `make diagrams` (requires Graphviz).

### Directory Structure

```
apex/
├── config/                 # Configuration files (YAML)
│   ├── base.yaml           # Main configuration (brokers, ports, risk limits)
│   ├── universe.yaml       # Primary symbol universe (sectors, subsets)
│   ├── risk_config.yaml    # Risk limits and signal thresholds
│   ├── demo.yaml           # Demo mode configuration
│   ├── regime_weights.yaml # Regime detector factor weights
│   ├── signals/            # Signal rule definitions
│   ├── backtest/           # Backtest YAML specs (15+ examples)
│   └── validation/         # Validation universe and optimized params
├── data/
│   ├── historical/         # Cached bar data (Parquet)
│   ├── positions/          # Manual position files
│   └── logs/               # Structured log output
├── docs/                   # Documentation
├── migrations/             # Database migrations
├── scripts/                # CLI utilities (16 scripts)
├── src/
│   ├── application/        # Orchestrator, coordinators, services (21 files)
│   ├── backtest/           # Backtest engines, feeds, optimization (84 files)
│   ├── domain/             # Business logic (231 files)
│   │   ├── events/         # PriorityEventBus (dual-lane)
│   │   ├── services/       # Risk, MDQC, Regime, Correlation, Sizing
│   │   ├── signals/        # 47+ indicators, rule engine, bar aggregation
│   │   ├── strategy/       # Strategy framework (8 strategies)
│   │   └── interfaces/     # Port definitions (DI)
│   ├── infrastructure/     # External integrations (117 files)
│   │   ├── adapters/       # IB, Futu, Yahoo
│   │   ├── stores/         # RCU stores (6 store types)
│   │   ├── persistence/    # Database repositories
│   │   └── reporting/      # HTML reports, heatmaps, email, packages
│   ├── models/             # Data models (9 files)
│   ├── runners/            # CLI runners (5 files)
│   ├── services/           # Application services (14 files)
│   ├── tui/                # Terminal dashboard (42 files)
│   ├── utils/              # Utility functions (8 files)
│   └── verification/       # Contract verification (5 files)
├── tests/                  # Test suites (118 files)
│   ├── unit/               # 81 unit test files across 12 subdirectories
│   ├── integration/        # 7 integration tests
│   └── partial/            # 12 partial/experimental tests
├── main.py                 # Primary entry point
└── pyproject.toml          # Package config
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.13+ | Required (uses latest features) |
| TA-Lib | 0.6.8+ | Required for indicators |
| IBKR TWS/Gateway | Latest | Required for live market data |
| PostgreSQL | 14+ | Optional, for persistence |
| TimescaleDB | 2.x | Optional, for time-series |
| Futu OpenD | Latest | Optional, for Futu positions |

---

## Installation

### Using Make (Recommended)

```bash
# One command to install everything
make install
source .venv/bin/activate
```

### Using uv (Manual)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
cd apex
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with all features
uv pip install -e ".[dev,observability]"
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Configure Brokers

Edit `config/base.yaml`:

```yaml
brokers:
  ibkr:
    host: 127.0.0.1
    port: 4001        # 7497=TWS Paper, 4001=Gateway Paper
    client_id: 1
```

### 2. Start IB Gateway/TWS

Ensure API connections are enabled in settings.

### 3. Run APEX

```bash
# Development mode with TUI
python main.py --env dev

# Production mode
python main.py --env prod

# Demo mode (offline, sample data)
python main.py --env demo

# Headless mode (no TUI)
python main.py --env dev --no-dashboard
```

### 4. Terminal Dashboard

The TUI displays real-time risk metrics across 6 views:

```
+-------------------------------------------------------------------+
| APEX Risk Monitor | dev | 2025-03-15 10:30:45 HKT | IB:* FU:o    |
+-------------------------------------------------------------------+
| Portfolio Summary                                                  |
|   NAV: $1,234,567    Unrealized P&L: +$12,345   Daily: +$5,678   |
|   Delta: 25,000      Gamma: 1,234    Vega: 8,765  Theta: -567    |
+-------------------------------------------------------------------+
| Risk Signals                                                       |
|   ! SOFT: Portfolio delta at 82% of limit                         |
|   * HARD: TSLA notional exceeds limit                             |
+-------------------------------------------------------------------+
```

**TUI Views:**

| Key | View | Description |
|-----|------|-------------|
| `1` | Summary | Portfolio P&L, Greeks aggregation, risk summary |
| `2` | Positions | Position details, P&L breakdown by symbol |
| `3` | Signals | Real-time trading signals, signal history |
| `4` | Signal Introspection | Live signal pipeline visibility, rule evaluation |
| `5` | Data | Market data health, indicator values |
| `6` | Lab | Experimental features, diagnostic tools |

**Keyboard:** `1-6` switch views, `q` quit, `Ctrl+C` graceful shutdown

---

## CLI Reference

### Main Entry Point (`main.py`)

```bash
# Monitor mode (default)
python main.py --env dev
python main.py --env prod --no-dashboard
python main.py --env dev --metrics-port 9090

# Backtest mode
python main.py --mode backtest --strategy ma_cross --symbols AAPL \
    --start 2024-01-01 --end 2024-06-30
python main.py --mode backtest --spec config/backtest/ma_cross_example.yaml

# Trading mode (live execution)
python main.py --mode trading --dry-run  # Paper trading

# Common options
--config <path>        # Custom config file
--verbose, -v          # DEBUG level logging
--log-level [DEBUG|INFO|WARNING|ERROR]
```

### Backtest Runner

```bash
# Single backtest with ApexEngine (full simulation)
python -m src.backtest.runner --strategy ma_cross --symbols AAPL \
    --start 2024-01-01 --end 2024-06-30

# VectorBT engine (100x faster for parameter sweeps)
python -m src.backtest.runner --strategy ma_cross --symbols AAPL \
    --start 2024-01-01 --end 2024-06-30 --engine vectorbt

# Backtrader engine
python -m src.backtest.runner --strategy ma_cross --symbols AAPL \
    --start 2024-01-01 --end 2024-06-30 --engine backtrader

# Systematic experiment with YAML spec
python -m src.backtest.runner --spec config/backtest/examples/ta_metrics.yaml

# Behavioral gate validation
python -m src.backtest.runner --behavioral \
    --start 2018-01-01 --end 2025-12-31

# Behavioral predefined case studies
python -m src.backtest.runner --behavioral-cases

# List all registered strategies
python -m src.backtest.runner --list-strategies

# Custom parameters
python -m src.backtest.runner --strategy ma_cross --symbols AAPL MSFT \
    --start 2024-01-01 --end 2024-06-30 --params fast_period=10 slow_period=30

# Offline mode (fail if data gaps)
python -m src.backtest.runner --strategy ma_cross --symbols AAPL \
    --start 2024-01-01 --end 2024-06-30 --coverage-mode check
```

### Signal Runner

```bash
# Makefile shortcuts (preferred)
make signals-test         # Quick test (20 symbols) + HTTP server
make signals              # Full pipeline (update caps, retrain, report)
make signals-deploy       # Deploy to GitHub Pages
make signals-deploy-quick # Deploy without retraining
make signals-serve        # Serve existing report at localhost:8080
make signals-push         # Push existing out/signals to gh-pages

# Direct runner commands
python -m src.runners.signal_runner --live --symbols AAPL TSLA QQQ
python -m src.runners.signal_runner --live --symbols AAPL --timeframes 1m 5m 1h 1d
python -m src.runners.signal_runner --live --symbols AAPL --html-output report.html
python -m src.runners.signal_runner --backfill --symbols AAPL --days 365
python -m src.runners.signal_runner --live --symbols AAPL --with-persistence
python -m src.runners.signal_runner --live --universe config/universe.yaml
python -m src.runners.signal_runner --retrain-models --universe config/universe.yaml
python -m src.runners.signal_runner --update-market-caps --universe config/universe.yaml
```

### History Loader

```bash
# Load broker history
python scripts/history_loader.py --broker ib --days 30
python scripts/history_loader.py --broker futu --days 30 --market US
python scripts/history_loader.py --broker all --dry-run
python scripts/history_loader.py --broker ib --from-date 2024-01-01 --to-date 2024-06-30
```

### TrendPulse Validation

```bash
# Makefile shortcuts
make tp-validate          # Full 3-stage validation (36 symbols)
make tp-holdout           # Holdout only (faster)
make tp-optimize          # Phase 1 Optuna optimization
make tp-universe          # Full universe backtest + HTML report
make tp-universe-quick    # Quick test (12 symbols)

# Direct runner commands
python scripts/trend_pulse_validate.py
python scripts/trend_pulse_validate.py --skip-full
python scripts/trend_pulse_universe.py
python scripts/trend_pulse_universe.py --subset quick_test
```

### Validation Runner (Regime Detector)

```bash
# Makefile shortcuts (preferred)
make validate-fast          # PR gate (10 symbols, fast)
make validate               # Full validation suite (3 steps)

# Direct runner commands
python -m src.runners.validation_runner fast --symbols SPY QQQ AAPL --timeframes 1d
python -m src.runners.validation_runner full --universe config/universe.yaml --outer-folds 5
python -m src.runners.validation_runner holdout --universe config/universe.yaml
python -m src.runners.validation_runner optimize --universe config/universe.yaml --inner-trials 30
```

---

## Strategy Backtesting

APEX includes a production-ready strategy framework with **live/backtest parity** and **triple-engine** support.

### Backtesting Engines

| Engine | Use Case | Speed | Features |
|--------|----------|-------|----------|
| **ApexEngine** | Full simulation | Baseline | Event-driven, realistic fills, live/backtest parity |
| **VectorBTEngine** | Parameter sweeps | 100x faster | Vectorized, Optuna integration, WFO |
| **Backtrader** | Classic backtesting | Fast | Mature ecosystem, many analyzers |

### Available Strategies

| Strategy | Registry Name | Description |
|----------|---------------|-------------|
| Moving Average Cross | `ma_cross` | Classic SMA/EMA crossover |
| Buy and Hold | `buy_and_hold` | Passive benchmark strategy |
| RSI Mean Reversion | `rsi_mean_reversion` | RSI-based with limit orders |
| Momentum Breakout | `momentum_breakout` | ATR-based with trailing stops |
| Pairs Trading | `pairs_trading` | Statistical arbitrage |
| Scheduled Rebalance | `scheduled_rebalance` | Time-based rebalancing |
| MTF RSI Trend | `mtf_rsi_trend` | Multi-timeframe RSI strategy |
| TA Metrics | `ta_metrics_strategy` | Technical analysis metrics |

### Creating Custom Strategies

```python
from src.domain.strategy.base import Strategy
from src.domain.strategy.registry import register_strategy

@register_strategy("my_strategy", description="My custom strategy")
class MyStrategy(Strategy):
    def on_bar(self, bar: BarData):
        if self.should_buy(bar):
            self.request_order(OrderRequest(
                symbol=bar.symbol,
                side="BUY",
                quantity=100,
                order_type="MARKET",
            ))
```

See [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md) for complete documentation.

### Behavioral Gate Validation

The behavioral gate system validates strategies against historical market episodes (crashes, rallies, choppy periods):

```bash
# Quick behavioral test with default parameters
make behavioral

# Full pipeline: Optuna optimization + walk-forward + clustering
make behavioral-full

# Predefined case studies (market episodes)
make behavioral-cases
```

### Backtest Module Structure

```
src/backtest/
├── runner.py                    # Thin CLI facade (python -m src.backtest.runner)
├── cli/                         # CLI parser and commands
├── core/                        # Experiment, run, trial data models + verification
├── _internal/                   # Internal structured utilities
├── data/feeds/                  # 9 feed types
│   ├── csv_feeds.py             # CSV loading (streaming + buffered)
│   ├── parquet_feeds.py         # Parquet with predicate pushdown
│   ├── ib_feeds.py              # IB Historical + BarCache
│   ├── memory_feeds.py          # InMemory, Cached, Fixture
│   ├── historical_feeds.py      # Historical store adapter
│   └── multi_timeframe.py       # Multi-timeframe aggregation
├── execution/
│   ├── engines/                 # VectorBT, Apex, Backtrader adapters
│   ├── parity/                  # Live/backtest drift detection
│   ├── single_backtest.py       # SingleBacktestRunner
│   └── systematic_experiment.py # Data prefetch + systematic runs
├── optimization/                # Bayesian, grid, behavioral objectives
├── config/loaders.py            # Config loaders
└── analysis/reporting/          # HTML report generation
```

---

## Technical Indicators (47+)

APEX includes a comprehensive signal pipeline with 47+ indicators powered by TA-Lib, organized across 5 categories plus the Regime Detector system.

### Indicator Categories

| Category | Count | Indicators |
|----------|-------|------------|
| **Trend** | 11 | SMA, EMA, MACD, ADX, SuperTrend, Ichimoku, PSAR, Aroon, TRIX, Vortex, ZeroLag, TrendPulse |
| **Momentum** | 13 | RSI, KDJ, CCI, MFI, ROC, Stochastic, Williams %R, TSI, Awesome, Ultimate, Momentum, RSI Harmonics, Dual MACD |
| **Volatility** | 8 | ATR, Bollinger Bands, Keltner, Donchian, StdDev, Historical Vol, Squeeze, Chaikin Vol |
| **Volume** | 9 | OBV, CMF, A/D Line, VWAP, Force Index, CVD, VPVR, Volume, Volume Ratio |
| **Pattern** | 6 | Candlestick, Chart Patterns, Fibonacci, Support/Resistance, Pivot Points, Trendlines |

### TrendPulse Indicator

A hybrid multi-factor trend scoring indicator that combines multiple trend signals:

```bash
# Validate TrendPulse parameters
make tp-validate          # Full 3-stage validation
make tp-universe          # Full universe backtest + HTML report
```

### Regime Detector (4-Regime Market Classification)

Hierarchical classification system that drives position sizing and signal filtering:

| Regime | Name | Trading Implication |
|--------|------|---------------------|
| **R0** | Healthy Uptrend | Full trading allowed |
| **R1** | Choppy/Extended | Reduced frequency, wider spreads |
| **R2** | Risk-Off | No new positions, reduce size |
| **R3** | Rebound Window | Small defined-risk positions only |

**Pipeline:** Component states (Trend/Vol/Chop/Extension/IV) -> Decision tree (priority-based) -> Hysteresis (stable transitions) -> Optional turning point ML -> Composite score (0-100)

### Signal Rules

Rules are evaluated on indicator updates and fire trading signals:

| Category | Rules |
|----------|-------|
| **Trend** | Golden/Death Cross, Price Cross MA, PSAR Flip, SuperTrend Flip, Ichimoku Breakout, ADX Confirmation |
| **Momentum** | RSI Overbought/Oversold, Stochastic Crossover, MACD Crossovers, CCI Extremes, Williams %R Range |
| **Volatility** | Bollinger Squeeze, Keltner Breakout, ATR Expansion |
| **Volume** | OBV Divergence, CMF Confirmation, VWAP Cross |
| **Divergence** | RSI Divergence, MACD Divergence, Price-Volume Divergence |

### Signal Rule Development

```python
from src.domain.signals.models import SignalRule, ConditionType

rule = SignalRule(
    id="rsi_oversold",
    indicator="RSI",
    timeframe="5m",
    condition_type=ConditionType.THRESHOLD_CROSS,
    threshold=30,
    direction="below",
    cooldown_bars=5,
    detect_initial=True,
)
```

---

## Risk Management

### Position Reconciliation

Compares positions across sources:
- **IBKR** (Interactive Brokers API)
- **Futu** (OpenD API)
- **Manual YAML** (`data/positions/manual.yaml`)

Detects: `MISSING`, `DRIFT`, `STALE`

### Market Data Quality Control (MDQC)

- **Staleness:** Flags data older than threshold
- **Bid/Ask sanity:** Ensures bid <= ask
- **Zero quotes:** Flags suspicious zero prices
- **Missing Greeks:** Tracks incomplete data

### Risk Limits

```yaml
risk_limits:
  max_total_gross_notional: 5_000_000
  max_per_underlying_notional:
    default: 1_000_000
    TSLA: 1_500_000
  portfolio_delta_range: [-50_000, 50_000]
  portfolio_vega_range: [-15_000, 15_000]
  max_margin_utilization: 0.60
  soft_breach_threshold: 0.80  # Warning at 80%
```

### Breach Detection

- **Soft breach:** Warning at configurable threshold (default 80%)
- **Hard breach:** Critical alert when limit exceeded
- Events published to event bus for downstream handlers

---

## Reporting

APEX generates comprehensive HTML reports deployed to GitHub Pages:

### Signal Reports

Multi-page HTML report package with:
- Per-symbol indicator analysis with interactive Plotly charts
- Signal heatmaps across the entire universe
- Regime analysis overlays
- Dual MACD section
- TrendPulse trend scoring
- Confluence analysis

### Email Summaries

Automated email summary generation with:
- Market highlights and key movers
- Heatmap renderings
- Configurable scheduling

### Regime Validation Reports

HTML dashboards showing:
- Regime classification accuracy
- Turning point detection quality
- Parameter optimization results
- Cross-validation metrics

---

## Persistence Layer

PostgreSQL/DuckDB storage for historical data and warm-start.

### Features

- **Historical Data:** Futu orders/deals/fees, IB executions/commissions
- **Incremental Sync:** Track sync state per broker/account
- **Warm-Start:** Restore state from database on startup
- **Periodic Snapshots:** Configurable interval capture
- **Parquet Historical Store:** Local Parquet files for bar data caching
- **DuckDB Coverage Store:** Embedded analytics for data coverage tracking

See [docs/PERSISTENCE_LAYER.md](docs/PERSISTENCE_LAYER.md) for database setup.

---

## Observability

### OpenTelemetry + Prometheus Metrics

Exposed at `http://localhost:8000/metrics`:
- Risk metrics (P&L, Greeks, breaches)
- Adapter health (connection state, latency)
- Event bus performance (queue depths, processing time)

### Grafana Dashboards

Pre-configured dashboards in `config/grafana/`.

See [docs/OBSERVABILITY_SETUP.md](docs/OBSERVABILITY_SETUP.md) for setup.

---

## Development

### Testing

```bash
# Makefile shortcuts
make test               # Run unit tests
make test-all           # Run all tests (unit + integration)
make coverage           # Run tests with HTML coverage report

# Direct pytest commands
pytest                                    # All tests
pytest tests/unit/test_risk_engine.py    # Specific test file
pytest tests/unit/ -k "test_rule"        # Pattern matching
pytest tests/integration/                 # Integration tests
pytest -m "not slow"                      # Skip slow tests
pytest --cov=src --cov-report=html       # HTML coverage report
```

### Code Quality

```bash
# Makefile shortcuts (recommended)
make lint               # Check all (black, isort, flake8)
make format             # Auto-fix formatting
make type-check         # Run mypy
make quality            # All checks: lint + type-check + dead-code + complexity

# Direct commands
mypy src/ tests/        # Type checking (strict)
black src/ tests/       # Formatting (100-char lines)
isort src/ tests/       # Import sorting
flake8 src/ tests/      # Linting
```

**Important:** Always run `make format && make type-check` before committing to avoid CI failures.

### Performance Targets

| Position Count | Target Refresh |
|----------------|----------------|
| < 100 | < 100ms |
| 100-250 | < 250ms |
| 250-500 | < 500ms |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/base.yaml` | Main configuration (brokers, ports, API keys) |
| `config/universe.yaml` | **Primary universe** (symbols, sectors, subsets) |
| `config/risk_config.yaml` | Risk limits and signal thresholds |
| `config/demo.yaml` | Demo mode offline configuration |
| `config/regime_weights.yaml` | Regime detector factor weights |
| `config/gate_policy_clusters.yaml` | Behavioral gate policy clusters |
| `config/signals/rules.yaml` | Signal rule definitions |
| `config/signals/dev.yaml` | Development signal configuration |
| `config/validation/regime_universe.yaml` | Validation universe for regime detector |
| `config/validation/optimized_params.yaml` | Optuna-tuned parameters |
| `config/backtest/*.yaml` | Backtest specification examples (15+ files) |

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Development guidelines for AI assistants |
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | User manual |
| [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md) | Strategy development guide |
| [docs/PERSISTENCE_LAYER.md](docs/PERSISTENCE_LAYER.md) | Database setup and API reference |
| [docs/OBSERVABILITY_SETUP.md](docs/OBSERVABILITY_SETUP.md) | Observability setup guide |
| [docs/diagrams/](docs/diagrams/README.md) | Auto-generated architecture diagrams |
| [docs/indicators/](docs/indicators/) | Indicator documentation by category |
| [docs/rules/](docs/rules/) | Signal rule documentation |
| [docs/designs/](docs/designs/) | Architecture design documents |
| [docs/backtest/](docs/backtest/) | Backtest system documentation |
| [docs/reviews/](docs/reviews/) | Code reviews |

---

## Troubleshooting

### IB Connection Failed

1. Ensure TWS/Gateway is running
2. Check port: 7497 (TWS Paper), 7496 (TWS Live), 4001 (Gateway Paper)
3. Enable API in TWS: File -> Global Configuration -> API
4. Verify client ID is unique

### Missing Market Data

1. Check IBKR market data subscriptions
2. Verify symbol format
3. Review MDQC logs for staleness

### Position Reconciliation

1. Check `data/positions/manual.yaml` format
2. Verify position keys match across sources
3. Enable verbose logging: `-v`

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `make format && make type-check && make test`
5. Submit pull request

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Acknowledgments

Built with:
- [ib_async](https://github.com/ib-api-reloaded/ib_async) - Interactive Brokers API
- [futu-api](https://github.com/FutuOpenAPI/py-futu-api) - Futu OpenD API
- [textual](https://github.com/Textualize/textual) - Terminal UI framework
- [vectorbt](https://github.com/polakowo/vectorbt) - Vectorized backtesting
- [backtrader](https://github.com/mementum/backtrader) - Event-driven backtesting
- [TA-Lib](https://github.com/mrjbq7/ta-lib) - Technical analysis indicators
- [optuna](https://github.com/optuna/optuna) - Hyperparameter optimization
- [DuckDB](https://duckdb.org/) - Embedded analytics database
- [asyncpg](https://github.com/MagicStack/asyncpg) - PostgreSQL async client
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [OpenTelemetry](https://opentelemetry.io/) - Observability framework
- [pydantic](https://docs.pydantic.dev/) - Data validation
- [pandas-market-calendars](https://github.com/rsheftel/pandas_market_calendars) - Trading calendars
- [pytest](https://pytest.org/) - Testing framework
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
