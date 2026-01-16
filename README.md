# APEX - Live Risk Management & Backtesting System

**Production-grade portfolio risk monitoring and strategy backtesting for options and derivatives trading**

[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

APEX is a comprehensive risk management and backtesting platform designed for active traders managing options and derivatives portfolios. Built with a hexagonal architecture and event-driven patterns, it provides:

### Core Features

| Feature | Description |
|---------|-------------|
| **Real-time Risk Monitoring** | P&L (unrealized/daily), Greeks aggregation, concentration limits |
| **Multi-Broker Support** | Interactive Brokers, Futu OpenD with auto-reconnect |
| **Dual-Engine Backtesting** | ApexEngine (event-driven) + VectorBT (vectorized, 100x faster) |
| **44+ Technical Indicators** | Full TA-Lib integration with custom rule engine |
| **Terminal Dashboard** | Textual TUI with 6 views, keyboard navigation |
| **Event-Driven Architecture** | Priority event bus with fast/slow lanes |
| **Persistence Layer** | DuckDB/PostgreSQL with TimescaleDB support |
| **Observability** | Prometheus metrics, Grafana dashboards |

### Technical Highlights

- **95,000+ lines** of production Python code across 374 modules
- **Async-first** design using `asyncio` and `ib_async`
- **Thread-safe** RCU (Read-Copy-Update) stores for lock-free reads
- **Hexagonal architecture** with clear domain/infrastructure separation
- **65 test files** with 85% coverage requirement enforced

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│  Terminal Dashboard (Textual)  │  CLI Runners  │ Metrics Endpoint│
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│  Orchestrator  │  ReadinessManager  │  PriorityEventBus        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      DOMAIN LAYER                               │
│  RiskEngine  │  IndicatorEngine  │  Strategy Framework  │  MDQC │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                           │
│  IB Adapter  │  Futu Adapter  │  Backtest Engines  │  Stores    │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
apex/
├── config/               # Configuration files (YAML)
│   ├── base.yaml         # Main configuration
│   ├── risk_config.yaml  # Risk limits and signals
│   ├── signals/          # Signal rule definitions
│   └── backtest/         # Backtest YAML specs (9 examples)
├── data/
│   ├── historical/       # Cached bar data (Parquet)
│   ├── positions/        # Manual position files
│   └── logs/             # Structured log output
├── docs/                 # Documentation
├── migrations/           # Database migrations
├── scripts/              # CLI utilities
├── src/
│   ├── application/      # Orchestrator, coordinators (17 files)
│   ├── backtest/         # Backtest engines (51 files)
│   ├── domain/           # Business logic (168 files)
│   │   ├── events/       # PriorityEventBus
│   │   ├── services/     # RiskEngine, MDQC
│   │   ├── signals/      # 44+ indicators, rule engine
│   │   ├── strategy/     # Strategy framework
│   │   └── interfaces/   # Port definitions (DI)
│   ├── infrastructure/   # External integrations (62 files)
│   │   ├── adapters/     # IB, Futu, Yahoo
│   │   ├── stores/       # RCU stores
│   │   └── persistence/  # Database repositories
│   ├── models/           # Data models (9 files)
│   ├── runners/          # CLI runners (3 files)
│   ├── services/         # Application services (13 files)
│   └── tui/              # Terminal dashboard (42 files)
├── tests/                # Test suites (65 files)
├── main.py               # Primary entry point
└── pyproject.toml        # Package config
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.14+ | Required (uses latest features) |
| TA-Lib | 0.4.19+ | Required for indicators |
| IBKR TWS/Gateway | Latest | Required for live market data |
| PostgreSQL | 14+ | Optional, for persistence |
| TimescaleDB | 2.x | Optional, for time-series |
| Futu OpenD | Latest | Optional, for Futu positions |

---

## Installation

### Using uv (Recommended)

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
┌─────────────────────────────────────────────────────────────────┐
│ APEX Risk Monitor │ dev │ 2024-03-15 10:30:45 HKT │ IB:● FU:○  │
├─────────────────────────────────────────────────────────────────┤
│ Portfolio Summary                                               │
│   NAV: $1,234,567    Unrealized P&L: +$12,345   Daily: +$5,678 │
│   Delta: 25,000      Gamma: 1,234    Vega: 8,765  Theta: -567  │
├─────────────────────────────────────────────────────────────────┤
│ Risk Signals                                                    │
│   ⚠ SOFT: Portfolio delta at 82% of limit                      │
│   ● HARD: TSLA notional exceeds limit                          │
└─────────────────────────────────────────────────────────────────┘
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
# Live signal generation
python -m src.runners.signal_runner --live --symbols AAPL TSLA QQQ

# Multiple timeframes
python -m src.runners.signal_runner --live --symbols AAPL --timeframes 1m 5m 1h 1d

# Generate HTML report
python -m src.runners.signal_runner --live --symbols AAPL --html-output report.html

# Backfill historical signals
python -m src.runners.signal_runner --backfill --symbols AAPL --days 365

# With database persistence
python -m src.runners.signal_runner --live --symbols AAPL --with-persistence
```

### History Loader

```bash
# Load broker history
python scripts/history_loader.py --broker ib --days 30
python scripts/history_loader.py --broker futu --days 30 --market US
python scripts/history_loader.py --broker all --dry-run
python scripts/history_loader.py --broker ib --from-date 2024-01-01 --to-date 2024-06-30
```

---

## Strategy Backtesting

APEX includes a production-ready strategy framework with **live/backtest parity** and **dual-engine** support.

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

---

## Technical Indicators (44+)

APEX includes a comprehensive signal pipeline with 44+ indicators powered by TA-Lib.

### Indicator Categories

| Category | Count | Indicators |
|----------|-------|------------|
| **Trend** | 10 | SMA, EMA, MACD, ADX, SuperTrend, Ichimoku, PSAR, Aroon, TRIX, Vortex, ZeroLag |
| **Momentum** | 12 | RSI, KDJ, CCI, MFI, ROC, Williams %R, TSI, Awesome, Ultimate, Momentum, RSI Harmonics |
| **Volatility** | 8 | ATR, Bollinger Bands, Keltner, Donchian, StdDev, Historical Vol, Squeeze, Chaikin Vol |
| **Volume** | 8 | OBV, CMF, A/D Line, VWAP, Force Index, CVD, VPVR, Volume Ratio |
| **Pattern** | 6 | Candlestick, Chart Patterns, Fibonacci, Support/Resistance, Pivot Points, Trendlines |

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

## Persistence Layer

PostgreSQL/DuckDB storage for historical data and warm-start.

### Features

- **Historical Data:** Futu orders/deals/fees, IB executions/commissions
- **Incremental Sync:** Track sync state per broker/account
- **Warm-Start:** Restore state from database on startup
- **Periodic Snapshots:** Configurable interval capture

See [docs/PERSISTENCE_LAYER.md](docs/PERSISTENCE_LAYER.md) for database setup.

---

## Observability

### Prometheus Metrics

Exposed at `http://localhost:8000/metrics`:
- Risk metrics (P&L, Greeks, breaches)
- Adapter health (connection state, latency)
- Event bus performance (queue depths, processing time)

### Grafana Dashboards

Pre-configured dashboards in `config/prometheus/`.

See [docs/OBSERVABILITY_SETUP.md](docs/OBSERVABILITY_SETUP.md) for setup.

---

## Development

### Testing

```bash
# All tests with coverage (85% enforced)
pytest

# Specific test file
pytest tests/unit/test_risk_engine.py

# Pattern matching
pytest tests/unit/ -k "test_rule"

# Integration tests
pytest tests/integration/

# Skip slow tests
pytest -m "not slow"

# HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
mypy src/              # Type checking (strict)
black src/ tests/      # Formatting (100-char lines)
isort src/ tests/      # Import sorting
flake8 src/ tests/     # Linting
```

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
| `config/risk_config.yaml` | Risk limits and signal thresholds |
| `config/demo.yaml` | Demo mode offline configuration |
| `config/signals/rules.yaml` | Signal rule definitions |
| `config/signals/universe.yaml` | Trading universe definition |
| `config/backtest/*.yaml` | Backtest specification examples |

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Development guidelines for AI assistants |
| [docs/PERSISTENCE_LAYER.md](docs/PERSISTENCE_LAYER.md) | Database setup and API reference |
| [docs/indicators/](docs/indicators/) | Indicator documentation by category |
| [docs/rules/](docs/rules/) | Signal rule documentation |
| [docs/designs/](docs/designs/) | Architecture design documents |
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
4. Run test suite and quality checks
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
- [pytest](https://pytest.org/) - Testing framework
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
