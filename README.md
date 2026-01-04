# APEX - Live Risk Management & Backtesting System

**Production-grade portfolio risk monitoring and strategy backtesting for options and derivatives trading**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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
| **Strategy Backtesting** | Live/backtest parity - same code runs in both modes |
| **Terminal Dashboard** | Rich TUI with multiple views, keyboard navigation |
| **Event-Driven Architecture** | Priority event bus with fast/slow lanes |
| **Persistence Layer** | DuckDB/PostgreSQL with TimescaleDB support |
| **Observability** | Prometheus metrics, Grafana dashboards |

### Technical Highlights

- **46,000+ lines** of production Python code across 172 modules
- **Async-first** design using `asyncio` and `ib_async`
- **Thread-safe** in-memory stores with `RLock`
- **Hexagonal architecture** with clear domain/infrastructure separation
- **Comprehensive test suite** with unit, integration, and partial tests

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│  Terminal Dashboard (rich)  │  CLI Runners  │  Metrics Endpoint │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│  Orchestrator  │  ReadinessManager  │  PriorityEventBus        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      DOMAIN LAYER                               │
│  RiskEngine  │  RuleEngine  │  Strategy Framework  │  MDQC      │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                           │
│  IB Adapter  │  Futu Adapter  │  Backtest Engine  │  Stores     │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
apex/
├── config/               # Configuration files (YAML)
│   ├── base.yaml         # Main configuration
│   ├── backtest/         # Strategy backtest specs
│   └── prometheus/       # Alerting rules
├── data/
│   ├── positions/        # Manual position files
│   └── logs/             # Structured log output
├── docs/                 # Documentation
├── migrations/           # Database migrations
├── scripts/              # CLI tools
├── src/
│   ├── application/      # Orchestrator, event bus
│   ├── domain/           # Business logic
│   │   ├── events/       # PriorityEventBus
│   │   ├── services/     # RiskEngine, MDQC, RuleEngine
│   │   ├── strategy/     # Strategy framework
│   │   └── interfaces/   # Port definitions (DI)
│   ├── infrastructure/   # External integrations
│   │   ├── adapters/     # IB, Futu, Yahoo
│   │   ├── backtest/     # Backtest engine
│   │   └── persistence/  # Database repositories
│   ├── models/           # Data models
│   ├── runners/          # CLI runners
│   ├── services/         # Application services
│   └── tui/              # Terminal dashboard
├── tests/                # Test suites
├── main.py               # Entry point
└── pyproject.toml        # Package config
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.14 recommended |
| IBKR TWS/Gateway | Latest | Required for market data |
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
# Development mode
python orchestrator.py --env dev

# Production mode
python orchestrator.py --env prod

# Demo mode (offline, sample data)
python orchestrator.py --env demo
```

### 4. Terminal Dashboard

The TUI displays real-time risk metrics:

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

**Keyboard:** `1-6` switch views, `q` quit, `Ctrl+C` graceful shutdown

---

## Strategy Backtesting

APEX includes a production-ready strategy framework with **live/backtest parity**.

### Available Strategies

| Strategy | Registry Name | Description |
|----------|---------------|-------------|
| Moving Average Cross | `ma_cross` | Classic MA crossover |
| Buy and Hold | `buy_and_hold` | Passive benchmark |
| RSI Mean Reversion | `rsi_mean_reversion` | RSI with limit orders |
| Momentum Breakout | `momentum_breakout` | ATR-based with trailing stops |
| Pairs Trading | `pairs_trading` | Statistical arbitrage |
| Scheduled Rebalance | `scheduled_rebalance` | Time-based rebalancing |

### Running Backtests

```bash
# List strategies
python -m src.runners.backtest_runner --list-strategies

# CLI backtest
python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL \
    --start 2024-01-01 --end 2024-06-30

# From YAML spec
python -m src.runners.backtest_runner --spec config/backtest/ma_cross_example.yaml
```

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

### History Loading

```bash
# Load Futu history
python scripts/history_loader.py --broker futu --days 30

# Load IB history
python scripts/history_loader.py --broker ib --days 30

# Dry run
python scripts/history_loader.py --broker all --dry-run
```

See [docs/PERSISTENCE_LAYER.md](docs/PERSISTENCE_LAYER.md) for database setup.

---

## Observability

### Prometheus Metrics

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
# All tests with coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_risk_engine.py

# Integration tests
pytest tests/integration/

# Open coverage report
open htmlcov/index.html
```

### Code Quality

```bash
mypy src/              # Type checking
black src/ tests/      # Formatting
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

## Documentation

| Document | Description |
|----------|-------------|
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | Complete user guide with CLI reference |
| [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md) | Strategy development & backtest guide |
| [docs/PERSISTENCE_LAYER.md](docs/PERSISTENCE_LAYER.md) | Database setup and API reference |
| [docs/OBSERVABILITY_SETUP.md](docs/OBSERVABILITY_SETUP.md) | Prometheus, Grafana, alerting |
| [CLAUDE.md](CLAUDE.md) | Development guidelines |
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
- [rich](https://github.com/Textualize/rich) - Terminal UI
- [DuckDB](https://duckdb.org/) - Embedded analytics database
- [asyncpg](https://github.com/MagicStack/asyncpg) - PostgreSQL async client
- [TA-Lib](https://github.com/mrjbq7/ta-lib) - Technical analysis
- [pytest](https://pytest.org/) - Testing framework
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
