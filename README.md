# Live Risk Management System (APEX)

**Real-time portfolio risk monitoring for options and derivatives trading**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

APEX is a real-time risk management system for monitoring options and derivatives portfolios. It provides:

- **Real-time P&L calculation** (unrealized & daily)
- **Greeks aggregation** (delta, gamma, vega, theta) using IBKR Greeks
- **Position reconciliation** across multiple sources (IBKR vs manual YAML)
- **Market data quality control** (staleness, bid/ask validation)
- **Risk limit monitoring** with soft/hard breach detection
- **Spot shock scenarios** for stress testing
- **Terminal dashboard** with rich UI
- **Health monitoring** and auto-reconnect

## Architecture

The project follows a Hexagonal (Ports and Adapters) architecture, which separates the core business logic from external concerns like data sources and user interfaces. This makes the system modular and easier to maintain.

*   **Domain (`src/domain`):** Contains the core business logic, including services for risk calculation (`RiskEngine`), data validation (`MDQC`), and rule checking (`RuleEngine`).
*   **Application (`src/application`):** The `Orchestrator` acts as the central coordinator, managing the flow of data between the domain and infrastructure layers.
*   **Infrastructure (`src/infrastructure`):** Handles all external interactions. This includes adapters for connecting to external systems (like Interactive Brokers) and in-memory stores for managing application state.
*   **Presentation (`src/presentation`):** The `TerminalDashboard` provides a text-based UI to display risk information.

### Directory Structure

```
apex/
├── config/               # Configuration files
│   ├── risk_config.yaml  # Main configuration
│   ├── base.yaml         # Base config with database/snapshot settings
│   └── dev.yaml          # Dev overrides (future)
├── migrations/           # Database schema migrations
│   ├── 001_initial_schema.sql
│   └── runner.py         # Migration runner
├── scripts/              # CLI tools
│   └── history_loader.py # Load historical data from brokers
├── src/
│   ├── models/           # Data models (Position, MarketData, RiskSnapshot)
│   ├── domain/
│   │   ├── interfaces/   # Provider interfaces (DI)
│   │   └── services/     # Domain services (RiskEngine, MDQC, RuleEngine)
│   ├── infrastructure/
│   │   ├── adapters/     # IB adapter, file loader
│   │   ├── persistence/  # PostgreSQL/TimescaleDB layer
│   │   │   ├── database.py      # Connection manager (asyncpg)
│   │   │   └── repositories/    # Repository pattern implementations
│   │   ├── stores/       # Thread-safe data stores
│   │   └── monitoring/   # Health monitor, watchdog
│   ├── services/         # Business logic services
│   │   ├── history_loader_service.py  # History loading orchestrator
│   │   ├── snapshot_service.py        # Periodic snapshot capture
│   │   └── warm_start_service.py      # Startup state restoration
│   ├── application/      # Orchestrator, event bus
│   ├── presentation/     # Terminal dashboard
│   └── utils/            # Structured logger
├── data/
│   ├── positions/        # Manual position files
│   └── logs/             # Log output
├── docs/                 # Documentation
│   └── PERSISTENCE_LAYER.md  # Detailed persistence layer docs
├── tests/
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── legacy/               # Old implementation (reference)
├── main.py               # Entry point
└── pyproject.toml        # uv project config
```

### Data Flow

1.  The `Orchestrator` fetches position data from Interactive Brokers and local files.
2.  The `Reconciler` service compares this data with the existing positions stored in memory.
3.  Market data is fetched and stored.
4.  The `RiskEngine` processes the data to produce a `RiskSnapshot`.
5.  The `RuleEngine` checks the snapshot for any rule breaches.
6.  The `TerminalDashboard` displays the final risk snapshot.

## Prerequisites

- **Python 3.10+**
- **Interactive Brokers** TWS or IB Gateway (Paper Trading or Live)
- **PostgreSQL 14+** (optional, for persistence layer)
- **TimescaleDB** extension (optional, for time-series optimization)
- **uv** package manager (recommended) or pip

## Installation

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
cd apex

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

## Configuration

Edit `config/risk_config.yaml` to configure:

### 1. IBKR Connection

```yaml
ibkr:
  host: 127.0.0.1
  port: 7497  # 7497=TWS Paper, 7496=TWS Live, 4001=Gateway Paper
  client_id: 1
```

### 2. Risk Limits

```yaml
risk_limits:
  max_total_gross_notional: 5_000_000
  max_per_underlying_notional:
    default: 1_000_000
    TSLA: 1_500_000
  portfolio_delta_range: [-50_000, 50_000]
  portfolio_vega_range: [-15_000, 15_000]
  max_margin_utilization: 0.60
```

### 3. Database (Optional)

Configure in `config/base.yaml`:

```yaml
database:
  dsn: "postgresql://user:password@localhost:5432/apex_risk"
  min_pool_size: 2
  max_pool_size: 10
  command_timeout: 30

snapshots:
  position_interval_sec: 60      # Capture positions every minute
  account_interval_sec: 60       # Capture account state every minute
  risk_interval_sec: 300         # Capture risk metrics every 5 minutes
  capture_on_shutdown: true      # Save final snapshot on shutdown
  retention_days: 30             # Keep snapshots for 30 days
```

### 4. Manual Positions (Optional)

Edit `data/positions/manual.yaml` to add manual positions:

```yaml
positions:
  - symbol: AAPL
    underlying: AAPL
    asset_type: STOCK
    quantity: 100
    avg_price: 175.50
    multiplier: 1
```

## Usage

### Start the System

```bash
# Development mode (default)
python main.py --env dev

# Production mode
python main.py --env prod

# Headless mode (no dashboard)
python main.py --no-dashboard
```

### Terminal Dashboard

The system displays a real-time terminal UI with:

- **Portfolio Summary**: P&L, Greeks, notional exposure, concentration
- **Limit Breaches**: Soft/hard breach alerts with percentages
- **Component Health**: Connection status, data quality

Press `Ctrl+C` to shutdown gracefully.

## Testing

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_risk_engine.py

# Run with verbose output
pytest -v

# Generate HTML coverage report
pytest --cov-report=html
open htmlcov/index.html
```

## Code Quality

```bash
# Type checking
mypy src/

# Code formatting
black src/ tests/

# Import sorting
isort src/ tests/

# Linting
flake8 src/ tests/
```

## Development Workflow

1. **Make changes** to source code
2. **Run tests**: `pytest`
3. **Check types**: `mypy src/`
4. **Format code**: `black .`
5. **Commit** changes

## Key Features

### Position Reconciliation

Compares positions from:
- **IBKR** (Interactive Brokers API)
- **Manual YAML** file (`data/positions/manual.yaml`)
- **Cached** state (previous snapshot)

Detects:
- **MISSING**: Position in one source but not another
- **DRIFT**: Quantity mismatch between sources
- **STALE**: Position not updated for threshold period

### Market Data Quality Control (MDQC)

Validates:
- **Staleness**: Flags data older than 10 seconds
- **Bid/Ask sanity**: Ensures bid ≤ ask
- **Zero quotes**: Flags suspicious zero prices
- **Missing Greeks**: Tracks positions without Greeks

### Greeks Handling (MVP)

- **IBKR Greeks only** (no local BSM/Bachelier calculation)
- Mark positions as `DATA_MISSING` when Greeks unavailable
- Stock delta defaults to 1.0

### Risk Limits

- **Total gross notional**: Portfolio-wide exposure limit
- **Per-underlying notional**: Single-name concentration limits
- **Portfolio Greeks ranges**: Delta, vega, theta bounds
- **Margin utilization**: Max margin usage percentage
- **Concentration**: Max single-name percentage

### Breach Detection

- **Soft breach**: Warning at 80% of limit (configurable)
- **Hard breach**: Critical alert when limit exceeded
- **Events published** to event bus for alerts

### Persistence Layer

PostgreSQL/TimescaleDB storage for historical data and warm-start capability.

**Features:**
- **Historical Data Storage**: Futu orders/deals/fees, IB executions/commissions
- **Incremental Sync**: Track sync state per broker/account for efficient updates
- **Warm-Start**: Restore positions and account state from database snapshots on startup
- **Periodic Snapshots**: Capture positions, accounts, and risk metrics at configurable intervals

**History Loading CLI:**

```bash
# Load Futu historical data (last 30 days)
python scripts/history_loader.py --broker futu --account YOUR_ACC --days 30

# Load IB historical data
python scripts/history_loader.py --broker ib --account YOUR_ACC --days 30

# Load from all brokers
python scripts/history_loader.py --broker all --days 30

# Dry run (show what would be loaded)
python scripts/history_loader.py --broker futu --dry-run
```

**Database Setup:**

```bash
# Create database
createdb apex_risk

# Enable TimescaleDB extension (optional, for time-series optimization)
psql apex_risk -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Run migrations
python -c "
import asyncio
from src.infrastructure.persistence import get_database
from migrations.runner import MigrationRunner

async def migrate():
    db = await get_database('postgresql://user:pass@localhost/apex_risk')
    runner = MigrationRunner(db)
    await runner.run()
    await db.close()

asyncio.run(migrate())
"
```

See `docs/PERSISTENCE_LAYER.md` for comprehensive documentation.

## Performance Targets

- **< 100ms** refresh for < 100 positions
- **< 250ms** for 100-250 positions
- **< 500ms** for 250-500 positions

## Acceptance Criteria

- ✅ P&L accuracy > 99.9% vs TWS
- ✅ Auto-reconnect success rate 100%
- ✅ No crashes during 8-hour run
- ✅ Memory growth < 20% over 8 hours
- ✅ Test coverage > 85%

## Project Status

**Current**: Skeleton implementation with complete structure
**MVP Scope**: Core monitoring features (see CLAUDE.md)
**Deferred to v1.2**: What-if simulator, IV shocks, hedge optimization

See `CLAUDE.md` for detailed implementation guidelines.

## Logging

Structured JSON logging to `data/logs/live_risk.log`:

```json
{
  "timestamp": "2024-03-15T10:30:45.123Z",
  "level": "INFO",
  "category": "RISK",
  "message": "Portfolio delta breach detected",
  "data": {"delta": 55000, "limit": 50000}
}
```

Categories: `SYSTEM`, `RISK`, `TRADING`, `DATA`, `ALERT`

## Troubleshooting

### IB Connection Failed

1. Ensure TWS/Gateway is running
2. Check port number (7497 for Paper, 7496 for Live)
3. Enable API connections in TWS settings
4. Verify client ID is unique

### Missing Market Data

1. Check IBKR market data subscriptions
2. Verify symbols are correct
3. Check MDQC logs for staleness alerts

### Position Reconciliation Issues

1. Review `data/positions/manual.yaml` format
2. Check position keys match: `(symbol, underlying, asset_type, expiry, strike, right)`
3. Monitor reconciliation alerts in logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run test suite and code quality checks
5. Submit pull request

## License

MIT License - see LICENSE file

## Documentation

| Document | Description |
|----------|-------------|
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | Complete user guide with CLI reference |
| [docs/PERSISTENCE_LAYER.md](docs/PERSISTENCE_LAYER.md) | Database setup and API reference |
| [docs/OBSERVABILITY_SETUP.md](docs/OBSERVABILITY_SETUP.md) | Prometheus, Grafana, and alerting setup |
| [CLAUDE.md](CLAUDE.md) | Development guidelines |

## Support

For issues and questions:
- File an issue on GitHub
- Review `docs/USER_MANUAL.md` for usage instructions
- See `docs/PERSISTENCE_LAYER.md` for database setup
- Check `CLAUDE.md` for implementation guidance

## Acknowledgments

Built with:
- [ib_async](https://github.com/ib-api-reloaded/ib_async) - Interactive Brokers API client
- [futu-api](https://github.com/FutuOpenAPI/py-futu-api) - Futu OpenD API client
- [asyncpg](https://github.com/MagicStack/asyncpg) - Fast PostgreSQL client for asyncio
- [rich](https://github.com/Textualize/rich) - Terminal UI library
- [pytest](https://pytest.org/) - Testing framework
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
