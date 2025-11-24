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

```
apex/
├── config/               # Configuration files
│   ├── risk_config.yaml  # Main configuration
│   ├── base.yaml         # Base config (future)
│   └── dev.yaml          # Dev overrides (future)
├── src/
│   ├── models/           # Data models (Position, MarketData, RiskSnapshot)
│   ├── domain/
│   │   ├── interfaces/   # Provider interfaces (DI)
│   │   └── services/     # Domain services (RiskEngine, MDQC, RuleEngine)
│   ├── infrastructure/
│   │   ├── adapters/     # IB adapter, file loader
│   │   ├── stores/       # Thread-safe data stores
│   │   └── monitoring/   # Health monitor, watchdog
│   ├── application/      # Orchestrator, event bus
│   ├── presentation/     # Terminal dashboard
│   └── utils/            # Structured logger
├── data/
│   ├── positions/        # Manual position files
│   └── logs/             # Log output
├── tests/
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── legacy/               # Old implementation (reference)
├── main.py               # Entry point
└── pyproject.toml        # uv project config
```

## Prerequisites

- **Python 3.10+**
- **Interactive Brokers** TWS or IB Gateway (Paper Trading or Live)
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

### 3. Manual Positions (Optional)

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

## Support

For issues and questions:
- File an issue on GitHub
- Review `CLAUDE.md` for implementation guidance
- Check `PRD_v1.1_Suggestions_and_Improvements.md` for detailed requirements

## Acknowledgments

Built with:
- [ib_async](https://github.com/ib-api-reloaded/ib_async) - Interactive Brokers API client
- [rich](https://github.com/Textualize/rich) - Terminal UI library
- [pytest](https://pytest.org/) - Testing framework
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
