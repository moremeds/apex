# Quick Start Guide

## ✅ Setup Complete

Your Live Risk Management System skeleton is ready!

## 📦 Installed Packages

**Core Dependencies:**
- ✅ `ib-async` 2.0.1 - Interactive Brokers API client
- ✅ `rich` 14.2.0 - Terminal UI library
- ✅ `pyyaml` 6.0.3 - Configuration parsing

**Dev Dependencies:**
- ✅ `pytest` 9.0.1 - Testing framework
- ✅ `pytest-cov` 7.0.0 - Coverage reporting
- ✅ `pytest-asyncio` 1.3.0 - Async test support
- ✅ `mypy` 1.18.2 - Type checking
- ✅ `black` 25.11.0 - Code formatting
- ✅ `isort` 7.0.0 - Import sorting
- ✅ `flake8` 7.3.0 - Linting

## 🚀 Running the System

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 2. Configure IBKR Connection

Edit `config/risk_config.yaml`:

```yaml
ibkr:
  host: 127.0.0.1
  port: 7497  # 7497=TWS Paper, 4001=Gateway Paper
  client_id: 1
```

### 3. Start IB TWS or Gateway

- Open TWS or IB Gateway
- Enable API connections (Global Config → API → Settings)
- Use Paper Trading account for testing

### 4. Run the System

```bash
# Development mode
python orchestrator.py --env dev

# Production mode
python orchestrator.py --env prod

# Headless mode (no dashboard)
python orchestrator.py --no-dashboard
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_risk_engine.py -v

# View coverage report
open htmlcov/index.html
```

## 🛠️ Development Commands

```bash
# Type checking
mypy src/

# Code formatting
black src/ tests/

# Import sorting
isort src/ tests/

# Linting
flake8 src/ tests/

# All quality checks at once
mypy src/ && black src/ tests/ && isort src/ tests/ && flake8 src/ tests/
```

## 📊 Project Structure

```
apex/
├── src/                    # Main source code
│   ├── models/             # Data models (Position, MarketData, etc.)
│   ├── domain/
│   │   ├── interfaces/     # Provider interfaces (DI)
│   │   └── services/       # Business logic (RiskEngine, MDQC, etc.)
│   ├── infrastructure/
│   │   ├── adapters/       # IB adapter, file loader
│   │   ├── stores/         # Thread-safe data stores
│   │   └── monitoring/     # Health monitor, watchdog
│   ├── application/        # Orchestrator, event bus
│   ├── presentation/       # Terminal dashboard
│   └── utils/              # Structured logger
├── config/                 # Configuration management
├── data/
│   ├── positions/          # Manual position YAML files
│   └── logs/               # Log output
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
└── legacy/                 # Old implementation (reference)
```

## 📝 Current Test Status

- ✅ **4/4 tests passing**
- ⚠️ **Coverage: 26%** (skeleton implementation)
- 🎯 **Target: 85%** (for MVP completion)

## 🔍 Verification

Check that everything works:

```bash
# Verify imports
python -c "
from src.models.position import Position
from src.infrastructure.adapters import IbAdapter
from config.config_manager import ConfigManager
print('✅ All imports successful!')
"

# Run tests
pytest tests/ -v --no-cov
```

## 📚 Next Steps

### 1. Complete IB Adapter Implementation
- Implement full position fetching from IBKR
- Implement market data subscription
- Add Greeks fetching (IBKR-sourced only)
- Implement account info fetching

### 2. Add More Tests
- Unit tests for all domain services
- Integration tests with IBKR Paper Trading
- Test coverage target: 85%

### 3. Implement Full Workflow
- Test position reconciliation
- Verify limit breach detection
- Test dashboard rendering
- Validate P&L calculations

### 4. Production Readiness
- 4+ hour soak test with Paper Trading
- Auto-reconnect testing
- Memory leak testing (8-hour run)
- Performance validation (< 100ms for < 100 positions)

## 🐛 Troubleshooting

### Virtual Environment Issues

If imports fail, make sure you're using the venv Python:

```bash
# Activate venv
source .venv/bin/activate

# Or use venv Python directly
.venv/bin/python orchestrator.py
```

### IBKR Connection Issues

1. Ensure TWS/Gateway is running
2. Check port number matches config
3. Enable API in TWS settings: Global Config → API → Settings
4. Check client ID is unique

### Import Errors

```bash
# Reinstall package
uv pip install -e .

# Or reinstall all dependencies
uv pip install -e ".[dev]"
```

## 📖 Documentation

- `README.md` - Full documentation
- `CLAUDE.md` - Development guidelines for AI assistants
- `PRD_v1.1_Suggestions_and_Improvements.md` - Detailed requirements
- `config/risk_config.yaml` - Configuration reference

## ✨ Features Implemented

✅ **Models Layer**
- Position with source tracking
- MarketData with Greeks & quality flags
- ReconciliationIssue (MISSING/DRIFT/STALE)
- RiskSnapshot with portfolio metrics
- AccountInfo with margin calculations

✅ **Domain Layer**
- Provider interfaces (DI pattern)
- RiskEngine (P&L, Greeks, expiry buckets)
- Reconciler (position reconciliation)
- MDQC (market data quality control)
- RuleEngine (soft/hard breach detection)
- SimpleSuggester (top contributors)
- SimpleShockEngine (spot shock scenarios)

✅ **Infrastructure Layer**
- IbAdapter skeleton (ib_async integration)
- FileLoader (YAML position loading)
- Thread-safe stores (Position, MarketData, Account)
- HealthMonitor (component health tracking)
- Watchdog (connection & freshness monitoring)

✅ **Application Layer**
- Orchestrator (main workflow)
- EventBus (pub-sub pattern)

✅ **Presentation Layer**
- TerminalDashboard (rich library UI)

✅ **Configuration**
- ConfigManager (env-based loading)
- Complete risk_config.yaml

✅ **Testing**
- pytest setup with fixtures
- Coverage reporting
- Unit test examples

## 🎯 Success!

Your project skeleton is complete and ready for implementation! 🚀

Start coding by implementing the IB adapter methods or adding more tests.
