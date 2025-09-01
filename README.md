# Apex - High-Performance Strategy Backtesting System

## Overview

Apex is a production-ready, high-performance backtesting system designed for quantitative trading research. Built with a focus on speed (vectorbt-first approach), data quality, and progressive enhancement from MVP to full ML capabilities.

## Key Features

- **High Performance**: Vectorized operations using vectorbt for sub-second backtests
- **Data Quality Framework**: Comprehensive 5-dimensional quality scoring system
- **Strategy Validation**: Walk-forward analysis, overfitting prevention, parameter stability checks
- **Progressive Enhancement**: Start simple with basic strategies, scale to ML-powered systems
- **Production Ready**: Monitoring, logging, error recovery from day one
- **Multi-Engine Support**: Primary vectorbt engine with planned Backtrader support

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/apex.git
cd apex

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# Or: .venv\Scripts\activate  # On Windows

# Install dependencies with uv
uv pip install -e ".[dev,ml]"

# Or simply sync with the lock file
uv sync

# Setup environment
uv run python scripts/setup_environment.py

# Download sample data
uv run python scripts/download_sample_data.py
```

### Run Your First Backtest

```python
from apex.engine import EngineHarness
from apex.strategies.vectorbt import MovingAverageCrossStrategy
from apex.data.providers import YahooDataProvider
import asyncio

async def main():
    # Load data
    provider = YahooDataProvider()
    data = await provider.fetch_data("SPY", "2020-01-01", "2023-12-31")
    
    # Configure and run
    config = {'initial_cash': 100000, 'commission': 0.001}
    harness = EngineHarness(config)
    strategy = MovingAverageCrossStrategy(fast_period=20, slow_period=50)
    result = harness.run_backtest(strategy, data)
    
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")

asyncio.run(main())
```

## Project Structure

```
apex/
├── src/apex/                 # Main package
│   ├── core/                 # Core functionality
│   ├── data/                 # Data management
│   ├── strategies/           # Trading strategies
│   ├── engine/              # Backtesting engine
│   ├── analysis/            # Analysis and reporting
│   └── api/                 # REST API (Phase 3)
├── tests/                    # Test suite
├── configs/                  # Configuration files
├── notebooks/               # Jupyter notebooks
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
```

## Data Quality Framework

The system implements a comprehensive 5-dimensional quality scoring system:

1. **Completeness**: Missing data detection and handling
2. **Consistency**: OHLC relationship validation
3. **Validity**: Outlier detection and correction
4. **Timeliness**: Gap analysis and frequency validation
5. **Uniqueness**: Duplicate detection and removal

Quality scores below 0.7 trigger configurable handling (warn/block/fix).

## Available Strategies

### MVP Strategies (Phase 1)
- **Moving Average Cross**: Classic trend-following strategy
- **RSI Mean Reversion**: Overbought/oversold trading
- **Breakout Strategy**: Support/resistance breakout trading

### ML Strategies (Phase 2)
- Feature engineering pipeline
- XGBoost/LightGBM models
- Purged cross-validation
- Walk-forward optimization

## Development

### Dependency Management

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update dependencies
uv lock --upgrade

# Sync your environment with the lock file
uv sync
```

### Running Tests

```bash
# Run all tests with coverage
make test

# Or directly with uv
uv run pytest tests/ -v --cov=apex

# Run specific test categories
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/performance/ -v
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Or directly with uv
uv run black src/ tests/
uv run ruff check src/
uv run mypy src/ --strict
```

### Building Documentation

```bash
make docs
```

## Performance Targets

- Simple backtest: < 0.5s per year of data
- ML backtest: < 5s per year of data
- Memory usage: < 1GB per million rows
- API latency (p99): < 100ms

## Development Roadmap

### Phase 0: Foundation (Weeks 1-3) ✅
- [x] Project structure setup
- [ ] Data pipeline with quality scoring
- [ ] Core types and base classes

### Phase 1: MVP (Weeks 4-9)
- [ ] Vectorbt engine integration
- [ ] 3 basic strategies
- [ ] Walk-forward validation
- [ ] Basic reporting

### Phase 2: ML & Advanced (Weeks 10-17)
- [ ] Feature engineering pipeline
- [ ] ML strategy framework
- [ ] Purged cross-validation
- [ ] Advanced validation

### Phase 3: Production (Weeks 18-22)
- [ ] REST API with FastAPI
- [ ] Monitoring and alerting
- [ ] Performance optimization
- [ ] Additional engines

## Configuration

See `configs/default.yaml` for configuration options. Environment-specific configs can be placed in `configs/environments/`.

## API Documentation

API documentation will be available at `/docs` when the FastAPI server is running (Phase 3).

## Contributing

Please read [CONTRIBUTING.md](docs/development/contributing.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions, please use the GitHub issue tracker.

## Acknowledgments

- Built with [vectorbt](https://github.com/polakowo/vectorbt) for high-performance backtesting
- Data quality framework inspired by industry best practices
- Progressive enhancement approach for sustainable development