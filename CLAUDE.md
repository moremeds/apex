# CLAUDE.md

## Code Architecture Standards

### Hard Requirements 硬性指标

1. **File Length Limits**
   - Dynamic languages (Python, JavaScript, TypeScript): Maximum 200 lines per file
   - Static languages (Java, Go, Rust): Maximum 250 lines per file
   - Maximum 8 files per directory - create subdirectories when exceeded

### BAD TASTE to  AVOID

1.  除了硬性指标以外，还需要时刻关注优雅的架构设计，避免出现以下可能侵蚀我们代码质量的「坏味道」：
  （1）僵化 (Rigidity): 系统难以变更，任何微小的改动都会引发一连串的连锁修改。
  （2）冗余 (Redundancy): 同样的代码逻辑在多处重复出现，导致维护困难且容易产生不一致。
  （3）循环依赖 (Circular Dependency): 两个或多个模块互相纠缠，形成无法解耦的“死结”，导致难以测试与复用。
  （4）脆弱性 (Fragility): 对代码一处的修改，导致了系统中其他看似无关部分功能的意外损坏。
  （5）晦涩性 (Obscurity): 代码意图不明，结构混乱，导致阅读者难以理解其功能和设计。
  （6）数据泥团 (Data Clump): 多个数据项总是一起出现在不同方法的参数中，暗示着它们应该被组合成一个独立的对象。
  （7）不必要的复杂性 (Needless Complexity): 用“杀牛刀”去解决“杀鸡”的问题，过度设计使系统变得臃肿且难以理解。

- 【非常重要！！】无论是你自己编写代码，还是阅读或审核他人代码时，都要严格遵守上述硬性指标，以及时刻关注优雅的架构设计。
- 【非常重要！！】无论何时，一旦你识别出那些可能侵蚀我们代码质量的「坏味道」，都应当立即询问用户是否需要优化，并给出合理的优化建议
 
### Critical Rules
- **ALWAYS** enforce file length limits and directory organization
- **IMMEDIATELY** flag any code bad taste and propose optimizations
- **NEVER** compromise on these architectural standards


## Project Overview

Apex is a high-performance strategy backtesting system designed for quantitative trading research. The project follows a production-ready architecture with emphasis on speed (vectorbt-first approach), data quality, and progressive enhancement from MVP to full ML capabilities.

## Development Commands

### Environment Setup
```bash
# Install Python 3.12+ and uv first, then:
uv venv                              # Create virtual environment
source .venv/bin/activate            # Activate it (Unix/macOS)
uv sync                              # Sync dependencies with lock file
uv run python scripts/setup_environment.py  # Complete environment setup
uv run python scripts/download_sample_data.py  # Download test data
```

### Testing
```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=apex --cov-report=html

# Run specific test categories
uv run pytest tests/unit/ -v               # Unit tests only
uv run pytest tests/integration/ -v        # Integration tests
uv run pytest tests/performance/ -v        # Performance benchmarks

# Run a single test file
uv run pytest tests/unit/test_data/test_quality.py -v

# Run with specific markers
uv run pytest -m "not slow" -v            # Skip slow tests
```

### Code Quality
```bash
# Format code
uv run black src/ tests/
uv run ruff check --fix src/

# Type checking
uv run mypy src/ --strict

# Linting
uv run ruff check src/
```

### Development Workflow
```bash
make venv          # Create virtual environment with uv
make sync          # Sync dependencies with uv.lock
make install-dev   # Install development environment with uv
make test          # Run test suite with uv
make lint          # Run linting and type checking with uv
make format        # Format code with uv
make clean         # Clean build artifacts
make run-backtest  # Run sample backtest with uv
```

## Task Management (Task Master)

This project uses Task Master AI for task management. Key commands:

```bash
# View and manage tasks
/project:tm/status                  # Project dashboard
/project:tm/next                    # Get next recommended task
/project:tm/list                    # List all tasks
/project:tm/show <id>              # Show task details

# Update task status
/project:tm/set-status/to-in-progress <id>
/project:tm/set-status/to-done <id>

# Parse PRD and generate tasks
/project:tm/parse-prd backtester-prd-final.md

# Run workflows
/project:tm/workflows/smart-flow    # Adaptive workflow
/project:tm/workflows/auto-implement # AI implementation
```

## Architecture Overview

### Core Layers

1. **Data Layer** (`src/apex/data/`)
   - **Providers**: Yahoo Finance, CCXT (crypto), CSV loaders
   - **Quality Framework**: Comprehensive scoring system with 5 quality dimensions
   - **Caching**: Redis-backed caching with file fallback
   - **Feature Store**: Technical indicators and ML features

2. **Strategy Layer** (`src/apex/strategies/`)
   - **Engine-Specific**: Strategies are written for specific engines (vectorbt, backtrader)
   - **Validation**: Walk-forward analysis, overfitting prevention, parameter stability
   - **Optimization**: Grid search with Bayesian optimization planned

3. **Engine Layer** (`src/apex/engine/`)
   - **Harness Pattern**: Unified interface across different backtesting engines
   - **Vectorbt Engine**: Primary engine for speed and vectorized operations
   - **Portfolio Management**: Position sizing, risk management, execution simulation

4. **Analysis Layer** (`src/apex/analysis/`)
   - **Metrics**: Sharpe, Sortino, Calmar ratios, max drawdown, win rate
   - **Visualization**: Plotly-based interactive charts
   - **Reports**: Markdown and HTML report generation

### Key Design Patterns

1. **Engine Harness Pattern**: 
   - Abstracts engine differences behind unified interface
   - Strategies inherit from engine-specific base classes
   - Results returned in standardized format

2. **Data Quality Gates**:
   - Mandatory quality scoring before processing
   - Configurable enforcement (warn/block/fix)
   - Detailed quality reports with recommendations

3. **Progressive Enhancement**:
   - MVP first (weeks 1-9): Basic backtesting with 3 strategies
   - ML features (weeks 10-17): Feature engineering, purged CV
   - Production (weeks 18-26): REST API, monitoring, optimization

## Development Phases & Timeline

### Current Status
Project is in setup phase, implementing foundation based on PRD v4.0.

### Phase 0: Foundation (Weeks 1-3) - CURRENT
- Project structure setup
- Data pipeline with quality scoring
- Core types and base classes

### Phase 1: MVP (Weeks 4-9)
- Vectorbt engine integration
- 3 basic strategies (MA Cross, RSI, Breakout)
- Walk-forward validation
- Basic reporting

### Phase 2: ML & Advanced (Weeks 10-17)
- Feature engineering pipeline
- ML strategy framework
- Purged cross-validation
- Advanced validation (Monte Carlo, statistical significance)

### Phase 3: Production (Weeks 18-22)
- REST API with FastAPI
- Monitoring and alerting
- Performance optimization
- Additional engines (Backtrader)

## Critical Implementation Details

### Data Quality Framework
The system implements a 5-dimensional quality scoring system:
1. **Completeness**: Missing data detection
2. **Consistency**: OHLC relationship validation
3. **Validity**: Outlier detection
4. **Timeliness**: Gap analysis
5. **Uniqueness**: Duplicate detection

Quality scores below 0.7 trigger handling based on enforcement level.

### Strategy Implementation
Strategies must inherit from engine-specific base classes:
- `VectorbtStrategy`: For vectorized strategies
- `BacktraderStrategy`: For event-driven strategies (future)

Each strategy must implement:
- `generate_signals()`: Returns entry/exit signals
- `get_parameters()`: Returns current parameters
- `validate_data()`: Checks data requirements

### Performance Targets
- Simple backtest: < 0.5s per year of data
- ML backtest: < 5s per year of data
- Memory usage: < 1GB per million rows
- API latency (p99): < 100ms

## MCP Servers Configuration

The project has several MCP servers enabled:
- **task-master-ai**: Task management and project coordination
- **sequential-thinking**: Complex problem solving and analysis
- **context7**: Library documentation and best practices
- **codacy**: Code quality analysis

## Important Files & Locations

- **PRD**: `backtester-prd-final.md` - Complete product requirements
- **Task Commands**: `.claude/TM_COMMANDS_GUIDE.md` - Task Master command reference
- **Settings**: `.claude/settings.local.json` - Local Claude settings
- **Data Cache**: `~/.apex/cache/` (default)
- **Results**: `./results/` - Backtest results and reports
- **Configs**: `configs/` - Strategy and environment configurations

## Testing Requirements

### Coverage Targets
- MVP (Week 9): 80% unit test coverage
- ML Phase (Week 17): 85% coverage
- Production (Week 22): 90% coverage

### Test Categories
- **Unit Tests**: Core functionality isolation
- **Integration Tests**: Pipeline and component interaction
- **Performance Tests**: Speed and memory benchmarks
- **Statistical Tests**: Strategy validation correctness

## Common Development Tasks

### Adding a New Strategy
1. Create strategy file in `src/apex/strategies/vectorbt/`
2. Inherit from `VectorbtStrategy`
3. Implement `generate_signals()` method
4. Add configuration in `configs/strategies/`
5. Write unit tests in `tests/unit/test_strategies/`
6. Add to walk-forward validation suite

### Adding a Data Provider
1. Create provider in `src/apex/data/providers/`
2. Inherit from base provider class
3. Implement data fetching and normalization
4. Add quality scoring integration
5. Implement caching logic
6. Write integration tests

### Running a Backtest
```python
from apex.engine import EngineHarness
from apex.strategies.vectorbt import MovingAverageCrossStrategy
from apex.data.providers import YahooDataProvider

# Load data
provider = YahooDataProvider()
data = await provider.fetch_data("SPY", start_date, end_date)

# Configure and run
config = {'initial_cash': 100000, 'commission': 0.001}
harness = EngineHarness(config)
strategy = MovingAverageCrossStrategy(fast_period=20, slow_period=50)
result = harness.run_backtest(strategy, data)
```

## Project Principles

1. **No False Abstractions**: Strategies are engine-specific, not falsely portable
2. **Quality First**: Data quality and validation are mandatory, not optional
3. **Progressive Enhancement**: Start simple (MVP), add complexity iteratively
4. **Performance Focused**: Vectorized operations, async I/O, efficient caching
5. **Production Ready**: Monitoring, logging, error recovery from day one

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
