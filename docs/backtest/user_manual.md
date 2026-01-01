# Apex Backtesting Framework - User Manual

This manual covers the systematic backtesting framework for strategy experimentation, validation, and optimization.

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Configuration](#2-configuration)
3. [Running Experiments](#3-running-experiments)
4. [Data Splitting](#4-data-splitting)
5. [Statistical Validation](#5-statistical-validation)
6. [Results & Analysis](#6-results--analysis)
7. [Strategy Development](#7-strategy-development)
8. [Advanced Topics](#8-advanced-topics)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Start

### 1.1 Installation

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Verify installation
python -c "from src.backtest import SystematicRunner; print('OK')"
```

### 1.2 First Experiment (5-minute tutorial)

Create a YAML spec file `my_experiment.yaml`:

```yaml
name: "My_First_Experiment"
strategy: "ma_cross"

parameters:
  fast_period: {type: range, min: 10, max: 30, step: 5}
  slow_period: {type: range, min: 50, max: 100, step: 10}

universe:
  type: static
  symbols: ["AAPL", "MSFT", "GOOGL"]

temporal:
  primary_method: walk_forward
  train_days: 252
  test_days: 63
  folds: 3
  purge_days: 5

optimization:
  method: grid
  metric: sharpe

reproducibility:
  random_seed: 42
  data_version: "v1.0"
```

Run the experiment:

```bash
python -m src.runners.systematic_backtest_runner --spec my_experiment.yaml
```

### 1.3 Running from CLI

```bash
# Run experiment from YAML
python -m src.runners.systematic_backtest_runner --spec config/backtest/my_experiment.yaml

# Dry run (show what would execute)
python -m src.runners.systematic_backtest_runner --spec config/backtest/my_experiment.yaml --dry-run

# List available strategies
python -m src.runners.backtest_runner --list-strategies

# Run built-in example
python -m src.runners.systematic_backtest_runner --spec config/backtest/examples/ta_metrics_experiment.yaml
```

---

## 2. Configuration

### 2.1 Experiment Specification (YAML format)

An experiment spec defines everything needed for systematic backtesting:

```yaml
name: "Experiment_Name"           # Required: unique experiment name
description: "Optional description"
strategy: "strategy_name"         # Required: registered strategy name

parameters: {...}                 # Parameter search space
universe: {...}                   # Symbol universe configuration
temporal: {...}                   # Time splits (WFO, CPCV)
optimization: {...}               # Optimization method and constraints
profiles: {...}                   # Execution profiles (costs, slippage)
reproducibility: {...}            # Random seed, data version
```

### 2.2 Parameter Definitions

Three parameter types are supported:

```yaml
parameters:
  # Range: generates [10, 15, 20, 25, 30]
  lookback:
    type: range
    min: 10
    max: 30
    step: 5

  # Categorical: discrete choices
  position_sizing:
    type: categorical
    values: ["equal", "risk_parity", "volatility_inverse"]

  # Fixed: single value (not optimized)
  stop_loss:
    type: fixed
    value: 0.02
```

### 2.3 Universe Configuration

```yaml
# Static universe (explicit symbols)
universe:
  type: static
  symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# Dynamic universe (rules-based, future feature)
universe:
  type: dynamic
  rules: "top_100_by_adv"
  as_of: "split_start"
  filters:
    min_price: 5.0
    min_adv: 1000000

# Index-based universe (future feature)
universe:
  type: index
  index: "SP500"
```

### 2.4 Temporal Settings

Walk-forward optimization with purge/embargo gaps:

```yaml
temporal:
  primary_method: walk_forward    # or "expanding", "sliding"
  train_days: 252                 # Training window (1 year)
  test_days: 63                   # Test window (3 months)
  step_days: 63                   # Step size (defaults to test_days)
  folds: 5                        # Number of WFO folds

  # Gap settings for preventing look-ahead bias
  purge_days: 5                   # Gap between train and test
  embargo_days: 2                 # Gap after test
  label_horizon_days: 0           # Trade resolution horizon

  # Date range
  start_date: "2020-01-01"
  end_date: "2025-12-30"
```

**Timeline visualization:**

```
|--- TRAIN ---|-- PURGE --|--- TEST ---|-- EMBARGO --|--- NEXT TRAIN ...
     252d          5d          63d           2d
```

### 2.5 Optimization Methods

```yaml
# Grid search (exhaustive)
optimization:
  method: grid
  metric: sharpe
  direction: maximize
  constraints:
    - metric: p10_sharpe
      operator: ">="
      value: 0.0
    - metric: median_max_dd
      operator: "<="
      value: 0.20

# Bayesian optimization (for large parameter spaces)
optimization:
  method: bayesian
  sampler: TPE
  pruner: ASHA
  n_trials: 200
  timeout_hours: 8
  metric: sharpe
  direction: maximize
```

### 2.6 Execution Profiles

Define cost assumptions for screening vs. validation:

```yaml
profiles:
  fast_track:
    name: screening
    version: v1
    slippage_bps: 0              # No slippage for fast screening
    commission_per_share: 0
    fill_model: close

  realistic:
    name: validation
    version: v1
    slippage_bps: 10.0           # 10 bps slippage
    commission_per_share: 0.01   # $0.01 per share
    min_commission: 1.0          # $1.00 minimum
    fill_model: vwap
    volume_limit_pct: 0.05       # Max 5% of volume
```

---

## 3. Running Experiments

### 3.1 CLI Usage

```bash
# Basic run
python -m src.runners.systematic_backtest_runner \
    --spec config/backtest/my_experiment.yaml

# With parallel workers
python -m src.runners.systematic_backtest_runner \
    --spec config/backtest/my_experiment.yaml \
    --workers 8

# Store results in specific database
python -m src.runners.systematic_backtest_runner \
    --spec config/backtest/my_experiment.yaml \
    --db results/experiments.db
```

### 3.2 Python API

```python
from src.backtest import ExperimentSpec, SystematicRunner, RunnerConfig

# Load specification
spec = ExperimentSpec.from_yaml("config/backtest/my_experiment.yaml")

# Configure runner
config = RunnerConfig(
    parallel_workers=4,
    db_path="results/backtest.db",
)

# Create and run
runner = SystematicRunner(config)
experiment_id = runner.run(
    spec=spec,
    backtest_fn=my_backtest_function,  # Your backtest implementation
)

print(f"Experiment completed: {experiment_id}")
```

### 3.3 Parallel Execution

The framework automatically parallelizes across CPU cores:

```python
from src.backtest import ParallelRunner, ParallelConfig

config = ParallelConfig(
    max_workers=8,           # CPU cores to use
    chunk_size=10,           # Runs per batch
    timeout_per_run=300,     # 5 min timeout per run
    max_retries=2,           # Retry transient failures
    retry_base_delay=1.0,    # Exponential backoff base
)

runner = ParallelRunner(config)
results = runner.run_all(run_specs, backtest_fn)
```

**Retry Policy:**
- Transient errors (timeout, resource exhaustion) are automatically retried
- Deterministic errors (ValueError, TypeError) fail immediately
- Exponential backoff with jitter prevents thundering herd

---

## 4. Data Splitting

### 4.1 Walk-Forward Optimization

Walk-forward validation trains on historical data and tests on subsequent periods:

```python
from src.backtest import WalkForwardSplitter, SplitConfig

config = SplitConfig(
    train_days=252,          # 1 year training
    test_days=63,            # 3 months testing
    folds=5,                 # 5 walk-forward folds
    purge_days=5,            # Gap between train/test
    embargo_days=2,          # Gap after test
    label_horizon_days=20,   # For multi-day trades
)

splitter = WalkForwardSplitter(config)

for train_window, test_window in splitter.split("2020-01-01", "2024-12-31"):
    # train_window: fit model on this period
    # test_window: validate on this period
    print(f"Train: {train_window.train_start} to {train_window.train_end}")
    print(f"Test: {test_window.test_start} to {test_window.test_end}")
```

**Key parameters:**
- `purge_days`: Minimum gap to prevent data leakage
- `embargo_days`: Gap after test for model decay
- `label_horizon_days`: For strategies with multi-day trades

### 4.2 Combinatorial Purged Cross-Validation (CPCV)

CPCV provides more paths for robust PBO calculation:

```python
from src.backtest import CPCVSplitter, CPCVConfig

config = CPCVConfig(
    n_groups=8,              # Split data into 8 groups
    n_test_groups=2,         # 2 groups for test each path
    purge_days=5,
    embargo_days=2,
)

splitter = CPCVSplitter(config)
print(f"Total paths: {splitter.get_path_count()}")  # C(8,2) = 28 paths

for train_windows, test_windows, path in splitter.split("2020-01-01", "2024-12-31"):
    # Each path has multiple train and test windows
    pass
```

### 4.3 Trading Calendar Integration

The framework uses trading calendars for accurate day counting:

```python
from src.backtest import get_calendar, WeekdayCalendar

# Get NYSE calendar (if pandas-market-calendars installed)
calendar = get_calendar("NYSE")

# Fallback to weekday-only calendar
calendar = WeekdayCalendar()

# Count trading days
trading_days = calendar.count_trading_days("2024-01-01", "2024-12-31")
```

---

## 5. Statistical Validation

### 5.1 Probability of Backtest Overfit (PBO)

PBO measures the probability that the best in-sample strategy underperforms out-of-sample:

```python
from src.backtest import PBOCalculator

calc = PBOCalculator(n_simulations=1000)

# IS and OOS Sharpe ratios for each strategy variant
is_sharpes = [1.5, 1.2, 0.9, 1.1, 0.8]
oos_sharpes = [0.8, 1.0, 0.7, 0.9, 0.6]

pbo = calc.calculate(is_sharpes, oos_sharpes)

print(f"PBO: {pbo:.2%}")
# Interpretation:
# - PBO < 25%: Low overfitting risk
# - PBO 25-50%: Moderate risk
# - PBO > 50%: High overfitting risk
```

### 5.2 Deflated Sharpe Ratio (DSR)

DSR adjusts for multiple testing bias:

```python
from src.backtest import DSRCalculator

calc = DSRCalculator()

# Calculate DSR
dsr, p_value = calc.calculate(
    observed_sharpe=1.5,     # Best observed Sharpe
    n_trials=100,            # Number of strategies tested
    n_observations=756,      # 3 years of daily returns
    skewness=0.0,            # Return skewness
    kurtosis=3.0,            # Return kurtosis
)

print(f"DSR: {dsr:.2%}")
print(f"P-value: {p_value:.4f}")

# From returns directly
dsr, p_value = calc.calculate_from_returns(
    returns=daily_returns,   # numpy array
    n_trials=100,
    annualization_factor=252,
)
```

**Interpretation:**
- DSR > 95%: High confidence the strategy has genuine skill
- DSR 80-95%: Moderate confidence
- DSR < 80%: May be data mining artifact

### 5.3 Monte Carlo Trade Reshuffling

Test if equity curve could arise by chance:

```python
import pandas as pd
from src.backtest import MonteCarloSimulator

sim = MonteCarloSimulator(n_simulations=1000, seed=42)

trades = pd.DataFrame({"pnl": [100, -50, 200, -30, 150, -80]})
result = sim.reshuffle_trades(trades, initial_equity=10000)

print(f"Original return: {result.original_total_return:.2%}")
print(f"P-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")

# Equity curve percentiles for plotting
# result.equity_p5, result.equity_p50, result.equity_p95
```

---

## 6. Results & Analysis

### 6.1 Understanding Trial Aggregates

Each trial (parameter combination) produces aggregated statistics:

| Metric | Description |
|--------|-------------|
| `median_sharpe` | Median Sharpe across all runs |
| `p10_sharpe` | 10th percentile (worst-case) |
| `p90_sharpe` | 90th percentile (best-case) |
| `mad_sharpe` | Median Absolute Deviation |
| `median_max_dd` | Median maximum drawdown |
| `stability_score` | Combined stability metric |
| `degradation_ratio` | IS vs OOS performance gap |
| `is_median_sharpe` | In-sample median |
| `oos_median_sharpe` | Out-of-sample median |

### 6.2 Querying DuckDB Results

```python
import duckdb

# Connect to results database
conn = duckdb.connect("results/backtest.db")

# Top trials by Sharpe
top_trials = conn.execute("""
    SELECT
        trial_id,
        params,
        median_sharpe,
        p10_sharpe,
        median_max_dd,
        stability_score
    FROM trials
    WHERE experiment_id = ?
    ORDER BY median_sharpe DESC
    LIMIT 10
""", [experiment_id]).fetchdf()

# Individual run metrics
runs = conn.execute("""
    SELECT
        symbol,
        window_id,
        sharpe,
        max_drawdown,
        total_trades,
        is_oos
    FROM runs
    WHERE trial_id = ?
    ORDER BY window_id
""", [trial_id]).fetchdf()

# Aggregate by symbol
symbol_stats = conn.execute("""
    SELECT
        symbol,
        AVG(sharpe) as avg_sharpe,
        MIN(sharpe) as min_sharpe,
        MAX(sharpe) as max_sharpe,
        COUNT(*) as n_runs
    FROM runs
    WHERE experiment_id = ?
    GROUP BY symbol
""", [experiment_id]).fetchdf()
```

### 6.3 Constraint Validation

```python
from src.backtest import ConstraintValidator, Constraint

validator = ConstraintValidator()

constraints = [
    Constraint(metric="p10_sharpe", operator=">=", value=0.0),
    Constraint(metric="median_max_dd", operator="<=", value=0.20),
    Constraint(metric="total_trades", operator=">=", value=10),
]

# Validate trial results
passes, failures = validator.validate(trial_aggregates, constraints)

for failure in failures:
    print(f"Failed: {failure.metric} {failure.operator} {failure.value}")
```

---

## 7. Strategy Development

### 7.1 Registering a New Strategy

```python
from src.domain.strategy.base import Strategy
from src.domain.strategy.registry import register_strategy

@register_strategy("my_strategy", description="My custom strategy")
class MyStrategy(Strategy):

    def __init__(self, fast_period: int = 10, slow_period: int = 50):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_bar(self, bar):
        # Calculate indicators
        fast_ma = self.indicators.sma(bar.close, self.fast_period)
        slow_ma = self.indicators.sma(bar.close, self.slow_period)

        # Generate signals
        if fast_ma > slow_ma and not self.has_position(bar.symbol):
            self.request_order(
                symbol=bar.symbol,
                side="BUY",
                quantity=100,
                order_type="MARKET"
            )
        elif fast_ma < slow_ma and self.has_position(bar.symbol):
            self.request_order(
                symbol=bar.symbol,
                side="SELL",
                quantity=100,
                order_type="MARKET"
            )
```

### 7.2 Strategy Interface

Key methods available in strategies:

```python
class Strategy:
    # Market data
    def on_bar(self, bar: BarData) -> None: ...
    def on_tick(self, tick: TickData) -> None: ...

    # Order management
    def request_order(self, order: OrderRequest) -> None: ...
    def cancel_order(self, order_id: str) -> None: ...

    # Position queries
    def has_position(self, symbol: str) -> bool: ...
    def get_position(self, symbol: str) -> Position: ...

    # Indicator access
    self.indicators.sma(data, period)
    self.indicators.ema(data, period)
    self.indicators.rsi(data, period)
    self.indicators.atr(high, low, close, period)
```

---

## 8. Advanced Topics

### 8.1 Label Horizon for Multi-Day Trades

For swing trading strategies where trades take multiple days to resolve:

```yaml
temporal:
  train_days: 504
  test_days: 126
  purge_days: 5            # Minimum purge
  label_horizon_days: 20   # Trades may take 20 days to resolve
  # Effective purge = max(purge_days, label_horizon_days) = 20 days
```

This prevents data leakage when trade outcomes depend on future prices.

### 8.2 Code Version Tracking

Experiment IDs include the git commit SHA for reproducibility:

```python
from src.backtest import get_git_sha, generate_experiment_id

# Auto-detected from git
sha = get_git_sha()  # e.g., "57eabc7f"

# Included in experiment ID generation
exp_id = generate_experiment_id(
    name="test",
    strategy="ma_cross",
    parameters={...},
    universe={...},
    temporal={...},
    data_version="v1",
    code_version=sha,  # Optional, auto-detected if None
)
```

### 8.3 Thread-Safe Database Writes

For parallel experiments, use the WriteQueue:

```python
from src.backtest.data.storage import DatabaseManager, WriteQueue, WriterConfig

db = DatabaseManager("results.db")
db.initialize_schema()

config = WriterConfig(
    batch_size=100,              # Records per batch
    batch_timeout_seconds=1.0,   # Flush timeout
    max_retries=3,               # Retry on failure
)

with WriteQueue(db, config) as queue:
    # Safe to call from multiple threads
    queue.insert("runs", {"run_id": "run_123", "sharpe": 1.5})
    queue.insert("runs", {"run_id": "run_124", "sharpe": 0.8})

    # Explicit flush
    queue.flush()
```

### 8.4 Custom Trading Calendars

```python
from src.backtest.data.calendar import TradingCalendar
from datetime import date

class LSECalendar(TradingCalendar):
    """London Stock Exchange calendar."""

    @property
    def name(self) -> str:
        return "LSE"

    def is_trading_day(self, d: date) -> bool:
        # Implement LSE holiday logic
        if d.weekday() >= 5:  # Weekend
            return False
        # Add UK holidays...
        return True
```

### 8.5 VectorBT Engine (Fast Screening)

VectorBT provides 100-1000x faster backtesting using NumPy vectorization. Use it for:
- Fast parameter space exploration (10,000+ combinations)
- Initial screening before detailed validation
- Simple strategies (MA crossover, RSI, momentum)

```python
from src.backtest.execution import VectorBTEngine, VectorBTConfig

# Create engine with configuration
config = VectorBTConfig(
    strategy_type="ma_cross",   # Built-in strategy
    init_cash=100000.0,
    freq="1D",
)
engine = VectorBTEngine(config)

# Single run
result = engine.run(run_spec, data=price_data)

# Vectorized batch (much faster for same symbol)
results = engine.run_batch(run_specs, {"AAPL": aapl_data})
```

**Built-in Strategies:**
- `ma_cross`: Moving average crossover (fast_period, slow_period)
- `rsi`: RSI overbought/oversold (rsi_period, rsi_oversold, rsi_overbought)
- `momentum`: Price momentum (lookback_days, momentum_threshold)

**Register Custom Strategy:**
```python
def my_signals(data, params):
    """Return (entries, exits) boolean Series."""
    close = data["close"]
    threshold = params.get("threshold", 0.02)
    returns = close.pct_change()
    entries = returns > threshold
    exits = returns < -threshold
    return entries, exits

engine.register_strategy("my_strategy", my_signals)
```

### 8.6 Parity Harness (Engine Comparison)

The parity harness detects drift between different execution paths:

```python
from src.backtest.execution import (
    StrategyParityHarness,
    ParityConfig,
    VectorBTEngine,
)

# Create engines to compare
vectorbt_engine = VectorBTEngine()
apex_engine = ApexEngine()  # Full-featured engine

# Configure tolerances
config = ParityConfig(
    sharpe_tolerance=0.05,      # 5% relative difference allowed
    return_tolerance=0.01,      # 1% absolute return difference
    max_dd_tolerance=0.02,      # 2% drawdown difference
    trade_count_tolerance=2,    # Allow 2 trade difference
)

# Create harness
harness = StrategyParityHarness(
    reference_engine=apex_engine,
    test_engine=vectorbt_engine,
    config=config,
)

# Compare single run
parity = harness.compare(run_spec, data=price_data)

if not parity.is_parity:
    print(f"Parity failed: {parity.summary}")
    for drift in parity.drift_detected:
        print(f"  {drift.field}: {drift.message}")

# Batch comparison
results = harness.compare_batch(run_specs, data_dict)
failures = [r for r in results if not r.is_parity]

# Generate report
report = harness.generate_report(results)
print(report)
```

**Drift Types:**
| Type | Critical | Description |
|------|----------|-------------|
| `STATUS_MISMATCH` | Yes | Different execution status |
| `PNL_MISMATCH` | Yes | Returns beyond tolerance |
| `PRICE_EXECUTION` | Yes | Different fill prices |
| `TRADE_COUNT` | No | Different number of trades |
| `METRIC_MISMATCH` | No | Computed metrics differ |

### 8.7 Two-Stage Pipeline

Use VectorBT for fast screening, then validate top candidates with the full engine:

```yaml
# config/backtest/examples/06_two_stage.yaml
name: "Two_Stage_Pipeline"
strategy: "ma_cross"

parameters:
  fast_period: {type: range, min: 5, max: 50, step: 5}    # 10 values
  slow_period: {type: range, min: 50, max: 200, step: 10}  # 16 values
  # = 160 combinations per symbol

optimization:
  method: two_stage

  screening:
    engine: vectorbt        # Fast screening
    profile: fast_track     # No slippage/commission
    metric: sharpe
    top_n: 20               # Keep top 20 candidates

  validation:
    engine: apex            # Full validation
    profile: realistic      # With execution costs
    metric: stability_score
```

```python
from src.backtest.execution import VectorBTEngine, SystematicRunner

# Stage 1: Fast screening with VectorBT
vectorbt = VectorBTEngine()
screening_results = vectorbt.run_batch(all_specs, data)

# Sort and select top candidates
ranked = sorted(screening_results, key=lambda r: r.metrics.sharpe, reverse=True)
top_candidates = ranked[:20]

# Stage 2: Full validation with Apex engine
validation_results = apex_engine.run_batch(
    [create_spec(r) for r in top_candidates],
    data
)
```

---

## 9. Troubleshooting

### 9.1 Common Errors

**"Cannot import scipy"**
```bash
pip install scipy>=1.11.0
```

**"DuckDB not found"**
```bash
pip install duckdb>=0.9.0
```

**"Strategy not registered"**
```python
# Ensure strategy module is imported before running
from src.strategies import ma_cross  # Registers @register_strategy
```

**"Insufficient data for folds"**
- Reduce `train_days` or `test_days`
- Reduce number of `folds`
- Expand date range

### 9.2 Performance Tuning

**Slow parallel execution:**
```python
config = ParallelConfig(
    max_workers=min(8, cpu_count() - 1),  # Leave 1 core free
    chunk_size=20,                         # Larger batches
)
```

**High memory usage:**
- Reduce `max_workers`
- Use smaller date ranges per experiment
- Enable database write queue (batches writes)

### 9.3 Memory Management

For large experiments:

```python
from src.backtest import ParallelRunner, WriteQueue

# Use streaming mode for very large experiments
runner = ParallelRunner(config)
for result in runner.run_streaming(run_specs, backtest_fn):
    # Process results incrementally
    write_queue.insert("runs", result.to_dict())
```

---

## Appendix A: Full Configuration Reference

```yaml
# Complete experiment specification
name: string                    # Required
description: string             # Optional
strategy: string                # Required

parameters:
  param_name:
    type: range | categorical | fixed
    min: float                  # For range
    max: float                  # For range
    step: float                 # For range
    values: list                # For categorical
    value: any                  # For fixed

universe:
  type: static | dynamic | index
  symbols: list[string]         # For static
  rules: string                 # For dynamic
  as_of: string                 # For dynamic
  index: string                 # For index

temporal:
  primary_method: walk_forward | expanding | sliding
  train_days: int
  test_days: int
  step_days: int                # Optional
  folds: int
  purge_days: int
  embargo_days: int
  label_horizon_days: int       # Optional, default 0
  secondary_method: cpcv | monte_carlo | none
  cpcv_groups: int
  cpcv_test_groups: int
  start_date: string            # YYYY-MM-DD
  end_date: string              # YYYY-MM-DD

optimization:
  method: grid | bayesian | random
  metric: string
  direction: maximize | minimize
  sampler: string               # For bayesian
  pruner: string                # For bayesian
  n_trials: int                 # For bayesian
  timeout_hours: float          # For bayesian
  constraints:
    - metric: string
      operator: ">=" | "<=" | ">" | "<" | "=="
      value: float

profiles:
  profile_name:
    name: string
    version: string
    slippage_bps: float
    commission_per_share: float
    min_commission: float
    fill_model: string
    volume_limit_pct: float
    market_impact_model: string

reproducibility:
  random_seed: int
  data_version: string
  code_version: string          # Optional, auto-detected
```

---

## Appendix B: Metrics Reference

| Metric | Description | Range |
|--------|-------------|-------|
| `sharpe` | Sharpe ratio (annualized) | -inf to +inf |
| `sortino` | Sortino ratio (downside risk) | -inf to +inf |
| `calmar` | Calmar ratio (return/max DD) | -inf to +inf |
| `total_return` | Total return percentage | -1 to +inf |
| `cagr` | Compound annual growth rate | -1 to +inf |
| `max_drawdown` | Maximum drawdown | -1 to 0 |
| `avg_drawdown` | Average drawdown | -1 to 0 |
| `win_rate` | Winning trade percentage | 0 to 1 |
| `profit_factor` | Gross profit / gross loss | 0 to +inf |
| `expectancy` | Expected value per trade | -inf to +inf |
| `sqn` | System Quality Number | -inf to +inf |
| `total_trades` | Number of trades | 0 to +inf |
| `exposure_pct` | Time in market percentage | 0 to 1 |
