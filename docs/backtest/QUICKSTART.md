# Backtesting Quick Reference

## CLI Commands

```bash
# Run experiment
python -m src.runners.systematic_backtest_runner --spec config/backtest/my_experiment.yaml

# Dry run
python -m src.runners.systematic_backtest_runner --spec config/backtest/my_experiment.yaml --dry-run

# List strategies
python -m src.runners.backtest_runner --list-strategies
```

## Python API

```python
from src.backtest import (
    ExperimentSpec, SystematicRunner, RunnerConfig,
    WalkForwardSplitter, SplitConfig,
    PBOCalculator, DSRCalculator, MonteCarloSimulator,
)

# Load and run experiment
spec = ExperimentSpec.from_yaml("experiment.yaml")
runner = SystematicRunner(RunnerConfig(parallel_workers=4))
exp_id = runner.run(spec, backtest_fn)

# Walk-forward splitting
config = SplitConfig(train_days=252, test_days=63, folds=5, purge_days=5)
splitter = WalkForwardSplitter(config)
for train_window, test_window in splitter.split("2020-01-01", "2024-12-31"):
    # train_window, test_window are TimeWindow objects
    pass

# Statistical validation
pbo = PBOCalculator().calculate(is_sharpes, oos_sharpes)
dsr, pval = DSRCalculator().calculate(sharpe, n_trials, n_observations)
mc = MonteCarloSimulator(n_simulations=1000).reshuffle_trades(trades_df)

# VectorBT fast screening (100-1000x faster)
from src.backtest.execution import VectorBTEngine, VectorBTConfig
engine = VectorBTEngine(VectorBTConfig(strategy_type="ma_cross"))
result = engine.run(run_spec, data=price_df)
results = engine.run_batch(specs, {"AAPL": aapl_df})  # Vectorized

# Parity testing between engines
from src.backtest.execution import StrategyParityHarness, ParityConfig
harness = StrategyParityHarness(reference_engine, test_engine)
parity = harness.compare(run_spec)
assert parity.is_parity, parity.summary
```

## YAML Spec Template

```yaml
name: "My_Experiment"
strategy: "strategy_name"

parameters:
  param1: {type: range, min: 10, max: 50, step: 10}
  param2: {type: categorical, values: [a, b, c]}
  param3: {type: fixed, value: 100}

universe:
  type: static
  symbols: [AAPL, MSFT, GOOGL]

temporal:
  primary_method: walk_forward
  train_days: 252
  test_days: 63
  folds: 5
  purge_days: 5
  embargo_days: 2
  label_horizon_days: 0
  start_date: "2020-01-01"
  end_date: "2025-12-30"

optimization:
  method: grid
  metric: sharpe
  direction: maximize
  constraints:
    - {metric: p10_sharpe, operator: ">=", value: 0.0}
    - {metric: median_max_dd, operator: "<=", value: 0.20}

reproducibility:
  random_seed: 42
  data_version: "v1.0"
```

## Key Metrics

| Metric | Good Value | Description |
|--------|------------|-------------|
| `median_sharpe` | > 1.0 | Median Sharpe ratio |
| `p10_sharpe` | > 0.0 | 10th percentile (worst-case) |
| `median_max_dd` | < 0.20 | Median max drawdown |
| `pbo` | < 0.25 | Probability of overfit |
| `dsr` | > 0.95 | Deflated Sharpe significance |
| `degradation_ratio` | < 0.40 | IS→OOS performance drop |

## Example Files

```
config/backtest/examples/
├── 01_basic_wfo.yaml           # Basic walk-forward
├── 02_cpcv_validation.yaml     # CPCV for PBO
├── 03_label_horizon.yaml       # Multi-day trades
├── 04_monte_carlo.yaml         # Path dependency
├── 05_strict_validation.yaml   # Strict overfitting controls
├── 06_vectorbt_screening.yaml  # VectorBT fast screening
└── ta_metrics_experiment.yaml  # Full TA strategy
```

## Query Results (DuckDB)

```sql
-- Top trials
SELECT trial_id, median_sharpe, p10_sharpe, median_max_dd
FROM trials WHERE experiment_id = ? ORDER BY median_sharpe DESC LIMIT 10;

-- Run details
SELECT symbol, window_id, sharpe, is_oos FROM runs WHERE trial_id = ?;

-- Aggregate by symbol
SELECT symbol, AVG(sharpe), COUNT(*) FROM runs WHERE experiment_id = ? GROUP BY symbol;
```
