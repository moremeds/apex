# APEX Strategy Development & Backtest Guide

This guide covers the complete strategy development and backtesting framework in APEX.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Strategy Architecture](#strategy-architecture)
4. [Creating Strategies](#creating-strategies)
5. [Strategy Lifecycle](#strategy-lifecycle)
6. [Clock & Scheduler](#clock--scheduler)
7. [Data Feeds](#data-feeds)
8. [Backtest Configuration](#backtest-configuration)
9. [Running Backtests](#running-backtests)
10. [Backtest Results](#backtest-results)
11. [Example Strategies](#example-strategies)
12. [Best Practices](#best-practices)

---

## Overview

The APEX strategy framework provides:

- **Live/Backtest Parity**: Same strategy code works in both modes
- **Event-Driven Architecture**: React to bars, ticks, fills, and scheduled events
- **Multiple Data Sources**: IB historical, CSV, Parquet files
- **Simulated Execution**: Order matching with slippage and commission models
- **Comprehensive Metrics**: Sharpe, Sortino, drawdown, trade statistics

### Key Design Principles

1. **Abstraction**: Strategies use interfaces, not implementations
2. **Determinism**: Backtests produce identical results on replay
3. **Extensibility**: Easy to add new indicators, data sources, execution models

---

## Quick Start

### Run an Existing Strategy

```bash
# List available strategies
python -m src.runners.backtest_runner --list-strategies

# Run MA Cross on AAPL
python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL \
    --start 2024-01-01 --end 2024-06-30

# Run from spec file
python -m src.runners.backtest_runner --spec config/backtest/ma_cross_example.yaml
```

### CLI Options

```
--strategy NAME       Strategy name from registry
--symbols SYM1,SYM2   Comma-separated symbols
--start YYYY-MM-DD    Backtest start date
--end YYYY-MM-DD      Backtest end date
--capital AMOUNT      Initial capital (default: 100000)
--data-source TYPE    ib | csv | parquet (default: ib)
--data-dir PATH       Directory for CSV/Parquet files
--bar-size SIZE       1m | 5m | 15m | 1h | 1d (default: 1d)
--params KEY=VALUE    Strategy parameters
--spec PATH           YAML spec file path
--list-strategies     List available strategies
-v, --verbose         Enable debug logging
```

---

## Strategy Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     BacktestEngine                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ DataFeed    │  │ SimulatedClock│  │ SimulatedExecution │     │
│  │ (IB/CSV/PQ) │  │              │  │                    │     │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬──────────┘     │
│         │                │                     │                │
│         │         ┌──────┴───────┐             │                │
│         │         │  Scheduler   │             │                │
│         │         └──────┬───────┘             │                │
│         │                │                     │                │
│         └────────────────┼─────────────────────┘                │
│                          │                                      │
│                   ┌──────┴──────┐                               │
│                   │ Strategy    │                               │
│                   │ Context     │                               │
│                   └──────┬──────┘                               │
│                          │                                      │
│                   ┌──────┴──────┐                               │
│                   │  Strategy   │  ◄── Your code here           │
│                   └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### Core Classes

| Class | Purpose |
|-------|---------|
| `Strategy` | Abstract base class for all strategies |
| `StrategyContext` | Injected context with clock, scheduler, positions |
| `Clock` | Time abstraction (System/Simulated) |
| `Scheduler` | Time-based action scheduling |
| `DataFeed` | Historical data loading and streaming |
| `SimulatedExecution` | Order matching and fill simulation |
| `BacktestEngine` | Orchestrates the backtest loop |
| `BacktestResult` | Computed metrics and trade log |

---

## Creating Strategies

### Basic Strategy Template

```python
from typing import List
from src.domain.strategy.base import Strategy, StrategyContext
from src.domain.strategy.registry import register_strategy
from src.domain.events.domain_events import QuoteTick, BarData, TradeFill
from src.domain.interfaces.execution_provider import OrderRequest


@register_strategy(
    "my_strategy",
    description="My custom trading strategy",
    author="Your Name",
    version="1.0",
)
class MyStrategy(Strategy):
    """
    Docstring describing strategy logic.
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        # Custom parameters with defaults
        param1: int = 10,
        param2: float = 0.5,
    ):
        super().__init__(strategy_id, symbols, context)

        # Store parameters
        self.param1 = param1
        self.param2 = param2

        # Initialize state
        self._prices = {s: [] for s in symbols}

    def on_start(self) -> None:
        """Called when strategy starts. Initialize resources here."""
        pass

    def on_stop(self) -> None:
        """Called when strategy stops. Cleanup resources here."""
        pass

    def on_tick(self, tick: QuoteTick) -> None:
        """
        Process each price tick.

        Args:
            tick: QuoteTick with bid/ask/last prices
        """
        pass

    def on_bar(self, bar: BarData) -> None:
        """
        Process each completed bar.

        Args:
            bar: BarData with OHLCV
        """
        symbol = bar.symbol
        self._prices[symbol].append(bar.close)

        # Your strategy logic here
        if self._should_buy(symbol):
            self.request_order(OrderRequest(
                symbol=symbol,
                side="BUY",
                quantity=100,
                order_type="MARKET",
            ))

    def on_fill(self, fill: TradeFill) -> None:
        """
        Handle execution fill.

        Args:
            fill: TradeFill with execution details
        """
        print(f"Filled: {fill.side} {fill.quantity} {fill.symbol} @ {fill.price}")

    def _should_buy(self, symbol: str) -> bool:
        """Custom logic to determine buy signal."""
        # Implement your signal logic
        return False
```

### Registering Strategies

Use the `@register_strategy` decorator:

```python
@register_strategy(
    "strategy_name",           # Unique identifier
    description="Description", # Shown in --list-strategies
    author="Author Name",      # Optional
    version="1.0",             # Optional
)
class MyStrategy(Strategy):
    pass
```

The strategy is automatically discovered when the module is imported.

### Accessing Strategy Context

The `StrategyContext` provides access to:

```python
# Time operations
current_time = self.context.now()           # Current datetime
timestamp = self.context.clock.timestamp()  # Unix timestamp

# Position information
position = self.context.get_position("AAPL")
quantity = self.context.get_position_quantity("AAPL")
has_pos = self.context.has_position("AAPL")
is_long = self.context.is_long("AAPL")

# Market data
quote = self.context.get_quote("AAPL")
mid = self.context.get_mid_price("AAPL")

# Scheduling (see Scheduler section)
self.context.scheduler.schedule_daily(...)
```

### Submitting Orders

Two methods for order submission:

#### 1. Direct Order Request

```python
from src.domain.interfaces.execution_provider import OrderRequest

# Market order
order = OrderRequest(
    symbol="AAPL",
    side="BUY",           # "BUY" or "SELL"
    quantity=100,
    order_type="MARKET",
    client_order_id="my-order-001",  # Optional
)
self.request_order(order)

# Limit order
order = OrderRequest(
    symbol="AAPL",
    side="BUY",
    quantity=100,
    order_type="LIMIT",
    limit_price=149.50,
)
self.request_order(order)
```

#### 2. Trading Signal (for risk layer integration)

```python
from src.domain.strategy.base import TradingSignal

signal = TradingSignal(
    signal_id="sig-001",
    symbol="AAPL",
    direction="LONG",      # "LONG", "SHORT", "FLAT"
    strength=0.8,          # 0.0 to 1.0
    target_quantity=100,
    target_price=150.0,
    reason="MA crossover",
    timestamp=self.context.now(),
)
self.emit_signal(signal)
```

---

## Strategy Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                           │
│    strategy = MyStrategy(id, symbols, context, **params)   │
│    State: INITIALIZED                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. START                                                    │
│    strategy.start()                                         │
│    → Calls on_start()                                       │
│    State: RUNNING                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. RUNNING (main loop)                                      │
│    For each bar/tick:                                       │
│      → on_bar(bar) or on_tick(tick)                        │
│      → on_fill(fill) for executions                        │
│      → Scheduled callbacks fire                            │
│    State: RUNNING                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. STOP                                                     │
│    strategy.stop()                                          │
│    → Calls on_stop()                                        │
│    State: STOPPED                                           │
└─────────────────────────────────────────────────────────────┘
```

### Event Methods

| Method | When Called | Use Case |
|--------|-------------|----------|
| `on_start()` | Strategy starts | Initialize state, schedule actions |
| `on_tick(tick)` | Each price tick | High-frequency signal logic |
| `on_bar(bar)` | Each completed bar | Daily/hourly strategy logic |
| `on_fill(fill)` | Order executed | Track trades, update state |
| `on_stop()` | Strategy stops | Cleanup, logging |

---

## Clock & Scheduler

### Clock Abstraction

The clock enables live/backtest parity:

```python
# In strategy code (works in both modes)
current_time = self.context.now()
```

| Mode | Clock Type | Behavior |
|------|------------|----------|
| Live | `SystemClock` | Real system time |
| Backtest | `SimulatedClock` | Advances with bar timestamps |

### Scheduler

Schedule time-based actions:

```python
from datetime import time

def on_start(self):
    # Daily action at specific time
    self.context.scheduler.schedule_daily(
        action_id="daily_rebalance",
        callback=self.rebalance,
        time_of_day=time(15, 55),  # 3:55 PM
    )

    # Action after each bar close
    self.context.scheduler.schedule_on_bar_close(
        action_id="check_risk",
        callback=self.check_risk_limits,
    )

def rebalance(self):
    """Called daily at 3:55 PM in both live and backtest."""
    pass

def check_risk_limits(self):
    """Called after each bar."""
    pass
```

### Schedule Types

| Method | Description |
|--------|-------------|
| `schedule_daily(id, callback, time_of_day)` | Daily at specific time |
| `schedule_on_bar_close(id, callback)` | After each bar |
| `schedule_before_close(id, callback, minutes)` | Before market close |

---

## Data Feeds

### IB Historical Data (Default)

```python
# Connection settings from config/base.yaml
data:
  source: ib
  bar_size: 1d
  start_date: "2024-01-01"
  end_date: "2024-06-30"
```

IB connection configured in `config/base.yaml`:

```yaml
brokers:
  ibkr:
    host: 127.0.0.1
    port: 4001        # 4001=Gateway Live, 7497=TWS Paper
    client_id: 1
```

### CSV Data

```bash
python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL \
    --data-source csv --data-dir ./data/csv \
    --start 2024-01-01 --end 2024-06-30
```

Expected format (`data/csv/AAPL.csv`):

```csv
date,open,high,low,close,volume
2024-01-02,150.0,151.0,149.0,150.5,1000000
2024-01-03,150.5,152.0,150.0,151.0,950000
```

### Parquet Data

```bash
python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL \
    --data-source parquet --data-dir ./data/parquet \
    --start 2024-01-01 --end 2024-06-30
```

---

## Backtest Configuration

### YAML Spec File

```yaml
# config/backtest/my_strategy.yaml

# Strategy definition
strategy:
  name: ma_cross                    # Registry name
  id: ma-cross-aapl-2024           # Unique run ID
  params:
    short_window: 10
    long_window: 50
    position_size: 100

# Universe
universe:
  symbols:
    - AAPL
    - MSFT

# Data configuration
data:
  source: ib                        # ib | csv | parquet
  bar_size: 1d                      # 1m | 5m | 15m | 1h | 1d
  start_date: "2024-01-01"
  end_date: "2024-06-30"
  # For CSV/Parquet:
  # data_dir: "./data"

# Execution settings
execution:
  initial_capital: 100000
  currency: USD
  allowed_order_types:
    - MARKET
    - LIMIT

# Reality model (fees & slippage)
reality_model:
  fee_model:
    type: fixed
    commission_per_share: 0.005
    min_commission: 1.0
  slippage_model:
    type: constant
    slippage_bps: 5

# Risk controls (optional)
risk:
  enabled: false
  max_position_size: 1000

# Reporting
reporting:
  analyzers:
    - sharpe
    - drawdown
    - trades
  persist_to_db: false
  output_dir: "results/backtests"

# Metadata
metadata:
  author: "Strategy Team"
  description: "MA crossover backtest"
  tags:
    - momentum
    - equities
```

### Running with Spec

```bash
python -m src.runners.backtest_runner --spec config/backtest/my_strategy.yaml
```

---

## Running Backtests

### Programmatic Usage

```python
import asyncio
from datetime import date
from src.runners.backtest_runner import BacktestRunner

async def run_backtest():
    runner = BacktestRunner(
        strategy_name="ma_cross",
        symbols=["AAPL", "MSFT"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        initial_capital=100000,
        data_source="ib",
        bar_size="1d",
        strategy_params={
            "short_window": 10,
            "long_window": 50,
        },
    )

    result = await runner.run()
    result.print_summary()
    return result

asyncio.run(run_backtest())
```

### From Spec File

```python
from src.runners.backtest_runner import BacktestRunner

runner = BacktestRunner.from_spec("config/backtest/my_strategy.yaml")
result = await runner.run()
```

---

## Backtest Results

### Metrics Calculated

```python
result = await runner.run()

# Performance metrics
result.total_return_pct        # Total return percentage
result.cagr                    # Compound annual growth rate
result.sharpe_ratio            # Risk-adjusted return
result.sortino_ratio           # Downside risk-adjusted return
result.max_drawdown_pct        # Maximum drawdown percentage
result.volatility              # Annualized volatility

# Trade metrics
result.total_trades            # Total number of trades
result.winning_trades          # Number of winning trades
result.win_rate                # Win rate percentage
result.profit_factor           # Gross profit / gross loss
result.avg_trade_pnl           # Average P&L per trade

# Print summary
result.print_summary()

# Export
result.to_json()               # JSON string
result.to_dict()               # Dictionary
```

### Sample Output

```
============================================================
BACKTEST RESULTS: ma_cross
============================================================
Period:         2024-01-01 to 2024-06-30 (125 trading days)
Initial:        $100,000.00
Final:          $108,450.00
------------------------------------------------------------
PERFORMANCE
  Total Return:     8.45%
  CAGR:            17.2%
  Best Day:         2.3%
  Worst Day:       -1.8%
------------------------------------------------------------
RISK
  Sharpe Ratio:     1.45
  Sortino Ratio:    2.10
  Max Drawdown:    -5.2%
  Volatility:      15.3%
------------------------------------------------------------
TRADES
  Total Trades:     24
  Win Rate:        62.5%
  Profit Factor:    1.85
  Avg Trade:       $352.08
============================================================
```

---

## Example Strategies

### 1. Moving Average Cross (`ma_cross`)

Classic trend-following strategy.

```python
# Entry: Short MA crosses above Long MA
# Exit: Short MA crosses below Long MA
```

Parameters:
- `short_window`: Short MA period (default: 10)
- `long_window`: Long MA period (default: 50)
- `position_size`: Shares per trade (default: 100)

### 2. RSI Mean Reversion (`rsi_reversion`)

Counter-trend with limit orders.

```python
# Entry: RSI < 30 (oversold) → Buy with limit order
# Exit: RSI > 70 (overbought) → Sell with limit order
```

Parameters:
- `rsi_period`: RSI period (default: 14)
- `oversold`: Oversold threshold (default: 30)
- `overbought`: Overbought threshold (default: 70)

### 3. Momentum Breakout (`momentum_breakout`)

ATR-based trend following with trailing stops.

```python
# Entry: Price breaks above N-period high
# Stop: ATR-based trailing stop
```

Parameters:
- `lookback`: Channel period (default: 20)
- `atr_period`: ATR period (default: 14)
- `atr_multiplier`: Stop distance in ATRs (default: 2.0)

### 4. Pairs Trading (`pairs_trading`)

Statistical arbitrage on correlated pairs.

```python
# Entry: Z-score exceeds threshold
# Exit: Z-score returns to mean
```

Parameters:
- `lookback`: Mean/std calculation period (default: 20)
- `entry_zscore`: Entry threshold (default: 2.0)
- `exit_zscore`: Exit threshold (default: 0.5)

### 5. Scheduled Rebalance (`scheduled_rebalance`)

Time-based portfolio rebalancing.

```python
# Action: Rebalance to target weights daily at 3:55 PM
# Trigger: When drift exceeds threshold
```

Parameters:
- `target_weights`: Dict of symbol → weight
- `rebalance_threshold`: Drift threshold (default: 0.05)

---

## Best Practices

### 1. State Management

```python
def __init__(self, ...):
    # Initialize all state in __init__
    self._prices = {}
    self._last_signal = {}
    self._indicators = {}
```

### 2. Use Context, Not Globals

```python
# Good
current_time = self.context.now()
position = self.context.get_position(symbol)

# Bad
import datetime
current_time = datetime.datetime.now()  # Won't work in backtest!
```

### 3. Handle Missing Data

```python
def on_bar(self, bar):
    if bar.close is None or bar.close <= 0:
        return  # Skip invalid bars
```

### 4. Avoid Lookahead Bias

```python
# Good - use data up to current bar
prices = self._prices[:-1]  # Exclude current

# Bad - using future data
prices = self._prices  # Includes current bar
```

### 5. Log Important Events

```python
import logging
logger = logging.getLogger(__name__)

def on_fill(self, fill):
    logger.info(f"[{self.strategy_id}] FILL: {fill.side} {fill.quantity} @ {fill.price}")
```

### 6. Test with Multiple Periods

```bash
# Test different market conditions
python -m src.runners.backtest_runner --strategy my_strategy --symbols AAPL \
    --start 2020-01-01 --end 2020-12-31  # COVID crash

python -m src.runners.backtest_runner --strategy my_strategy --symbols AAPL \
    --start 2021-01-01 --end 2021-12-31  # Bull market
```

---

## Troubleshooting

### IB Connection Issues

```
Error: Failed to connect to IB at 127.0.0.1:4001
```

1. Ensure IB Gateway/TWS is running
2. Check port in `config/base.yaml`
3. Enable API connections in IB settings

### No Data Returned

```
IbHistoricalDataFeed loaded 0 total bars
```

1. Check symbol is valid
2. Verify market data subscription
3. Check date range is valid (not future)

### Strategy Not Found

```
Unknown strategy: my_strategy
```

1. Ensure module is imported in `__init__.py`
2. Check `@register_strategy` decorator
3. Verify strategy name matches

---

## See Also

- [README.md](../README.md) - Project overview
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [Example specs](../config/backtest/) - Sample configurations
