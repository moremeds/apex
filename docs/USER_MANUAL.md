# APEX User Manual

Complete guide to installing, configuring, and running the APEX Live Risk Management System.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running APEX](#running-apex)
5. [History Loader](#history-loader)
6. [Backtest Runner](#backtest-runner)
7. [Terminal Dashboard](#terminal-dashboard)
8. [Common Use Cases](#common-use-cases)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e .

# 2. Configure brokers (edit config/base.yaml)
#    - Set IBKR host/port
#    - Set Futu host/port (if using)

# 3. Start IBKR TWS or Gateway

# 4. Run APEX
python main.py --env dev

# 5. (Optional) Load historical data
python scripts/history_loader.py --broker futu --days 30
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.14 recommended |
| PostgreSQL | 14+ | Optional, for persistence |
| TimescaleDB | 2.x | Optional, for time-series |
| IBKR TWS/Gateway | Latest | Required for market data |
| Futu OpenD | Latest | Optional, for Futu positions |

### Install with uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install APEX
uv pip install -e .

# Install with all optional features
uv pip install -e ".[dev,observability]"
```

### Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Database Setup (Optional)

```bash
# Create database
createdb apex_risk

# Enable TimescaleDB (optional)
psql apex_risk -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Set password environment variable
export APEX_DB_PASSWORD="your_password"
```

---

## Configuration

### Configuration Files

| File | Purpose |
|------|---------|
| `config/base.yaml` | Main configuration (brokers, limits, database) |
| `config/dev.yaml` | Development overrides |
| `config/prod.yaml` | Production overrides |
| `config/demo.yaml` | Demo mode (offline, sample data) |
| `data/positions/manual.yaml` | Manual position entries |

### Broker Configuration

#### Interactive Brokers

```yaml
brokers:
  ibkr:
    enabled: true
    host: 127.0.0.1
    port: 4001        # See port reference below
    client_id: 1
```

**Port Reference:**
| Port | Environment |
|------|-------------|
| 7497 | TWS Paper Trading |
| 7496 | TWS Live Trading |
| 4001 | IB Gateway Paper |
| 4000 | IB Gateway Live |

#### Futu OpenD

```yaml
brokers:
  futu:
    enabled: true
    host: 127.0.0.1
    port: 11111
    security_firm: FUTUSECURITIES  # FUTUSECURITIES, FUTUINC, FUTUSG, FUTUAU
    trd_env: REAL                   # REAL or SIMULATE
    filter_trdmarket: US            # US, HK, CN, SG, JP, AU
```

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
  max_concentration_pct: 0.30
  soft_breach_threshold: 0.80  # Warning at 80%
```

### Database Configuration

```yaml
database:
  type: timescaledb
  host: localhost
  port: 5432
  database: apex_risk
  user: apex
  password: ""  # Use APEX_DB_PASSWORD env var

  pool:
    min_connections: 2
    max_connections: 10

  timescale:
    enabled: true
    chunk_interval: "1 month"
    compression_enabled: true
    compression_after: "7 days"
```

### Snapshot Configuration

```yaml
snapshots:
  position_interval_sec: 60   # Capture every minute
  account_interval_sec: 60
  risk_interval_sec: 60
  capture_on_shutdown: true
  retention_days: 365
```

### Display Timezone

```yaml
display:
  timezone: "Asia/Hong_Kong"  # IANA timezone
  datetime_format: "%Y-%m-%d %H:%M:%S %Z"
```

---

## Running APEX

### Main Command

```bash
python main.py [OPTIONS]
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--env` | `dev` | Environment: `dev`, `prod`, or `demo` |
| `--config` | - | Custom config file path |
| `--no-dashboard` | false | Run headless (no TUI) |
| `--metrics-port` | 8000 | Prometheus metrics port (0 to disable) |
| `--verbose, -v` | false | Enable DEBUG logging |
| `--log-level` | INFO | Log level: DEBUG, INFO, WARNING, ERROR |

### Examples

```bash
# Development mode (default)
python main.py --env dev

# Production mode
python main.py --env prod

# Demo mode (offline, sample positions)
python main.py --env demo

# Headless mode (no dashboard)
python main.py --no-dashboard

# Custom config file
python main.py --config /path/to/custom.yaml

# Verbose logging
python main.py -v

# Disable metrics endpoint
python main.py --metrics-port 0

# Full example with multiple options
python main.py --env prod --log-level DEBUG --metrics-port 9090
```

### Environment Modes

| Mode | Description |
|------|-------------|
| `dev` | Development - connects to paper trading |
| `prod` | Production - connects to live trading |
| `demo` | Demo - offline mode with sample positions |

---

## History Loader

Load historical order/execution data from brokers into the database.

### Command

```bash
python scripts/history_loader.py [OPTIONS]
```

### Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--broker` | Yes | - | Broker: `futu`, `ib`, or `all` |
| `--account` | No | all | Specific account ID |
| `--market` | No | US | Futu market: US, HK, CN, SG, JP, AU |
| `--days` | No | 30 | Lookback period in days |
| `--from-date` | No | - | Start date (YYYY-MM-DD) |
| `--to-date` | No | today | End date (YYYY-MM-DD) |
| `--dry-run` | No | false | Preview without writing |
| `--force` | No | false | Full reload (ignore sync state) |
| `--config` | No | config/base.yaml | Config file path |
| `--verbose, -v` | No | false | Enable DEBUG logging |

### Examples

```bash
# Load Futu US orders (last 30 days)
python scripts/history_loader.py --broker futu --market US --days 30

# Load Futu HK orders for specific account
python scripts/history_loader.py --broker futu --account ACC123 --market HK

# Load IB executions
python scripts/history_loader.py --broker ib --days 7

# Load from all brokers
python scripts/history_loader.py --broker all --days 30

# Custom date range
python scripts/history_loader.py --broker futu --from-date 2024-01-01 --to-date 2024-03-31

# Dry run (preview)
python scripts/history_loader.py --broker futu --dry-run

# Force full reload (ignore incremental sync)
python scripts/history_loader.py --broker futu --force

# Verbose output
python scripts/history_loader.py --broker futu -v
```

### Output

```
2024-03-15 10:30:00 [INFO] History Loader starting
2024-03-15 10:30:00 [INFO]   Broker: futu
2024-03-15 10:30:00 [INFO]   Date range: 2024-02-15 to 2024-03-15
2024-03-15 10:30:00 [INFO]   Market: US
2024-03-15 10:30:05 [INFO] Database connected
...
2024-03-15 10:30:45 [INFO] ============================================================
2024-03-15 10:30:45 [INFO] Load Summary:
2024-03-15 10:30:45 [INFO]   Status: SUCCESS
2024-03-15 10:30:45 [INFO]   Orders loaded: 156
2024-03-15 10:30:45 [INFO]   Deals/Executions loaded: 203
2024-03-15 10:30:45 [INFO]   Fees/Commissions loaded: 156
2024-03-15 10:30:45 [INFO]   Duration: 40.23s
```

---

## Backtest Runner

Run historical strategy backtests using real IB market data.

### Quick Start

```bash
# List available strategies
python -m src.runners.backtest_runner --list-strategies

# Run a backtest with CLI options
python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL \
    --start 2024-01-01 --end 2024-06-30

# Run from YAML spec file
python -m src.runners.backtest_runner --spec config/backtest/ma_cross_example.yaml
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | - | Strategy name from registry |
| `--symbols` | - | Comma-separated symbols (e.g., `AAPL,MSFT`) |
| `--start` | - | Start date (YYYY-MM-DD) |
| `--end` | - | End date (YYYY-MM-DD) |
| `--capital` | 100000 | Initial capital |
| `--data-source` | ib | Data source: `ib`, `csv`, `parquet` |
| `--data-dir` | - | Directory for CSV/Parquet files |
| `--bar-size` | 1d | Bar size: `1m`, `5m`, `15m`, `1h`, `1d` |
| `--params` | - | Strategy params (KEY=VALUE) |
| `--spec` | - | YAML spec file path |
| `--list-strategies` | - | List available strategies |
| `-v, --verbose` | false | Enable debug logging |

### Available Strategies

| Strategy | Description |
|----------|-------------|
| `ma_cross` | Moving average crossover (trend-following) |
| `buy_and_hold` | Passive buy and hold benchmark |
| `rsi_reversion` | RSI mean reversion with limit orders |
| `momentum_breakout` | ATR-based breakout with trailing stops |
| `pairs_trading` | Statistical arbitrage on correlated pairs |
| `scheduled_rebalance` | Time-based portfolio rebalancing |

### YAML Spec File Format

```yaml
# config/backtest/my_strategy.yaml

strategy:
  name: ma_cross              # Registry name
  id: backtest-001            # Unique run ID
  params:
    short_window: 10
    long_window: 50

universe:
  symbols:
    - AAPL
    - MSFT

data:
  source: ib                  # ib | csv | parquet
  bar_size: 1d
  start_date: "2024-01-01"
  end_date: "2024-06-30"

execution:
  initial_capital: 100000
  currency: USD

reality_model:
  fee_model:
    type: fixed
    commission_per_share: 0.005
  slippage_model:
    type: constant
    slippage_bps: 5

reporting:
  analyzers:
    - sharpe
    - drawdown
    - trades
  output_dir: "results/backtests"
```

### Examples

```bash
# MA Cross on single stock
python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL \
    --start 2024-01-01 --end 2024-06-30

# Pairs trading on AAPL/MSFT
python -m src.runners.backtest_runner --strategy pairs_trading --symbols AAPL,MSFT \
    --start 2024-01-01 --end 2024-06-30 --params lookback=20 entry_zscore=2.0

# Use CSV data instead of IB
python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL \
    --data-source csv --data-dir ./data/csv \
    --start 2024-01-01 --end 2024-06-30

# Run from spec file
python -m src.runners.backtest_runner --spec config/backtest/pairs_trading_example.yaml

# Verbose output for debugging
python -m src.runners.backtest_runner --spec config/backtest/ma_cross_example.yaml -v
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

### IB Connection for Backtests

The backtest runner loads IB connection settings from `config/base.yaml`:

```yaml
brokers:
  ibkr:
    host: 127.0.0.1
    port: 4001        # 4001=Gateway, 7497=TWS Paper
    client_id: 1
```

Ensure IB Gateway or TWS is running before starting a backtest with `data.source: ib`.

### Creating Custom Strategies

See [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) for the complete strategy development guide, including:

- Strategy architecture and lifecycle
- Creating custom strategies with `@register_strategy`
- Clock and scheduler usage
- Order submission and fill handling
- Best practices and troubleshooting

---

## Terminal Dashboard

The TUI displays real-time risk metrics when running APEX.

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ APEX Risk Monitor │ dev │ 2024-03-15 10:30:45 HKT │ IB:● FU:●  │
├─────────────────────────────────────────────────────────────────┤
│ Portfolio Summary                                               │
│   NAV: $1,234,567    Unrealized P&L: +$12,345   Daily: +$5,678 │
│   Delta: 25,000      Gamma: 1,234    Vega: 8,765  Theta: -567  │
├─────────────────────────────────────────────────────────────────┤
│ Risk Signals                                                    │
│   ⚠ SOFT: Portfolio delta at 82% of limit                      │
│   🛑 HARD: TSLA notional exceeds limit                         │
├─────────────────────────────────────────────────────────────────┤
│ Positions (15)                                                  │
│   Symbol  Qty    Mark   P&L     Delta  Gamma  DTE              │
│   TSLA    100  $245.50 +$1,234  100    0      -                │
│   ...                                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Status Indicators

| Symbol | Meaning |
|--------|---------|
| ● (green) | Connected |
| ○ (red) | Disconnected |
| ◐ (yellow) | Connecting/Degraded |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Graceful shutdown |
| `q` | Quit (same as Ctrl+C) |

---

## Common Use Cases

### 1. Daily Monitoring (Production)

```bash
# Start with production config
python main.py --env prod

# In another terminal, check metrics
curl http://localhost:8000/metrics
```

### 2. Paper Trading Development

```bash
# Connect to TWS Paper (port 7497)
# Edit config/base.yaml: port: 7497

python main.py --env dev -v
```

### 3. Backfill Historical Data

```bash
# Initial setup - load 90 days of history
python scripts/history_loader.py --broker all --days 90

# Daily sync (incremental, respects last sync time)
python scripts/history_loader.py --broker all --days 1
```

### 4. Demo/Testing (No Broker Connection)

```bash
# Run with sample positions (no live connection)
python main.py --env demo
```

### 5. Headless Server Mode

```bash
# Run without TUI, expose metrics only
python main.py --env prod --no-dashboard --metrics-port 9090
```

### 6. Debug Connection Issues

```bash
# Verbose logging to diagnose issues
python main.py -v --log-level DEBUG
```

### 7. Multiple Environments

```bash
# Terminal 1: Paper trading
python main.py --env dev --metrics-port 8001

# Terminal 2: Production (different port)
python main.py --env prod --metrics-port 8002
```

---

## Troubleshooting

### Connection Issues

**Problem:** `Failed to connect to IB`

**Solutions:**
1. Verify TWS/Gateway is running
2. Check port matches your setup (7497, 7496, 4001, 4000)
3. Enable API in TWS: File → Global Configuration → API → Settings
4. Ensure "Enable ActiveX and Socket Clients" is checked
5. Check client_id is unique (not used by another application)

**Problem:** `Futu connection failed`

**Solutions:**
1. Verify Futu OpenD is running
2. Check port (default: 11111)
3. Verify security_firm matches your account
4. Check trd_env (REAL vs SIMULATE)

### Database Issues

**Problem:** `Connection refused` to PostgreSQL

**Solutions:**
1. Check PostgreSQL is running: `pg_isready`
2. Verify database exists: `psql -l | grep apex_risk`
3. Check credentials in config
4. Set APEX_DB_PASSWORD environment variable

**Problem:** Migration failed

**Solutions:**
1. Check PostgreSQL version (14+ required)
2. Install TimescaleDB extension
3. Review error in migration log
4. Try `--force` to skip sync state

### Data Issues

**Problem:** No positions showing

**Solutions:**
1. Verify broker connection (check status indicators)
2. Check manual.yaml file format
3. Enable verbose logging: `-v`
4. Check logs in `./logs/`

**Problem:** Missing market data

**Solutions:**
1. Check IBKR market data subscriptions
2. Verify symbols are correct format
3. Check MDQC logs for staleness alerts
4. Review `mdqc.stale_seconds` setting

### Performance Issues

**Problem:** Slow dashboard refresh

**Solutions:**
1. Check position count (>250 may be slower)
2. Reduce `dashboard.refresh_interval_sec`
3. Check network latency to brokers
4. Review metrics at `/metrics` endpoint

---

## Log Files

| File | Contents |
|------|----------|
| `logs/{env}_system.log` | System events, startup/shutdown |
| `logs/{env}_adapter.log` | Broker connection events |
| `logs/{env}_risk.log` | Risk calculations, breaches |
| `logs/{env}_data.log` | Market data, position updates |
| `logs/{env}_perf.log` | Performance metrics |
| `logs/{env}_risk_alerts.jsonl` | Risk alert audit trail |

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `APEX_DB_PASSWORD` | Database password |
| `APEX_ENV` | Default environment (dev/prod) |
| `APEX_LOG_LEVEL` | Default log level |

---

## Quick Reference Card

```
# Start APEX
python main.py [--env dev|prod|demo] [-v] [--no-dashboard]

# History Loader
python scripts/history_loader.py --broker futu|ib|all [--days N] [--dry-run]

# Common flags
--env dev|prod|demo    # Environment mode
-v, --verbose          # Debug logging
--no-dashboard         # Headless mode
--metrics-port N       # Prometheus port (0=disable)
--config FILE          # Custom config
--dry-run              # Preview only
--force                # Full reload
```

---

## Related Documentation

- [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) - Strategy development and backtest guide
- [PERSISTENCE_LAYER.md](PERSISTENCE_LAYER.md) - Database and repository documentation
- [OBSERVABILITY_SETUP.md](OBSERVABILITY_SETUP.md) - Prometheus, Grafana, and alerting setup
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [README.md](../README.md) - Project overview
