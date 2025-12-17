# APEX Persistence Layer Manual

This document provides comprehensive documentation for the APEX persistence layer, which handles database operations, history loading, and warm-start capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Database Setup](#database-setup)
4. [Configuration](#configuration)
5. [Repositories](#repositories)
6. [History Loading](#history-loading)
7. [Snapshot & Warm-Start](#snapshot--warm-start)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The persistence layer provides:

- **PostgreSQL/TimescaleDB** storage for trading data
- **Async connection pooling** via asyncpg
- **Repository pattern** for clean data access
- **History loading** from Futu and IB brokers
- **Warm-start** capability from database snapshots
- **Time-series storage** for risk analytics

### Key Features

| Feature | Description |
|---------|-------------|
| Raw Order Storage | Store orders from Futu (3-table design) and IB (2-table design) |
| Incremental Sync | Track last sync time per broker/account/market |
| Rate Limiting | Futu API rate limiting (10 req/30s) |
| UPSERT Pattern | Idempotent writes using ON CONFLICT |
| Snapshot Capture | Periodic position/account/risk snapshots |
| Warm-Start | Fast startup from last known state |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ HistoryLoader   │  │ SnapshotService │  │ WarmStartService│ │
│  │    Service      │  │                 │  │                 │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
┌───────────┼─────────────────────┼─────────────────────┼─────────┐
│           │           Repository Layer                │         │
│  ┌────────┴────────┐  ┌────────┴────────┐  ┌────────┴────────┐ │
│  │ FutuOrderRepo   │  │ PositionSnapshot│  │ RiskSnapshot    │ │
│  │ FutuDealRepo    │  │     Repo        │  │     Repo        │ │
│  │ FutuFeeRepo     │  │ AccountSnapshot │  │ SignalRepo      │ │
│  │ IbExecutionRepo │  │     Repo        │  │ BacktestRepo    │ │
│  │ IbCommissionRepo│  │                 │  │                 │ │
│  │ SyncStateRepo   │  │                 │  │                 │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
┌───────────┴─────────────────────┴─────────────────────┴─────────┐
│                      Database Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Database Class                           ││
│  │  - Connection Pool (asyncpg)                                ││
│  │  - Query Execution (fetch, execute, fetchval)               ││
│  │  - Transaction Support                                      ││
│  │  - Health Checks                                            ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
            │
┌───────────┴──────────────────────────────────────────────────────┐
│                  PostgreSQL / TimescaleDB                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ futu_raw_*   │  │ ib_raw_*     │  │ *_snapshots  │           │
│  │ (3 tables)   │  │ (2 tables)   │  │ (3 tables)   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ sync_state   │  │ risk_signals │  │ backtests    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Database Setup

### Prerequisites

1. **PostgreSQL 14+** with TimescaleDB extension
2. **Database created** with appropriate user permissions

### Installation

```bash
# Install PostgreSQL (macOS)
brew install postgresql@14

# Install TimescaleDB
brew tap timescale/tap
brew install timescaledb

# Start PostgreSQL
brew services start postgresql@14

# Create database and user
psql postgres -c "CREATE USER apex WITH PASSWORD 'your_password';"
psql postgres -c "CREATE DATABASE apex_risk OWNER apex;"
psql apex_risk -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
```

### Running Migrations

```python
import asyncio
from config.models import DatabaseConfig
from src.infrastructure.persistence.database import Database
from migrations.runner import MigrationRunner

async def run_migrations():
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="apex_risk",
        user="apex",
        password="your_password",
    )

    db = Database(config)
    await db.connect()

    runner = MigrationRunner(db, migrations_dir="migrations")
    applied = await runner.run()

    print(f"Applied {len(applied)} migrations")
    await db.close()

asyncio.run(run_migrations())
```

### Database Schema

The initial migration (`001_initial_schema.sql`) creates these tables:

| Table | Purpose |
|-------|---------|
| `futu_raw_orders` | Futu orders from history_order_list_query() |
| `futu_raw_deals` | Futu executions from history_deal_list_query() |
| `futu_raw_fees` | Futu fees from order_fee_query() |
| `ib_raw_executions` | IB executions from reqExecutions() |
| `ib_raw_commissions` | IB commission reports |
| `risk_signals` | Risk engine breach/warning signals |
| `trade_signals` | Trading strategy signals |
| `backtests` | Backtest results storage |
| `position_snapshots` | Position snapshots for warm-start |
| `account_snapshots` | Account snapshots for warm-start |
| `risk_snapshots` | Risk metric time-series |
| `sync_state` | Incremental sync tracking |
| `exchange_info` | Exchange reference data |

---

## Configuration

### Database Configuration

In `config/base.yaml`:

```yaml
database:
  type: timescaledb
  host: localhost
  port: 5432
  database: apex_risk
  user: apex
  password: ""  # Set via APEX_DB_PASSWORD env var

  pool:
    min_connections: 2
    max_connections: 10

  timescale:
    enabled: true
    chunk_interval: "1 month"
    compression_enabled: true
    compression_after: "7 days"
```

### Display Timezone

```yaml
display:
  timezone: "Asia/Hong_Kong"  # Dashboard display timezone
  date_format: "%Y-%m-%d"
  time_format: "%H:%M:%S"
  datetime_format: "%Y-%m-%d %H:%M:%S %Z"
```

### Snapshot Configuration

```yaml
snapshots:
  position_interval_sec: 60   # Capture every 60 seconds
  account_interval_sec: 60
  risk_interval_sec: 60
  capture_on_shutdown: true   # Capture final snapshot on exit
  retention_days: 365         # Keep 1 year of history
  compression_after_days: 7   # Compress old data
```

### History Loader Configuration

```yaml
history_loader:
  default_lookback_days: 30
  batch_size: 100
  futu_rate_limit:
    requests_per_window: 10
    window_seconds: 30
```

---

## Repositories

### Base Repository Pattern

All repositories extend `BaseRepository[T]` which provides:

```python
class BaseRepository(ABC, Generic[T]):
    # Core methods
    async def find_by_id(id_value) -> Optional[T]
    async def find_all(limit, offset) -> List[T]
    async def count() -> int
    async def exists(**conditions) -> bool
    async def find_where(limit, order_by, **conditions) -> List[T]
    async def find_one_where(**conditions) -> Optional[T]

    # Write methods
    async def insert(entity: T) -> T
    async def upsert(entity: T) -> T
    async def upsert_many(entities: List[T]) -> int
    async def delete(id_value) -> bool
    async def delete_where(**conditions) -> int
```

### Repository Classes

| Repository | Entity | Key Methods |
|------------|--------|-------------|
| `FutuOrderRepository` | `FutuRawOrder` | `find_by_order_id()`, `find_by_date_range()`, `find_filled_orders()` |
| `FutuDealRepository` | `FutuRawDeal` | `find_by_deal_id()`, `find_by_order_id()`, `get_total_volume_by_code()` |
| `FutuFeeRepository` | `FutuRawFee` | `find_by_order_id()`, `get_total_fees()`, `get_missing_fee_order_ids()` |
| `IbExecutionRepository` | `IbRawExecution` | `find_by_exec_id()`, `find_by_perm_id()`, `find_options_by_underlying()` |
| `IbCommissionRepository` | `IbRawCommission` | `find_by_exec_id()`, `get_total_commissions()` |
| `SyncStateRepository` | `SyncState` | `get_state()`, `update_sync_complete()`, `needs_sync()` |
| `RiskSignalRepository` | `RiskSignal` | `find_by_type()`, `find_breaches()`, `get_signal_counts_by_type()` |
| `TradeSignalRepository` | `TradeSignal` | `find_by_source()`, `find_pending()`, `mark_executed()` |
| `BacktestRepository` | `Backtest` | `find_by_strategy()`, `find_top_performers()`, `get_strategy_summary()` |
| `PositionSnapshotRepository` | `PositionSnapshot` | `get_latest()`, `get_by_time_range()`, `cleanup_old()` |
| `AccountSnapshotRepository` | `AccountSnapshot` | `get_latest()`, `get_by_time_range()`, `cleanup_old()` |
| `RiskSnapshotRepository` | `RiskSnapshotRecord` | `get_latest()`, `get_time_series()`, `get_daily_summary()` |

### Usage Example

```python
from src.infrastructure.persistence.database import Database
from src.infrastructure.persistence.repositories import FutuOrderRepository

async def example():
    db = Database(config)
    await db.connect()

    repo = FutuOrderRepository(db)

    # Find orders
    orders = await repo.find_by_account("ACC123", market="US", limit=100)

    # Find filled orders since a date
    filled = await repo.find_filled_orders("ACC123", start_date=datetime(2024, 1, 1))

    # UPSERT an order
    order = FutuOrderRepository.from_futu_order(api_data, "ACC123", "US")
    await repo.upsert(order)

    await db.close()
```

---

## History Loading

### CLI Usage

```bash
# Load Futu history for specific account
python scripts/history_loader.py --broker futu --account ACC123 --market US --days 30

# Load IB history (current-day only via API)
python scripts/history_loader.py --broker ib --account U1234567

# Load from all configured brokers
python scripts/history_loader.py --broker all --days 30

# Dry run (preview without writing)
python scripts/history_loader.py --broker futu --account ACC123 --dry-run

# Force full reload (ignore last sync time)
python scripts/history_loader.py --broker futu --account ACC123 --force

# Custom date range
python scripts/history_loader.py --broker futu --account ACC123 --from-date 2024-01-01 --to-date 2024-03-01
```

### Programmatic Usage

```python
from src.services import HistoryLoaderService
from src.infrastructure.persistence.database import Database

async def load_history():
    db = Database(config)
    await db.connect()

    service = HistoryLoaderService(db=db, config=app_config)

    # Load Futu history
    result = await service.load_futu_history(
        account_id="ACC123",
        market="US",
        from_date=date(2024, 1, 1),
        to_date=date(2024, 3, 1),
    )

    print(f"Status: {result.status}")
    print(f"Orders: {result.orders_loaded}")
    print(f"Deals: {result.deals_loaded}")
    print(f"Fees: {result.fees_loaded}")

    await db.close()
```

### Rate Limiting

Futu API has rate limits (10 requests per 30 seconds). The history loader automatically handles this:

```python
class RateLimiter:
    def __init__(self, requests_per_window: int, window_seconds: int):
        # Tracks request times and waits if necessary

    async def acquire(self):
        # Blocks until rate limit allows another request
```

### Incremental Sync

The `sync_state` table tracks the last sync time for each broker/account/data_type/market combination:

```python
# Check if sync is needed
needs_sync = await sync_state_repo.needs_sync(
    broker="FUTU",
    account_id="ACC123",
    data_type="futu_orders",
    market="US",
    max_age_hours=24.0,
)

# Get last record time for incremental loading
last_time = await sync_state_repo.get_last_record_time(
    broker="FUTU",
    account_id="ACC123",
    data_type="futu_orders",
    market="US",
)
```

---

## Snapshot & Warm-Start

### Snapshot Service

Periodically captures position, account, and risk snapshots:

```python
from src.services import SnapshotService

# Initialize with callbacks
service = SnapshotService(
    db=db,
    config=snapshot_config,
    get_positions_callback=lambda: get_current_positions(),
    get_account_callback=lambda: get_current_accounts(),
    get_risk_snapshot_callback=lambda: get_current_risk(),
)

# Start periodic capture
await service.start()

# Manually capture now
await service.capture_all_now()

# Cleanup old snapshots
deleted = await service.cleanup()
print(f"Deleted: {deleted}")

# Stop service (captures final snapshot if configured)
await service.stop()
```

### Warm-Start Service

Restores state from database snapshots on startup:

```python
from src.services import WarmStartService

service = WarmStartService(
    db=db,
    max_age_seconds=3600,  # Only load snapshots < 1 hour old
)

# Define brokers to restore
brokers = [
    {"name": "IB", "account_id": "U1234567"},
    {"name": "FUTU", "account_id": "ACC123"},
]

# Perform warm start
result = await service.warm_start(
    brokers=brokers,
    on_positions_loaded=lambda broker, acc, positions: restore_positions(positions),
    on_account_loaded=lambda broker, acc, account: restore_account(account),
    on_risk_loaded=lambda risk_data: restore_risk(risk_data),
)

print(f"Loaded {result.positions_loaded} positions")
print(f"Loaded {result.accounts_loaded} accounts")
print(f"Risk snapshot: {'loaded' if result.risk_snapshot_loaded else 'not found'}")
```

### Snapshot Status

Check available snapshots before warm start:

```python
status = await service.get_snapshot_status(brokers)
for snap in status["position_snapshots"]:
    print(f"{snap['broker']}/{snap['account_id']}: {snap['position_count']} positions, age={snap['age_seconds']:.0f}s")
```

---

## API Reference

### Database Class

```python
class Database:
    def __init__(self, config: DatabaseConfig)

    # Connection management
    async def connect() -> None
    async def close() -> None
    async def health_check() -> bool

    # Query execution
    async def execute(query, *args, timeout=None) -> str
    async def executemany(query, args, timeout=None) -> None
    async def fetch(query, *args, timeout=None) -> List[Record]
    async def fetchrow(query, *args, timeout=None) -> Optional[Record]
    async def fetchval(query, *args, column=0, timeout=None) -> Any

    # Transactions
    @asynccontextmanager
    async def transaction() -> AsyncIterator[Connection]

    @asynccontextmanager
    async def acquire() -> AsyncIterator[Connection]

    # Utilities
    async def table_exists(table_name) -> bool
    async def get_table_row_count(table_name) -> int
    async def get_pool_stats() -> Dict[str, int]
```

### MigrationRunner Class

```python
class MigrationRunner:
    def __init__(self, db: Database, migrations_dir: str = "migrations")

    async def run(target_version=None) -> List[Migration]
    async def get_pending_migrations() -> List[Migration]
    async def get_applied_migrations() -> List[Migration]
    async def get_current_version() -> Optional[str]
    async def reset() -> None  # For testing only
```

### HistoryLoaderService Class

```python
class HistoryLoaderService:
    def __init__(self, db, config, dry_run=False)

    async def load_futu_history(account_id, market, from_date, to_date, force) -> LoadResult
    async def load_ib_history(account_id, from_date, to_date, force) -> LoadResult
    async def load_all_history(from_date, to_date, force) -> LoadResult
    async def get_sync_status() -> List[Dict]

@dataclass
class LoadResult:
    status: str  # SUCCESS, PARTIAL, FAILED
    orders_loaded: int
    deals_loaded: int
    fees_loaded: int
    duration_seconds: float
    errors: List[str]
```

### SnapshotService Class

```python
class SnapshotService:
    def __init__(self, db, config, get_positions_callback, get_account_callback, get_risk_snapshot_callback)

    async def start() -> None
    async def stop() -> None
    async def capture_all_now() -> None
    async def capture_positions_now() -> None
    async def capture_accounts_now() -> None
    async def capture_risk_now() -> None
    async def cleanup() -> Dict[str, int]
```

### WarmStartService Class

```python
class WarmStartService:
    def __init__(self, db, max_age_seconds=3600)

    async def warm_start(brokers, on_positions_loaded, on_account_loaded, on_risk_loaded) -> WarmStartResult
    async def get_snapshot_status(brokers) -> Dict[str, Any]
    async def clear_all_snapshots() -> Dict[str, int]

@dataclass
class WarmStartResult:
    success: bool
    positions_loaded: int
    accounts_loaded: int
    risk_snapshot_loaded: bool
    snapshot_age_seconds: Optional[float]
    error: Optional[str]
```

---

## Troubleshooting

### Connection Issues

**Problem**: `ConnectionError: Failed to connect to database`

**Solutions**:
1. Verify PostgreSQL is running: `pg_isready`
2. Check connection string in config
3. Verify user has permissions: `psql -U apex -d apex_risk`
4. Check firewall/network settings

### Migration Failures

**Problem**: `MigrationError: Migration 001 failed`

**Solutions**:
1. Check SQL syntax in migration file
2. Verify TimescaleDB extension is installed
3. Run `SELECT version()` to check PostgreSQL version
4. Check migration log for specific error

### Rate Limiting

**Problem**: Futu API returns rate limit errors

**Solutions**:
1. Ensure `futu_rate_limit` is configured correctly
2. Don't run multiple history loaders in parallel
3. Wait for rate limit window to reset (30 seconds)

### Stale Snapshots

**Problem**: Warm start loads old/stale data

**Solutions**:
1. Check `max_age_seconds` parameter
2. Verify snapshot service is running
3. Check `capture_on_shutdown` is enabled
4. Review snapshot timestamps in database

### Missing Data

**Problem**: History loader reports 0 records

**Solutions**:
1. Verify broker connection is working
2. Check date range parameters
3. Confirm account ID is correct
4. Review broker API permissions

---

## Best Practices

1. **Always use transactions** for multi-table updates
2. **Use UPSERT** for idempotent writes (safe for retries)
3. **Check sync state** before loading to avoid duplicates
4. **Configure retention** to prevent unbounded table growth
5. **Monitor pool stats** to tune connection pool size
6. **Use warm-start** for fast restarts in production
7. **Capture snapshots on shutdown** for data continuity

---

## Related Documentation

- [05_persistence_layer.md](design/05_persistence_layer.md) - Design document
- [05a_persistence_sql.md](design/05a_persistence_sql.md) - SQL schema reference
- [05b_persistence_repositories.md](design/05b_persistence_repositories.md) - Repository implementations
- [05c_persistence_loaders.md](design/05c_persistence_loaders.md) - History loader details
