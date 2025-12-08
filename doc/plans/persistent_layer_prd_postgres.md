# Option Risk Engine - Persistent Layer PRD v0.1

**Author:** Risk Engine Team  
**Date:** December 2025  
**Status:** Implementation Ready  
**Storage Backend:** PostgreSQL

---

## Executive Summary

This document defines the architecture, data model, and implementation specifications for the Persistent Layer of the Option Risk Engine. The layer unifies historical and real-time trading data from **Futu (Moomoo)** and **Interactive Brokers (IB)** into a normalized schema optimized for risk calculation, backtesting, and trading attribution.

**Core Design Principles:**
- *Raw Data 尽可能宽*: Preserve all original API payloads for audit/replay
- *Normalized Data 尽可能严*: Strict schema for downstream consumption
- *IB 历史靠报表*: Use Flex Reports as source of truth for IB history
- *Futu 历史靠 API*: Leverage full API capabilities for Futu history
- *策略识别靠时间窗口聚类*: Rule-based strategy classification with upgrade path

---

## 1. Architecture Overview

### 1.1 Module Decomposition

```
risk_engine/
├── persistent/
│   ├── adapters/           # Broker-specific data fetchers
│   │   ├── base.py
│   │   ├── futu_adapter.py
│   │   └── ib_adapter.py
│   ├── normalize/          # Raw → Norm transformations
│   │   ├── base.py
│   │   ├── futu_normalizer.py
│   │   ├── ib_normalizer.py
│   │   └── time_utils.py
│   ├── classify/           # Strategy identification
│   │   └── strategy_classifier_v1.py
│   ├── storage/            # PostgreSQL operations
│   │   ├── postgres_store.py
│   │   ├── schemas.py
│   │   └── queries.py
│   ├── reconcile/          # Order-Trade-Fee validation
│   │   └── reconciler.py
│   └── orchestrator/       # Pipeline coordination
│       ├── loader.py
│       ├── incremental.py
│       └── full_reload.py
```

### 1.2 Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EXTRACT PHASE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Futu OpenD                          │  IB Gateway / Flex Reports       │
│  ├─ history_order_list_query         │  ├─ T+0: reqExecutions           │
│  ├─ history_deal_list_query          │  ├─ T+0: commissionReport        │
│  └─ order_fee_query (batch 400)      │  └─ T-1: Flex Query XML          │
└───────────────┬──────────────────────┴───────────────┬──────────────────┘
                │                                      │
                ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAW PERSIST (Immediate)                          │
│  • orders_raw_futu / orders_raw_ib                                      │
│  • trades_raw_futu / trades_raw_ib                                      │
│  • fees_raw_futu / fees_raw_ib                                          │
│  • Full JSONB payloads preserved                                        │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRANSFORM PHASE                                  │
│  • Timezone conversion (Exchange-local → UTC storage)                   │
│  • Field normalization (broker-agnostic schema)                         │
│  • ID unification (order_uid, trade_uid, fee_uid)                       │
│  • Option symbol parsing (underlying, strike, expiry, right)            │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFY PHASE                                   │
│  • Time-window clustering (same underlying, ≤5s delta)                  │
│  • Rule-based strategy detection (spreads, straddles, etc.)             │
│  • Confidence scoring                                                    │
│  • strategy_id assignment                                               │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOAD PHASE (Idempotent)                          │
│  • orders_norm, trades_norm, fees_norm                                  │
│  • order_strategy_map                                                   │
│  • ON CONFLICT DO UPDATE (upsert semantics)                             │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RECONCILE PHASE                                  │
│  • Order.filled_qty == SUM(Trade.qty)                                   │
│  • Every filled order has fee record                                    │
│  • Anomaly flagging (orphan trades, missing fees)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Broker API Deep Dive

### 2.1 Futu OpenAPI

Futu provides three complementary historical data APIs that must be used together:

#### 2.1.1 API Comparison Matrix

| API | Level | Key Fields | Limitations | Use For |
|-----|-------|------------|-------------|---------|
| `history_order_list_query` | Order | order_id, code, qty, price, status, create_time, update_time, fill_qty, fill_avg_price | Real trading only | Order lifecycle, fill summary |
| `history_deal_list_query` | Deal/Execution | deal_id, order_id, qty, price, create_time | Real trading only | Actual fills, true P&L, slippage |
| `order_fee_query` | Fee | order_id, fee_amount, fee_details[] | **Max 400 orders/request**, 10 req/30s, Orders since 2018-01-01 only | Commission breakdown |

#### 2.1.2 Critical API Constraints (Confirmed from Official Docs)

**order_fee_query:**
- Maximum 400 order_ids per request
- Rate limit: 10 requests per 30 seconds per account
- Only orders after 2018-01-01 supported
- Use `acc_id` NOT `acc_index` (index can shift when accounts added/removed)

**history_deal_list_query:**
- Real trading environment only (no paper trading)
- Returns partial fills as separate records
- `order_id` links back to parent order

#### 2.1.3 Understanding the Raw Payload

Futu APIs return pandas DataFrames. The `payload JSONB` field stores the **complete row as JSON**:

```python
# Futu returns DataFrame like:
ret, data = trd_ctx.history_order_list_query(start=start, end=end, acc_id=acc_id)

# Example row from data DataFrame:
{
    "order_id": "6664320708369556828",
    "order_id_ex": "20210330_15680495_SQSWWgSYCStLVb7BDmx7kgAARgy31Nc1",
    "code": "US.AAPL",
    "stock_name": "Apple Inc",
    "trd_side": "BUY",
    "order_type": "NORMAL",
    "order_status": "FILLED_ALL",
    "qty": 100.0,
    "price": 125.50,
    "create_time": "2021-03-30 09:34:23.628",
    "update_time": "2021-03-30 09:34:24.016",
    "dealt_qty": 100.0,
    "dealt_avg_price": 125.48,
    "aux_price": 0.0,
    "trail_type": "NONE",
    "trail_value": 0.0,
    "trail_spread": 0.0,
    "currency": "USD",
    "remark": "",
    "time_in_force": "DAY",
    "fill_outside_rth": false,
    # ... potentially more fields Futu may add in future
}

# We store this entire dict as JSONB
# This preserves ALL fields even ones we don't currently normalize
```

**Why store raw payload?**
1. **Audit trail** - Exact broker data for compliance/debugging
2. **Future-proofing** - New fields automatically captured
3. **Re-normalization** - Fix bugs without re-fetching from broker
4. **Field discovery** - Query JSONB to find useful fields you missed

#### 2.1.4 Optimal Futu Fetch Pipeline

```python
def futu_historical_backfill(start_date: str, end_date: str, acc_id: int):
    """
    Complete Futu history pipeline with proper ordering
    """
    # 1. Fetch Orders (source of order_ids for fee query)
    ret, orders_df = trd_ctx.history_order_list_query(
        start=start_date, 
        end=end_date, 
        acc_id=acc_id,
        status_filter_list=[OrderStatus.FILLED_ALL, OrderStatus.FILLED_PART]
    )
    
    # Convert DataFrame rows to list of dicts for JSONB storage
    orders_raw = orders_df.to_dict('records')
    persist_raw('orders_raw_futu', orders_raw)
    
    # 2. Fetch Deals (execution-level truth)
    ret, deals_df = trd_ctx.history_deal_list_query(
        start=start_date,
        end=end_date,
        acc_id=acc_id
    )
    deals_raw = deals_df.to_dict('records')
    persist_raw('trades_raw_futu', deals_raw)
    
    # 3. Fetch Fees (batched, max 400 per request)
    order_ids = orders_df['order_id'].unique().tolist()
    fees_list = []
    BATCH_SIZE = 400
    
    for i in range(0, len(order_ids), BATCH_SIZE):
        batch = order_ids[i:i+BATCH_SIZE]
        ret, fees_df = trd_ctx.order_fee_query(
            order_id_list=batch,
            acc_id=acc_id
        )
        if ret == RET_OK:
            fees_list.extend(fees_df.to_dict('records'))
        # Rate limiting: 10 req / 30s max
        if (i // BATCH_SIZE + 1) % 10 == 0:
            time.sleep(30)
    
    persist_raw('fees_raw_futu', fees_list)
    
    # 4. Validation
    validate_order_deal_consistency(orders_df, deals_df)
```

#### 2.1.5 Futu Time Field Handling

Futu returns timestamps as strings with milliseconds:

```python
# Example: "2025-01-15 09:31:12.456"

# Raw storage: preserve original string
create_time_raw_str = "2025-01-15 09:31:12.456"

# Normalized: parse with timezone
from zoneinfo import ZoneInfo

EXCHANGE_TZ = {
    'US': 'America/New_York',
    'HK': 'Asia/Hong_Kong', 
    'CN': 'Asia/Shanghai',
    'SG': 'Asia/Singapore',
    'JP': 'Asia/Tokyo'
}

def parse_futu_timestamp(raw: str, market: str) -> datetime:
    """
    Parse Futu timestamp string to timezone-aware datetime
    """
    if not raw:
        return None
    tz = ZoneInfo(EXCHANGE_TZ.get(market, 'UTC'))
    # Format: "YYYY-MM-DD HH:MM:SS.mmm"
    dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
    return dt.replace(tzinfo=tz)
```

### 2.2 Interactive Brokers (ib_async)

#### 2.2.1 Critical Limitation: No Long-Term Historical Orders via API

**Confirmed from official sources and ib_async discussions:**

> "IBKR gateway/tws APIs only report trades/executions for the most recent 1-2 calendar days then they stop showing up in the API execution list."

This is a fundamental design limitation of the IB API - it's optimized for real-time trading, not historical data retrieval.

#### 2.2.2 Dual-Path Strategy: API + Flex Reports

| Data Source | Time Window | Contents | Reliability |
|-------------|-------------|----------|-------------|
| `reqExecutions()` | T+0 (current session) | Real-time fills | High |
| `commissionReport` events | T+0 (current session) | Real-time fees | High |
| `reqCompletedOrders()` | ~7 days (unreliable) | Recent completed orders | Medium |
| **Flex Query XML** | **Full history** | Orders, Trades, Fees | **Highest** |

#### 2.2.3 IB Real-Time Data Capture

```python
from ib_async import IB, ExecutionFilter

class IbRealtimeCapture:
    def __init__(self, ib: IB):
        self.ib = ib
        self.pending_commissions = {}
        
        # Subscribe to execution events
        ib.execDetailsEvent += self.on_exec_details
        ib.commissionReportEvent += self.on_commission
        
    def on_exec_details(self, trade, fill):
        """Capture execution as it happens - store full object as payload"""
        exec_record = {
            'exec_id': fill.execution.execId,
            'perm_id': fill.execution.permId,
            'account': fill.execution.acctNumber,
            'symbol': fill.contract.symbol,
            'side': fill.execution.side,
            'qty': fill.execution.shares,
            'price': fill.execution.price,
            'time': str(fill.execution.time),
            'exchange': fill.execution.exchange,
            'order_ref': fill.execution.orderRef,
        }
        # Store the complete execution object as payload
        payload = {
            'execution': fill.execution.__dict__,
            'contract': fill.contract.__dict__,
        }
        persist_raw('trades_raw_ib', exec_record, payload)
        
    def on_commission(self, trade, fill, report):
        """Capture commission linked to execution"""
        fee_record = {
            'exec_id': report.execId,
            'commission': report.commission,
            'currency': report.currency,
            'realized_pnl': report.realizedPNL,
        }
        payload = report.__dict__
        persist_raw('fees_raw_ib', fee_record, payload)
```

#### 2.2.4 IB Flex Report Integration

Use the `ibflex` library for historical data:

```python
from ibflex import client, parser

def fetch_ib_flex_history(token: str, query_id: str) -> dict:
    """
    Download and parse IB Flex Report
    
    Setup in IB Account Management:
    1. Performance & Reports > Flex Queries > Custom Flex Queries
    2. Create query with: Trades, Orders, Transaction Fees
    3. Generate API token
    """
    # Download XML
    xml_data = client.download(token, query_id)
    
    # Parse to Python objects
    response = parser.parse(xml_data)
    
    result = {
        'orders': [],
        'trades': [],
        'fees': []
    }
    
    for stmt in response.FlexStatements:
        # Trades include commission data
        for trade in stmt.Trades:
            result['trades'].append({
                'trade_id': trade.tradeID,
                'order_id': trade.orderID,
                'perm_id': trade.ibOrderID,
                'symbol': trade.symbol,
                'underlying': trade.underlyingSymbol,
                'sec_type': trade.assetCategory,  # STK, OPT, FUT
                'strike': trade.strike,
                'expiry': trade.expiry,
                'right': trade.putCall,
                'side': trade.buySell,
                'qty': trade.quantity,
                'price': trade.tradePrice,
                'commission': trade.ibCommission,
                'trade_time': trade.dateTime,
                # Store full trade object as payload
                'payload': trade.__dict__,
            })
    
    return result
```

---

## 3. PostgreSQL Schema Design

### 3.1 Why PostgreSQL

- **JSONB native**: First-class support for semi-structured data with indexing
- **Mature upsert**: Robust `ON CONFLICT` handling
- **ACID compliant**: Strong consistency guarantees
- **Partitioning**: Native table partitioning for time-series data
- **Extensions**: TimescaleDB for time-series, pg_cron for scheduling
- **Async support**: Works well with asyncpg for high-throughput ingestion
- **Operational familiarity**: Well-understood ops, monitoring, backup patterns

### 3.2 Schema Definition

```sql
-- ============================================================
-- RAW LAYER: Preserve original API payloads
-- Purpose: Audit trail, re-processing capability, debug
-- ============================================================

-- Futu Raw Orders
CREATE TABLE IF NOT EXISTS orders_raw_futu (
    broker TEXT NOT NULL DEFAULT 'FUTU',
    acc_id BIGINT NOT NULL,
    order_id TEXT NOT NULL,
    
    -- The complete API response row as JSONB
    -- Contains ALL fields from Futu's history_order_list_query
    -- Example: {"order_id": "123", "code": "US.AAPL", "qty": 100, ...}
    payload JSONB NOT NULL,
    
    -- Extracted for indexing (also in payload)
    create_time_raw_str TEXT,
    update_time_raw_str TEXT,
    
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (acc_id, order_id)
);

-- Futu Raw Trades/Deals
CREATE TABLE IF NOT EXISTS trades_raw_futu (
    broker TEXT NOT NULL DEFAULT 'FUTU',
    acc_id BIGINT NOT NULL,
    deal_id TEXT NOT NULL,
    order_id TEXT,
    
    -- Complete API response from history_deal_list_query
    payload JSONB NOT NULL,
    
    trade_time_raw_str TEXT,
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (acc_id, deal_id)
);

-- Futu Raw Fees
CREATE TABLE IF NOT EXISTS fees_raw_futu (
    broker TEXT NOT NULL DEFAULT 'FUTU',
    acc_id BIGINT NOT NULL,
    order_id TEXT NOT NULL,
    
    -- Extracted for quick access
    fee_amount NUMERIC(20, 8),
    
    -- Complete API response from order_fee_query
    -- Contains fee_details array: [("Commission", 5.85), ("Platform Fee", 2.7), ...]
    payload JSONB NOT NULL,
    
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (acc_id, order_id)
);

-- IB Raw Orders (captured in real-time or from Flex)
CREATE TABLE IF NOT EXISTS orders_raw_ib (
    broker TEXT NOT NULL DEFAULT 'IB',
    account TEXT NOT NULL,
    perm_id BIGINT,
    client_order_id TEXT,
    order_ref TEXT,
    
    -- Complete order object from IB API or Flex Report
    payload JSONB NOT NULL,
    
    create_time_raw_str TEXT,
    update_time_raw_str TEXT,
    source TEXT DEFAULT 'API',  -- 'API' or 'FLEX'
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account, COALESCE(perm_id::TEXT, '__NA__'), COALESCE(client_order_id, '__NA__'))
);

-- IB Raw Trades/Executions
CREATE TABLE IF NOT EXISTS trades_raw_ib (
    broker TEXT NOT NULL DEFAULT 'IB',
    account TEXT NOT NULL,
    exec_id TEXT NOT NULL,
    perm_id BIGINT,
    order_ref TEXT,
    
    -- Complete execution + contract objects
    payload JSONB NOT NULL,
    
    trade_time_raw_str TEXT,
    source TEXT DEFAULT 'API',
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account, exec_id)
);

-- IB Raw Fees (from commissionReport or Flex)
CREATE TABLE IF NOT EXISTS fees_raw_ib (
    broker TEXT NOT NULL DEFAULT 'IB',
    account TEXT NOT NULL,
    exec_id TEXT NOT NULL,
    
    commission NUMERIC(20, 8),
    currency TEXT,
    realized_pnl NUMERIC(20, 8),
    
    -- Complete commission report object
    payload JSONB,
    
    source TEXT DEFAULT 'API',
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account, exec_id)
);

-- ============================================================
-- NORMALIZED LAYER: Broker-agnostic schema for analytics
-- Purpose: Risk calculation, backtesting, reporting
-- ============================================================

CREATE TABLE IF NOT EXISTS orders_norm (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    order_uid TEXT NOT NULL,  -- Unified: {broker}_{account}_{order_id}
    
    -- Instrument
    instrument_type TEXT,      -- STOCK, OPTION, FUTURE, FOREX
    symbol TEXT NOT NULL,
    underlying TEXT,
    exchange TEXT,
    
    -- Option-specific
    strike NUMERIC(20, 4),
    expiry DATE,
    option_right TEXT,         -- CALL, PUT
    
    -- Order details
    side TEXT NOT NULL,        -- BUY, SELL
    qty NUMERIC(20, 4),
    limit_price NUMERIC(20, 8),
    order_type TEXT,           -- MARKET, LIMIT, STOP, etc.
    time_in_force TEXT,        -- DAY, GTC, IOC, etc.
    
    -- Status
    status TEXT,               -- WORKING, FILLED, CANCELLED, REJECTED
    filled_qty NUMERIC(20, 4),
    avg_fill_price NUMERIC(20, 8),
    
    -- Timestamps (UTC)
    create_time_utc TIMESTAMPTZ,
    update_time_utc TIMESTAMPTZ,
    
    -- Metadata
    order_reconstructed BOOLEAN DEFAULT FALSE,
    
    -- Reference back to raw table (for debugging/audit)
    raw_ref JSONB,  -- {"table": "orders_raw_futu", "acc_id": 123, "order_id": "456"}
    
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (broker, account_id, order_uid)
);

CREATE TABLE IF NOT EXISTS trades_norm (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    trade_uid TEXT NOT NULL,   -- Unified: {broker}_{account}_{deal_id/exec_id}
    order_uid TEXT,            -- FK to orders_norm
    
    -- Instrument
    instrument_type TEXT,
    symbol TEXT NOT NULL,
    underlying TEXT,
    
    -- Option-specific
    strike NUMERIC(20, 4),
    expiry DATE,
    option_right TEXT,
    
    -- Execution details
    side TEXT NOT NULL,
    qty NUMERIC(20, 4) NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    exchange TEXT,
    
    -- Timestamp (UTC)
    trade_time_utc TIMESTAMPTZ NOT NULL,
    
    -- Metadata
    raw_ref JSONB,
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (broker, account_id, trade_uid)
);

CREATE TABLE IF NOT EXISTS fees_norm (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    fee_uid TEXT NOT NULL,     -- Order-level or execution-level depending on broker
    order_uid TEXT,
    trade_uid TEXT,
    
    -- Fee breakdown
    fee_type TEXT,             -- COMMISSION, PLATFORM, EXCHANGE, SEC, TAF, OTHER
    amount NUMERIC(20, 8) NOT NULL,
    currency TEXT DEFAULT 'USD',
    
    -- Metadata
    raw_ref JSONB,
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (broker, account_id, fee_uid, fee_type)
);

-- ============================================================
-- STRATEGY LAYER: Trade grouping and classification
-- ============================================================

CREATE TABLE IF NOT EXISTS order_strategy_map (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    order_uid TEXT NOT NULL,
    
    strategy_id TEXT NOT NULL,      -- Hash of leg combination
    strategy_type TEXT NOT NULL,    -- directional, bull_put_spread, etc.
    strategy_name TEXT,             -- Human-readable name
    confidence NUMERIC(5, 4),       -- 0.0 to 1.0
    leg_index INTEGER,              -- Position in multi-leg strategy
    legs JSONB,                     -- All legs in this strategy
    
    classify_version TEXT DEFAULT 'v1',
    updated_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (broker, account_id, order_uid)
);

-- ============================================================
-- SIGNAL LAYER: Extensible for future risk/trading signals
-- ============================================================

CREATE TABLE IF NOT EXISTS risk_signals (
    account_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,      -- VIX_SPIKE, CONCENTRATION, THETA_DECAY, etc.
    payload JSONB NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    version TEXT,
    PRIMARY KEY (account_id, signal_id)
);

CREATE TABLE IF NOT EXISTS trading_signals (
    account_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,      -- ENTRY, EXIT, HEDGE, ROLL
    payload JSONB NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    version TEXT,
    PRIMARY KEY (account_id, signal_id)
);

-- ============================================================
-- POSITION SNAPSHOTS: Point-in-time state
-- ============================================================

CREATE TABLE IF NOT EXISTS positions_snapshot (
    snapshot_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    snapshot_time_utc TIMESTAMPTZ NOT NULL,
    
    symbol TEXT NOT NULL,
    underlying TEXT,
    instrument_type TEXT,
    
    -- Position
    qty NUMERIC(20, 4),
    avg_cost NUMERIC(20, 8),
    market_value NUMERIC(20, 4),
    unrealized_pnl NUMERIC(20, 4),
    
    -- Option Greeks (if applicable)
    delta NUMERIC(10, 6),
    gamma NUMERIC(10, 6),
    theta NUMERIC(10, 6),
    vega NUMERIC(10, 6),
    
    -- Metadata
    payload JSONB,
    
    PRIMARY KEY (snapshot_id, account_id, symbol)
);

-- ============================================================
-- INDEXES for query performance
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_orders_norm_symbol ON orders_norm(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_norm_underlying ON orders_norm(underlying);
CREATE INDEX IF NOT EXISTS idx_orders_norm_create_time ON orders_norm(create_time_utc);
CREATE INDEX IF NOT EXISTS idx_orders_norm_status ON orders_norm(status);

CREATE INDEX IF NOT EXISTS idx_trades_norm_symbol ON trades_norm(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_norm_trade_time ON trades_norm(trade_time_utc);
CREATE INDEX IF NOT EXISTS idx_trades_norm_order ON trades_norm(order_uid);

CREATE INDEX IF NOT EXISTS idx_strategy_map_type ON order_strategy_map(strategy_type);
CREATE INDEX IF NOT EXISTS idx_strategy_map_strategy_id ON order_strategy_map(strategy_id);

-- JSONB indexes for payload queries
CREATE INDEX IF NOT EXISTS idx_orders_raw_futu_payload ON orders_raw_futu USING GIN (payload);
CREATE INDEX IF NOT EXISTS idx_trades_raw_futu_payload ON trades_raw_futu USING GIN (payload);

-- ============================================================
-- PARTITIONING (Optional - for high volume)
-- ============================================================

-- Example: Partition trades_norm by month
-- CREATE TABLE trades_norm (
--     ...
-- ) PARTITION BY RANGE (trade_time_utc);
--
-- CREATE TABLE trades_norm_2025_01 PARTITION OF trades_norm
--     FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

### 3.3 PostgreSQL Upsert Pattern

```python
import asyncpg
from typing import List, Dict, Any
import json

class PostgresStore:
    def __init__(self, dsn: str):
        """
        Args:
            dsn: PostgreSQL connection string
                 e.g., "postgresql://user:pass@localhost:5432/risk_engine"
        """
        self.dsn = dsn
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)
        await self._init_schema()
    
    async def _init_schema(self):
        """Create tables if not exist"""
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)  # The SQL above
    
    async def upsert_orders_raw_futu(self, records: List[Dict[str, Any]]):
        """
        Upsert raw Futu orders with full payload preservation
        """
        if not records:
            return
        
        sql = """
        INSERT INTO orders_raw_futu (acc_id, order_id, payload, create_time_raw_str, update_time_raw_str)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (acc_id, order_id) 
        DO UPDATE SET 
            payload = EXCLUDED.payload,
            create_time_raw_str = EXCLUDED.create_time_raw_str,
            update_time_raw_str = EXCLUDED.update_time_raw_str,
            ingest_ts = NOW()
        """
        
        async with self.pool.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r['acc_id'],
                    r['order_id'],
                    json.dumps(r),  # Store entire record as payload
                    r.get('create_time'),
                    r.get('update_time'),
                )
                for r in records
            ])
    
    async def upsert_orders_norm(self, records: List[Dict[str, Any]]):
        """
        Upsert normalized orders
        """
        if not records:
            return
        
        sql = """
        INSERT INTO orders_norm (
            broker, account_id, order_uid, instrument_type, symbol, underlying,
            exchange, strike, expiry, option_right, side, qty, limit_price,
            order_type, time_in_force, status, filled_qty, avg_fill_price,
            create_time_utc, update_time_utc, order_reconstructed, raw_ref
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
        ON CONFLICT (broker, account_id, order_uid) 
        DO UPDATE SET 
            status = EXCLUDED.status,
            filled_qty = EXCLUDED.filled_qty,
            avg_fill_price = EXCLUDED.avg_fill_price,
            update_time_utc = EXCLUDED.update_time_utc,
            ingest_ts = NOW()
        """
        
        async with self.pool.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r['broker'], r['account_id'], r['order_uid'],
                    r.get('instrument_type'), r['symbol'], r.get('underlying'),
                    r.get('exchange'), r.get('strike'), r.get('expiry'),
                    r.get('option_right'), r['side'], r.get('qty'),
                    r.get('limit_price'), r.get('order_type'), r.get('time_in_force'),
                    r.get('status'), r.get('filled_qty'), r.get('avg_fill_price'),
                    r.get('create_time_utc'), r.get('update_time_utc'),
                    r.get('order_reconstructed', False),
                    json.dumps(r.get('raw_ref', {})),
                )
                for r in records
            ])
    
    async def get_data_boundaries(self, table: str, time_col: str) -> tuple:
        """Get earliest and latest timestamps for incremental load"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT MIN({time_col}), MAX({time_col}) FROM {table}
            """)
            return row[0], row[1]
    
    async def query_raw_payload(self, table: str, jsonb_path: str, value: str):
        """
        Query raw tables by JSONB field
        
        Example: Find all orders for a specific symbol
            query_raw_payload('orders_raw_futu', 'code', 'US.AAPL')
        """
        async with self.pool.acquire() as conn:
            return await conn.fetch(f"""
                SELECT * FROM {table}
                WHERE payload->>'{jsonb_path}' = $1
            """, value)
```

### 3.4 Synchronous Alternative (psycopg2)

```python
import psycopg2
from psycopg2.extras import execute_values, Json

class PostgresStoreSyncf:
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False
    
    def upsert_orders_raw_futu(self, records: List[Dict[str, Any]]):
        """Bulk upsert using execute_values for performance"""
        if not records:
            return
        
        sql = """
        INSERT INTO orders_raw_futu (acc_id, order_id, payload, create_time_raw_str, update_time_raw_str)
        VALUES %s
        ON CONFLICT (acc_id, order_id) 
        DO UPDATE SET 
            payload = EXCLUDED.payload,
            ingest_ts = NOW()
        """
        
        values = [
            (r['acc_id'], r['order_id'], Json(r), r.get('create_time'), r.get('update_time'))
            for r in records
        ]
        
        with self.conn.cursor() as cur:
            execute_values(cur, sql, values)
        self.conn.commit()
```

---

## 4. Strategy Classification Rules v1

### 4.1 Classification Algorithm

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import hashlib

class StrategyType(Enum):
    # Single-leg
    DIRECTIONAL = "directional"
    
    # Vertical spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    
    # Volatility plays
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    
    # Calendar/Diagonal
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    
    # Income strategies
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    CASH_SECURED_PUT = "cash_secured_put"
    
    # Multi-leg complex
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    
    # Fallback
    UNKNOWN = "unknown"

@dataclass
class LegInfo:
    order_uid: str
    symbol: str
    underlying: str
    instrument_type: str
    side: str  # BUY, SELL
    qty: float
    strike: Optional[float]
    expiry: Optional[str]
    option_right: Optional[str]  # CALL, PUT
    trade_time: datetime

@dataclass
class StrategyResult:
    strategy_type: StrategyType
    strategy_id: str
    confidence: float
    legs: List[LegInfo]
    name: str

class StrategyClassifierV1:
    """
    Rule-based strategy classifier using time-window clustering
    """
    
    TIME_WINDOW_SECONDS = 5  # Legs within 5s considered same strategy
    
    def classify(self, legs: List[LegInfo]) -> StrategyResult:
        """
        Main classification entry point
        """
        if len(legs) == 0:
            return self._unknown(legs, 0.0)
        
        if len(legs) == 1:
            return self._classify_single(legs[0])
        
        options = [l for l in legs if l.instrument_type == 'OPTION']
        stocks = [l for l in legs if l.instrument_type == 'STOCK']
        
        if len(stocks) == 1 and len(options) == 1:
            return self._classify_stock_option(stocks[0], options[0])
        
        if len(stocks) == 0 and len(options) >= 2:
            return self._classify_option_spread(options)
        
        return self._unknown(legs, 0.2)
    
    def _classify_single(self, leg: LegInfo) -> StrategyResult:
        return StrategyResult(
            strategy_type=StrategyType.DIRECTIONAL,
            strategy_id=self._generate_id([leg]),
            confidence=0.95,
            legs=[leg],
            name=f"{'Long' if leg.side == 'BUY' else 'Short'} {leg.symbol}"
        )
    
    def _classify_stock_option(self, stock: LegInfo, opt: LegInfo) -> StrategyResult:
        if stock.side == 'BUY' and opt.side == 'SELL' and opt.option_right == 'CALL':
            return StrategyResult(
                strategy_type=StrategyType.COVERED_CALL,
                strategy_id=self._generate_id([stock, opt]),
                confidence=0.90,
                legs=[stock, opt],
                name=f"Covered Call on {stock.underlying}"
            )
        
        if stock.side == 'BUY' and opt.side == 'BUY' and opt.option_right == 'PUT':
            return StrategyResult(
                strategy_type=StrategyType.PROTECTIVE_PUT,
                strategy_id=self._generate_id([stock, opt]),
                confidence=0.90,
                legs=[stock, opt],
                name=f"Protective Put on {stock.underlying}"
            )
        
        return self._unknown([stock, opt], 0.3)
    
    def _classify_option_spread(self, options: List[LegInfo]) -> StrategyResult:
        options = sorted(options, key=lambda x: (x.expiry or '', x.strike or 0))
        
        if len(options) == 2:
            return self._classify_two_leg(options[0], options[1])
        
        if len(options) == 4:
            return self._classify_four_leg(options)
        
        return self._unknown(options, 0.2)
    
    def _classify_two_leg(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        same_expiry = leg1.expiry == leg2.expiry
        same_right = leg1.option_right == leg2.option_right
        same_strike = leg1.strike == leg2.strike
        
        # Vertical spread
        if same_expiry and same_right and not same_strike:
            return self._classify_vertical(leg1, leg2)
        
        # Straddle
        if same_expiry and same_strike and not same_right:
            return self._classify_straddle(leg1, leg2)
        
        # Strangle
        if same_expiry and not same_strike and not same_right:
            return self._classify_strangle(leg1, leg2)
        
        # Calendar
        if not same_expiry and same_strike and same_right:
            return StrategyResult(
                strategy_type=StrategyType.CALENDAR_SPREAD,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.80,
                legs=[leg1, leg2],
                name=f"Calendar {leg1.option_right} {leg1.strike}"
            )
        
        # Diagonal
        if not same_expiry and not same_strike and same_right:
            return StrategyResult(
                strategy_type=StrategyType.DIAGONAL_SPREAD,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.75,
                legs=[leg1, leg2],
                name=f"Diagonal {leg1.option_right}"
            )
        
        return self._unknown([leg1, leg2], 0.3)
    
    def _classify_vertical(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        if leg1.strike > leg2.strike:
            leg1, leg2 = leg2, leg1
        
        is_call = leg1.option_right == 'CALL'
        buy_lower = leg1.side == 'BUY'
        
        if is_call:
            if buy_lower:
                strategy_type = StrategyType.BULL_CALL_SPREAD
                name = f"Bull Call {leg1.strike}/{leg2.strike}"
            else:
                strategy_type = StrategyType.BEAR_CALL_SPREAD
                name = f"Bear Call {leg1.strike}/{leg2.strike}"
        else:
            if buy_lower:
                strategy_type = StrategyType.BEAR_PUT_SPREAD
                name = f"Bear Put {leg1.strike}/{leg2.strike}"
            else:
                strategy_type = StrategyType.BULL_PUT_SPREAD
                name = f"Bull Put {leg1.strike}/{leg2.strike}"
        
        return StrategyResult(
            strategy_type=strategy_type,
            strategy_id=self._generate_id([leg1, leg2]),
            confidence=0.90,
            legs=[leg1, leg2],
            name=name
        )
    
    def _classify_straddle(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        both_buy = leg1.side == 'BUY' and leg2.side == 'BUY'
        both_sell = leg1.side == 'SELL' and leg2.side == 'SELL'
        
        if both_buy:
            return StrategyResult(
                strategy_type=StrategyType.LONG_STRADDLE,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.90,
                legs=[leg1, leg2],
                name=f"Long Straddle {leg1.strike}"
            )
        elif both_sell:
            return StrategyResult(
                strategy_type=StrategyType.SHORT_STRADDLE,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.90,
                legs=[leg1, leg2],
                name=f"Short Straddle {leg1.strike}"
            )
        
        return self._unknown([leg1, leg2], 0.4)
    
    def _classify_strangle(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        both_buy = leg1.side == 'BUY' and leg2.side == 'BUY'
        both_sell = leg1.side == 'SELL' and leg2.side == 'SELL'
        
        put_leg = leg1 if leg1.option_right == 'PUT' else leg2
        call_leg = leg2 if leg1.option_right == 'PUT' else leg1
        
        if both_buy:
            return StrategyResult(
                strategy_type=StrategyType.LONG_STRANGLE,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.85,
                legs=[put_leg, call_leg],
                name=f"Long Strangle {put_leg.strike}/{call_leg.strike}"
            )
        elif both_sell:
            return StrategyResult(
                strategy_type=StrategyType.SHORT_STRANGLE,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.85,
                legs=[put_leg, call_leg],
                name=f"Short Strangle {put_leg.strike}/{call_leg.strike}"
            )
        
        return self._unknown([leg1, leg2], 0.4)
    
    def _classify_four_leg(self, options: List[LegInfo]) -> StrategyResult:
        puts = sorted([o for o in options if o.option_right == 'PUT'], key=lambda x: x.strike)
        calls = sorted([o for o in options if o.option_right == 'CALL'], key=lambda x: x.strike)
        
        if len(puts) != 2 or len(calls) != 2:
            return self._unknown(options, 0.3)
        
        # Iron Condor
        if (puts[0].side == 'BUY' and puts[1].side == 'SELL' and
            calls[0].side == 'SELL' and calls[1].side == 'BUY'):
            return StrategyResult(
                strategy_type=StrategyType.IRON_CONDOR,
                strategy_id=self._generate_id(options),
                confidence=0.85,
                legs=options,
                name=f"Iron Condor {puts[0].strike}/{puts[1].strike}/{calls[0].strike}/{calls[1].strike}"
            )
        
        # Iron Butterfly
        if puts[1].strike == calls[0].strike:
            return StrategyResult(
                strategy_type=StrategyType.IRON_BUTTERFLY,
                strategy_id=self._generate_id(options),
                confidence=0.80,
                legs=options,
                name=f"Iron Butterfly @ {puts[1].strike}"
            )
        
        return self._unknown(options, 0.3)
    
    def _unknown(self, legs: List[LegInfo], confidence: float) -> StrategyResult:
        return StrategyResult(
            strategy_type=StrategyType.UNKNOWN,
            strategy_id=self._generate_id(legs),
            confidence=confidence,
            legs=legs,
            name="Unknown/Custom Strategy"
        )
    
    def _generate_id(self, legs: List[LegInfo]) -> str:
        leg_strs = sorted([
            f"{l.symbol}|{l.side}|{l.strike}|{l.expiry}|{l.option_right}"
            for l in legs
        ])
        return hashlib.sha256('|'.join(leg_strs).encode()).hexdigest()[:16]
```

### 4.2 Time-Window Grouping

```python
def group_trades_by_strategy(trades: List[LegInfo], window_seconds: int = 5) -> List[List[LegInfo]]:
    """
    Group trades into potential strategy legs based on:
    1. Same underlying
    2. Trade times within window_seconds of each other
    """
    if not trades:
        return []
    
    trades = sorted(trades, key=lambda t: (t.underlying, t.trade_time))
    
    groups = []
    current_group = [trades[0]]
    
    for trade in trades[1:]:
        prev = current_group[-1]
        
        if (trade.underlying == prev.underlying and
            (trade.trade_time - prev.trade_time).total_seconds() <= window_seconds):
            current_group.append(trade)
        else:
            groups.append(current_group)
            current_group = [trade]
    
    groups.append(current_group)
    return groups
```

---

## 5. Reload and Incremental Logic

### 5.1 Configuration

```yaml
# config/persistent.yaml
persistent:
  storage:
    type: postgres
    dsn: "postgresql://risk_user:password@localhost:5432/risk_engine"
    pool_min: 2
    pool_max: 10
    
  reload:
    full_reload: false
    lookback_days_default: 365
    
  futu:
    acc_ids: [12345678]
    batch_fee_size: 400
    fee_rate_limit_window: 30
    
  ib:
    accounts: ["U1234567"]
    flex_token: "YOUR_TOKEN"
    flex_query_id: "123456"
    execution_backfill_days: 365
```

### 5.2 Orchestrator Implementation

```python
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import yaml

class PersistenceOrchestrator:
    def __init__(self, config_path: str = 'config/persistent.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)['persistent']
        
        self.store = PostgresStore(self.config['storage']['dsn'])
        self.futu_adapter = FutuAdapter(self.config['futu'])
        self.ib_adapter = IbAdapter(self.config['ib'])
        self.classifier = StrategyClassifierV1()
    
    async def run(self, full_reload: Optional[bool] = None):
        """Main entry point"""
        await self.store.connect()
        
        should_reload = full_reload if full_reload is not None else self.config['reload']['full_reload']
        
        if should_reload:
            await self._full_reload()
        else:
            await self._incremental_load()
        
        await self._reconcile()
        await self._classify_strategies()
    
    async def _full_reload(self):
        print("Starting full reload...")
        
        await self.store.truncate_normalized_tables()
        
        lookback = self.config['reload']['lookback_days_default']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback)
        
        await self._fetch_futu_history(start_date, end_date)
        await self._fetch_ib_history(start_date, end_date)
        await self._normalize_all()
    
    async def _incremental_load(self):
        print("Starting incremental load...")
        
        _, latest = await self.store.get_data_boundaries('trades_norm', 'trade_time_utc')
        
        if latest:
            start = latest - timedelta(hours=1)
        else:
            start = datetime.now() - timedelta(days=self.config['reload']['lookback_days_default'])
        
        end = datetime.now()
        
        await self._fetch_futu_history(start, end)
        await self._fetch_ib_realtime()
        
        if await self._should_refresh_flex():
            await self._fetch_ib_flex_history()
        
        await self._normalize_incremental(start)
    
    async def _reconcile(self):
        reconciler = Reconciler(self.store)
        anomalies = await reconciler.run()
        
        if anomalies:
            print(f"Found {len(anomalies)} reconciliation issues")
    
    async def _classify_strategies(self):
        trades = await self.store.get_unclassified_trades()
        groups = group_trades_by_strategy(trades)
        
        results = [self.classifier.classify(g) for g in groups]
        await self.store.upsert_strategy_mappings(results)

# CLI Entry point
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-reload', action='store_true')
    parser.add_argument('--config', default='config/persistent.yaml')
    args = parser.parse_args()
    
    orchestrator = PersistenceOrchestrator(args.config)
    asyncio.run(orchestrator.run(full_reload=args.full_reload))
```

---

## 6. Verification Checklist

### 6.1 Data Completeness

```sql
-- Futu order count vs raw
SELECT 
    (SELECT COUNT(*) FROM orders_raw_futu) AS raw_count,
    (SELECT COUNT(*) FROM orders_norm WHERE broker = 'FUTU') AS norm_count;

-- Futu fee coverage
SELECT COUNT(*) AS missing_fees
FROM orders_norm o 
LEFT JOIN fees_norm f ON o.order_uid = f.order_uid 
WHERE o.broker = 'FUTU' 
  AND o.status = 'FILLED' 
  AND f.fee_uid IS NULL;

-- IB execution coverage
SELECT COUNT(*) AS missing_commissions
FROM trades_raw_ib t 
LEFT JOIN fees_raw_ib f ON t.exec_id = f.exec_id 
WHERE f.exec_id IS NULL;
```

### 6.2 Order-Trade-Fee Reconciliation

```sql
-- Check filled qty matches trade sum
SELECT 
    o.order_uid,
    o.filled_qty AS order_filled,
    COALESCE(SUM(t.qty), 0) AS trade_sum,
    o.filled_qty - COALESCE(SUM(t.qty), 0) AS discrepancy
FROM orders_norm o
LEFT JOIN trades_norm t ON o.order_uid = t.order_uid
WHERE o.status IN ('FILLED', 'FILLED_PART')
GROUP BY o.order_uid, o.filled_qty
HAVING ABS(o.filled_qty - COALESCE(SUM(t.qty), 0)) > 0.001;
```

### 6.3 Query Raw Payload for Debugging

```sql
-- Find all original fields Futu returned for a specific order
SELECT payload 
FROM orders_raw_futu 
WHERE order_id = '6664320708369556828';

-- Discover what fields are available
SELECT DISTINCT jsonb_object_keys(payload) 
FROM orders_raw_futu 
LIMIT 50;

-- Find orders with a specific remark
SELECT * FROM orders_raw_futu 
WHERE payload->>'remark' LIKE '%hedge%';
```

### 6.4 Strategy Classification Validation

```sql
-- Classification coverage
SELECT 
    strategy_type,
    COUNT(*) AS count,
    ROUND(AVG(confidence)::numeric, 3) AS avg_confidence
FROM order_strategy_map
GROUP BY strategy_type
ORDER BY count DESC;

-- Unknown strategies for review
SELECT 
    osm.strategy_id,
    osm.confidence,
    o.symbol,
    o.side,
    o.qty,
    o.create_time_utc
FROM order_strategy_map osm
JOIN orders_norm o ON osm.order_uid = o.order_uid
WHERE osm.strategy_type = 'unknown'
ORDER BY o.create_time_utc DESC
LIMIT 50;
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Days 1-2)
- [ ] Set up PostgreSQL with schema
- [ ] Implement FutuAdapter with batch fee fetching
- [ ] Implement FutuNormalizer with timezone handling
- [ ] Run 90-day historical backfill

### Phase 2: IB Integration (Days 3-4)
- [ ] Implement IB real-time capture
- [ ] Set up Flex Query and ibflex parser
- [ ] Implement IbNormalizer
- [ ] Merge IB data into normalized tables

### Phase 3: Classification (Day 5)
- [ ] Implement StrategyClassifierV1
- [ ] Run classification on historical data
- [ ] Validate with manual sample review

### Phase 4: Operations (Day 6)
- [ ] Implement incremental load logic
- [ ] Build reconciliation reports
- [ ] Create monitoring/alerting

### Phase 5: Integration (Day 7)
- [ ] Connect to RiskEngine main pipeline
- [ ] Verify Greeks calculation inputs
- [ ] End-to-end test with live data

---

## Appendix A: Key API References

### Futu OpenAPI
- history_order_list_query: https://openapi.futunn.com/futu-api-doc/en/trade/get-history-order-list.html
- history_deal_list_query: https://openapi.futunn.com/futu-api-doc/en/trade/get-history-order-fill-list.html
- order_fee_query: https://openapi.futunn.com/futu-api-doc/en/trade/order-fee-query.html

### IB / ib_async
- ib_async GitHub: https://github.com/ib-api-reloaded/ib_async
- Flex Web Service: https://www.interactivebrokers.com/campus/ibkr-api-page/flex-web-service/
- ibflex parser: https://pypi.org/project/ibflex/

### PostgreSQL
- UPSERT: https://www.postgresql.org/docs/current/sql-insert.html#SQL-ON-CONFLICT
- JSONB: https://www.postgresql.org/docs/current/datatype-json.html
- asyncpg: https://magicstack.github.io/asyncpg/

---

## Appendix B: Error Handling Matrix

| Error Type | Detection | Recovery |
|------------|-----------|----------|
| Futu rate limit | RetCode -1 | Exponential backoff, max 3 retries |
| Futu partial response | Incomplete DataFrame | Re-fetch with narrower date range |
| IB disconnect | `ib.disconnect()` event | Reconnect and resume from last checkpoint |
| IB session reset | executions() returns stale data | Force Flex report refresh |
| PostgreSQL constraint violation | ON CONFLICT triggers | Upsert with DO UPDATE |
| Timezone parse error | datetime exception | Log, mark record for manual review |
| Strategy classification failure | Unknown type | Assign "unknown", flag for review |

---

*Document Version: 0.1*  
*Last Updated: December 2025*
