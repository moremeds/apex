# Option Risk Engine - Persistent Layer PRD v0.1

**Author:** Risk Engine Team
**Date:** December 2025
**Status:** Implementation Ready
**Storage Backend:** DuckDB (Primary)

---

## Implementation Adjustments (December 2025)

**Context**: The existing system already has significant persistence infrastructure. These adjustments align the plan with what exists.

### What Already Exists (No Changes Needed)
- `src/infrastructure/persistence/duckdb_adapter.py` - v2 schema, thread-safe
- `src/infrastructure/persistence/persistence_manager.py` - Event-driven sync
- `src/infrastructure/adapters/futu/adapter.py` - Full history fetch + fee batching
- Orders/trades tables in schema v2

### Adjustments Made
1. **Directory Structure**: Use `src/infrastructure/persistence/` not `risk_engine/persistent/`
2. **Schema Evolution**: Add v3 migration to existing adapter (not new DB)
3. **Futu Adapter**: Already has `fetch_orders()`, `fetch_trades()` with fees - add raw payload hook
4. **IB Flex**: Implement `flex_parser.py` for historical backfill
5. **Smart Backfill**: Check if data exists before full reload

### New Files to Create
- `src/infrastructure/persistence/repositories/raw_data_repo.py`
- `src/infrastructure/persistence/classify/strategy_classifier.py`
- `src/infrastructure/persistence/pipeline/historical_loader.py`
- `src/infrastructure/adapters/ib/flex_parser.py`

### Deferred to v2
- `risk_signals`, `trading_signals` tables (have alert_repo already)
- `positions_snapshot` table (have position_repo already)

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
│   ├── storage/            # DuckDB operations
│   │   ├── duckdb_store.py
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

#### 2.1.3 Optimal Futu Fetch Pipeline

```python
def futu_historical_backfill(start_date: str, end_date: str, acc_id: int):
    """
    Complete Futu history pipeline with proper ordering
    """
    # 1. Fetch Orders (source of order_ids for fee query)
    orders_df = trd_ctx.history_order_list_query(
        start=start_date, 
        end=end_date, 
        acc_id=acc_id,
        status_filter_list=[OrderStatus.FILLED_ALL, OrderStatus.FILLED_PART]
    )
    persist_raw('orders_raw_futu', orders_df)
    
    # 2. Fetch Deals (execution-level truth)
    deals_df = trd_ctx.history_deal_list_query(
        start=start_date,
        end=end_date,
        acc_id=acc_id
    )
    persist_raw('trades_raw_futu', deals_df)
    
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
            fees_list.append(fees_df)
        # Rate limiting: 10 req / 30s max
        if (i // BATCH_SIZE + 1) % 10 == 0:
            time.sleep(30)
    
    all_fees_df = pd.concat(fees_list)
    persist_raw('fees_raw_futu', all_fees_df)
    
    # 4. Validation
    validate_order_deal_consistency(orders_df, deals_df)
```

#### 2.1.4 Futu Time Field Handling

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
        """Capture execution as it happens"""
        exec_record = {
            'exec_id': fill.execution.execId,
            'perm_id': fill.execution.permId,
            'account': fill.execution.acctNumber,
            'symbol': fill.contract.symbol,
            'side': fill.execution.side,
            'qty': fill.execution.shares,
            'price': fill.execution.price,
            'time': fill.execution.time,  # Already timezone-aware
            'exchange': fill.execution.exchange,
            'order_ref': fill.execution.orderRef,
        }
        persist_raw('trades_raw_ib', exec_record)
        
    def on_commission(self, trade, fill, report):
        """Capture commission linked to execution"""
        fee_record = {
            'exec_id': report.execId,
            'commission': report.commission,
            'currency': report.currency,
            'realized_pnl': report.realizedPNL,
        }
        persist_raw('fees_raw_ib', fee_record)
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
            })
    
    return result
```

---

## 3. DuckDB Schema Design

### 3.1 Why DuckDB

- **OLAP-optimized**: Columnar storage ideal for aggregations
- **Embedded**: No server process, perfect for single-user risk engine
- **Parquet native**: Easy export for archival/sharing
- **SQL standard**: Familiar query interface
- **Upsert support**: `INSERT ... ON CONFLICT` for idempotent writes

### 3.2 Schema Definition

```sql
-- ============================================================
-- RAW LAYER: Preserve original API payloads
-- ============================================================

-- Futu Raw Orders
CREATE TABLE IF NOT EXISTS orders_raw_futu (
    broker VARCHAR DEFAULT 'FUTU',
    acc_id BIGINT NOT NULL,
    order_id VARCHAR NOT NULL,
    payload JSON NOT NULL,
    create_time_raw_str VARCHAR,
    update_time_raw_str VARCHAR,
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (acc_id, order_id)
);

-- Futu Raw Trades/Deals
CREATE TABLE IF NOT EXISTS trades_raw_futu (
    broker VARCHAR DEFAULT 'FUTU',
    acc_id BIGINT NOT NULL,
    deal_id VARCHAR NOT NULL,
    order_id VARCHAR,
    payload JSON NOT NULL,
    trade_time_raw_str VARCHAR,
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (acc_id, deal_id)
);

-- Futu Raw Fees
CREATE TABLE IF NOT EXISTS fees_raw_futu (
    broker VARCHAR DEFAULT 'FUTU',
    acc_id BIGINT NOT NULL,
    order_id VARCHAR NOT NULL,
    fee_amount DECIMAL(20, 8),
    fee_details JSON,
    payload JSON NOT NULL,
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (acc_id, order_id)
);

-- IB Raw Orders (captured in real-time or from Flex)
CREATE TABLE IF NOT EXISTS orders_raw_ib (
    broker VARCHAR DEFAULT 'IB',
    account VARCHAR NOT NULL,
    perm_id BIGINT,
    client_order_id VARCHAR,
    order_ref VARCHAR,
    payload JSON NOT NULL,
    create_time_raw_str VARCHAR,
    update_time_raw_str VARCHAR,
    source VARCHAR DEFAULT 'API',  -- 'API' or 'FLEX'
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (account, COALESCE(perm_id, -1), COALESCE(client_order_id, '__NA__'))
);

-- IB Raw Trades/Executions
CREATE TABLE IF NOT EXISTS trades_raw_ib (
    broker VARCHAR DEFAULT 'IB',
    account VARCHAR NOT NULL,
    exec_id VARCHAR NOT NULL,
    perm_id BIGINT,
    order_ref VARCHAR,
    payload JSON NOT NULL,
    trade_time_raw_str VARCHAR,
    source VARCHAR DEFAULT 'API',
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (account, exec_id)
);

-- IB Raw Fees (from commissionReport or Flex)
CREATE TABLE IF NOT EXISTS fees_raw_ib (
    broker VARCHAR DEFAULT 'IB',
    account VARCHAR NOT NULL,
    exec_id VARCHAR NOT NULL,
    commission DECIMAL(20, 8),
    currency VARCHAR,
    realized_pnl DECIMAL(20, 8),
    payload JSON,
    source VARCHAR DEFAULT 'API',
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (account, exec_id)
);

-- ============================================================
-- NORMALIZED LAYER: Broker-agnostic schema
-- ============================================================

CREATE TABLE IF NOT EXISTS orders_norm (
    broker VARCHAR NOT NULL,
    account_id VARCHAR NOT NULL,
    order_uid VARCHAR NOT NULL,  -- Unified: {broker}_{account}_{order_id}
    
    -- Instrument
    instrument_type VARCHAR,      -- STOCK, OPTION, FUTURE, FOREX
    symbol VARCHAR NOT NULL,
    underlying VARCHAR,
    exchange VARCHAR,
    
    -- Option-specific
    strike DECIMAL(20, 4),
    expiry DATE,
    option_right VARCHAR,         -- CALL, PUT
    
    -- Order details
    side VARCHAR NOT NULL,        -- BUY, SELL
    qty DECIMAL(20, 4),
    limit_price DECIMAL(20, 8),
    order_type VARCHAR,           -- MARKET, LIMIT, STOP, etc.
    time_in_force VARCHAR,        -- DAY, GTC, IOC, etc.
    
    -- Status
    status VARCHAR,               -- WORKING, FILLED, CANCELLED, REJECTED
    filled_qty DECIMAL(20, 4),
    avg_fill_price DECIMAL(20, 8),
    
    -- Timestamps (UTC)
    create_time_utc TIMESTAMP WITH TIME ZONE,
    update_time_utc TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    order_reconstructed BOOLEAN DEFAULT FALSE,
    raw_ref JSON,                 -- Reference to raw record
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (broker, account_id, order_uid)
);

CREATE TABLE IF NOT EXISTS trades_norm (
    broker VARCHAR NOT NULL,
    account_id VARCHAR NOT NULL,
    trade_uid VARCHAR NOT NULL,   -- Unified: {broker}_{account}_{deal_id/exec_id}
    order_uid VARCHAR,            -- FK to orders_norm
    
    -- Instrument
    instrument_type VARCHAR,
    symbol VARCHAR NOT NULL,
    underlying VARCHAR,
    
    -- Option-specific
    strike DECIMAL(20, 4),
    expiry DATE,
    option_right VARCHAR,
    
    -- Execution details
    side VARCHAR NOT NULL,
    qty DECIMAL(20, 4) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    exchange VARCHAR,
    
    -- Timestamp (UTC)
    trade_time_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Metadata
    raw_ref JSON,
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (broker, account_id, trade_uid)
);

CREATE TABLE IF NOT EXISTS fees_norm (
    broker VARCHAR NOT NULL,
    account_id VARCHAR NOT NULL,
    fee_uid VARCHAR NOT NULL,     -- Order-level or execution-level depending on broker
    order_uid VARCHAR,
    trade_uid VARCHAR,
    
    -- Fee breakdown
    fee_type VARCHAR,             -- COMMISSION, PLATFORM, EXCHANGE, SEC, TAF, OTHER
    amount DECIMAL(20, 8) NOT NULL,
    currency VARCHAR DEFAULT 'USD',
    
    -- Metadata
    raw_ref JSON,
    ingest_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (broker, account_id, fee_uid, fee_type)
);

-- ============================================================
-- STRATEGY LAYER: Trade grouping and classification
-- ============================================================

CREATE TABLE IF NOT EXISTS order_strategy_map (
    broker VARCHAR NOT NULL,
    account_id VARCHAR NOT NULL,
    order_uid VARCHAR NOT NULL,
    
    strategy_id VARCHAR NOT NULL,      -- Hash of leg combination
    strategy_type VARCHAR NOT NULL,    -- directional, bull_put_spread, etc.
    strategy_name VARCHAR,             -- Human-readable name
    confidence DECIMAL(5, 4),          -- 0.0 to 1.0
    leg_index INTEGER,                 -- Position in multi-leg strategy
    legs JSON,                         -- All legs in this strategy
    
    classify_version VARCHAR DEFAULT 'v1',
    updated_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (broker, account_id, order_uid)
);

-- ============================================================
-- SIGNAL LAYER: Extensible for future risk/trading signals
-- ============================================================

CREATE TABLE IF NOT EXISTS risk_signals (
    account_id VARCHAR NOT NULL,
    signal_id VARCHAR NOT NULL,
    signal_type VARCHAR NOT NULL,      -- VIX_SPIKE, CONCENTRATION, THETA_DECAY, etc.
    payload JSON NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL,
    version VARCHAR,
    PRIMARY KEY (account_id, signal_id)
);

CREATE TABLE IF NOT EXISTS trading_signals (
    account_id VARCHAR NOT NULL,
    signal_id VARCHAR NOT NULL,
    signal_type VARCHAR NOT NULL,      -- ENTRY, EXIT, HEDGE, ROLL
    payload JSON NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL,
    version VARCHAR,
    PRIMARY KEY (account_id, signal_id)
);

-- ============================================================
-- POSITION SNAPSHOTS: Point-in-time state
-- ============================================================

CREATE TABLE IF NOT EXISTS positions_snapshot (
    snapshot_id VARCHAR NOT NULL,
    account_id VARCHAR NOT NULL,
    snapshot_time_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    
    symbol VARCHAR NOT NULL,
    underlying VARCHAR,
    instrument_type VARCHAR,
    
    -- Position
    qty DECIMAL(20, 4),
    avg_cost DECIMAL(20, 8),
    market_value DECIMAL(20, 4),
    unrealized_pnl DECIMAL(20, 4),
    
    -- Option Greeks (if applicable)
    delta DECIMAL(10, 6),
    gamma DECIMAL(10, 6),
    theta DECIMAL(10, 6),
    vega DECIMAL(10, 6),
    
    -- Metadata
    payload JSON,
    
    PRIMARY KEY (snapshot_id, account_id, symbol)
);

-- ============================================================
-- INDEXES for query performance
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_orders_norm_symbol ON orders_norm(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_norm_underlying ON orders_norm(underlying);
CREATE INDEX IF NOT EXISTS idx_orders_norm_create_time ON orders_norm(create_time_utc);

CREATE INDEX IF NOT EXISTS idx_trades_norm_symbol ON trades_norm(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_norm_trade_time ON trades_norm(trade_time_utc);
CREATE INDEX IF NOT EXISTS idx_trades_norm_order ON trades_norm(order_uid);

CREATE INDEX IF NOT EXISTS idx_strategy_map_type ON order_strategy_map(strategy_type);
CREATE INDEX IF NOT EXISTS idx_strategy_map_strategy_id ON order_strategy_map(strategy_id);
```

### 3.3 DuckDB Idempotent Write Pattern

```python
import duckdb
from typing import List, Dict, Any

class DuckDBStore:
    def __init__(self, db_path: str = 'risk_engine.duckdb'):
        self.conn = duckdb.connect(db_path)
        self._init_schema()
    
    def upsert_orders_norm(self, records: List[Dict[str, Any]]):
        """
        Idempotent upsert for normalized orders
        
        DuckDB upsert: INSERT ... ON CONFLICT (pk) DO UPDATE SET ...
        Note: Input must NOT have duplicates within the same batch
        """
        if not records:
            return
            
        # De-duplicate within batch (keep latest by ingest_ts)
        seen = {}
        for r in records:
            key = (r['broker'], r['account_id'], r['order_uid'])
            seen[key] = r
        records = list(seen.values())
        
        # Build parameterized query
        columns = list(records[0].keys())
        placeholders = ', '.join(['?' for _ in columns])
        update_clause = ', '.join([f'{c} = excluded.{c}' for c in columns if c not in ('broker', 'account_id', 'order_uid')])
        
        sql = f"""
        INSERT INTO orders_norm ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (broker, account_id, order_uid) 
        DO UPDATE SET {update_clause}
        """
        
        for r in records:
            values = [r[c] for c in columns]
            self.conn.execute(sql, values)
        
        self.conn.commit()
    
    def get_data_boundaries(self, table: str, time_col: str) -> tuple:
        """Get earliest and latest timestamps for incremental load"""
        result = self.conn.execute(f"""
            SELECT MIN({time_col}), MAX({time_col}) FROM {table}
        """).fetchone()
        return result[0], result[1]
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
        
        Input: List of legs grouped by (underlying, time_window)
        Output: Strategy classification with confidence
        """
        if len(legs) == 0:
            return self._unknown(legs, 0.0)
        
        # Single leg
        if len(legs) == 1:
            return self._classify_single(legs[0])
        
        # Multi-leg: separate by instrument type
        options = [l for l in legs if l.instrument_type == 'OPTION']
        stocks = [l for l in legs if l.instrument_type == 'STOCK']
        
        # Stock + Option combinations
        if len(stocks) == 1 and len(options) == 1:
            return self._classify_stock_option(stocks[0], options[0])
        
        # Pure option combinations
        if len(stocks) == 0 and len(options) >= 2:
            return self._classify_option_spread(options)
        
        return self._unknown(legs, 0.2)
    
    def _classify_single(self, leg: LegInfo) -> StrategyResult:
        """Single-leg position"""
        return StrategyResult(
            strategy_type=StrategyType.DIRECTIONAL,
            strategy_id=self._generate_id([leg]),
            confidence=0.95,
            legs=[leg],
            name=f"{'Long' if leg.side == 'BUY' else 'Short'} {leg.symbol}"
        )
    
    def _classify_stock_option(self, stock: LegInfo, opt: LegInfo) -> StrategyResult:
        """Stock + Option combinations"""
        
        # Covered Call: Long stock + Short call
        if stock.side == 'BUY' and opt.side == 'SELL' and opt.option_right == 'CALL':
            return StrategyResult(
                strategy_type=StrategyType.COVERED_CALL,
                strategy_id=self._generate_id([stock, opt]),
                confidence=0.90,
                legs=[stock, opt],
                name=f"Covered Call on {stock.underlying}"
            )
        
        # Protective Put: Long stock + Long put
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
        """Pure option multi-leg strategies"""
        
        # Sort by strike for consistent analysis
        options = sorted(options, key=lambda x: (x.expiry or '', x.strike or 0))
        
        # 2-leg spreads
        if len(options) == 2:
            return self._classify_two_leg(options[0], options[1])
        
        # 4-leg: Iron Condor / Iron Butterfly
        if len(options) == 4:
            return self._classify_four_leg(options)
        
        return self._unknown(options, 0.2)
    
    def _classify_two_leg(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        """Two-leg option spreads"""
        
        same_expiry = leg1.expiry == leg2.expiry
        same_right = leg1.option_right == leg2.option_right
        same_strike = leg1.strike == leg2.strike
        
        # Vertical spread: same expiry, same right, different strikes
        if same_expiry and same_right and not same_strike:
            return self._classify_vertical(leg1, leg2)
        
        # Straddle: same expiry, same strike, different right
        if same_expiry and same_strike and not same_right:
            return self._classify_straddle(leg1, leg2)
        
        # Strangle: same expiry, different strikes, different rights
        if same_expiry and not same_strike and not same_right:
            return self._classify_strangle(leg1, leg2)
        
        # Calendar: different expiry, same strike, same right
        if not same_expiry and same_strike and same_right:
            return StrategyResult(
                strategy_type=StrategyType.CALENDAR_SPREAD,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.80,
                legs=[leg1, leg2],
                name=f"Calendar {leg1.option_right} {leg1.strike}"
            )
        
        # Diagonal: different expiry, different strikes, same right
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
        """Vertical spread classification"""
        
        # Ensure leg1 has lower strike
        if leg1.strike > leg2.strike:
            leg1, leg2 = leg2, leg1
        
        is_call = leg1.option_right == 'CALL'
        buy_lower = leg1.side == 'BUY'
        
        if is_call:
            if buy_lower:
                # Buy lower call, sell higher call = Bull Call Spread
                strategy_type = StrategyType.BULL_CALL_SPREAD
                name = f"Bull Call {leg1.strike}/{leg2.strike}"
            else:
                # Sell lower call, buy higher call = Bear Call Spread
                strategy_type = StrategyType.BEAR_CALL_SPREAD
                name = f"Bear Call {leg1.strike}/{leg2.strike}"
        else:  # PUT
            if buy_lower:
                # Buy lower put, sell higher put = Bear Put Spread
                strategy_type = StrategyType.BEAR_PUT_SPREAD
                name = f"Bear Put {leg1.strike}/{leg2.strike}"
            else:
                # Sell lower put, buy higher put = Bull Put Spread
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
        """Straddle classification"""
        
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
        """Strangle classification"""
        
        both_buy = leg1.side == 'BUY' and leg2.side == 'BUY'
        both_sell = leg1.side == 'SELL' and leg2.side == 'SELL'
        
        # Ensure put has lower strike
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
        """Iron Condor / Iron Butterfly"""
        
        puts = sorted([o for o in options if o.option_right == 'PUT'], key=lambda x: x.strike)
        calls = sorted([o for o in options if o.option_right == 'CALL'], key=lambda x: x.strike)
        
        if len(puts) != 2 or len(calls) != 2:
            return self._unknown(options, 0.3)
        
        # Iron Condor: OTM put spread + OTM call spread
        # Structure: Buy low put, Sell mid-low put, Sell mid-high call, Buy high call
        if (puts[0].side == 'BUY' and puts[1].side == 'SELL' and
            calls[0].side == 'SELL' and calls[1].side == 'BUY'):
            
            return StrategyResult(
                strategy_type=StrategyType.IRON_CONDOR,
                strategy_id=self._generate_id(options),
                confidence=0.85,
                legs=options,
                name=f"Iron Condor {puts[0].strike}/{puts[1].strike}/{calls[0].strike}/{calls[1].strike}"
            )
        
        # Iron Butterfly: ATM straddle + OTM wings
        if puts[1].strike == calls[0].strike:  # Middle strikes match
            return StrategyResult(
                strategy_type=StrategyType.IRON_BUTTERFLY,
                strategy_id=self._generate_id(options),
                confidence=0.80,
                legs=options,
                name=f"Iron Butterfly @ {puts[1].strike}"
            )
        
        return self._unknown(options, 0.3)
    
    def _unknown(self, legs: List[LegInfo], confidence: float) -> StrategyResult:
        """Fallback for unrecognized patterns"""
        return StrategyResult(
            strategy_type=StrategyType.UNKNOWN,
            strategy_id=self._generate_id(legs),
            confidence=confidence,
            legs=legs,
            name="Unknown/Custom Strategy"
        )
    
    def _generate_id(self, legs: List[LegInfo]) -> str:
        """Generate deterministic strategy ID from legs"""
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
    
    # Sort by (underlying, trade_time)
    trades = sorted(trades, key=lambda t: (t.underlying, t.trade_time))
    
    groups = []
    current_group = [trades[0]]
    
    for trade in trades[1:]:
        prev = current_group[-1]
        
        # Same underlying and within time window
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
    type: duckdb
    path: ./data/risk_engine.duckdb
    
  reload:
    full_reload: false              # Set true to truncate and reload
    lookback_days_default: 365      # Default history depth
    
  futu:
    acc_ids: [12345678]             # Futu account IDs
    batch_fee_size: 400             # Max orders per fee query
    fee_rate_limit_window: 30       # Seconds between fee batch bursts
    
  ib:
    accounts: ["U1234567"]          # IB account IDs
    flex_token: "YOUR_TOKEN"
    flex_query_id: "123456"
    execution_backfill_days: 365    # Flex report lookback
```

### 5.2 Orchestrator Implementation

```python
from datetime import datetime, timedelta
from typing import Optional
import yaml

class PersistenceOrchestrator:
    def __init__(self, config_path: str = 'config/persistent.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)['persistent']
        
        self.store = DuckDBStore(self.config['storage']['path'])
        self.futu_adapter = FutuAdapter(self.config['futu'])
        self.ib_adapter = IbAdapter(self.config['ib'])
        self.classifier = StrategyClassifierV1()
    
    def run(self, full_reload: Optional[bool] = None):
        """
        Main entry point
        
        Args:
            full_reload: Override config setting
        """
        should_reload = full_reload if full_reload is not None else self.config['reload']['full_reload']
        
        if should_reload:
            self._full_reload()
        else:
            self._incremental_load()
        
        self._reconcile()
        self._classify_strategies()
    
    def _full_reload(self):
        """Truncate norm tables and reload from raw"""
        print("Starting full reload...")
        
        # Truncate normalized tables (keep raw for audit)
        self.store.truncate_normalized_tables()
        
        # Determine date range
        lookback = self.config['reload']['lookback_days_default']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback)
        
        # Fetch and persist raw data
        self._fetch_futu_history(start_date, end_date)
        self._fetch_ib_history(start_date, end_date)
        
        # Normalize
        self._normalize_all()
    
    def _incremental_load(self):
        """Load only new data since last sync"""
        print("Starting incremental load...")
        
        # Get current data boundaries
        _, latest_futu = self.store.get_data_boundaries('trades_norm', 'trade_time_utc')
        _, latest_ib = self.store.get_data_boundaries('trades_raw_ib', 'trade_time_raw_str')
        
        # Futu: incremental from latest
        if latest_futu:
            start = latest_futu - timedelta(hours=1)  # Overlap for safety
        else:
            start = datetime.now() - timedelta(days=self.config['reload']['lookback_days_default'])
        
        end = datetime.now()
        
        self._fetch_futu_history(start, end)
        self._fetch_ib_realtime()  # IB real-time is always "today"
        
        # Check if we need Flex report refresh
        if self._should_refresh_flex():
            self._fetch_ib_flex_history()
        
        self._normalize_incremental(start)
    
    def _should_refresh_flex(self) -> bool:
        """Check if IB Flex report needs refresh (e.g., daily)"""
        # Implementation: check last flex ingest timestamp
        pass
    
    def _reconcile(self):
        """Run order-trade-fee reconciliation"""
        reconciler = Reconciler(self.store)
        anomalies = reconciler.run()
        
        if anomalies:
            print(f"Found {len(anomalies)} reconciliation issues:")
            for a in anomalies[:10]:
                print(f"  - {a}")
    
    def _classify_strategies(self):
        """Run strategy classification on all trades"""
        # Get unclassified trades
        trades = self.store.get_unclassified_trades()
        
        # Group by time window
        groups = group_trades_by_strategy(trades)
        
        # Classify each group
        results = []
        for group in groups:
            result = self.classifier.classify(group)
            results.append(result)
        
        # Persist mappings
        self.store.upsert_strategy_mappings(results)
```

---

## 6. Verification Checklist

### 6.1 Data Completeness

| Check | Query/Method | Expected |
|-------|--------------|----------|
| Futu order count | Compare `orders_norm` vs App history | Match ±1% |
| Futu deal count | Compare `trades_norm` vs App history | Match exactly |
| Futu fee coverage | `SELECT COUNT(*) FROM orders_norm o LEFT JOIN fees_norm f ON o.order_uid = f.order_uid WHERE o.status = 'FILLED' AND f.fee_uid IS NULL` | = 0 |
| IB execution count | Compare `trades_raw_ib` vs Flex report | Match exactly |
| IB commission coverage | `SELECT COUNT(*) FROM trades_raw_ib t LEFT JOIN fees_raw_ib f ON t.exec_id = f.exec_id WHERE f.exec_id IS NULL` | = 0 |

### 6.2 Order-Trade-Fee Reconciliation

```sql
-- Check: Filled orders should have matching trade quantities
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

-- Check: All filled orders should have fee records
SELECT o.order_uid, o.symbol, o.create_time_utc
FROM orders_norm o
LEFT JOIN fees_norm f ON o.order_uid = f.order_uid
WHERE o.status = 'FILLED' AND f.fee_uid IS NULL;
```

### 6.3 Timestamp Consistency

```sql
-- Check: Trade time should not be before order create time
SELECT 
    t.trade_uid,
    o.create_time_utc AS order_time,
    t.trade_time_utc AS trade_time
FROM trades_norm t
JOIN orders_norm o ON t.order_uid = o.order_uid
WHERE t.trade_time_utc < o.create_time_utc;

-- Check: DST handling (find trades near DST boundaries)
SELECT *
FROM trades_norm
WHERE trade_time_utc::date IN ('2025-03-09', '2025-11-02')  -- US DST dates
ORDER BY trade_time_utc;
```

### 6.4 Strategy Classification Validation

```sql
-- Check: Classification coverage
SELECT 
    strategy_type,
    COUNT(*) AS count,
    AVG(confidence) AS avg_confidence
FROM order_strategy_map
GROUP BY strategy_type
ORDER BY count DESC;

-- Check: Unknown strategies for manual review
SELECT 
    osm.strategy_id,
    osm.strategy_type,
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

### 6.5 Idempotency Test

```bash
# Run twice and verify no duplicate data
python -m risk_engine.persistent.orchestrator --incremental
python -m risk_engine.persistent.orchestrator --incremental

# Count should be identical
duckdb risk_engine.duckdb "SELECT COUNT(*) FROM orders_norm"
duckdb risk_engine.duckdb "SELECT COUNT(*) FROM trades_norm"
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Days 1-2)
1. [ ] Set up DuckDB database with schema
2. [ ] Implement FutuAdapter with batch fee fetching
3. [ ] Implement FutuNormalizer with timezone handling
4. [ ] Run 90-day historical backfill

### Phase 2: IB Integration (Days 3-4)
1. [ ] Implement IB real-time capture
2. [ ] Set up Flex Query and ibflex parser
3. [ ] Implement IbNormalizer
4. [ ] Merge IB data into normalized tables

### Phase 3: Classification (Day 5)
1. [ ] Implement StrategyClassifierV1
2. [ ] Run classification on historical data
3. [ ] Validate with manual sample review

### Phase 4: Operations (Day 6)
1. [ ] Implement incremental load logic
2. [ ] Build reconciliation reports
3. [ ] Create monitoring/alerting

### Phase 5: Integration (Day 7)
1. [ ] Connect to RiskEngine main pipeline
2. [ ] Verify Greeks calculation inputs
3. [ ] End-to-end test with live data

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

### DuckDB
- INSERT ON CONFLICT: https://duckdb.org/docs/stable/sql/statements/insert
- Partitioned writes: https://duckdb.org/docs/stable/data/partitioning/partitioned_writes

---

## Appendix B: Error Handling Matrix

| Error Type | Detection | Recovery |
|------------|-----------|----------|
| Futu rate limit | RetCode -1 | Exponential backoff, max 3 retries |
| Futu partial response | Incomplete DataFrame | Re-fetch with narrower date range |
| IB disconnect | `ib.disconnect()` event | Reconnect and resume from last checkpoint |
| IB session reset | executions() returns stale data | Force Flex report refresh |
| DuckDB constraint violation | ON CONFLICT triggers | Upsert with DO UPDATE |
| Timezone parse error | datetime exception | Log, mark record for manual review |
| Strategy classification failure | Unknown type | Assign "unknown", flag for review |

---

*Document Version: 0.1*  
*Last Updated: December 2025*