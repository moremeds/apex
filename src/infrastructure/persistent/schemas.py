"""
PostgreSQL schema definitions for the persistent layer.

Design Principles:
- Raw Data: Preserve all original API payloads for audit/replay (JSONB)
- Normalized Data: Strict schema for downstream consumption
- Idempotent: All inserts use ON CONFLICT DO UPDATE (upsert)
"""

# =============================================================================
# RAW LAYER: Preserve original API payloads
# =============================================================================

ORDERS_RAW_FUTU = """
CREATE TABLE IF NOT EXISTS orders_raw_futu (
    broker TEXT NOT NULL DEFAULT 'FUTU',
    acc_id BIGINT NOT NULL,
    order_id TEXT NOT NULL,

    -- The complete API response row as JSONB
    payload JSONB NOT NULL,

    -- Extracted fields for indexing and querying (also in payload)
    code TEXT,                          -- Security code (e.g., "US.AAPL", "US.AAPL240119C190000")
    stock_name TEXT,                    -- Security name
    trd_side TEXT,                      -- Trade direction: BUY, SELL, BUY_BACK, SELL_SHORT
    order_type TEXT,                    -- Order type: NORMAL, MARKET, ABSOLUTE_LIMIT, etc.
    order_status TEXT,                  -- Order status: SUBMITTED, FILLED_ALL, CANCELLED_ALL, etc.
    qty NUMERIC(20, 4),                 -- Order quantity
    price NUMERIC(20, 8),               -- Limit price
    dealt_qty NUMERIC(20, 4),           -- Filled quantity
    dealt_avg_price NUMERIC(20, 8),     -- Average fill price

    -- Timestamps (raw strings from API, in US Eastern time)
    create_time_raw_str TEXT,
    update_time_raw_str TEXT,
    -- Converted to UTC for queries
    create_time_utc TIMESTAMPTZ,
    update_time_utc TIMESTAMPTZ,

    -- Additional order fields
    aux_price NUMERIC(20, 8),           -- Auxiliary price (stop price for stop orders)
    trail_type TEXT,                    -- Trailing type
    trail_value NUMERIC(20, 8),         -- Trailing value
    trail_spread NUMERIC(20, 8),        -- Trailing spread
    time_in_force TEXT,                 -- Time in force: DAY, GTC, etc.
    fill_outside_rth BOOLEAN,           -- Allow fill outside regular trading hours
    remark TEXT,                        -- Order remarks
    last_err_msg TEXT,                  -- Last error message

    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (acc_id, order_id)
);
"""

TRADES_RAW_FUTU = """
CREATE TABLE IF NOT EXISTS trades_raw_futu (
    broker TEXT NOT NULL DEFAULT 'FUTU',
    acc_id BIGINT NOT NULL,
    deal_id TEXT NOT NULL,
    order_id TEXT,

    -- Complete API response from history_deal_list_query
    payload JSONB NOT NULL,

    -- Extracted fields for indexing and querying (also in payload)
    code TEXT,                          -- Security code
    stock_name TEXT,                    -- Security name
    trd_side TEXT,                      -- Trade direction: BUY, SELL, BUY_BACK, SELL_SHORT
    qty NUMERIC(20, 4),                 -- Filled quantity
    price NUMERIC(20, 8),               -- Execution price
    status TEXT,                        -- Deal status

    -- Counter broker info
    counter_broker_id TEXT,             -- Counter broker ID
    counter_broker_name TEXT,           -- Counter broker name

    -- Timestamps (raw string in US Eastern time)
    trade_time_raw_str TEXT,
    -- Converted to UTC
    trade_time_utc TIMESTAMPTZ,

    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (acc_id, deal_id)
);
"""

FEES_RAW_FUTU = """
CREATE TABLE IF NOT EXISTS fees_raw_futu (
    broker TEXT NOT NULL DEFAULT 'FUTU',
    acc_id BIGINT NOT NULL,
    order_id TEXT NOT NULL,

    -- Extracted for quick access
    fee_amount NUMERIC(20, 8),          -- Total fee amount
    fee_list JSONB,                     -- Breakdown: [["Commission", 0.99], ["Platform Fee", 0.0], ...]

    -- Complete API response from order_fee_query
    payload JSONB NOT NULL,

    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (acc_id, order_id)
);
"""

ORDERS_RAW_IB = """
CREATE TABLE IF NOT EXISTS orders_raw_ib (
    broker TEXT NOT NULL DEFAULT 'IB',
    account TEXT NOT NULL,
    perm_id BIGINT,
    client_order_id TEXT,
    order_ref TEXT,
    order_id_composite TEXT NOT NULL,  -- Generated: perm_id or client_order_id

    -- Complete order object from IB API or Flex Report
    payload JSONB NOT NULL,

    create_time_raw_str TEXT,
    update_time_raw_str TEXT,
    source TEXT DEFAULT 'API',  -- 'API' or 'FLEX'
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account, order_id_composite)
);
"""

TRADES_RAW_IB = """
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
"""

FEES_RAW_IB = """
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
"""

# =============================================================================
# NORMALIZED LAYER: Broker-agnostic schema for analytics
# =============================================================================

APEX_ORDER = """
CREATE TABLE IF NOT EXISTS apex_order (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    order_uid TEXT NOT NULL,  -- Unified: {broker}_{account}_{order_id}

    -- Instrument
    instrument_type TEXT,      -- STOCK, OPTION, FUTURE, FOREX
    symbol TEXT NOT NULL,
    stock_name TEXT,           -- Security name from broker
    underlying TEXT,
    exchange TEXT,

    -- Option-specific
    strike NUMERIC(20, 4),
    expiry DATE,
    option_right TEXT,         -- CALL, PUT

    -- Order details
    side TEXT NOT NULL,        -- BUY, SELL
    trd_side TEXT,             -- Original broker side: BUY, SELL, BUY_BACK, SELL_SHORT
    qty NUMERIC(20, 4),
    limit_price NUMERIC(20, 8),
    order_type TEXT,           -- MARKET, LIMIT, STOP, etc.
    order_type_raw TEXT,       -- Original broker order type
    time_in_force TEXT,        -- DAY, GTC, IOC, etc.

    -- Stop/Trailing order fields
    aux_price NUMERIC(20, 8),  -- Auxiliary price (stop price)
    trail_type TEXT,           -- Trailing type
    trail_value NUMERIC(20, 8),-- Trailing value
    trail_spread NUMERIC(20, 8),-- Trailing spread

    -- Order options
    fill_outside_rth BOOLEAN,  -- Allow fill outside regular trading hours

    -- Status
    status TEXT,               -- WORKING, FILLED, CANCELLED, REJECTED
    status_raw TEXT,           -- Original broker status
    filled_qty NUMERIC(20, 4),
    avg_fill_price NUMERIC(20, 8),

    -- Error/Remarks
    last_err_msg TEXT,         -- Last error message
    remark TEXT,               -- Order remarks

    -- Timestamps (UTC)
    create_time_utc TIMESTAMPTZ,
    update_time_utc TIMESTAMPTZ,

    -- Fees (aggregated from fees_norm)
    total_fee NUMERIC(20, 8) DEFAULT 0,

    -- Metadata
    order_reconstructed BOOLEAN DEFAULT FALSE,

    -- Reference back to raw table
    raw_ref JSONB,

    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (broker, account_id, order_uid)
);
"""

APEX_TRADES = """
CREATE TABLE IF NOT EXISTS apex_trades (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    trade_uid TEXT NOT NULL,   -- Unified: {broker}_{account}_{deal_id/exec_id}
    order_uid TEXT,            -- FK to apex_order

    -- Instrument
    instrument_type TEXT,
    symbol TEXT NOT NULL,
    stock_name TEXT,           -- Security name from broker
    underlying TEXT,

    -- Option-specific
    strike NUMERIC(20, 4),
    expiry DATE,
    option_right TEXT,

    -- Execution details
    side TEXT NOT NULL,        -- Normalized: BUY, SELL
    trd_side TEXT,             -- Original broker side: BUY, SELL, BUY_BACK, SELL_SHORT
    qty NUMERIC(20, 4) NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    exchange TEXT,
    status TEXT,               -- Deal status from broker

    -- Counter broker info (for order routing analysis)
    counter_broker_id TEXT,
    counter_broker_name TEXT,

    -- Position effect and PnL (for strategy classification)
    position_effect TEXT,      -- OPEN, CLOSE
    realized_pnl NUMERIC(20, 4),

    -- Timestamp (UTC)
    trade_time_utc TIMESTAMPTZ NOT NULL,
    update_time_utc TIMESTAMPTZ,

    -- Metadata
    raw_ref JSONB,
    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (broker, account_id, trade_uid)
);
"""

APEX_FEES = """
CREATE TABLE IF NOT EXISTS apex_fees (
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
"""

# =============================================================================
# STRATEGY LAYER: Trade grouping and classification
# =============================================================================

APEX_STRATEGY_ANALYSIS = """
CREATE TABLE IF NOT EXISTS apex_strategy_analysis (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    order_uid TEXT NOT NULL,        -- The opening order ID (primary identifier)

    -- Strategy classification
    strategy_id TEXT NOT NULL,      -- Hash of leg combination for grouping
    strategy_type TEXT NOT NULL,    -- Pattern: long_call, bull_put_spread, iron_condor, etc.
    strategy_name TEXT,             -- Human-readable: "Long Call on AAPL 150"
    strategy_outcome TEXT,          -- Outcome: take_profit, stop_loss, open, close_flat
    is_closed BOOLEAN DEFAULT FALSE,-- Whether the strategy has been fully closed

    -- Timing
    trade_duration TEXT,            -- INTRADAY, SWING, UNKNOWN

    -- All involved orders (open and close)
    involved_orders JSONB,          -- Array of {order_uid, side, position_effect, price, qty}

    -- Classification metadata
    confidence NUMERIC(5, 4),       -- 0.0 to 1.0
    legs JSONB,                     -- All legs with details

    -- Timestamps
    open_time TIMESTAMPTZ,          -- When position was opened
    close_time TIMESTAMPTZ,         -- When position was closed (if closed)
    updated_time TIMESTAMPTZ,       -- Last update time from source

    classify_version TEXT DEFAULT 'v1',
    insert_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (broker, account_id, order_uid)
);
"""

# =============================================================================
# SIGNAL LAYER: Extensible for risk/trading signals
# =============================================================================

RISK_SIGNALS = """
CREATE TABLE IF NOT EXISTS risk_signals (
    account_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,      -- VIX_SPIKE, CONCENTRATION, THETA_DECAY, etc.
    payload JSONB NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    version TEXT,
    PRIMARY KEY (account_id, signal_id)
);
"""

TRADING_SIGNALS = """
CREATE TABLE IF NOT EXISTS trading_signals (
    account_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,      -- ENTRY, EXIT, HEDGE, ROLL
    payload JSONB NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    version TEXT,
    PRIMARY KEY (account_id, signal_id)
);
"""

# =============================================================================
# POSITION SNAPSHOTS: Point-in-time state
# =============================================================================

POSITIONS_SNAPSHOT = """
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
"""

# =============================================================================
# INDEXES for query performance
# =============================================================================

INDEXES = """
-- Raw orders indexes (for fast queries on extracted fields)
CREATE INDEX IF NOT EXISTS idx_orders_raw_futu_code ON orders_raw_futu(code);
CREATE INDEX IF NOT EXISTS idx_orders_raw_futu_status ON orders_raw_futu(order_status);
CREATE INDEX IF NOT EXISTS idx_orders_raw_futu_create_time ON orders_raw_futu(create_time_utc);
CREATE INDEX IF NOT EXISTS idx_orders_raw_futu_update_time ON orders_raw_futu(update_time_utc);

-- Raw trades indexes
CREATE INDEX IF NOT EXISTS idx_trades_raw_futu_code ON trades_raw_futu(code);
CREATE INDEX IF NOT EXISTS idx_trades_raw_futu_trade_time ON trades_raw_futu(trade_time_utc);
CREATE INDEX IF NOT EXISTS idx_trades_raw_futu_order ON trades_raw_futu(order_id);

-- Orders normalized indexes
CREATE INDEX IF NOT EXISTS idx_apex_order_symbol ON apex_order(symbol);
CREATE INDEX IF NOT EXISTS idx_apex_order_underlying ON apex_order(underlying);
CREATE INDEX IF NOT EXISTS idx_apex_order_create_time ON apex_order(create_time_utc);
CREATE INDEX IF NOT EXISTS idx_apex_order_update_time ON apex_order(update_time_utc);
CREATE INDEX IF NOT EXISTS idx_apex_order_status ON apex_order(status);
CREATE INDEX IF NOT EXISTS idx_apex_order_trd_side ON apex_order(trd_side);

-- Trades normalized indexes
CREATE INDEX IF NOT EXISTS idx_apex_trades_symbol ON apex_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_apex_trades_trade_time ON apex_trades(trade_time_utc);
CREATE INDEX IF NOT EXISTS idx_apex_trades_order ON apex_trades(order_uid);
CREATE INDEX IF NOT EXISTS idx_apex_trades_trd_side ON apex_trades(trd_side);
CREATE INDEX IF NOT EXISTS idx_apex_trades_position_effect ON apex_trades(position_effect);

-- Strategy analysis indexes
CREATE INDEX IF NOT EXISTS idx_apex_strategy_type ON apex_strategy_analysis(strategy_type);
CREATE INDEX IF NOT EXISTS idx_apex_strategy_id ON apex_strategy_analysis(strategy_id);
CREATE INDEX IF NOT EXISTS idx_apex_strategy_duration ON apex_strategy_analysis(trade_duration);
CREATE INDEX IF NOT EXISTS idx_apex_strategy_outcome ON apex_strategy_analysis(strategy_outcome);
CREATE INDEX IF NOT EXISTS idx_apex_strategy_is_closed ON apex_strategy_analysis(is_closed);
CREATE INDEX IF NOT EXISTS idx_apex_strategy_open_time ON apex_strategy_analysis(open_time);

-- JSONB indexes for payload queries
CREATE INDEX IF NOT EXISTS idx_orders_raw_futu_payload ON orders_raw_futu USING GIN (payload);
CREATE INDEX IF NOT EXISTS idx_trades_raw_futu_payload ON trades_raw_futu USING GIN (payload);

-- Position snapshot indexes
CREATE INDEX IF NOT EXISTS idx_positions_snapshot_time ON positions_snapshot(snapshot_time_utc);
CREATE INDEX IF NOT EXISTS idx_positions_snapshot_account ON positions_snapshot(account_id);
"""

# =============================================================================
# Combined schema for initialization
# =============================================================================

ALL_TABLES = [
    # Raw tables
    ORDERS_RAW_FUTU,
    TRADES_RAW_FUTU,
    FEES_RAW_FUTU,
    ORDERS_RAW_IB,
    TRADES_RAW_IB,
    FEES_RAW_IB,
    # Normalized tables (apex_* naming)
    APEX_ORDER,
    APEX_TRADES,
    APEX_FEES,
    # Strategy
    APEX_STRATEGY_ANALYSIS,
    # Signals
    RISK_SIGNALS,
    TRADING_SIGNALS,
    # Snapshots
    POSITIONS_SNAPSHOT,
    # Indexes
    INDEXES,
]

SCHEMA_SQL = "\n".join(ALL_TABLES)


def get_schema_sql() -> str:
    """Return the complete schema SQL for initialization."""
    return SCHEMA_SQL
