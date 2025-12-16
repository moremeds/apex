-- Migration 003: Performance Indexes
-- Create indexes for common query patterns

-- ============================================================================
-- Futu Orders Indexes
-- ============================================================================

-- Primary access pattern: account + market + time (for history loading)
CREATE INDEX IF NOT EXISTS idx_futu_orders_account_market_time
    ON futu_raw_orders (account_id, market, create_time DESC);

-- Filter by symbol
CREATE INDEX IF NOT EXISTS idx_futu_orders_code
    ON futu_raw_orders (code);

-- Filter by status (for finding filled orders)
CREATE INDEX IF NOT EXISTS idx_futu_orders_status
    ON futu_raw_orders (order_status)
    WHERE order_status = 'FILLED_ALL';

-- Updated time for incremental sync
CREATE INDEX IF NOT EXISTS idx_futu_orders_updated
    ON futu_raw_orders (account_id, market, updated_time DESC);

-- ============================================================================
-- Futu Deals Indexes
-- ============================================================================

-- Primary access pattern: account + market + time
CREATE INDEX IF NOT EXISTS idx_futu_deals_account_market_time
    ON futu_raw_deals (account_id, market, create_time DESC);

-- Join with orders
CREATE INDEX IF NOT EXISTS idx_futu_deals_order_id
    ON futu_raw_deals (order_id);

-- Filter by symbol
CREATE INDEX IF NOT EXISTS idx_futu_deals_code
    ON futu_raw_deals (code);

-- ============================================================================
-- Futu Fees Indexes
-- ============================================================================

-- Primary lookup
CREATE INDEX IF NOT EXISTS idx_futu_fees_order_account
    ON futu_raw_fees (order_id, account_id);

-- ============================================================================
-- IB Executions Indexes
-- ============================================================================

-- Primary access pattern: account + time
CREATE INDEX IF NOT EXISTS idx_ib_executions_account_time
    ON ib_raw_executions (account_id, exec_time DESC);

-- Filter by symbol
CREATE INDEX IF NOT EXISTS idx_ib_executions_symbol
    ON ib_raw_executions (symbol);

-- Join with commissions
CREATE INDEX IF NOT EXISTS idx_ib_executions_exec_id
    ON ib_raw_executions (exec_id);

-- ============================================================================
-- IB Commissions Indexes
-- ============================================================================

-- Primary lookup
CREATE INDEX IF NOT EXISTS idx_ib_commissions_exec_id
    ON ib_raw_commissions (exec_id);

-- ============================================================================
-- Signal Indexes
-- ============================================================================

-- Risk signals by type and time
CREATE INDEX IF NOT EXISTS idx_risk_signals_type_time
    ON risk_signals (signal_type, signal_time DESC);

-- Risk signals by level (for breach filtering)
CREATE INDEX IF NOT EXISTS idx_risk_signals_level
    ON risk_signals (signal_level)
    WHERE signal_level IN ('HARD', 'SOFT');

-- Trade signals by source and time
CREATE INDEX IF NOT EXISTS idx_trade_signals_source_time
    ON trade_signals (signal_source, signal_time DESC);

-- GIN indexes for JSONB payload search
CREATE INDEX IF NOT EXISTS idx_risk_signals_payload_gin
    ON risk_signals USING GIN (payload);

CREATE INDEX IF NOT EXISTS idx_trade_signals_payload_gin
    ON trade_signals USING GIN (payload);

-- ============================================================================
-- Backtest Indexes
-- ============================================================================

-- By strategy
CREATE INDEX IF NOT EXISTS idx_backtests_strategy
    ON backtests (strategy_name, start_time DESC);

-- By status
CREATE INDEX IF NOT EXISTS idx_backtests_status
    ON backtests (status);

-- GIN index for metrics search
CREATE INDEX IF NOT EXISTS idx_backtests_metrics_gin
    ON backtests USING GIN (metrics);

-- ============================================================================
-- Snapshot Indexes
-- ============================================================================

-- Position snapshots: broker + account + time (for warm-start)
CREATE INDEX IF NOT EXISTS idx_position_snapshots_broker_account
    ON position_snapshots (broker, account_id, snapshot_time DESC);

-- Account snapshots: broker + account + time
CREATE INDEX IF NOT EXISTS idx_account_snapshots_broker_account
    ON account_snapshots (broker, account_id, snapshot_time DESC);

-- Risk snapshots: time (for historical analysis)
CREATE INDEX IF NOT EXISTS idx_risk_snapshots_time
    ON risk_snapshots (snapshot_time DESC);

-- ============================================================================
-- Sync State Indexes
-- ============================================================================

-- Primary access pattern
CREATE INDEX IF NOT EXISTS idx_sync_state_broker_data_type
    ON sync_state (broker, data_type);

-- Find stale syncs
CREATE INDEX IF NOT EXISTS idx_sync_state_last_sync
    ON sync_state (last_sync_time)
    WHERE last_sync_status != 'COMPLETED';

-- ============================================================================
-- Exchange Info Index
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_exchange_info_symbol
    ON exchange_info (symbol);
