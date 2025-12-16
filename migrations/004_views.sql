-- Migration 004: Convenience Views
-- Views for common query patterns and reporting

-- ============================================================================
-- Futu Order Summary View
-- ============================================================================

CREATE OR REPLACE VIEW v_futu_order_summary AS
SELECT
    o.account_id,
    o.market,
    o.code,
    o.stock_name,
    COUNT(*) as order_count,
    COUNT(*) FILTER (WHERE o.order_status = 'FILLED_ALL') as filled_count,
    COUNT(*) FILTER (WHERE o.order_status = 'CANCELLED_ALL') as cancelled_count,
    SUM(o.dealt_qty) as total_dealt_qty,
    SUM(o.dealt_qty * o.dealt_avg_price) as total_dealt_value,
    MIN(o.create_time) as first_order_time,
    MAX(o.create_time) as last_order_time
FROM futu_raw_orders o
GROUP BY o.account_id, o.market, o.code, o.stock_name;

-- ============================================================================
-- Futu Deal Summary View (Daily P&L by Symbol)
-- ============================================================================

CREATE OR REPLACE VIEW v_futu_daily_pnl AS
SELECT
    d.account_id,
    d.market,
    d.code,
    d.stock_name,
    DATE(d.create_time) as trade_date,
    SUM(CASE WHEN d.trd_side = 'BUY' THEN d.qty ELSE 0 END) as buy_qty,
    SUM(CASE WHEN d.trd_side = 'SELL' THEN d.qty ELSE 0 END) as sell_qty,
    SUM(CASE WHEN d.trd_side = 'BUY' THEN d.qty * d.price ELSE 0 END) as buy_value,
    SUM(CASE WHEN d.trd_side = 'SELL' THEN d.qty * d.price ELSE 0 END) as sell_value,
    COUNT(*) as deal_count
FROM futu_raw_deals d
GROUP BY d.account_id, d.market, d.code, d.stock_name, DATE(d.create_time);

-- ============================================================================
-- Futu Fee Summary View
-- ============================================================================

CREATE OR REPLACE VIEW v_futu_fee_summary AS
SELECT
    f.account_id,
    DATE(o.create_time) as trade_date,
    o.market,
    COUNT(DISTINCT f.order_id) as order_count,
    SUM(f.fee_amount) as total_fees,
    SUM(f.commission) as total_commission,
    SUM(f.platform_fee) as total_platform_fee,
    SUM(f.sec_fee) as total_sec_fee,
    SUM(f.taf_fee) as total_taf_fee,
    SUM(f.stamp_duty) as total_stamp_duty
FROM futu_raw_fees f
JOIN futu_raw_orders o ON f.order_id = o.order_id AND f.account_id = o.account_id
GROUP BY f.account_id, DATE(o.create_time), o.market;

-- ============================================================================
-- IB Execution Summary View
-- ============================================================================

CREATE OR REPLACE VIEW v_ib_execution_summary AS
SELECT
    e.account_id,
    e.symbol,
    e.sec_type,
    DATE(e.exec_time) as trade_date,
    SUM(CASE WHEN e.side = 'BOT' THEN e.shares ELSE 0 END) as bought_shares,
    SUM(CASE WHEN e.side = 'SLD' THEN e.shares ELSE 0 END) as sold_shares,
    SUM(CASE WHEN e.side = 'BOT' THEN e.shares * e.price ELSE 0 END) as bought_value,
    SUM(CASE WHEN e.side = 'SLD' THEN e.shares * e.price ELSE 0 END) as sold_value,
    COUNT(*) as execution_count,
    SUM(COALESCE(c.commission, 0)) as total_commission
FROM ib_raw_executions e
LEFT JOIN ib_raw_commissions c ON e.exec_id = c.exec_id
GROUP BY e.account_id, e.symbol, e.sec_type, DATE(e.exec_time);

-- ============================================================================
-- Risk Signal Summary View (Recent Breaches)
-- ============================================================================

CREATE OR REPLACE VIEW v_risk_signal_summary AS
SELECT
    signal_type,
    signal_level,
    COUNT(*) as signal_count,
    MIN(signal_time) as first_signal,
    MAX(signal_time) as last_signal,
    COUNT(*) FILTER (WHERE signal_time > NOW() - INTERVAL '1 hour') as last_hour_count,
    COUNT(*) FILTER (WHERE signal_time > NOW() - INTERVAL '24 hours') as last_day_count
FROM risk_signals
WHERE signal_time > NOW() - INTERVAL '7 days'
GROUP BY signal_type, signal_level
ORDER BY last_signal DESC;

-- ============================================================================
-- Sync State Overview View
-- ============================================================================

CREATE OR REPLACE VIEW v_sync_state_overview AS
SELECT
    broker,
    account_id,
    data_type,
    market,
    last_sync_time,
    last_record_time,
    records_synced,
    last_sync_status,
    CASE
        WHEN last_sync_status = 'FAILED' THEN 'ERROR'
        WHEN last_sync_time < NOW() - INTERVAL '24 hours' THEN 'STALE'
        WHEN last_sync_status = 'IN_PROGRESS' THEN 'RUNNING'
        ELSE 'OK'
    END as health_status,
    EXTRACT(EPOCH FROM (NOW() - last_sync_time)) / 3600 as hours_since_sync
FROM sync_state
ORDER BY broker, account_id, data_type, market;

-- ============================================================================
-- Latest Snapshots View (For Warm-Start)
-- ============================================================================

CREATE OR REPLACE VIEW v_latest_position_snapshots AS
SELECT DISTINCT ON (broker, account_id)
    id,
    snapshot_time,
    broker,
    account_id,
    positions,
    position_count
FROM position_snapshots
ORDER BY broker, account_id, snapshot_time DESC;

CREATE OR REPLACE VIEW v_latest_account_snapshots AS
SELECT DISTINCT ON (broker, account_id)
    id,
    snapshot_time,
    broker,
    account_id,
    account_data
FROM account_snapshots
ORDER BY broker, account_id, snapshot_time DESC;

-- ============================================================================
-- Risk Snapshot Time Series View (For Charts)
-- ============================================================================

CREATE OR REPLACE VIEW v_risk_snapshot_hourly AS
SELECT
    time_bucket('1 hour', snapshot_time) as hour,
    AVG(portfolio_value)::numeric(16,2) as avg_portfolio_value,
    AVG(total_delta)::numeric(12,2) as avg_delta,
    AVG(unrealized_pnl)::numeric(14,2) as avg_unrealized_pnl,
    AVG(daily_pnl)::numeric(14,2) as avg_daily_pnl,
    MAX(position_count) as max_positions,
    COUNT(*) as snapshot_count
FROM risk_snapshots
WHERE snapshot_time > NOW() - INTERVAL '7 days'
GROUP BY hour
ORDER BY hour DESC;

-- ============================================================================
-- Combined Trade History View (Both Brokers)
-- ============================================================================

CREATE OR REPLACE VIEW v_combined_trades AS
SELECT
    'FUTU' as broker,
    d.account_id,
    d.market,
    d.code as symbol,
    d.stock_name,
    d.trd_side as side,
    d.qty as quantity,
    d.price,
    d.qty * d.price as value,
    d.create_time as trade_time
FROM futu_raw_deals d
UNION ALL
SELECT
    'IB' as broker,
    e.account_id,
    NULL as market,
    e.symbol,
    NULL as stock_name,
    e.side,
    e.shares as quantity,
    e.price,
    e.shares * e.price as value,
    e.exec_time as trade_time
FROM ib_raw_executions e;
