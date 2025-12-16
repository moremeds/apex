-- Migration 002: Create TimescaleDB Hypertables
-- Converts time-series tables to hypertables for efficient querying and compression

-- ============================================================================
-- Futu Raw Tables (High-Volume History)
-- ============================================================================

-- Orders hypertable (monthly chunks)
SELECT create_hypertable('futu_raw_orders', 'create_time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Deals hypertable (monthly chunks)
SELECT create_hypertable('futu_raw_deals', 'create_time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ============================================================================
-- IB Raw Tables
-- ============================================================================

-- Executions hypertable (monthly chunks)
SELECT create_hypertable('ib_raw_executions', 'exec_time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Commissions hypertable (monthly chunks)
SELECT create_hypertable('ib_raw_commissions', 'exec_time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- ============================================================================
-- Signal Tables (Medium-Frequency)
-- ============================================================================

-- Risk signals hypertable (weekly chunks for faster queries)
SELECT create_hypertable('risk_signals', 'signal_time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Trade signals hypertable (weekly chunks)
SELECT create_hypertable('trade_signals', 'signal_time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ============================================================================
-- Snapshot Tables (High-Frequency, Warm-Start)
-- ============================================================================

-- Position snapshots (daily chunks for fast warm-start lookups)
SELECT create_hypertable('position_snapshots', 'snapshot_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Account snapshots (daily chunks)
SELECT create_hypertable('account_snapshots', 'snapshot_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Risk snapshots (daily chunks - highest frequency for historical analysis)
SELECT create_hypertable('risk_snapshots', 'snapshot_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ============================================================================
-- Compression Policies (Compress older data to save space)
-- ============================================================================

-- Futu orders: compress after 7 days, segment by account/market for query efficiency
ALTER TABLE futu_raw_orders SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'account_id, market'
);
SELECT add_compression_policy('futu_raw_orders', INTERVAL '7 days', if_not_exists => TRUE);

-- Futu deals: compress after 7 days
ALTER TABLE futu_raw_deals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'account_id, market'
);
SELECT add_compression_policy('futu_raw_deals', INTERVAL '7 days', if_not_exists => TRUE);

-- IB executions: compress after 7 days
ALTER TABLE ib_raw_executions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'account_id'
);
SELECT add_compression_policy('ib_raw_executions', INTERVAL '7 days', if_not_exists => TRUE);

-- IB commissions: compress after 7 days
ALTER TABLE ib_raw_commissions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'account_id'
);
SELECT add_compression_policy('ib_raw_commissions', INTERVAL '7 days', if_not_exists => TRUE);

-- Risk signals: compress after 7 days
ALTER TABLE risk_signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'signal_type'
);
SELECT add_compression_policy('risk_signals', INTERVAL '7 days', if_not_exists => TRUE);

-- Risk snapshots: compress after 7 days (keep recent for fast queries)
ALTER TABLE risk_snapshots SET (
    timescaledb.compress
);
SELECT add_compression_policy('risk_snapshots', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- Data Retention Policies (Optional - uncomment to enable auto-deletion)
-- ============================================================================

-- Uncomment to auto-delete old data:
-- SELECT add_retention_policy('futu_raw_orders', INTERVAL '365 days', if_not_exists => TRUE);
-- SELECT add_retention_policy('futu_raw_deals', INTERVAL '365 days', if_not_exists => TRUE);
-- SELECT add_retention_policy('risk_signals', INTERVAL '90 days', if_not_exists => TRUE);
-- SELECT add_retention_policy('risk_snapshots', INTERVAL '365 days', if_not_exists => TRUE);
-- SELECT add_retention_policy('position_snapshots', INTERVAL '30 days', if_not_exists => TRUE);
-- SELECT add_retention_policy('account_snapshots', INTERVAL '30 days', if_not_exists => TRUE);
