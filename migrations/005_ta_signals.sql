-- Migration 005: Technical Analysis Signal Persistence
-- Creates tables for TA signals, indicator values, and confluence scores
-- with TimescaleDB hypertables and LISTEN/NOTIFY triggers for real-time TUI updates

-- ============================================================================
-- TA Trading Signals Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ta_signals (
    time TIMESTAMPTZ NOT NULL,
    signal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    category TEXT NOT NULL,          -- momentum, trend, volatility, volume, pattern
    indicator TEXT NOT NULL,
    direction TEXT NOT NULL,         -- buy, sell, alert
    strength INTEGER NOT NULL,
    priority TEXT NOT NULL,          -- high, medium, low
    trigger_rule TEXT NOT NULL,
    current_value DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    previous_value DOUBLE PRECISION,
    message TEXT,
    cooldown_until TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable with 1-day chunks (high-frequency signal data)
SELECT create_hypertable('ta_signals', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_ta_signals_symbol_time
    ON ta_signals (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_ta_signals_category_time
    ON ta_signals (category, time DESC);
CREATE INDEX IF NOT EXISTS idx_ta_signals_indicator_time
    ON ta_signals (indicator, time DESC);
CREATE INDEX IF NOT EXISTS idx_ta_signals_signal_id
    ON ta_signals (signal_id, time DESC);

-- ============================================================================
-- Indicator Values Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS indicator_values (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    indicator TEXT NOT NULL,
    state JSONB NOT NULL,            -- Flexible: {value: 45.2, zone: "oversold", ...}
    previous_state JSONB,
    bar_close DOUBLE PRECISION,      -- Reference price at this timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable('indicator_values', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index for charting queries: filter by symbol/timeframe/indicator, order by time
CREATE INDEX IF NOT EXISTS idx_indicator_values_lookup
    ON indicator_values (symbol, timeframe, indicator, time DESC);

-- ============================================================================
-- Confluence Scores Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS confluence_scores (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    alignment_score DOUBLE PRECISION NOT NULL,  -- -1.0 to +1.0
    bullish_count INTEGER NOT NULL,
    bearish_count INTEGER NOT NULL,
    neutral_count INTEGER NOT NULL,
    total_indicators INTEGER NOT NULL,
    dominant_direction TEXT,         -- bullish, bearish, neutral
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable('confluence_scores', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index for TUI queries
CREATE INDEX IF NOT EXISTS idx_confluence_symbol_tf_time
    ON confluence_scores (symbol, timeframe, time DESC);

-- ============================================================================
-- Compression Policies (compress after 7 days to save space)
-- ============================================================================

-- TA Signals: segment by symbol and indicator for efficient queries
ALTER TABLE ta_signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, indicator'
);
SELECT add_compression_policy('ta_signals', INTERVAL '7 days', if_not_exists => TRUE);

-- Indicator Values: segment by symbol, timeframe, and indicator
ALTER TABLE indicator_values SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, timeframe, indicator'
);
SELECT add_compression_policy('indicator_values', INTERVAL '7 days', if_not_exists => TRUE);

-- Confluence Scores: segment by symbol and timeframe
ALTER TABLE confluence_scores SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, timeframe'
);
SELECT add_compression_policy('confluence_scores', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- NOTIFY Triggers for Real-Time TUI Updates
-- ============================================================================

-- Function to notify on new TA signal
CREATE OR REPLACE FUNCTION notify_ta_signal_insert()
RETURNS TRIGGER AS $$
BEGIN
    -- Send lightweight JSON payload via PostgreSQL NOTIFY
    -- TUI listens to 'ta_signal_updates' channel
    PERFORM pg_notify('ta_signal_updates', json_build_object(
        'type', 'signal',
        'signal_id', NEW.signal_id,
        'symbol', NEW.symbol,
        'timeframe', NEW.timeframe,
        'direction', NEW.direction,
        'indicator', NEW.indicator,
        'strength', NEW.strength,
        'time', NEW.time
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger fires after every INSERT on ta_signals
DROP TRIGGER IF EXISTS ta_signal_insert_notify ON ta_signals;
CREATE TRIGGER ta_signal_insert_notify
    AFTER INSERT ON ta_signals
    FOR EACH ROW EXECUTE FUNCTION notify_ta_signal_insert();

-- Function to notify on new indicator value (optional - may be chatty)
CREATE OR REPLACE FUNCTION notify_indicator_insert()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('indicator_updates', json_build_object(
        'type', 'indicator',
        'symbol', NEW.symbol,
        'timeframe', NEW.timeframe,
        'indicator', NEW.indicator,
        'time', NEW.time
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for indicator updates (can be disabled if too chatty)
DROP TRIGGER IF EXISTS indicator_insert_notify ON indicator_values;
CREATE TRIGGER indicator_insert_notify
    AFTER INSERT ON indicator_values
    FOR EACH ROW EXECUTE FUNCTION notify_indicator_insert();

-- Function to notify on confluence update
CREATE OR REPLACE FUNCTION notify_confluence_insert()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('confluence_updates', json_build_object(
        'type', 'confluence',
        'symbol', NEW.symbol,
        'timeframe', NEW.timeframe,
        'alignment_score', NEW.alignment_score,
        'dominant_direction', NEW.dominant_direction,
        'time', NEW.time
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for confluence updates
DROP TRIGGER IF EXISTS confluence_insert_notify ON confluence_scores;
CREATE TRIGGER confluence_insert_notify
    AFTER INSERT ON confluence_scores
    FOR EACH ROW EXECUTE FUNCTION notify_confluence_insert();

-- ============================================================================
-- Optional: Retention Policies (uncomment to enable auto-deletion)
-- ============================================================================

-- Keep TA signals for 90 days
-- SELECT add_retention_policy('ta_signals', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep indicator values for 90 days (charting needs historical data)
-- SELECT add_retention_policy('indicator_values', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep confluence scores for 90 days
-- SELECT add_retention_policy('confluence_scores', INTERVAL '90 days', if_not_exists => TRUE);
