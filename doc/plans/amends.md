# Persistence Amendments

**Status: IMPLEMENTED** (2025-12-08)

## Changes Made

### 1. Fee Capture (Critical) - DONE ✅

- **Futu fees**: `historical_loader.py:303-359` - Added `_fetch_and_persist_futu_fees()` with throttled batching
- **IB commissions**: `ib/adapter.py:619-658` - Added `execDetailsEvent` and `commissionReportEvent` subscriptions
- **Persistence**: `persistence_manager.py:667-709` - Added `_on_commission_report()` handler

### 2. Rate Limiting (Critical) - DONE ✅

- **Futu fee throttle**: `futu/adapter.py:559-562` - Added 3-second delay between 400-order batches (10 req/30s)
- **Historical loader**: Uses same throttle pattern

### 3. Raw Payload Storage (Major) - DONE ✅

- **Futu**: `futu/adapter.py:723-830` - Added `fetch_orders_raw()`, `fetch_deals_raw()`, `fetch_order_fees()`
- **IB**: Real-time fills now persist `raw_payload` via `TRADE_EXECUTED` event

### 4. IB Flex Refresh Schedule (Major) - DONE ✅

- **Config**: `historical_loader.py:49,60-68` - Added `flex_refresh_hours` parameter (default: 24)
- **Check**: `historical_loader.py:629-669` - Added `_should_refresh_flex()` method
- **Integration**: `historical_loader.py:224-233` - Flex refresh in `incremental_load()`

### 5. Fee Coverage Reconciliation (Medium) - DONE ✅

- **Method**: `persistence_manager.py:951-997` - Added `check_fee_coverage()` for Futu and IB

### 6. Event Types (Medium) - DONE ✅

- **New event**: `event_bus.py:33` - Added `COMMISSION_REPORT` event type

## Open Questions Resolved

1. **Normalized tables**: Keep existing `orders`/`trades` as normalized layer. Raw tables (`*_raw`) serve as audit trail. No additional `*_norm` tables needed.

2. **Flex schedule**: Configure via `HistoricalLoader(flex_refresh_hours=24)`. Token/query_id passed as constructor args. Daily refresh during `incremental_load()`.

## Original Findings (Archived)



  Gap Analysis Summary

  Overall Status: ~75% Complete

  ---
  Critical Gaps

  1. Normalized Layer Tables Not Implemented

  | Plan                                | Implementation                    |
  |-------------------------------------|-----------------------------------|
  | orders_norm, trades_norm, fees_norm | Using legacy orders/trades tables |

  Impact: No separation between raw audit trail and normalized data for downstream consumption.

  Suggestion: Add v4 migration with normalized tables:
  CREATE TABLE orders_norm (...);  -- Broker-agnostic schema with UTC timestamps
  CREATE TABLE trades_norm (...);
  CREATE TABLE fees_norm (...);

  2. IB Real-Time Event Capture Missing

  The plan (Section 2.2.3) specifies:
  ib.execDetailsEvent += self.on_exec_details
  ib.commissionReportEvent += self.on_commission

  Current state: Only batch reqExecutionsAsync() fetching, no real-time event subscription.

  Suggestion: Add to IbAdapter:
  def _subscribe_events(self):
      self.ib.execDetailsEvent += self._on_exec_details
      self.ib.commissionReportEvent += self._on_commission/c

  3. Incomplete Reconciliation Logic

  reconcile_orders_with_trades() checks:
  - ✅ MISSING_TRADES
  - ✅ QUANTITY_MISMATCH
  - ❌ Missing fee coverage (filled orders without fees)
  - ❌ Timestamp consistency (trade_time < order_time)

  Suggestion: Add to persistence_manager.py:
  def _check_fee_coverage(self):
      """Check all FILLED orders have fee records."""
      ...

  def _check_timestamp_consistency(self):
      """Check trade_time >= order_create_time."""
      ...

  ---
  Medium Gaps

  4. Configuration Not in YAML

  Plan Section 5.1 specifies persistent.yaml with:
  - reload.full_reload, lookback_days_default
  - futu.batch_fee_size, fee_rate_limit_window
  - ib.flex_token, ib.flex_query_id

  Current: Hardcoded in PersistenceConfig dataclass.

  Suggestion: Add to config/risk_config.yaml:
  persistence:
    historical:
      enabled: true
      ib_flex_token: ${IB_FLEX_TOKEN}
      ib_flex_query_id: "123456"
      full_reload_days: 365

  5. Historical Loader Incomplete

  historical_loader.py has stubs but needs:
  - _fetch_futu_history() with raw payload hook
  - _fetch_ib_flex_history() integration
  - Normalization pipeline

  ---
  What's Working Well ✅

  | Feature                                    | Status                 |
  |--------------------------------------------|------------------------|
  | Raw layer tables (v3 schema)               | ✅ Fully implemented    |
  | Futu fee batching (400 max, 10 req/30s)    | ✅ Excellent            |
  | Timezone handling (UTC conversion)         | ✅ Excellent            |
  | Strategy classifier (StrategyClassifierV1) | ✅ Complete             |
  | Time-window grouping (5s)                  | ✅ Implemented          |
  | Order strategy map table                   | ✅ Implemented          |
  | FlexParser structure                       | ✅ Basic implementation |

  ---
  Recommendations

  Priority 1 - Data Integrity

  1. Add normalized tables (orders_norm, trades_norm, fees_norm)
  2. Complete reconciliation logic (fee + timestamp checks)
  3. Finish historical_loader.py implementation

  Priority 2 - Real-Time Data

  1. Wire up IB execDetailsEvent and commissionReportEvent
  2. Persist raw payloads on real-time events

  Priority 3 - Configuration

  1. Move historical config to YAML
  2. Make time-window grouping configurable

  ---
  Quick Wins

  1. Add missing reconciliation queries to persistence_manager.py:
  # Add after line 873
  def _check_fee_coverage(self) -> List[Dict]:
      """Check all FILLED orders have fee records."""
      return self.db.fetch_all("""
          SELECT o.order_id, o.source, o.symbol
          FROM orders o
          LEFT JOIN fees_raw_futu f ON o.order_id = f.order_id
          WHERE o.status = 'FILLED' 
            AND o.source = 'FUTU' 
            AND f.order_id IS NULL
      """)

  2. Make time window configurable:
  # In strategy_classifier.py
  def __init__(self, time_window_seconds: int = 5):
      self.time_window_seconds = time_window_seconds



