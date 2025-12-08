# Persistent Layer PRD Review & Integration Plan

**Review Date:** December 2025
**Status:** In Review

---

## 1. Executive Summary

The PRD proposes a PostgreSQL-based persistent layer for historical order/trade data with:
- Raw data preservation (JSONB payloads)
- Normalized schema for analytics
- Strategy classification (rule-based)
- Dual broker support (Futu + IB)
- Standalone backfill capability

**Overall Assessment:** Well-designed, fits the existing architecture. Several integration points align well with current codebase.

---

## 2. Alignment with Current System

### 2.1 What Aligns Well

| PRD Component | Current System | Assessment |
|---------------|----------------|------------|
| Module path `persistent/` | Would go under `src/infrastructure/` | ✅ Fits layered architecture |
| Order/Trade models | `src/models/order.py` already defines `Order`, `Trade` | ✅ Can reuse/extend |
| Futu adapter | `src/infrastructure/adapters/futu/adapter.py` has `fetch_orders_raw()`, `fetch_deals_raw()` | ✅ Direct integration |
| IB adapter | `src/infrastructure/adapters/ib/adapter.py` exists with similar interface | ✅ Direct integration |
| Timezone handling | `src/utils/timezone.py` exists | ✅ Can leverage |
| Strategy detection | `src/domain/services/strategy_detector.py` exists | ⚠️ May overlap with `classify/` |

### 2.2 Gaps to Address

| Gap | Description | Priority |
|-----|-------------|----------|
| **PostgreSQL dependency** | Currently no DB dependency in the project | High |
| **Async vs Sync** | PRD shows asyncpg, but also psycopg2 sync. Current adapters are async | Medium |
| **Fee model** | PRD has separate `fees_norm` table; current `Trade` has `commission` field | Medium |
| **Strategy classifier overlap** | PRD's `strategy_classifier_v1.py` overlaps with existing `strategy_detector.py` | Low |
| **Position snapshots** | PRD adds `positions_snapshot` table; Gap Analysis identified this as complete gap | High |

### 2.3 Architectural Fit

The PRD's module structure should map to existing architecture:

```
Current:                          PRD Proposed:
src/infrastructure/               risk_engine/persistent/
├── adapters/                     ├── adapters/
│   ├── futu/                     │   ├── futu_adapter.py      → Extend existing
│   └── ib/                       │   └── ib_adapter.py        → Extend existing
├── stores/                       ├── storage/
│   ├── position_store.py         │   ├── postgres_store.py    → NEW
│   └── market_data_store.py      │   └── schemas.py           → NEW
└── monitoring/                   ├── normalize/               → NEW
                                  ├── classify/                → Merge with strategy_detector
                                  ├── reconcile/               → NEW
                                  └── orchestrator/            → NEW
```

**Recommended structure:**
```
src/infrastructure/
├── persistent/                   # NEW top-level module
│   ├── __init__.py
│   ├── store.py                  # PostgresStore
│   ├── schemas.py                # SQL schema definitions
│   ├── normalizers/
│   │   ├── base.py
│   │   ├── futu.py
│   │   └── ib.py
│   ├── reconciler.py
│   └── orchestrator.py           # Backfill & incremental load
```

---

## 3. Integration Points

### 3.1 Reuse Existing Raw Fetch Methods

The Futu adapter already has:
```python
async def fetch_orders_raw(self, days_back: int = 30, ...) -> List[Dict]
async def fetch_deals_raw(self, days_back: int = 30) -> List[Dict]
async def fetch_order_fees(self, order_ids: List[str]) -> List[Dict]
```

These can be used directly by the persistent layer without modification.

### 3.2 Extend for IB Flex Reports

The IB adapter needs:
- Flex report download/parse capability (PRD references `ibflex` library)
- Current `src/infrastructure/adapters/ib/flex_parser.py` exists but unclear if complete

### 3.3 Strategy Classification

PRD's `StrategyClassifierV1` overlaps with existing `StrategyDetector`. Options:
1. **Merge**: Enhance existing `strategy_detector.py` with PRD's algorithm
2. **Replace**: Use PRD's version entirely
3. **Coexist**: Keep both (not recommended - creates confusion)

**Recommendation:** Option 1 (Merge) - the existing detector can be enhanced with the time-window clustering logic.

---

## 4. Standalone Backfill Capability

The PRD's CLI entry point is designed for standalone operation:
```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-reload', action='store_true)
    parser.add_argument('--config', default='config/persistent.yaml')
    ...
```

**Implementation approach:**
1. Create `scripts/backfill.py` as standalone entry point
2. Reuse existing adapters via import
3. Can run independently of main orchestrator

---

## 5. Clarifications Needed

### Q1: Database Choice
The PRD uses PostgreSQL. Questions:
- Do you want PostgreSQL specifically, or would SQLite be acceptable for MVP (simpler ops)?
- If PostgreSQL, do you have an existing instance, or should we include Docker setup?

### Q2: Async vs Sync for Backfill
- Should the backfill script be async (matches existing adapters) or sync (simpler for one-time batch)?
- The PRD shows both `asyncpg` and `psycopg2` patterns

### Q3: Strategy Classification Integration
- Should we merge the PRD's classifier into existing `strategy_detector.py`?
- Or keep them separate (one for real-time positions, one for historical trades)?

### Q4: Fee Granularity
- PRD normalizes fees to separate `fees_norm` table with breakdown by fee_type
- Current `Trade.commission` is a single float
- Do you need the fee breakdown (COMMISSION, PLATFORM, EXCHANGE, etc.)?

### Q5: Position Snapshots
- PRD includes `positions_snapshot` table for point-in-time state
- This would also address the Gap Analysis "Snapshot Persistence" gap
- Should this be part of Phase 1, or separate feature?

### Q6: IB Historical Data
- PRD notes IB API only provides ~1-2 days of execution history
- For longer history, Flex Reports are required
- Do you have IB Flex Query already configured, or need setup instructions?

### Q7: Backfill Scope
- PRD suggests 365-day default lookback
- Futu `order_fee_query` only supports orders since 2018-01-01
- What's your expected historical data range?

---

## 6. Proposed Implementation Phases

### Phase 1: Foundation (Recommended MVP)
- [ ] PostgreSQL schema creation (raw + norm tables)
- [ ] `PostgresStore` with upsert methods
- [ ] Futu normalizer (leverage existing raw fetch)
- [ ] Standalone backfill CLI

### Phase 2: IB Integration
- [ ] IB Flex report integration
- [ ] IB normalizer
- [ ] Merge IB data into normalized tables

### Phase 3: Classification & Reconciliation
- [ ] Strategy classifier (merge with existing or new)
- [ ] Order-Trade-Fee reconciliation
- [ ] Anomaly flagging

### Phase 4: Position Snapshots
- [ ] `positions_snapshot` table
- [ ] Periodic snapshot capture
- [ ] Historical query capability

### Phase 5: Live Integration
- [ ] Hook into main orchestrator
- [ ] Incremental load on each cycle
- [ ] Real-time event-driven updates

---

## 7. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| PostgreSQL ops overhead | Medium | Consider SQLite for MVP if simpler ops preferred |
| IB Flex API complexity | Medium | Start with Futu-only, add IB in Phase 2 |
| Schema migrations | Low | Use versioned SQL files, not ORM |
| Rate limiting during backfill | Medium | PRD already accounts for Futu rate limits |
| Strategy misclassification | Low | Start with high-confidence rules only |

---

## 8. Dependencies to Add

```toml
# pyproject.toml additions
asyncpg = "^0.29.0"      # Async PostgreSQL driver
psycopg2-binary = "^2.9" # Sync PostgreSQL driver (for backfill script)
ibflex = "^0.15"         # IB Flex report parser (Phase 2)
```

---

## 9. Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | PostgreSQL | Full PRD spec, JSONB, production-ready |
| Strategy classifier | Replace existing | Use PRD's `StrategyClassifierV1`, remove `strategy_detector.py` |
| Fee model | Full breakdown | Separate `fees_norm` table with fee_type column |
| Backfill range | 365 days | Full year of history |
| IB integration | Phase 1 | Flex Query already configured |
| Position snapshots | Phase 1 | Include `positions_snapshot` table |
| PG instance | Existing | User has PostgreSQL available |

---

## 10. Final Implementation Plan

### Phase 1: Complete Persistent Layer (Single Phase)

Since IB is ready and snapshots are wanted in Phase 1, we'll implement everything together:

#### Step 1: Infrastructure Setup
- [ ] Create `src/infrastructure/persistent/` module structure
- [ ] Add `asyncpg`, `ibflex` to dependencies
- [ ] Create `config/persistent.yaml` configuration file

#### Step 2: PostgreSQL Schema
- [ ] Create `schemas.py` with all SQL DDL:
  - Raw tables: `orders_raw_futu`, `trades_raw_futu`, `fees_raw_futu`
  - Raw tables: `orders_raw_ib`, `trades_raw_ib`, `fees_raw_ib`
  - Normalized: `orders_norm`, `trades_norm`, `fees_norm`
  - Strategy: `order_strategy_map`
  - Signals: `risk_signals`, `trading_signals`
  - Snapshots: `positions_snapshot`
- [ ] Include all indexes as per PRD

#### Step 3: PostgresStore Implementation
- [ ] `store.py` with `PostgresStore` class
- [ ] Connection pooling via `asyncpg.create_pool()`
- [ ] Upsert methods for all raw and normalized tables
- [ ] Query methods: `get_data_boundaries()`, `query_raw_payload()`
- [ ] Snapshot methods: `save_position_snapshot()`, `get_snapshot_at()`

#### Step 4: Normalizers
- [ ] `normalizers/base.py` - abstract normalizer interface
- [ ] `normalizers/futu.py` - Futu raw → normalized conversion
  - Timezone handling (US/HK market detection)
  - Option symbol parsing
  - UID generation: `FUTU_{acc_id}_{order_id}`
- [ ] `normalizers/ib.py` - IB raw → normalized conversion
  - Flex report parsing via `ibflex`
  - Handle both API and Flex sources
  - UID generation: `IB_{account}_{exec_id}`

#### Step 5: Strategy Classifier
- [ ] Create `classify/strategy_classifier.py` from PRD
- [ ] Implement `StrategyType` enum with all strategy types
- [ ] Time-window grouping (5-second window)
- [ ] Classification rules for spreads, straddles, condors, etc.
- [ ] Delete existing `src/domain/services/strategy_detector.py`

#### Step 6: Reconciler
- [ ] `reconciler.py` with order-trade-fee validation
- [ ] Check: `filled_qty == SUM(trade.qty)`
- [ ] Check: every filled order has fee record
- [ ] Anomaly flagging and reporting

#### Step 7: Orchestrator & CLI
- [ ] `orchestrator.py` with `PersistenceOrchestrator` class
- [ ] Support both full reload and incremental load
- [ ] Create `scripts/backfill.py` standalone entry point:
  ```bash
  python scripts/backfill.py --full-reload --config config/persistent.yaml
  python scripts/backfill.py --days 30  # Incremental
  ```

#### Step 8: Integration with Main System
- [ ] Add persistent layer to main `Orchestrator` startup
- [ ] Hook position snapshot capture to risk engine cycle
- [ ] Add incremental trade sync on each loop (optional)

### File Structure After Implementation

```
src/infrastructure/
├── persistent/
│   ├── __init__.py
│   ├── store.py                  # PostgresStore class
│   ├── schemas.py                # SQL DDL definitions
│   ├── normalizers/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract normalizer
│   │   ├── futu.py               # Futu normalizer
│   │   └── ib.py                 # IB normalizer
│   ├── classify/
│   │   ├── __init__.py
│   │   └── strategy_classifier.py
│   ├── reconciler.py
│   └── orchestrator.py
scripts/
└── backfill.py                   # Standalone CLI entry point
config/
└── persistent.yaml               # Persistence configuration
```

### Configuration Template

```yaml
# config/persistent.yaml
persistent:
  storage:
    dsn: "postgresql://risk_user:password@localhost:5432/apex_risk"
    pool_min: 2
    pool_max: 10

  reload:
    full_reload: false
    lookback_days_default: 365

  futu:
    # Will use acc_id from existing futu adapter config
    batch_fee_size: 400
    fee_rate_limit_window: 30

  ib:
    # Will use account from existing ib adapter config
    flex_token: "${IB_FLEX_TOKEN}"
    flex_query_id: "${IB_FLEX_QUERY_ID}"

  snapshots:
    enabled: true
    interval_seconds: 300  # Snapshot every 5 minutes
```

---

## 11. Testing Strategy

### Unit Tests
- [ ] Normalizer tests: Futu/IB raw → normalized conversion
- [ ] Strategy classifier tests: All strategy type detection
- [ ] Reconciler tests: Missing trade/fee detection

### Integration Tests
- [ ] PostgreSQL upsert idempotency
- [ ] Full backfill with rate limit handling
- [ ] IB Flex download and parse

### Validation Queries (as per PRD Section 6)
```sql
-- Data completeness
SELECT COUNT(*) FROM orders_raw_futu;
SELECT COUNT(*) FROM orders_norm WHERE broker = 'FUTU';

-- Fee coverage
SELECT COUNT(*) AS missing_fees
FROM orders_norm o
LEFT JOIN fees_norm f ON o.order_uid = f.order_uid
WHERE o.broker = 'FUTU' AND o.status = 'FILLED' AND f.fee_uid IS NULL;

-- Strategy classification coverage
SELECT strategy_type, COUNT(*), ROUND(AVG(confidence)::numeric, 3)
FROM order_strategy_map GROUP BY strategy_type;
```

---

## 12. Conclusion

The implementation plan is ready. Key points:

1. **Single-phase implementation** since all components are needed together
2. **Standalone backfill capability** via `scripts/backfill.py`
3. **Full PRD compliance** with PostgreSQL, JSONB, fee breakdown, snapshots
4. **Replaces existing** `strategy_detector.py` with PRD's classifier
5. **Integrates with existing adapters** - reuses `fetch_orders_raw()`, `fetch_deals_raw()`

Estimated implementation effort: 5-7 days as per PRD roadmap.
