# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Live Risk Management System** for options and derivatives trading. It provides real-time portfolio risk monitoring with PnL calculation, Greeks aggregation, concentration limits, and expiry bucket analysis.

**Language:** Python 3.14
**Runtime:** Standalone Python application (no web framework)

## Architecture

The codebase follows a layered architecture with clear separation of concerns:

### Core Components

1. **models/** - Data models (dataclasses) for the domain
   - `Position`: Unified model for stocks, options, futures with composite key for deduplication
   - `MarketData`: Market quotes (bid/ask/last/mid) plus Greeks (delta/gamma/vega/theta)
   - `AccountInfo`: Account balance and margin data
   - `RiskSnapshot`: Aggregated portfolio risk metrics output

2. **stores/** - Thread-safe in-memory data stores
   - `PositionStore`: Keyed by Position.key() tuple for reconciliation
   - `MarketDataStore`: Keyed by symbol
   - `AccountStore`: Latest account snapshot
   - All stores use `threading.RLock` for concurrent access safety

3. **engines/** - Risk calculation logic
   - `RiskEngine`: Core aggregation engine (`build_snapshot()` is the main entry point)
     - Computes portfolio-level unrealized/daily PnL
     - Aggregates Greeks (delta, gamma, vega, theta) across all positions
     - Groups risk by underlying, strategy tag, and expiry buckets
     - Calculates concentration metrics (single-name, gamma/vega near-term)

4. **config/** - Configuration loading
   - `loader.py`: YAML config parser (loads `risk_config.yaml`)
   - Returns `AppConfig` with properties for risk_limits, ibkr_config, dashboard_config

### Risk Engine Details (engines/risk_engine.py)

The `RiskEngine.build_snapshot()` method is the heart of the system:

- **Expiry Buckets:** Positions are grouped into `0DTE`, `1_7D`, `8_30D`, `31_90D`, `90D_PLUS` based on days to expiry
- **Greeks Fallback:** For stocks without market data, delta defaults to 1.0
- **Missing Data Handling:** Positions without market data are skipped (no price = no contribution)
- **Notional Calculations:**
  - Notional = mark × quantity × multiplier
  - Gamma notional = gamma × mark² × 0.01 × quantity × multiplier
  - Vega notional = vega × quantity × multiplier

### Configuration (risk_config.yaml)

The `risk_config.yaml` file defines:

- **IBKR connection:** host, port, client_id for Interactive Brokers TWS/Gateway
- **Risk limits:** Per-underlying notional caps, portfolio Greek ranges, margin utilization, concentration thresholds
- **Scenarios:** Spot/IV shock scenarios for stress testing
- **MDQC (Market Data Quality Control):** Staleness thresholds, bid/ask sanity checks
- **Watchdog:** Reconnection backoff, missing MD ratio limits

## Data Flow

```
External Sources (IBKR/Manual YAML)
         ↓
    Stores (thread-safe)
         ↓
    RiskEngine.build_snapshot()
         ↓
   RiskSnapshot (output)
         ↓
  Dashboard/Alerts/Logging
```

## Key Patterns

- **Position Deduplication:** Use `Position.key()` tuple for reconciliation across sources (IB vs manual)
- **Thread Safety:** Always use store locks when accessing shared state
- **Market Data Fallback:** `MarketData.effective_mid()` returns mid if available, else last
- **Error Handling:** Missing market data doesn't crash - positions are skipped gracefully

## Development Status & Roadmap

This is currently a **skeleton implementation**. The PRD (v1.1) outlines a comprehensive system with the following components to be implemented:

### MVP Scope (Simplified per PRD recommendations)

**Core Features to Implement:**
- IBKR adapter with auto-reconnect
- Position reconciliation (IBKR vs manual YAML)
- Market Data Quality Control (MDQC) - staleness, bid/ask validation
- Real-time P&L calculation (unrealized & daily)
- Greeks aggregation (using IBKR Greeks only - no local BSM/Bachelier fallback)
- Rule engine with soft/hard breach detection
- Terminal dashboard (using `rich` library)
- Structured JSON logging
- Health monitoring (watchdog)

**Simplified/Deferred Features:**
- Greeks calculation: Use IBKR Greeks exclusively; mark as MISSING if unavailable (no multi-model fallback)
- Scenario shocks: Spot shocks only (defer IV/combined shocks to v1.2)
- Suggester: Top contributors diagnosis only (defer optimization/hedging efficiency to v1.2)
- What-if simulator: Deferred to v1.2 (pre-trade tool, not core monitoring)

### Architectural Principles (from PRD)

1. **Layered Architecture:**
   ```
   Presentation (Terminal UI) → Application (Orchestrator) → Domain (Risk Engine) → Infrastructure (Adapters)
   ```

2. **Dependency Injection:** Domain layer depends on interfaces, not concrete adapters
   - `PositionProvider` interface for position sources
   - `MarketDataProvider` interface for market data

3. **Event-Driven (Future):** Event bus for POSITION_CHANGED, LIMIT_BREACHED, CONNECTION_LOST events

4. **Configuration Management:** Split into base.yaml, env-specific, risk_limits.yaml, secrets.yaml

### Implementation Guidelines

**Greeks Handling (Critical Decision):**
- **DO NOT** implement local BSM/Bachelier calculation in MVP
- Use IBKR Greeks exclusively
- Mark positions as `DATA_MISSING` when Greeks unavailable
- Rationale: Avoids complex market data infrastructure (risk-free rates, dividend curves)

**Position Reconciliation:**
- Compare IBKR positions vs manual YAML positions vs cached state
- Detect: MISSING (in one source but not another), DRIFT (quantity mismatch), STALE (no update)
- Use `Position.key()` tuple for matching across sources

**Market Data Quality Control:**
- Check staleness (default: 10 seconds threshold)
- Validate bid ≤ ask sanity
- Ignore zero quotes if configured
- Flag missing Greeks vs missing prices separately

**Performance Targets:**
- < 100ms refresh for < 100 positions
- < 250ms for 100-250 positions
- < 500ms for 250-500 positions

### Development Commands

**Testing (to be implemented):**
```bash
pytest --cov=. --cov-report=html
pytest --cov-fail-under=85  # Minimum coverage target
```

**Code Quality (to be set up):**
```bash
mypy .                      # Type checking
black --line-length 100 .   # Formatting
isort --profile black .     # Import sorting
flake8 .                    # Linting
```

**Running the System (future):**
```bash
python main.py --env dev    # Development mode
python main.py --env prod   # Production mode
```

## Testing Strategy

**Unit Tests (to be implemented):**
- Risk engine: P&L calculation accuracy, Greek aggregation
- Reconciler: Detect MISSING/DRIFT/STALE scenarios
- MDQC: Validate bid/ask, staleness, zero price detection
- Rule engine: Soft/hard breach classification

**Integration Tests:**
- IBKR Paper Trading soak test (4+ hours)
- Reconnection test (disconnect/reconnect IB Gateway)
- Breach simulation (lower limits to trigger alerts)

**Acceptance Criteria (MVP Success):**
- P&L accuracy > 99.9% vs TWS
- Auto-reconnect success rate 100% (10 tests)
- No crashes during 8-hour run
- Memory growth < 20% over 8 hours
- Test coverage > 85%

## Configuration Loading

```python
from config.loader import load_config
cfg = load_config("risk_config.yaml")
limits = cfg.risk_limits

# Future multi-config approach:
from config_manager import ConfigManager
config = ConfigManager(env="dev")  # Loads base.yaml + dev.yaml + risk_limits.yaml
```

## Key Decisions from PRD

**MVP Priority Adjustments:**
- **Removed from MVP:** What-if simulator, cross-asset hedging suggestions, multi-model Greeks, combined scenario shocks
- **Simplified in MVP:** Suggester (diagnosis only), scenario shocks (spot only), expiry buckets (can keep full granularity)
- **Must Have:** Position reconciliation, MDQC, IBKR auto-reconnect, rule engine, terminal dashboard

**Risk Mitigation:**
- Validate IBKR Greeks quality during Paper Trading phase
- If Greeks missing rate > 20%, reconsider fallback strategy
- Incremental rollout: Test with small position count first

## Time Savings from Simplification

By removing over-engineering from v1.1 original design:
- Greeks multi-model fallback: ~3 days saved
- Scenario shocks (IV/combined): ~2 days saved
- Suggester optimization logic: ~4 days saved
- What-if simulator: ~5 days saved
- **Total: ~14 days saved**
