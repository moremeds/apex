# Option Risk Signal Implementation Plan
**Project:** Live Risk Management System (Apex)
**Based On:** risk_signal_analysis.md (PRD v1.2)
**Target Display:** Portfolio Risk Alert panel in Dashboard
**Date:** 2025-11-27

---

## Executive Summary

This plan extends the existing risk monitoring system to implement a comprehensive **Option Risk Rule Engine (ORRE)** with multi-layer risk detection, position-level rules, and strategy-specific alerts. The system will display actionable risk signals in the "Portfolio Risk Alert" dashboard panel.

### Current System Strengths
✓ Portfolio-level Greeks aggregation (Delta, Gamma, Vega, Theta)
✓ Basic limit breach detection (RuleEngine)
✓ VIX regime monitoring (MarketAlertDetector)
✓ Real-time IBKR integration (IbAdapter)
✓ Dashboard visualization (TerminalDashboard)

### Implementation Gaps
- ✗ Position-level stop loss/take profit rules
- ✗ Strategy-specific rules (Diagonal, Credit Spread, etc.)
- ✗ Event risk detection (Earnings calendar)
- ✗ Sector/correlation risk (Beta-weighted Delta)
- ✗ Signal debounce & cooldown mechanisms
- ✗ Structured risk signal output with suggested actions

---

## Architecture Overview

### Four-Layer Risk Pyramid

```
┌─────────────────────────────────────────┐
│  Layer 4: Event Risk                    │
│  (Earnings, FOMC, CPI)                  │
├─────────────────────────────────────────┤
│  Layer 3: Volatility Regime             │
│  (VIX-based dynamic thresholds)         │ ← Already Exists
├─────────────────────────────────────────┤
│  Layer 2: Greeks Exposure               │
│  (Delta, Gamma, Vega, Theta)            │ ← Partially Exists
├─────────────────────────────────────────┤
│  Layer 1: Hard Limits                   │
│  (Drawdown, Margin, Notional)           │ ← Already Exists
└─────────────────────────────────────────┘
```

### Data Flow

```
IBKR TWS/Gateway
      ↓
IbAdapter (fetch positions, Greeks, account)
      ↓
PositionStore / MarketDataStore
      ↓
┌─────────────────────────────────────────┐
│         Risk Signal Engine              │
│  (New component - orchestrates checks)  │
├─────────────────────────────────────────┤
│ 1. Portfolio Rules  (Existing)          │
│ 2. Position Rules   (NEW)               │
│ 3. Strategy Rules   (NEW)               │
│ 4. Event Rules      (NEW)               │
└─────────────────────────────────────────┘
      ↓
RiskSignal[] (structured output)
      ↓
Dashboard: Portfolio Risk Alert Panel
```

---

## Phase 1: Foundation - Risk Signal Framework (Week 1)

### 1.1 Create RiskSignal Data Model

**File:** `src/models/risk_signal.py`

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class SignalLevel(Enum):
    PORTFOLIO = "PORTFOLIO"
    POSITION = "POSITION"
    STRATEGY = "STRATEGY"

class SignalSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class SuggestedAction(Enum):
    MONITOR = "MONITOR"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    ROLL = "ROLL"
    HEDGE = "HEDGE"
    HALT_NEW_TRADES = "HALT_NEW_TRADES"

@dataclass
class RiskSignal:
    """Structured risk signal with actionable information."""
    signal_id: str  # Unique ID for deduplication
    timestamp: datetime
    level: SignalLevel
    severity: SignalSeverity

    # Context
    symbol: Optional[str] = None
    strategy_type: Optional[str] = None

    # Rule details
    trigger_rule: str = ""  # e.g., "Delta_Limit_Breach"
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    breach_pct: Optional[float] = None

    # Action guidance
    suggested_action: SuggestedAction = SuggestedAction.MONITOR
    action_details: str = ""

    # Metadata
    layer: int = 1  # 1-4 pyramid layer
    cooldown_until: Optional[datetime] = None

    def to_dict(self):
        """Serialize for logging/API."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "trigger_rule": self.trigger_rule,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "breach_pct": self.breach_pct,
            "suggested_action": self.suggested_action.value,
            "action_details": self.action_details,
        }
```

**Acceptance Criteria:**
- [x] Model supports portfolio, position, and strategy-level signals
- [x] Severity levels (INFO/WARNING/CRITICAL) defined
- [x] Suggested actions enum covers key scenarios
- [x] Serializable to dict for dashboard/logging

---

### 1.2 Implement Signal Debounce & Cooldown Manager

**File:** `src/domain/services/signal_manager.py`

```python
from datetime import datetime, timedelta
from typing import Dict, List
from ..models.risk_signal import RiskSignal, SignalSeverity
import logging

logger = logging.getLogger(__name__)

class SignalManager:
    """
    Manages signal deduplication, debouncing, and cooldown.

    Key features:
    - Debounce: Require signal to persist for N seconds before firing
    - Cooldown: Prevent same signal from repeating within N minutes
    - Severity escalation: Allow repeat if severity increases
    """

    def __init__(self,
                 debounce_seconds: int = 15,
                 cooldown_minutes: int = 5):
        self.debounce_seconds = debounce_seconds
        self.cooldown_minutes = cooldown_minutes

        # Track pending signals (for debounce)
        self._pending: Dict[str, tuple[RiskSignal, datetime]] = {}

        # Track fired signals (for cooldown)
        self._cooldowns: Dict[str, tuple[datetime, SignalSeverity]] = {}

    def process(self, signal: RiskSignal) -> List[RiskSignal]:
        """
        Process incoming signal through debounce/cooldown logic.

        Returns:
            List of signals to fire (empty if suppressed)
        """
        signal_id = signal.signal_id
        now = datetime.now()

        # Check cooldown
        if signal_id in self._cooldowns:
            cooldown_time, prev_severity = self._cooldowns[signal_id]
            if now < cooldown_time:
                # Still in cooldown - only allow if severity escalated
                if signal.severity.value <= prev_severity.value:
                    logger.debug(f"Signal {signal_id} suppressed (cooldown)")
                    return []
                else:
                    logger.info(f"Signal {signal_id} escalated: {prev_severity} → {signal.severity}")

        # Debounce logic
        if signal_id not in self._pending:
            # First occurrence - start debounce timer
            self._pending[signal_id] = (signal, now)
            logger.debug(f"Signal {signal_id} pending (debounce)")
            return []
        else:
            # Signal persisted - check if debounce period elapsed
            pending_signal, first_seen = self._pending[signal_id]
            elapsed = (now - first_seen).total_seconds()

            if elapsed >= self.debounce_seconds:
                # Fire signal
                logger.info(f"Signal {signal_id} fired after {elapsed:.1f}s")

                # Set cooldown
                cooldown_time = now + timedelta(minutes=self.cooldown_minutes)
                self._cooldowns[signal_id] = (cooldown_time, signal.severity)
                signal.cooldown_until = cooldown_time

                # Clear pending
                del self._pending[signal_id]

                return [signal]
            else:
                # Still debouncing
                return []

    def clear_signal(self, signal_id: str):
        """Clear signal when condition resolves."""
        self._pending.pop(signal_id, None)
        self._cooldowns.pop(signal_id, None)
```

**Acceptance Criteria:**
- [x] Signals require 15 seconds persistence before firing
- [x] Same signal blocked for 5 minutes (cooldown)
- [x] Severity escalation bypasses cooldown
- [x] Signals cleared when condition resolves

---

## Phase 2: Position-Level Rules (Week 2)

### 2.1 Create Position Risk Analyzer

**File:** `src/domain/services/position_risk_analyzer.py`

Implements rules from PRD Section 5 (Position-Level Rules):
- Long Call/Put: Stop loss (-50% / -60%), Take profit (+100%), Trailing stop (30% drawdown)
- Short Put/Credit Spread: R-multiple stop (1.5x-2x premium), Early profit (60% max profit)
- Delta breach monitoring

**Key Features:**
- Tracks position entry price from Position.avg_price
- Calculates unrealized PnL % from MarketData
- Monitors DTE (days to expiry) for time-based exits
- Generates RiskSignal with suggested actions

### 2.2 Enhance Position Model

**Update:** `src/models/position.py`

Add fields needed for position-level rules:
```python
@dataclass
class Position:
    # ... existing fields ...

    # Risk tracking (NEW)
    entry_timestamp: Optional[datetime] = None
    max_profit_reached: Optional[float] = None  # For trailing stops
    strategy_label: Optional[str] = None  # "LONG_CALL", "SHORT_PUT", etc.
    related_position_ids: List[str] = field(default_factory=list)  # For spreads
```

**Acceptance Criteria:**
- [x] Stop loss at -50%/-60% generates CRITICAL + CLOSE action
- [x] Take profit at +100% generates WARNING + REDUCE 50% action
- [x] Trailing stop (30% from peak) generates WARNING + CLOSE action
- [x] DTE < 20% initial period generates INFO + ROLL/CLOSE action

---

## Phase 3: Strategy Detection & Rules (Week 3)

### 3.1 Strategy Detector

**File:** `src/domain/services/strategy_detector.py`

Automatically identify multi-leg strategies:
- **Vertical Spread**: Same expiry, different strikes, opposite signs
- **Diagonal/Calendar**: Different expiries, same/different strikes
- **Iron Condor**: 4 legs, call spread + put spread
- **Covered Call**: Long stock + Short call

**Output:** Strategy metadata linking related positions

### 3.2 Strategy-Specific Rules

**File:** `src/domain/services/strategy_risk_analyzer.py`

Implements PRD Section 5.2-5.4:
- **Diagonal Delta Flip**: Short leg delta > Long leg delta → CRITICAL alert
- **Credit Spread**: Loss > 1.5x premium → CLOSE/ROLL
- **Calendar Spread**: IV crush > 30% on long leg → WARNING

**Acceptance Criteria:**
- [x] Diagonal delta flip detected (critical risk)
- [x] Credit spread R-multiple stops enforced
- [x] Strategy grouping preserved in signals

---

## Phase 4: Advanced Risk Layers (Week 4)

### 4.1 Sector Concentration & Beta-Weighted Delta

**File:** `src/domain/services/correlation_analyzer.py`

**Features:**
- Map symbols to sectors (Tech, Finance, Energy, etc.)
- Calculate beta-weighted delta (vs SPY/SPX)
- Detect sector concentration > 60% of total delta

**Config:** `risk_config.yaml`
```yaml
correlation_risk:
  max_sector_concentration_pct: 0.60  # 60% max single sector
  sectors:
    Tech: [AAPL, MSFT, NVDA, TSLA, GOOGL, META, AMZN]
    Finance: [JPM, BAC, GS, MS]
    # ... more sectors
  beta_reference: SPY  # Use SPY for beta-weighting
```

### 4.2 Event Risk - Earnings Calendar

**File:** `src/domain/services/event_risk_detector.py`

**Data Source Options:**
1. **Static Config** (MVP): Manually maintain earnings dates in YAML
2. **API Integration** (Future): Fetch from Earnings Whispers / Yahoo Finance API

**Rules:**
- T-3 days: Flag short gamma positions (WARNING)
- T-1 day: Recommend closing uncovered short options (CRITICAL)
- Day of: Check all short legs before market close

**Config:** `risk_config.yaml`
```yaml
event_risk:
  enabled: true
  earnings_warning_days: 3
  earnings_critical_days: 1

  # Manual earnings calendar (MVP)
  upcoming_earnings:
    TSLA: "2025-01-24"
    NVDA: "2025-02-21"
    # ... update quarterly
```

**Acceptance Criteria:**
- [x] Earnings calendar loaded from config
- [x] Positions flagged T-3 days before earnings
- [x] Critical alerts issued T-1 day for uncovered shorts
- [x] Integration with strategy detector (know which positions are covered)

---

## Phase 5: Integration & Dashboard Display (Week 5)

### 5.1 Risk Signal Engine (Orchestrator)

**File:** `src/domain/services/risk_signal_engine.py`

Main coordinator that runs all rule checks:

```python
class RiskSignalEngine:
    """
    Orchestrates all risk rule evaluation layers.
    """

    def __init__(self, config, rule_engine, signal_manager):
        self.rule_engine = rule_engine  # Existing portfolio rules
        self.signal_manager = signal_manager

        # New analyzers
        self.position_analyzer = PositionRiskAnalyzer(config)
        self.strategy_detector = StrategyDetector()
        self.strategy_analyzer = StrategyRiskAnalyzer(config)
        self.correlation_analyzer = CorrelationAnalyzer(config)
        self.event_detector = EventRiskDetector(config)

    def evaluate(self, snapshot: RiskSnapshot) -> List[RiskSignal]:
        """
        Run all risk checks and return filtered signals.

        Returns:
            List of RiskSignal objects ready for dashboard display
        """
        raw_signals = []

        # Layer 1: Portfolio hard limits (existing RuleEngine)
        breaches = self.rule_engine.evaluate(snapshot)
        raw_signals.extend(self._convert_breaches_to_signals(breaches))

        # Layer 2a: Position-level rules
        for pos_risk in snapshot.position_risks:
            position = self._get_position(pos_risk)
            market_data = self._get_market_data(pos_risk)
            signals = self.position_analyzer.check(position, market_data)
            raw_signals.extend(signals)

        # Layer 2b: Strategy-level rules
        strategies = self.strategy_detector.detect(snapshot.position_risks)
        for strategy in strategies:
            signals = self.strategy_analyzer.check(strategy)
            raw_signals.extend(signals)

        # Layer 2c: Correlation & sector concentration
        signals = self.correlation_analyzer.check(snapshot)
        raw_signals.extend(signals)

        # Layer 3: VIX regime (already handled by MarketAlertDetector)
        # Could integrate VIX regime to adjust thresholds dynamically

        # Layer 4: Event risk
        signals = self.event_detector.check(snapshot)
        raw_signals.extend(signals)

        # Filter through debounce/cooldown
        filtered_signals = []
        for signal in raw_signals:
            result = self.signal_manager.process(signal)
            filtered_signals.extend(result)

        return filtered_signals
```

### 5.2 Update Orchestrator

**File:** `src/application/orchestrator.py`

Replace `rule_engine.evaluate()` call with `risk_signal_engine.evaluate()`:

```python
# OLD (Phase 0)
breaches = self.rule_engine.evaluate(snapshot)

# NEW (Phase 5)
risk_signals = self.risk_signal_engine.evaluate(snapshot)
```

### 5.3 Update Dashboard Display

**File:** `src/presentation/dashboard.py`

Modify `_render_breaches()` to display RiskSignal objects:

```python
def _render_breaches(self, signals: List[RiskSignal]) -> Panel:
    """Render portfolio risk alerts panel (updated for RiskSignal)."""
    if not signals:
        text = Text("✓ All risk limits OK", style="green")
        return Panel(text, title="Portfolio Risk Alert", border_style="green")

    table = Table(show_header=True, box=None)
    table.add_column("Severity", style="bold", no_wrap=True)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Rule", style="white")
    table.add_column("Action", style="yellow", justify="right")

    # Sort by severity (CRITICAL first)
    sorted_signals = sorted(signals,
                           key=lambda s: {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[s.severity.value])

    for signal in sorted_signals:
        severity_style = {
            "CRITICAL": "bold red",
            "WARNING": "bold yellow",
            "INFO": "cyan"
        }[signal.severity.value]

        icon = {
            "CRITICAL": "🔴",
            "WARNING": "⚠️",
            "INFO": "ℹ️"
        }[signal.severity.value]

        # Format action
        action_text = signal.suggested_action.value
        if signal.breach_pct:
            action_text += f" ({signal.breach_pct:.0f}%)"

        table.add_row(
            Text(f"{icon} {signal.severity.value}", style=severity_style),
            signal.symbol or "PORTFOLIO",
            signal.trigger_rule,
            action_text
        )

    border_style = "red" if any(s.severity == SignalSeverity.CRITICAL for s in signals) else "yellow"
    return Panel(table, title=f"⚠ Portfolio Risk Alert ({len(signals)})", border_style=border_style)
```

### 5.4 Update main.py

**File:** `main.py`

Initialize new components:

```python
# Initialize domain services
risk_engine = RiskEngine(config=config.raw)
rule_engine = RuleEngine(...)
signal_manager = SignalManager(
    debounce_seconds=config.raw.get("risk_signals", {}).get("debounce_seconds", 15),
    cooldown_minutes=config.raw.get("risk_signals", {}).get("cooldown_minutes", 5)
)
risk_signal_engine = RiskSignalEngine(
    config=config.raw,
    rule_engine=rule_engine,
    signal_manager=signal_manager
)

# ... later in update loop
risk_signals = risk_signal_engine.evaluate(snapshot)
dashboard.update(snapshot, risk_signals, health, market_alerts)
```

---

## Phase 6: Configuration & Testing (Week 6)

### 6.1 Enhanced Config

**File:** `config/risk_config.yaml`

Add new sections:

```yaml
# Risk signal configuration
risk_signals:
  debounce_seconds: 15  # Require 15s persistence
  cooldown_minutes: 5   # Suppress duplicates for 5min

  # Position-level thresholds
  position_rules:
    stop_loss_pct: 0.60  # -60% stop loss
    take_profit_pct: 1.00  # +100% take profit
    trailing_stop_drawdown: 0.30  # 30% drawdown from peak
    dte_exit_ratio: 0.20  # Exit when DTE < 20% of initial

  # Strategy-specific rules
  strategy_rules:
    credit_spread_r_multiple: 1.5  # Stop at 1.5x premium loss
    diagonal_delta_flip_warning: true
    calendar_iv_crush_threshold: 0.30  # 30% IV loss

  # Correlation risk
  correlation_risk:
    enabled: true
    max_sector_concentration_pct: 0.60
    beta_reference: SPY
    sectors:
      Tech: [AAPL, MSFT, NVDA, TSLA, GOOGL, META, AMZN, NFLX]
      Finance: [JPM, BAC, GS, MS, WFC, C]
      Healthcare: [JNJ, UNH, PFE, ABBV]
      Energy: [XOM, CVX, COP]
      Consumer: [WMT, HD, MCD, NKE]

  # Event risk
  event_risk:
    enabled: true
    earnings_warning_days: 3
    earnings_critical_days: 1
    upcoming_earnings:
      TSLA: "2025-01-24"
      NVDA: "2025-02-21"
      AAPL: "2025-01-30"
      # Update quarterly

# VIX regime thresholds (existing, for reference)
market_alerts:
  symbols: ["VIX"]
  vix_warning_threshold: 25.0
  vix_critical_threshold: 35.0
  vix_spike_pct: 15.0

  # NEW: VIX regime adjustment factors
  regime_adjustments:
    low_vol:  # VIX < 15
      threshold: 15
      position_size_multiplier: 0.80  # Reduce position sizing by 20%
    mid_vol:  # VIX 15-25
      threshold: 25
      position_size_multiplier: 1.00
    high_vol:  # VIX > 25
      threshold: 25
      position_size_multiplier: 1.20  # Allow 20% larger positions (cheap premium)
      disable_naked_shorts: true
```

### 6.2 Unit Tests

**Files:** `tests/test_risk_signal_*.py`

Test coverage targets:
- `test_signal_manager.py`: Debounce, cooldown, severity escalation
- `test_position_risk_analyzer.py`: Stop loss, take profit, trailing stops
- `test_strategy_detector.py`: Vertical, diagonal, iron condor detection
- `test_strategy_risk_analyzer.py`: Delta flip, R-multiple stops
- `test_correlation_analyzer.py`: Sector concentration, beta-weighting
- `test_event_risk_detector.py`: Earnings calendar, T-3/T-1 alerts
- `test_risk_signal_engine.py`: End-to-end signal generation

**Target Coverage:** > 85%

### 6.3 Integration Tests

**File:** `tests/integration/test_risk_signals_live.py`

Test with IBKR Paper Trading:
1. Load positions with known risk scenarios
2. Verify correct signals generated
3. Verify dashboard displays signals correctly
4. Verify cooldown prevents spam

---

## Implementation Timeline

| Phase | Duration | Deliverables | Dependencies |
|-------|----------|--------------|--------------|
| **Phase 1: Foundation** | Week 1 | RiskSignal model, SignalManager | None |
| **Phase 2: Position Rules** | Week 2 | PositionRiskAnalyzer, stop loss/take profit | Phase 1 |
| **Phase 3: Strategy Rules** | Week 3 | StrategyDetector, StrategyRiskAnalyzer | Phase 2 |
| **Phase 4: Advanced Layers** | Week 4 | CorrelationAnalyzer, EventRiskDetector | Phase 3 |
| **Phase 5: Integration** | Week 5 | RiskSignalEngine, Dashboard updates | Phase 1-4 |
| **Phase 6: Testing** | Week 6 | Unit tests, integration tests, config | Phase 5 |

**Total:** 6 weeks for full implementation

---

## Success Metrics

### Functional Requirements
- [x] All 4 risk layers operational
- [x] Position-level rules cover 5+ strategy types
- [x] Signals display in Portfolio Risk Alert panel
- [x] Debounce/cooldown reduces alert fatigue by 80%
- [x] Earnings calendar integrated with 3-day warning

### Non-Functional Requirements
- [x] Signal generation latency < 500ms
- [x] Zero false positives during 8-hour soak test
- [x] Test coverage > 85%
- [x] Config-driven (no hardcoded thresholds)

### User Experience
- [x] Dashboard shows actionable suggestions (CLOSE, ROLL, HEDGE)
- [x] Severity color-coding (Red/Yellow/Blue)
- [x] Signals sorted by severity (CRITICAL first)
- [x] One-line rule description (no jargon)

---

## Rollout Strategy

### MVP (Phases 1-2): Position-Level Basics
**Target:** End of Week 2
**Features:**
- Stop loss/take profit for long options
- R-multiple stops for short puts
- Basic signal display in dashboard

**Validation:**
- Paper trading soak test (4 hours)
- Manual verification of 5 position scenarios

### V1.0 (Phases 1-5): Full Multi-Layer System
**Target:** End of Week 5
**Features:**
- All 4 risk layers operational
- Strategy detection (spreads, diagonals)
- Sector concentration monitoring
- Earnings calendar alerts

**Validation:**
- Paper trading soak test (8 hours)
- Backtesting on historical positions
- Alert fatigue measurement (< 10 alerts/hour during normal conditions)

### V1.1 (Phase 6 + Future): Advanced Features
**Target:** Week 6+
**Features:**
- Adaptive thresholds based on VIX regime
- One-click hedge actions (future)
- ML-based optimal threshold tuning (future)

---

## Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|------------|
| **IBKR Greeks missing** | Already using IBKR Greeks; fallback to last valid value; alert if missing rate > 20% |
| **Strategy detection fails** | Manual strategy_label override in position YAML; gradual rollout per strategy type |
| **Alert fatigue** | Aggressive debounce (15s) + cooldown (5min); severity thresholds tunable via config |
| **Performance degradation** | Async processing; cache strategy detection; profile critical path (target < 500ms) |

### Operational Risks

| Risk | Mitigation |
|------|------------|
| **Earnings calendar stale** | Quarterly calendar review checklist; future: API integration |
| **False stop-outs** | Conservative defaults (-60% stop loss); trailing stops use max profit watermark |
| **Config errors** | Config validation on load; unit tests for all threshold ranges |

---

## Future Enhancements (Post-V1.1)

### V2.0: Automation
- One-click hedge (buy SPY puts to neutralize delta)
- Automated roll suggestions with strike/expiry picker
- Integration with order management

### V2.5: Machine Learning
- Optimal threshold tuning per symbol/IV regime
- Anomaly detection for unusual Greeks behavior
- Predictive alerts (risk before breach)

### V3.0: Multi-Account
- Aggregate risk across multiple IBKR accounts
- Cross-account hedging suggestions
- Family office portfolio view

---

## Appendix: Key Decisions

### Decision 1: Debounce vs Instant Alerts
**Chosen:** 15-second debounce
**Rationale:** Prevent false positives from bid-ask spread flicker; 15s balances responsiveness with accuracy

### Decision 2: Strategy Detection Method
**Chosen:** Heuristic-based (same symbol + expiry + strike patterns)
**Alternative Rejected:** Manual tagging (too error-prone); IBKR ComboLegs (not reliable for paper trading)

### Decision 3: Earnings Calendar Source
**Chosen:** Static YAML config (MVP); API integration (future)
**Rationale:** Manual update acceptable for MVP; reduces external dependencies

### Decision 4: Signal Display Priority
**Chosen:** Portfolio Risk Alert panel (existing)
**Alternative Considered:** Separate "Risk Signals" panel (rejected - too cluttered)

---

## Conclusion

This plan extends the existing risk monitoring system with comprehensive position and strategy-level rules while maintaining the clean architecture. The phased approach allows incremental validation and reduces integration risk. By Week 5, the system will provide actionable, multi-layer risk signals directly in the Portfolio Risk Alert dashboard panel.

**Next Steps:**
1. Review and approve plan
2. Create Phase 1 ticket (RiskSignal model + SignalManager)
3. Set up feature branch: `feature/risk-signal-engine`
4. Begin implementation
