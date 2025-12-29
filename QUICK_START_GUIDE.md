# Option Risk Signal Engine - Quick Start Guide

**Ready to use in 3 steps!** 🚀

---

## Step 1: Update Earnings Calendar (5 minutes)

Edit `config/risk_config.yaml` and update the `upcoming_earnings` section with current earnings dates:

```yaml
risk_signals:
  event_risk:
    upcoming_earnings:
      TSLA: "2025-01-24"  # Update quarterly
      NVDA: "2025-02-21"
      AAPL: "2025-01-30"
      # Add your symbols here
```

**Where to find earnings dates:**
- Yahoo Finance: `finance.yahoo.com/calendar/earnings`
- Earnings Whispers: `earningswhispers.com`
- Company IR pages

---

## Step 2: Run the System

```bash
# Start with default config
python orchestrator.py --env dev

# Or specify custom config
python orchestrator.py --config config/risk_config.yaml
```

The dashboard will now display enhanced risk alerts in the **"Portfolio Risk Alert"** panel!

---

## Step 3: Monitor Risk Signals

### What You'll See

**Normal Operation (No Risks):**
```
✓ All risk limits OK
```

**Active Risks Detected:**
```
🔴 Portfolio Risk Alert (3)
──────────────────────────────────────────
Severity        | Symbol | Rule                    | Action
🔴 CRITICAL     | TSLA   | Stop_Loss_Hit           | CLOSE (16%)
⚠️  WARNING     | NVDA   | Take_Profit_Hit         | REDUCE (50%)
ℹ️  INFO        | SPY    | Low_DTE                 | ROLL
```

### Signal Severity Levels

| Icon | Severity | Meaning | Typical Action |
|------|----------|---------|----------------|
| 🔴 | CRITICAL | Immediate action required | Close position, add hedge |
| ⚠️ | WARNING | Take action soon | Reduce size, roll position |
| ℹ️ | INFO | Monitor closely | Watch for changes |

---

## Understanding Risk Signals

### Position-Level Alerts

**Stop Loss Hit** (-60%)
- **Trigger:** Position down 60% or more from entry
- **Action:** CLOSE - Cut losses immediately
- **Example:** Bought call at $5.00, now at $2.00 (-60%)

**Take Profit Hit** (+100%)
- **Trigger:** Position up 100% or more from entry
- **Action:** REDUCE - Lock in 50% of gains
- **Example:** Bought put at $3.00, now at $6.50 (+117%)

**Trailing Stop Hit** (30% from peak)
- **Trigger:** Position dropped 30% from highest profit
- **Action:** CLOSE - Protect remaining gains
- **Example:** Was +150%, now at +105% (-30% from peak)

**Low DTE (Days to Expiry)**
- **Long Options:** 7 days → ROLL or CLOSE (avoid theta decay)
- **Short Options:** 3 days → CLOSE (avoid assignment risk)

### Strategy-Level Alerts

**Diagonal Delta Flip** (CRITICAL)
- **Trigger:** Short leg delta > long leg delta
- **Meaning:** Short option now ITM, high assignment risk
- **Action:** CLOSE or ROLL immediately

**Credit Spread R-Multiple Stop**
- **Trigger:** Loss exceeds 1.5x premium collected
- **Action:** CLOSE or ROLL to limit losses

**Calendar IV Crush**
- **Trigger:** Implied volatility dropped >30% on long leg
- **Action:** MONITOR - may need to exit early

### Portfolio-Level Alerts

**Sector Concentration** (>60%)
- **Trigger:** More than 60% delta exposure in single sector
- **Action:** HEDGE - Add diversification or sector hedge
- **Example:** 80% exposure in Tech stocks (TSLA, NVDA, AAPL)

**Earnings Event Risk**
- **T-3 Days:** WARNING - Short gamma positions flagged
- **T-1 Day:** CRITICAL - Close uncovered short options
- **Action:** CLOSE or ROLL before earnings

---

## Configuration Tips

### Adjust Stop Loss Threshold

Edit `config/risk_config.yaml`:

```yaml
risk_signals:
  position_rules:
    stop_loss_pct: 0.50  # Change to -50% stop loss (more aggressive)
```

### Modify Debounce/Cooldown

Reduce alert noise:

```yaml
risk_signals:
  debounce_seconds: 30  # Require 30s persistence (reduce false positives)
  cooldown_minutes: 10  # Suppress duplicates for 10 min
```

### Add Custom Sectors

Group your positions by industry:

```yaml
risk_signals:
  correlation_risk:
    sectors:
      Crypto: [COIN, MSTR, RIOT]
      Semiconductor: [NVDA, AMD, INTC, TSM]
      AI: [NVDA, MSFT, GOOGL, META]
```

### Disable Specific Analyzers

Turn off features you don't need:

```yaml
risk_signals:
  correlation_risk:
    enabled: false  # Disable sector concentration checks

  event_risk:
    enabled: false  # Disable earnings calendar alerts
```

---

## Common Scenarios

### Scenario 1: Long Call Losing Money

**Signal:**
```
🔴 CRITICAL | TSLA 250131C00300000 | Stop_Loss_Hit | CLOSE (18%)
```

**What happened:**
- You bought TSLA $300 calls at $5.00
- They're now at $1.90 (-62%)
- System triggered -60% stop loss

**Action:**
1. Review current TSLA price and outlook
2. If thesis invalidated → CLOSE position
3. If still bullish → Consider rolling to later expiry

---

### Scenario 2: Diagonal Spread Risk

**Signal:**
```
🔴 CRITICAL | NVDA | Diagonal_Delta_Flip | CLOSE
```

**What happened:**
- Your diagonal spread: Long Feb $500C, Short Jan $520C
- Stock rallied above $520
- Short Jan call now has higher delta than long Feb call
- Assignment risk on short leg

**Action:**
1. Close entire spread immediately, OR
2. Roll short leg to higher strike or later expiry
3. Do NOT wait - assignment risk is high

---

### Scenario 3: Earnings Coming Up

**Signal:**
```
⚠️ WARNING | AAPL | Earnings_Event_Risk | CLOSE
```

**What happened:**
- AAPL reports earnings in 2 days
- You have short AAPL options
- IV crush + assignment risk

**Action:**
1. Close short options before earnings (T-1 day latest)
2. Or roll to post-earnings expiry
3. Avoid holding through earnings if short gamma

---

### Scenario 4: Sector Concentration

**Signal:**
```
⚠️ WARNING | PORTFOLIO | Sector_Concentration:Tech | HEDGE (73%)
```

**What happened:**
- 73% of your portfolio delta is in Tech stocks
- Limit is 60%
- Correlated risk if sector drops

**Action:**
1. Reduce Tech positions, OR
2. Add hedges (buy SPY/QQQ puts), OR
3. Add positions in other sectors for diversification

---

## Testing Your Setup

### Verify Signals Work

1. **Create a test position in manual.yaml:**
```yaml
positions:
  - symbol: "TEST 250131C00100000"
    underlying: "TEST"
    asset_type: "OPTION"
    quantity: 10
    avg_price: 10.00  # Entry at $10
    # ... other fields
```

2. **Add fake market data** showing -70% loss in IbAdapter

3. **Expected:** Should see CRITICAL stop loss signal

4. **Test debounce:** Signal should persist for 15 seconds before firing

---

## Maintenance Schedule

### Weekly
- [ ] Review active signals in dashboard
- [ ] Adjust positions based on CRITICAL alerts
- [ ] Check for new earnings announcements

### Monthly
- [ ] Update earnings calendar in risk_config.yaml
- [ ] Review stop loss/take profit thresholds
- [ ] Analyze signal statistics (false positives)

### Quarterly
- [ ] Full earnings calendar refresh
- [ ] Sector mapping updates (new holdings)
- [ ] Threshold tuning based on backtest

---

## Troubleshooting

### "No signals showing up"

**Check:**
1. Is RiskSignalEngine initialized? (Check main.py logs: "Risk signal engine initialized")
2. Are positions loaded? (Check dashboard position count)
3. Is market data available? (Check health status panel)
4. Are all analyzers enabled in config? (Check risk_signals.*.enabled)

**Fix:**
```bash
# Verify config loads correctly
python -c "from config.config_manager import ConfigManager; c=ConfigManager('config','dev').load(); print(c.raw['risk_signals'])"
```

### "Too many alerts (alert fatigue)"

**Fix:**
```yaml
# Increase debounce/cooldown in risk_config.yaml
risk_signals:
  debounce_seconds: 30  # Up from 15
  cooldown_minutes: 15  # Up from 5
```

### "Earnings alerts not firing"

**Check:**
1. Is event_risk enabled in config?
2. Are earnings dates in YYYY-MM-DD format?
3. Are dates within 3 days of today?

**Fix:**
```yaml
risk_signals:
  event_risk:
    enabled: true
    upcoming_earnings:
      TSLA: "2025-01-24"  # Must be valid date
```

---

## Advanced Usage

### Get Signal Statistics

Check signal processing stats in logs:

```python
# In orchestrator or risk_signal_engine
stats = risk_signal_engine.get_stats()
print(stats)
# Output: {
#   'total_evaluations': 150,
#   'raw_signals': 25,
#   'filtered_signals': 8,
#   'layer1_signals': 2,  # Portfolio limits
#   'layer2_signals': 5,  # Position/strategy rules
#   'layer4_signals': 1,  # Earnings events
# }
```

### Programmatic Access

Access signals in your code:

```python
# Get latest signals
risk_signals = orchestrator.get_latest_risk_signals()

# Filter by severity
critical_signals = [s for s in risk_signals if s.severity == SignalSeverity.CRITICAL]

# Filter by symbol
tsla_signals = [s for s in risk_signals if s.symbol == "TSLA"]

# Get suggested actions
close_actions = [s for s in risk_signals if s.suggested_action == SuggestedAction.CLOSE]
```

---

## Need Help?

**Documentation:**
- Implementation Plan: `doc/plan/option_risk_signal_implementation_plan.md`
- Completion Summary: `IMPLEMENTATION_COMPLETE.md`
- Phase 1 Details: `doc/plan/phase1_completion_summary.md`

**Run Tests:**
```bash
pytest tests/unit/test_risk_signal*.py -v
pytest tests/unit/test_position_risk_analyzer.py -v
pytest tests/unit/test_strategy_detector.py -v
```

**Check Logs:**
```bash
tail -f logs/system.log  # System events
tail -f logs/market.log  # Market data updates
```

---

## Summary

✅ **You now have:**
- Multi-layer risk detection (4 layers)
- Position-aware stop loss/take profit
- Strategy-specific alerts (spreads, diagonals)
- Earnings calendar warnings
- Sector concentration monitoring
- Enhanced dashboard with actionable suggestions

🎯 **Next steps:**
1. Update earnings calendar
2. Run `python main.py --env dev`
3. Monitor Portfolio Risk Alert panel
4. Take action on CRITICAL signals

**Happy trading! 📈**
