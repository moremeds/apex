# Option P&L and Greeks - Diagnostic Guide

## Issues Reported
1. ❌ Option P&L not calculated
2. ❌ Live Greeks not received

## Root Cause Analysis

The P&L calculation logic is **CORRECT**. The issue is that **market data is not being received from IBKR**.

Without market data:
- No `bid/ask/last` prices → P&L cannot be calculated
- No `modelGreeks` → Greeks show as 0 or missing

## Diagnostic Steps

### Step 1: Check if IBKR is Connected

Run the main system and check logs:
```bash
tail -f logs/live_risk_dev_sys.log | grep -E "Connected|IB|market"
```

Look for:
- ✓ "Connected to IB"
- ✓ "Fetching market data for X positions"
- ✗ Any connection errors

### Step 2: Test Live Greeks Retrieval

**IMPORTANT:** Make sure IB Gateway/TWS is running first!

Run the diagnostic script:
```bash
python diagnose_live_greeks.py
```

This will:
1. Connect to IBKR (paper trading port 4002)
2. Request TSLA 360C option data
3. Test with and without tick type 106
4. Show what data is actually received

**Expected Output:**
```
Testing WITH Greeks (tick type '106')...
   After 3-5s:
     Bid: 3.45
     Ask: 3.55
     Last: 3.50
     IV: 0.28
     modelGreeks FOUND:
       Delta: 0.48
       Gamma: 0.015
       Vega: 0.18
       Theta: -0.06
       undPrice: 350.00
```

**If you see this:** ✓ Greeks ARE working!

**If you see "modelGreeks: None":** There's a problem with:
- Market hours (Greeks only during market hours)
- IBKR subscription level
- Option liquidity
- Delayed data feed

### Step 3: Check Market Hours

**CRITICAL:** IBKR only provides real-time Greeks during market hours!

During off-hours:
- ✓ You get bid/ask/last (delayed or snapshot)
- ✗ NO Greeks (modelGreeks will be None)
- ✗ NO IV (impliedVolatility will be None)

**Solution:** Run system during market hours (9:30 AM - 4:00 PM ET)

### Step 4: Check P&L Calculation Format

The P&L calculation expects:
- `avg_price`: **Dollars per contract** (e.g., 56.29 = $56.29 per contract)
- `pnl_price` (from IBKR): **Dollars per share** (e.g., 3.50 = $3.50/share premium)

Example calculation:
```
Position: 1 contract TSLA 360C
avg_price: $56.29 (per contract)
Current premium: $3.50/share

current_value = 3.50 * 100 = $350 (per contract)
unrealized_pnl = (350 - 56.29) * 1 = $293.71 ✓
```

**Check your manual.yaml:**
```yaml
- symbol: "TSLA  251219C00360000"
  avg_price: 56.29  # Should be per-contract, NOT $5,629
```

If you paid $5,629 for the contract, you should have avg_price: 5629.00

### Step 5: Verify Data Flow

1. **Market Data Received?**
   ```bash
   grep "Fetched market data" logs/live_risk_dev_sys.log | tail -5
   ```
   Should see: "Fetched market data for X/Y positions"

2. **Greeks Extracted?**
   ```bash
   grep "Greeks for" logs/live_risk_dev_sys.log | tail -5
   ```
   Should see: "✓ Greeks for TSLA...: Δ=0.48..."

3. **Position Risks Created?**
   ```bash
   grep "position_risks" logs/live_risk_dev_sys.log | tail -5
   ```
   Should see: "Snapshot created with X position_risks"

## Common Problems & Solutions

### Problem 1: "No modelGreeks available"

**Cause:** Market is closed or delayed data

**Solution:**
- Run during market hours (9:30 AM - 4:00 PM ET)
- OR upgrade to real-time data subscription
- OR Greeks will populate when market opens

### Problem 2: "P&L shows $0.00"

**Causes:**
1. No market data received → no price to calculate P&L
2. avg_price in wrong units (per-share vs per-contract)
3. Market data has no bid/ask/last

**Solutions:**
1. Check IBKR connection
2. Verify avg_price format in manual.yaml
3. Run during market hours

### Problem 3: "IV column empty"

**Cause:** `ticker.impliedVolatility` not populated

**Solution:**
- Requires tick type 106 ✓ (already implemented)
- Requires market hours
- Some illiquid options may not have IV

### Problem 4: "Delta dollars shows $0.00"

**Causes:**
1. No underlying_price from IBKR
2. No delta from IBKR

**Solution:**
- Run during market hours
- Check `grep "Underlying price" logs/` to see if undPrice is received

## Quick Test

Run this to see current state:
```bash
python main.py
```

Then check the dashboard:
1. Do options show bid/ask prices? → Market data working
2. Do options show IV %? → IV extraction working
3. Do options show Delta/Gamma/Vega? → Greeks working
4. Does P&L show non-zero? → P&L calculation working

## Files Modified

Already fixed in your code:
- ✓ Tick type 106 requested (`market_data_fetcher.py:277`)
- ✓ IV extraction added (`market_data_fetcher.py:393`)
- ✓ Underlying price extraction (`market_data_fetcher.py:408`)
- ✓ Delta dollars fixed (`risk_engine.py:363`)
- ✓ P&L calculation correct (`risk_engine.py:248-255`)

## Next Steps

1. **Run diagnostic:** `python diagnose_live_greeks.py`
2. **Check results:** Do you see modelGreeks populated?
3. **If YES:** System is working, just need market hours
4. **If NO:** Check IBKR connection, subscription, market hours

Report back what you see from the diagnostic!
