# Signal Verification Fixtures

Golden fixtures for the APEX signal verification framework.

## Directory Structure

```
signal_verification/
├── ohlcv/                    # OHLCV input data (CSV)
│   ├── trend_uptrend.csv     # 100 bars, clear uptrend
│   ├── trend_choppy.csv      # 100 bars, sideways/choppy
│   └── trend_crash.csv       # 100 bars, sharp drawdown + recovery
├── snapshots/                # Expected indicator outputs (JSONL)
│   ├── momentum_outputs.jsonl
│   └── trend_outputs.jsonl
├── baselines/                # Performance baselines
│   └── perf_baseline.json
└── README.md
```

## File Formats

### OHLCV (CSV)
- Human-readable for easy git diffs
- Columns: `timestamp,open,high,low,close,volume`
- Timestamp format: `YYYY-MM-DD` (daily bars)
- Minimal dependencies (no pyarrow required)

### Snapshots (JSONL)
- One JSON object per line
- Each line represents indicator output for one bar
- Format:
  ```json
  {"indicator": "rsi", "timestamp": "2024-01-02", "value": 45.5, "zone": "neutral"}
  ```

### Performance Baseline (JSON)
- p50/p95 runtime per indicator
- Used for degradation detection
- Warning at +30%, failure at +100%

## Warmup Period Handling

Indicator outputs during warmup are unstable. Golden comparisons:
1. Use `warmup_complete` flag from indicator output, OR
2. Skip first N bars based on indicator's `warmup_periods` attribute

This is documented per-indicator in the manifest.

## Updating Fixtures

When indicator logic changes intentionally:

1. Run verification in "regenerate" mode:
   ```bash
   python -m src.verification.signal_verifier --regenerate
   ```

2. Review diff carefully:
   ```bash
   git diff tests/fixtures/signal_verification/snapshots/
   ```

3. If changes are expected, commit the new fixtures

**Important**: Never auto-regenerate without human review. The whole point
of golden fixtures is to catch unintentional behavior changes.

## Tolerance Strategy

| Issue | Solution |
|-------|----------|
| EMA initialization differences | Compare only post-warmup rows |
| Floating point propagation | `abs_err=1e-6` AND `rel_err=1e-4` |
| Definition variants (ATR TR) | Document variant used in comments |
| Boundary conditions | Explicit edge case fixtures |

## Scenario Coverage

### trend_uptrend.csv
- Steady 0.5% daily gains
- Tests: RSI trending higher (not overbought), MACD positive histogram
- Expected: Healthy uptrend detection

### trend_choppy.csv
- Price oscillates in 97-103 range
- Tests: RSI mean-reverting around 50, ADX < 25
- Expected: Choppy/neutral regime detection

### trend_crash.csv
- Initial rally to 165, then 50% crash, then recovery
- Tests: RSI oversold extremes, volatility spike detection
- Expected: Risk-off regime during crash, recovery detection
