# ATR-Based Trading Strategy Guide

This document explains the ATR (Average True Range) based exit strategy implemented in the Apex Risk Management System.

## Overview

The ATR panel in the IB/Futu positions view (Tabs 3 & 4) provides:
- Stop loss and take profit levels based on ATR
- R-Multiple targets for systematic profit taking
- Trailing stop logic for letting winners run
- Timeframe-specific ATR calculations

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `w/s` | Select previous/next position |
| `+/-` | Increase/decrease ATR period |
| `t` | Cycle timeframe: Daily → 4H → 1H |
| `h` | Toggle help overlay |
| `r` | Reset to defaults (Period 14, Daily) |

---

## R-Multiple Targets (Static Exits)

R-Multiple targets are **fixed price levels** for taking profit, based on your initial risk.

### Definition

```
Risk (1R) = Entry Price - Stop Price
         = 1.5 × ATR (our default stop distance)
```

### Example

```
Entry: $483    Stop: $465    Risk (1R) = $18

1R Target: $501  (+$18)   ← "I risked $18, I want $18 back"
2R Target: $519  (+$36)   ← "I risked $18, I want $36 back"
3R Target: $537  (+$54)   ← "I risked $18, I want $54 back"
```

### Use Case

"I'll sell 50% at 2R, 50% at 3R" — **predetermined exits**

### Best For
- Ranging markets
- Quick trades
- When you want certainty

### Risk
- Might exit too early on a big trending move

---

## Trailing Stop (Dynamic Exit)

A trailing stop **moves with price**, locking in profits while giving the trade room to breathe.

### Definition

```
Trailing Stop = Highest Price Since Entry - (2 × ATR)
```

The trail only moves UP (for longs), never down.

### Example

```
Entry: $483
ATR: $12
Trail Distance: 2 × $12 = $24

Timeline:
─────────────────────────────────────────────────────────────────────

Day 1: TSLA at $483 (entry)
       Stop: $465 (fixed at 1.5× ATR below entry)

Day 3: TSLA moves to $495
       Stop: $465 (still fixed, hasn't hit 1R yet)

Day 5: TSLA hits $501 (1R achieved!)
       ✓ Trailing stop activates
       New Stop: $501 - $24 = $477
       → You've now locked in breakeven!

Day 7: TSLA rallies to $520
       Trailing Stop: $520 - $24 = $496
       → Locked in +$13/share profit

Day 10: TSLA peaks at $540, then reverses
        Trailing Stop: $540 - $24 = $516

Day 11: TSLA drops to $516
        → Stopped out at $516
        → Profit: $33/share (1.8R)
```

### Visual

```
Price
  │
540├─────────────────●←─ Peak
   │                ╱
530├───────────────╱
   │              ╱
520├─────────────●─────── Stop trails to $516
   │            ╱│
510├───────────╱ │
   │          ╱  │ 2× ATR gap
501├─────────●───┼────── 1R target (trailing activates)
   │        ╱    │
490├───────╱     │
   │      ╱
483├─────●─────────────── Entry
   │
465├─────────────────────  Initial Stop (1.5× ATR)
   │
   └────────────────────────────────────────── Time
```

### Use Case

"I don't know how far it'll go, I'll ride until it reverses" — **adaptive exit**

### Best For
- Trending markets
- Runners (stocks with momentum)
- When you want to maximize gains

### Risk
- Might give back some profits on sharp reversals

---

## Comparison: R-Targets vs Trailing Stop

| Aspect | R-Multiple Targets | Trailing Stop |
|--------|-------------------|---------------|
| Exit type | Fixed prices | Dynamic, follows price |
| Best for | Ranging markets, quick trades | Trending markets, runners |
| Psychology | "Take profit at X" | "Let it run, protect gains" |
| Risk | Might exit too early on big move | Might give back profits on reversal |

---

## Best Practice: Combine Both

The optimal strategy combines **scaling out with R-targets** and **trailing the remainder**.

### Example Strategy

```
Position: 300 shares TSLA @ $483, Stop $465

Plan:
├─ Sell 100 shares at 2R ($519)     ← Lock 2R on 1/3
├─ Sell 100 shares at 3R ($537)     ← Lock 3R on 1/3
└─ Trail remaining 100 at 2× ATR    ← Let 1/3 run for potential 5R-8R
```

### Why This Works

| Phase | Stop Behavior | Purpose |
|-------|--------------|---------|
| Entry → 1R | Fixed stop | Give trade room to develop |
| After 1R | Trail 2× ATR | Lock profits, let winners run |

### Benefits

- **Eliminates greed** — You have a systematic exit
- **Lets winners run** — Doesn't cut profits early
- **Protects gains** — Ratchets up as price rises
- **Respects volatility** — 2× ATR gives room for normal swings

---

## Timeframe Alignment

The ATR value depends on the timeframe of your bars:

| Timeframe | Typical ATR | Holding Period | Trade Style |
|-----------|-------------|----------------|-------------|
| Daily | 2.5% of price | Multi-day to weeks | Swing trading |
| 4H | ~1.25% of price | 1-5 days | Short-term swing |
| 1H | ~0.6% of price | Intraday to 1-2 days | Day trading |

**Important:** Match your ATR timeframe to your intended holding period!

- Using Daily ATR for a day trade = stop too wide
- Using 1H ATR for a swing trade = stop too tight

Press `[t]` to cycle through timeframes in the ATR panel.

---

## Panel Features

### ATR Panel Display

The panel uses **adaptive column widths** to ensure all prices display with 2 decimal places without truncation. Column width automatically adjusts based on the largest price value (8R level).

```
┌─────────────────────── ATR: TSLA | Daily ATR(14)=$12.08 (2.5%) ───────────────────────┐
│  TSLA         $483.37  ATR $12.08    (2.5%)  [Daily]              -                   │
│                                                                                        │
│   SL-2x  SL-1.5x  SMA21   Entry    1R      2R      3R      4R      5R      6R    ...  │
│  ─────────────────────────────────────────────────────────────────────────────────    │
│  459.21  465.25  468.87  483.37  501.49  519.61  537.73  555.85  573.97  592.09  ...  │
│    -5%     -4%     -3%      —      +4%     +7%    +11%    +15%    +19%    +22%   ...  │
│                                                                                        │
│  R-TARGETS                                             TRAILING STOP                   │
│  Risk (1R) = $18 (1.5×ATR)                             Trail = High - 2×ATR ($24)      │
│   Stop         $465      -$18  Exit all                                                │
│   1R           $501      +$18  Trail starts            Activates at 1R: $501           │
│   2R           $520      +$36  Sell 33%                                                │
│   3R           $538      +$54  Sell 33%                  $520 → stop $495              │
│   8R           $628     +$145  Max target                $556 → stop $532              │
│                                                          $592 → stop $568              │
│ ───────────────────────────────────────────────────────────────────────────────────   │
│   Risk: $18 (3.7%)             Max @8R: $145 (30%)          R:R @8R = 8.0              │
│                                                                                        │
│ [w/s] Sel  [+/-] Per(14)  [t] Daily  [h] Help  [r] Reset                               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

**Horizontal Price Bar includes:**
- **SL-2x**: Stop loss at 2× ATR below entry (emergency stop)
- **SL-1.5x**: Stop loss at 1.5× ATR below entry (default stop)
- **SMA21**: Simple Moving Average 21 estimate (~3% below current price)
- **Entry**: Current price
- **1R-8R**: R-Multiple profit targets

### Help Overlay

Press `[h]` to see a full educational overlay explaining:
- R-Multiple concepts with examples
- Trailing stop mechanics
- Side-by-side comparison
- Best practice scaling strategy
- Timeline walkthrough

---

## Implementation Notes

### ATR Calculation

Currently using **estimated ATR** based on typical volatility patterns:

```python
base_atr_pct = 0.025  # 2.5% for Daily 14-period
period_factor = sqrt(period / 14.0)  # Longer periods = higher ATR
timeframe_factor = {"Daily": 1.0, "4H": 0.5, "1H": 0.25}
estimated_atr = spot_price * base_atr_pct * period_factor * timeframe_factor
```

**Future Enhancement:** Add historical data service to fetch real OHLC bars and calculate true ATR using Wilder's smoothed average.

### Adaptive Column Width

The horizontal price bar uses adaptive column widths for consistent 2 decimal place display:
```python
# Calculate width based on largest price (8R level)
max_price = r_levels[8]
col_width = len(f"{max_price:.2f}") + 1  # +1 for padding
```

This ensures:
- Low-priced instruments ($24.05) use narrower columns
- High-priced instruments ($483.37) use wider columns
- All prices display with full 2 decimal precision

### Stop Distance

Default stop is **1.5× ATR** below entry:
- Gives enough room for normal price fluctuations
- Corresponds to roughly 3.7% for typical stocks
- Can be adjusted based on volatility regime

### Trail Distance

Default trail is **2× ATR** from highest price:
- Tighter than initial stop (locks in gains faster)
- Still respects volatility (won't get stopped by noise)
- Activates only after reaching 1R (breakeven)

### SMA21 Reference

The panel displays an estimated SMA21 (21-period Simple Moving Average):
```python
sma21 = current_price * 0.97  # Estimate ~3% below current for uptrend
```

This provides a technical reference point for:
- Support level awareness
- Mean reversion context
- Trailing stop reference in trending markets

**Future Enhancement:** Calculate actual SMA21 from historical price data.

---

## References

- [ATR Indicator Explained](https://www.investopedia.com/terms/a/atr.asp)
- [Position Sizing with ATR](https://www.tradingview.com/support/solutions/43000502115)
- [Trailing Stop Strategies](https://www.schwab.com/learn/story/trailing-stop-order)
