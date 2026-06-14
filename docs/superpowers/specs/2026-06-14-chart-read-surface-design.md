# Apex Chart Read Surface — Design Spec

**Date:** 2026-06-14
**Branch:** `feat/apex-chart-read-surface`
**Status:** approved (design decisions confirmed with user)

## Problem

The merged signal contract (`/ws/signals` + `/signals/{ticker}`) delivers only one
layer of a trading chart: the discrete signal markers (the `ta_signals` rows —
triangles, reversal/invalidation labels). To render a full chart — candles, the
moving-average / SuperTrend / Bollinger overlays, the oscillator panes
(RSI, MACD, OBV…), and the multi-timeframe confluence blocks — a consumer needs
**continuous per-bar time series**, which apex computes but does not expose.

**Hard constraint (user):** argon stores/saves nothing. It is a stateless renderer
that fetches *everything* it needs from apex on load / reconnect / scroll-back /
timeframe-switch. So apex must be the single read surface for the whole chart.

## What already exists (verified in code)

- **Bars:** `LivewireOhlcProvider.fetch_bars(symbol, timeframe, start, end)` reads
  full-history OHLCV from livewire bronze parquet (`ohlc_provider.py:64`). Apex
  consumes these internally; no route exposes them.
- **Indicator computation:** every indicator implements a *pure-functional*
  `calculate(df, params) -> DataFrame` over a bar DataFrame (`base.py:112`) plus
  `get_state(current, previous, params) -> dict` (`base.py:132`). The live
  `IndicatorEngine` calls `indicator.calculate(df, indicator.default_params)`
  (`indicator_engine.py:601`) — **default params, no YAML override** — so a
  compute-on-read path using `default_params` yields values *identical* to live.
  Registry: `get_indicator_registry().get(name)` (`registry.py:224`), 51 indicators.
- **Confluence:** persisted to `confluence_scores`; readable via
  `TASignalRepository.get_confluence_history(symbol, timeframe, start, end, limit)`.
- **Persistence is wired:** `TASignalService` writes `indicator_values`
  (`_persist_indicator`, `ta_signal_service.py:374`) and `confluence_scores`
  (`_persist_confluence`, `:299`) whenever a persistence repo is supplied — which
  the lifespan now does.

## Decisions

1. **Bars** → serve from livewire (full history). New `GET /bars/{ticker}`.
2. **Indicator series** → **compute-on-read (full depth)**. apex fetches the bar
   window (plus a warmup lead) from livewire and recomputes the indicator over it,
   returning a per-bar value series. Chosen over serve-from-DB because indicator
   values only exist in PG from when apex started watching; compute-on-read gives
   gap-free lines that always match the candles, and it's deterministic
   (pure function of bars + `default_params`).
3. **Confluence** → serve from `confluence_scores` (DB-backed). Confluence is a
   cross-indicator aggregate with debounce/state, *not* a pure function of one bar
   series, so historical depth = what's persisted. Documented as such.
4. **Contract rigor** → each new payload gets a JSON schema under
   `config/verification/schemas/` and is validated on egress, exactly like the
   signal payload. argon gets the same contract guarantee.

## Components

### New: `src/application/chart/indicator_compute.py`
`compute_indicator_series(provider, registry, symbol, timeframe, indicator, start, end, *, safety=3) -> list[dict]`
- Look up the indicator; raise `UnknownIndicatorError` if absent.
- Extend the fetch start back by `warmup_periods * timeframe_delta * safety` so the
  visible window has valid (non-NaN) values.
- `provider.fetch_bars(...)` → DataFrame (index = bar time, cols open/high/low/close/volume).
- `indicator.calculate(df, indicator.default_params)`; walk rows building
  `{time, state, bar_close}` via `get_state(current, previous)`.
- Coerce numpy/NaN to JSON-native (NaN → None).
- Trim to `time >= start` (drop the warmup lead). Empty bars → `[]`.

### New: `src/api/payload/chart.py`
`build_bars_payload`, `build_indicator_payload`, `build_confluence_payload` — pure
dict builders mirroring `payload/builder.py` (ISO timestamps, `count`, `generated_at`).

### New schemas (`config/verification/schemas/`)
`bars_payload.schema.json`, `indicator_series_payload.schema.json`,
`confluence_payload.schema.json`. Validated via the existing `validate.py` pattern
(generalised to take a schema path).

### New routes (`src/api/routes/chart.py`)
- `GET /bars/{ticker}?timeframe=&start=&end=` → 503 if no `ohlc_provider`.
- `GET /indicators/{ticker}?timeframe=&indicator=&start=&end=` → 503 if no provider;
  404/422 on unknown indicator.
- `GET /confluence/{ticker}?timeframe=&since=` → 503 if no `signal_repo`.

### Lifespan wiring (`src/api/server.py`)
Expose `app.state.ohlc_provider` (built from `APEX_LIVEWIRE_ROOT`) independent of the
streaming block so the read routes work even without a live subscription; reuse it in
the `SubscriptionManager`. Register the chart router in `create_app`.

## Payload shapes

```jsonc
// GET /bars/AAPL?timeframe=1d
{ "symbol":"AAPL","timeframe":"1d","count":1,"generated_at":"…",
  "bars":[{"time":"…","open":1.0,"high":2.0,"low":0.5,"close":1.5,"volume":1000,"vwap":1.3}] }

// GET /indicators/AAPL?timeframe=1d&indicator=rsi
{ "symbol":"AAPL","timeframe":"1d","indicator":"rsi","count":1,"generated_at":"…",
  "points":[{"time":"…","state":{"value":65.3,"zone":"neutral"},"bar_close":1.5}] }

// GET /confluence/AAPL?timeframe=1d
{ "symbol":"AAPL","timeframe":"1d","count":1,"generated_at":"…",
  "points":[{"time":"…","alignment_score":0.4,"bullish_count":3,"bearish_count":1,
             "neutral_count":2,"total_indicators":6,"dominant_direction":"bullish"}] }
```

## Out of scope (this PR)
- Live WS push of bars/indicators/confluence (signals already stream; chart-data
  WS deltas are a follow-up — REST backfill is enough for first render + polling).
- Compute-on-read **confluence** (would need replaying the cross-indicator analyzer
  per bar). DB-backed for now.
- Re-serving live quotes (right-edge ladder); that's xenon's surface.

## Testing
- Unit: compute service (synthetic-bar fake provider — structural assertions: one
  point per visible bar, warmup trimmed, bar_close aligned, unknown indicator
  raises, empty bars → []); payload builders (shape + schema-valid).
- Integration: each route 200 + schema-valid with fakes injected; 503 paths.
- Live: boot against real xenon + Postgres, capture real frames, confirm
  `indicator_values` / `confluence_scores` actually populate.
