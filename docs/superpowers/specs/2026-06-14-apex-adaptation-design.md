# Apex Adaptation — Design Spec

Status: Approved (design); pending spec review
Date: 2026-06-14
Branch: `feat/apex-adaptation`
Source of truth for the goal: `~/projects/livewire/docs/architecture/apex-adaptation.md`
Part of: the four-system decoupling (livewire · apex · argon · xenon)

---

## 1. Goal

Adapt apex from a 148K-LOC "Live Risk Management & Backtesting System" into a
focused **streaming TA service**: read bars from livewire + live ticks from
xenon, compute TA-Lib indicators + the rule-engine signal checklist + regime,
and serve TA signals over WS/REST to argon. Strip everything else.

```
  livewire ──(DuckDB-over-parquet bars)──┐
                                         ▼
  xenon ──(WS live ticks, ticket auth)──▶ apex ──(WS push + REST pull:
                                  TA-Lib + rules    signal_service_payload)──▶ argon
```

**Carve strategy:** strip in place — same repo, same git history. Delete removed
subsystems and refactor what remains; do not create a new repo.

## 2. Scope of this effort

This spec captures the **whole adaptation as a roadmap** (§3), but only a subset
gets detailed implementation plans now.

- **Detailed plans now:** Phases 0, 1, 2, 3 — the critical path to a live signal.
- **Roadmap only (planned later):** Phase 4 (live-in), Phase 6 (strip-down).
- **Out of scope entirely:** backtest modernization. The backtest cores
  (`src/backtest`, `domain/backtest`) are **left as-is, untouched** — neither
  modernized nor verified in the carve. (Decision: user, 2026-06-14.)

### What "leave backtest alone" implies

- The carve (Phase 0) does **not** include backtest in its extraction proof.
- `src/backtest` currently imports `infrastructure.adapters.ib.historical_adapter`.
  Phase 6 strip-down removes IB broker adapters — these collide. **Deferred
  decision (D5):** Phase 6 either keeps the one historical adapter backtest
  needs, or accepts backtest going dormant. Not resolved now.

## 3. Roadmap (whole goal)

Sequenced by dependency. Phase 5 (backtest modernize) is intentionally absent.

| Phase | Deliverable | Depends on | Plan now? |
|---|---|---|---|
| **0 · Carve** | Import-graph audit + extraction manifest + proof the TA-signal cores detach (compile isolated, core tests green) | — | ✅ detailed |
| **1 · Bars-in** | `LivewireOhlcProvider` (DuckDB-over-parquet) behind `market_data_provider`; retire own Parquet/IB/Futu history loaders | 0 | ✅ detailed |
| **2 · Subscription + compute** | Ref-counted subscribe/unsubscribe manager (§3.1 of source doc); seed-history-on-subscribe; incremental recompute per tick/bar | 1 | ✅ detailed |
| **3 · Signals-out** | FastAPI WS server + REST pull (`GET /signals/{ticker}?since=`); persist `ta_signals` (migration 005); validate against `signal_service_payload.schema.json` | 2 | ✅ detailed |
| **4 · Live-in** | WS client to xenon `ib_realtime_server.js`; tick→bar stitch; service-identity auth | 2 (consumes 3's pipeline) | roadmap |
| ~~5 · Backtest modernize~~ | **REMOVED — out of scope. Backtest left as-is.** | — | ✗ |
| **6 · Strip-down** | Remove broker exec adapters, TUI, CF frontend/`worker-assets`, risk monitor, event bus, orchestrator | done incrementally as 1–4 land | roadmap |

**Critical path to a live signal:** 0 → 1 → 2 → 3 (→ 4 for live ticks).

### Cross-team dependencies (flagged, not blocking now)

- **D1 — PG isolation (Phase 3):** apex's signal writes land in their own DB vs a
  shared instance — depends on whether xenon's order execution shares that
  Postgres. Blocked on xenon (`xenon-adaptation.md` §6.2). Sequenced late.
- **D2 — xenon service auth (Phase 4):** service-JWT ticket vs trusted-network
  internal WS. Blocked on xenon (`xenon-adaptation.md` §6.1).
- **D3 — HK/Asia bars (Phase 1):** keep Futu loader for markets livewire doesn't
  cover, or extend livewire? (source doc §6.4). Futu loader's fate folds in here.
- **D5 — backtest vs IB adapter removal (Phase 6):** see §2. (Numbered D5 to stay
  aligned with the source doc's open-decision list; there is no D4.)

## 4. Phase 0 — detailed design (the carve / scoping pass)

**Goal:** prove the harvest is real and produce the map every later phase
follows — without building any new feature. Strip-in-place; repo + history stay.

### 4.1 Keep-set under audit (TA-signal path only)

`domain/indicators`, `domain/signals`, `domain/strategy`,
`application/services/ta_signal_service.py`,
`application/orchestrator/signal_pipeline`, `config/signals/*.yaml`,
`config/regime_*.yaml`, `migrations/005_ta_signals.sql`.

(Backtest cores are explicitly **excluded** from the proof per §2.)

### 4.2 Verified starting coupling (seed scan, 2026-06-14)

- `domain/indicators`, `domain/strategy`: **zero** infra/application imports — clean.
- `domain/signals` is the coupled core. Imports to cut/follow:
  - `src.services.historical_data_manager` (×4) — becomes a port (Phase 1 fills with livewire).
  - `src.infrastructure.observability` (×3) — stub / no-op shim.
  - `src.application.services.turning_point.*` — decide: pull into keep-set or cut.
  - `src.application.orchestrator.signal_pipeline` — follow (in keep-set).
- `ta_signal_service` → `domain.events.EventType` enum + `SignalStateTracker` — domain-only; keep or replace event coupling with a direct call.

### 4.3 Approach (three steps)

1. **Import-graph audit.** Exhaustive per-module dependency graph for the
   keep-set. Classify every cross-layer edge: **clean** (domain-only) / **cut**
   (→infra/app, replace with a port) / **follow** (→another keep module).
2. **Extraction manifest.** One table tagging every top-level `src/` module
   **KEEP / DROP / MOVE / STUB**, with the coupling edges to cut and the port
   interface each becomes. Authoritative carve spec for phases 1–4 & 6.
3. **Proof-of-extraction.** Isolated import context where the keep-set imports
   with all DROP/infra modules **stubbed** (thin fakes behind existing port
   interfaces — fake historical provider, no-op observability shim). Run the
   core unit tests for indicators / signals / strategy against the stubs.

### 4.4 Deliverables

- `docs/superpowers/specs/2026-06-14-apex-phase0-carve.md` — audit report + manifest.
- Stub harness + passing TA-signal-core tests, committed.
- Ranked **coupling-cut list** that seeds Phase 1's plan.

### 4.5 Success criteria

- (a) every `src/` module classified in the manifest;
- (b) keep-set imports with infra stubbed;
- (c) TA-signal core unit tests pass in isolation;
- (d) zero unresolved "follow" edges left ambiguous.

### 4.6 Explicitly NOT in Phase 0

No livewire/xenon wiring, no WS code, no deletion of the drop-set (Phase 6), no
backtest changes whatsoever. Pure analysis + proof.

## 5. Phases 1–3 — design summary (detailed plans to follow)

### Phase 1 — Bars-in (`LivewireOhlcProvider`)

- Implement a DuckDB-over-parquet provider that reads livewire's per-ticker
  Hive-partitioned bronze (`asset_class=equity/symbol=AAPL/{1d,1m,5m,30m,1h}.parquet`)
  directly — the file layout *is* the contract (livewire doc §3, §5).
- Slot it behind apex's `market_data_provider` port (the seam exposed by the
  Phase 0 cut of `src/backtest` → `historical_adapter` and `domain/signals` →
  `historical_data_manager`).
- Reads are **subscribed-tickers only**, on demand for the historical seed.
- Retire apex's own Parquet store + IB/Futu history loaders for equities.
- **D3:** Futu loader kept only for HK/Asia until livewire covers them.

### Phase 2 — Subscription + compute (source doc §3.1)

- Ref-counted subscription manager. `subscribe(ticker)`: seed history from
  livewire, (Phase 4) open xenon live sub, compute TA, begin publishing — first
  payload carries historical seed + indicators so argon draws the full chart.
- Steady state: each new bar → incremental recompute → publish + persist one row.
- `unsubscribe` (refcount→0): stop computing, free resources, retain persisted
  signals for a short TTL then prune.
- Compute/persist **only** the subscribed set — store stays MB, not GB.

### Phase 3 — Signals-out

- FastAPI **WS server** (push) + **REST pull** (`GET /signals/{ticker}?since=…`).
- Persist to `ta_signals` (migration `005_ta_signals.sql`) — apex's table is the
  **authoritative** signal record; argon keeps a read-model copy via the contract.
- Every payload validates against `signal_service_payload.schema.json` (required:
  signal_id `{category}:{indicator}:{symbol}:{timeframe}`, symbol, category,
  indicator, direction buy/sell/alert, strength 0–100, priority, timeframe,
  trigger_rule, current_value, timestamp; plus lifecycle status active/invalidated).
- **D1:** own DB vs shared PG — sequenced late, pending xenon's answer.

## 6. Open decisions (consolidated)

| ID | Decision | Phase | Blocked on |
|---|---|---|---|
| D1 | Signal store: own DB vs shared Postgres | 3 | xenon §6.2 |
| D2 | xenon WS auth: service-JWT vs trusted-network | 4 | xenon §6.1 |
| D3 | HK/Asia bars: keep Futu loader vs extend livewire | 1 | product call |
| D5 | backtest vs IB-adapter removal | 6 | deferred |

## 7. Out of scope

Backtest modernization (entirely). New repo. Touching xenon/livewire/argon code
(apex consumes/produces via documented contracts only). Risk monitor, TUI, CF
frontend, broker execution — all removed in Phase 6, none enhanced.
