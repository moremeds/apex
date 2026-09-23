# Market-data MCP implementation plan

Date: 2026-09-22. Owner: Astra. Status: revised plan for review, not yet executed.
Contract: [design](../specs/2026-09-22-market-data-mcp-design.md).
Review: [findings and evidence](../specs/2026-09-22-market-data-mcp-review.md).

## 0. Delivery contract

- Build the full lake scope in two sequential PRs. PostgreSQL is excluded by the
  user's explicit correction; a follow-up task owns it.
- Implementation must use herd, as requested by the user. Astra owns task contracts,
  bounded worker file scopes, review, integration and final acceptance. Integrate
  and verify each dependency before assigning its dependent task; worker completion
  alone does not pass a gate. Cursor Grok workers/reviewers use 4.7.
- **PR1 requires full macmini real-lake verification before PR2 starts.** This is
  a user requirement, not an optional smoke test at the end of PR2.
- Preserve the existing uncommitted design/handover and unrelated main-checkout
  edits. Spec worktree rebased 2026-09-23 to master c690792b126eddfb63eb3f53fe98207675e0318b;
  see design §1.1 for what changed since v0.1.9.
- Implementation worktrees live under project .worktrees/. Record base SHA and
  dirty status before edits. No frozen subsystem changes or production lake writes.
- No commits/pushes/merge/deployment during this design review. On subsequent
  delivery authorization, explicitly force-add only the named deliverable docs;
  do not widen /docs/ ignore rules. No silent loss of ignored artifacts.
- Live data reads are authorized verification. Candidate code, temporary fixture
  mutations and report output live in an isolated scratch/worktree location outside
  the lake. Never replace the running image, restart production, or publish data
  merely to verify the candidate.

## 1. P0 — source and environment preflight

Owner: lead; bounded read-only evidence gathering may be delegated.

1. Recheck base vs remote and current contracts (design §1.1). Locate project rules, actual
   route signatures, schema files, tests and Docker dependency installation.
2. On macmini, inspect the running image identity and compose labels/mounts without
   dumping secret env values. Record actual lake roots, catalog identity, retained
   Silver/PIT revisions, repairs presence, and a free candidate loopback port.
3. Discover real sample symbols per asset/timeframe/residency using the catalog
   and bounded exact-file probes. Persist the inventory and explicit missing cells.
   The catalog is daily-only in the author's historical snapshot; do not infer
   intraday absence from its lack of rows. Expand bounded probes offline as needed.
4. Build a local minimal MCP v2 lifespan/auth/client probe using the intended api
   extra in an isolated environment. Lock the tested version. Do not install on
   production during planning. Source-check DuckDB cursor/interrupt semantics.
5. Freeze real, dated market fixtures from existing tests or the lake, with source
   and hash receipts. Copy PIT manifest fixtures from Livewire tests as explicitly
   test-only manifests; record upstream SHA.
6. Save the baseline performance workload and matrix definition before optimizing.
   Baseline and candidate use the same real artifacts, parameters, limits and host.

Exit: exact source contracts and run configuration are recorded; no guesses about
paths, SDK APIs, timestamps or deployed revision. Lack of real PIT data is recorded
early as a PR1 gate blocker, not deferred silently to PR2.

## 2. PR1 — lake queries and REST

Suggested branch: feat/lake-service-layer, based on the verified target branch.
Execute in this order; integrate each bounded task before assigning a dependent one.

| Step | Scope / expected files | Acceptance |
|---|---|---|
| P1.1 | Existing revisions.py, new pit_revisions.py and contract tests | Numbered/current Silver validation; retained artifacts; PIT manifest parse by explicit revision, served-artifact hash check, scope filtering, publisher_status echo; no lineage replay (design §3.4) |
| P1.2 | Existing coverage.py/reference.py/membership.py; new repairs.py where needed | Catalog rows/provenance; repairs degraded state; no future-known identity leakage; reuse existing reference behavior |
| P1.3 | application/lake shared queries/results/errors; provider seams | Request-local pins; bounded disk materialization; explicit price_mode; legacy-compatible output policy |
| P1.4 | Coverage/gap queries and catalog-backed futures discovery | All asset classes, explicit session/calendar uncertainty, edges and no-data distinguished |
| P1.5 | REST routes, payload builders, verification schemas, API docs and lake initialization | Full REST mapping, unchanged legacy defaults, additive provenance/pin fields, correct route precedence, source degradation |
| P1.6 | DuckDB A/B/C experiment and selected optimization (§4.5) | Real macmini timings and equality evidence; bounded memory/lifecycle; retain simplest measured passing option |
| P1.7 | Focused unit/carve checks and macmini verification runner | Matrix generated before run; expected failures explicit; resumable durable results; no silent skips |
| P1.8 | Candidate build, macmini matrix and independent review | All §4 gates satisfied on exact candidate; relevant CI green; no blocking review findings |

Keep REST access unchanged; mandatory MCP authentication belongs to PR2.
Compose mount reconciliation is part of the proposed change; production activation
is a separate deployment step. Preserve the original REST lifecycle, including
existing database/streaming features, without adding them to the lake container.

Known test entry points (exist at the baseline):

```sh
uv run pytest --no-cov tests/unit/api/test_bars_listing.py tests/unit/api/test_bulk_bars.py tests/unit/api/test_chart_payload.py tests/unit/infrastructure/livewire/test_ohlc_provider.py
uv run pytest tests/unit/ tests/carve/
make quality
```

Add only focused checks for the new contracts. Existing reference, listing, bulk,
membership and lifecycle regressions must remain protected; do not edit expected
results merely to accommodate an unintended behavior change. Tests with app-state
injection must still work without running the production lifespan.

## 3. PR2 — MCP transport and clients

**Entry gate:** PR1 §4 has passed; evidence covers the exact accepted candidate.
Prefer PR1 merged before branching PR2; rebase onto its accepted merge SHA and
verify the tree equivalence. A changed merge tree requires affected checks rerun.

1. Add tested SDK api dependency/lock, server/app/lifespan and four tool groups.
   MCP depends only on lake services and has no database/streaming startup.
2. Register the exact 20 tool names from design §5, typed inputs/output schema,
   bounded columnar responses and stable errors. Reuse PR1 queries.
3. Add SDK client tests, semantic REST parity, auth/Origin/Host checks, lifecycle
   shutdown/cancellation and HTTP transport integration. Do not rely only on an
   in-memory client, which bypasses network middleware.
4. Add portable compose/client examples, Makefile target and operator guide.
   Use configured tailnet publication, read-only mount and explicit image identity;
   keep host/account details and keys in private operator files.
5. Run local gates and independent review. Verify candidate MCP on macmini with
   every tool and the same argument matrix through the transport, preserving PR1
   data evidence. Recheck unknown/invalid/error combinations and response budgets.
6. After release/deployment authorization: pin exact image, verify running identity,
   direct tailnet client call and actual ChatGPT call through Secure MCP Tunnel.
   A connector listing is not successful tools/call. Report account/access blockers.

## 4. PR1 macmini verification — hard gate

### 4.1 Candidate isolation and reproducibility

Execute the candidate query code **on macmini**, not just requests from a laptop to
the old deployed API. Use an isolated checkout/container/process with read-only
lake/catalog access and a private loopback port. Run real registered routes with
production lifespan disabled and explicitly injected/managed LakeServices. Existing
env gates do not disable all subscriptions; do not start PG/xenon/subscriptions.
Use a lake-only env allowlist, not production API env. Verify production lifecycle
compatibility separately with mocks, not live side effects.
Record candidate SHA plus dirty diff hash (if any), dependency/image identity,
Python/DuckDB versions, source manifest hashes, catalog identity, start/end times
and runner arguments. An uncommitted run is provisional until tied to identical
committed bytes. No hidden writes to the lake or production compose.
Install from the lock and verify actual installed versions/image identity; copying
uv.lock beside an unconstrained pip install is insufficient.

Store matrix.json, results.jsonl, summary.md and performance.json under a timestamped
evidence directory outside the lake. Record its exact path in the delivery report;
retain a sanitized summary in the PR. Save each result as it finishes, not only
when the full run succeeds. Do not call a lost/unsaved run complete.

### 4.2 Full combination matrix

Generate an explicit case manifest before the run. No sampling/pairwise reduction
of the core product below. Collapse only dimensions a route does not accept, with
that exclusion justified by its contract. Absence of a real file is not N/A.

| Dimension | Enumerated values / partitions |
|---|---|
| asset_class | equity, volatility, fx, cmdty, futures, rates |
| timeframe | 1m, 5m, 30m, 1h, 1d, unsupported 4h; registry determines validity |
| process_config | raw and adjusted default, crossed with accepted per-request modes |
| price_mode | omitted, raw, adjusted |
| output_policy | legacy REST and bounded MCP; directly exercise both shared-query policies in PR1 |
| listing | listed, delisted, any; live-only, archive-only and dual for every accepted residency combination |
| revision | none, explicit current Silver, older retained Silver, published PIT (any publisher status; PROVEN is not a separate value, user 2026-09-23), both revision params (invalid) |
| window | default, start-only, end-only, bounded nonempty, bounded empty, outside artifact edges |
| limit | default, 1, exact available row count, below available count, maximum, nonpositive legacy/MCP policy |
| operation | single bars, bulk equity bars, rates series, session gaps (indicators excluded by user decision 2026-09-23) |

The capability matrix distinguishes accepted arguments from actual data availability.
For example: rates' valid operation is series; rates bars must reject. Non-equity
explicit adjusted must reject; omitted mode with adjusted process default follows
the existing non-equity raw fallback. Explicit historical intraday pins must reject. Bulk has no
PIT parameter and remains equity-only. Gaps have no
price/revision dimension because they assess observed session presence, not values.
Do not accidentally mark these negative cases as skipped. The manifest includes
expected code/status for every rejected combination, and explicit reason for
dimension exclusions.
PIT publisher status is not a matrix dimension: Apex has no code path that differs
between PROVEN and PARTIAL (design §3.4, it only echoes the status), so one published
PIT revision per index covers the PIT value. Assert the echoed status equals the
manifest's `status` field.

Within legal combinations choose real symbols/windows that contain data. Cover
both unchanged and changed values across retained daily Silver revisions; missing
positive evidence is BLOCKED_DATA, not a negative substitute. For listing=any
exercise live-only, archive-only and dual separately;
compare the actual per-date union to its two source files. A supported timeframe
without a real sample is BLOCKED_DATA. Report asset-class subtotals so equity
success cannot hide a missing rates/fx/volatility path.

Run all catalog/discovery, actions, delisting, security, index, membership history,
revision and lake-status routes in addition to the bars product. Cover all discovered
indices; do not use one index as universal proof.
For pageable surfaces test first/middle/final/empty pages and source identity change.
Reference combinations include before/within/after identity intervals, known_at
omitted/earlier/later, candidates false/true, action type and start/end filtering,
and symbol resolution across a real identity seam when retained evidence exists.

### 4.3 Value-level checks and negative cases

Each successful case compares returned timestamps, row count, ordering, nulls,
economic values, source basis, identity and revision metadata against an independent
read-only DuckDB/reference-manifest calculation over the same exact files. Do not
call the candidate service twice and describe that as independent verification.
Use explicit tolerance per floating field; identifiers/dates/counts are exact.
Record first/last timestamps and a normalized result hash per case.
Capture mutable source identities before/after candidate and oracle; persist and
boundedly retry invalidated comparisons on one generation. Unsettled comparisons
remain NOT_RUN, never PASS or a false numerical regression.

Additional negatives/boundaries: unknown symbol, absent source, corrupt manifest/hash,
escaped path/symlink, missing adjusted artifact, conflicting raw+revision, unknown
revision, naive timestamps, reversed window, oversized batch/output/input, invalid
limit, mixed-success batch, partial repairs, ticker ambiguity, early-close/DST/session
boundaries and no-data edges. Run corruption/replacement/unset-env cases in an
isolated scratch copy **on macmini**; never corrupt production inputs. Label these
CONTROLLED_CASE, separately from LIVE_DATA. They do not replace positive live cases.
Cover closures 2018-12-05/2025-01-09; typed invalid inputs expect 422, semantic
unsupported 4h expects 400. New lake/security routes must avoid the catch-all.
Bulk defaults must accept 200 symbols (50×200 within the 10000-row budget).

Concurrent checks: two different pins + ordinary default read; bulk while a new
current pointer is published in scratch; parent
connection shutdown waits for workers; cancellation cannot interrupt another cursor.
Catalog replacement and cache invalidation are exercised only in scratch.

### 4.4 Status and gate logic

Every expected case has exactly one recorded status:

- PASS: actual data/metadata matched its independent oracle.
- EXPECTED_REJECTION: the invalid combination returned the exact expected failure.
- BLOCKED_DATA: a legal case lacks real retained data/evidence.
- BLOCKED_DEPENDENCY: unresolved required runtime input or unsafe side effect.
- FAIL: wrong result, exception, timeout, provenance mismatch or failed comparison.
- NOT_RUN: no settled result yet.

Report planned/executed counts and lists, no unclassified cases. The gate passes
only with zero FAIL, NOT_RUN, BLOCKED_DATA and BLOCKED_DEPENDENCY, plus all
expected rejections verified. N/A is a matrix-generation exclusion justified by
the signature/capability contract, never a runtime escape hatch.

If production has no PIT or no needed residency/timeframe sample, leave PR1's live
gate open and name the exact upstream dependency. Fixture coverage remains useful
but cannot turn it green. Do not proceed to PR2, silently narrow the matrix, or
generate production data without explicit authority. Give the user the blocking
cells and concrete resolution needed from the upstream owner.

Default run is sequential; limit parallel reads to at most two after checking host
load and production ingest schedule. Checkpoint/resume large matrices. Record timeouts
as failures; do not quietly skip slow classes or run unbounded disk stress.
Persist generated counts by dimension and a pilot runtime/load estimate first;
schedule resumable batches. A large estimate does not authorize sampling.

### 4.5 Required performance experiment

The original performance goal remains in scope. Reuse DuckDB for actual Parquet
queries and coverage catalog for discovery; do not build Livewire temporary views.
Compare A existing per-call connection, B lifespan parent/per-worker cursor, then
C bounded result cache if repeated I/O is material. Prototype variants only within
this measured implementation task; do not ship parallel permanent implementations.

Workload includes every class/timeframe, raw/adjusted where valid, historical daily
pin, a published PIT revision (manifest parse + served-artifact hash, design §3.4),
catalog search/coverage, session gaps, single/bulk bars. A/B/C use identical query
semantics and the same artifact hash checks. Freeze
inputs, repeat at least 20 times per representative workload, report first request
separately and warm median/p95, peak RSS, failures and result hashes. Interleave
baseline/candidate to reduce filesystem-cache/order bias. Do not call a restarted
process an OS-cold cache or flush shared host caches.

Acceptance: equal results; no correctness/isolation regressions; warm p95 no worse
than baseline by more than max(10%, 50ms) per workload. Keep B/C only when the affected
workload's median improves >=20% without violating other workload budgets; otherwise
retain A with measured evidence. Missing the budget requires investigation or a
user-approved amended target, not an invented PASS. Record absolute latencies too:
passing a relative comparison does not establish an interactive absolute SLO.

Any cache must have a byte cap, all-source invalidation and selected-artifact hash
validation. Test identical mtime/size with atomic replacement and revision switches;
never trust mtime alone. No cache is also a valid measured conclusion, not permission
to skip the experiment. Record the selected option and rejected alternatives.

## 5. Review, delivery and rollback

Before finalizing each PR: self-review, required independent review, targeted checks,
full unit/carve and quality gates, then relevant CI. Reviewers see source plus exact
matrix evidence, not just a green summary. Preserve code/test/live/deployed distinctions.

PR1 documents the full macmini result, performance decision and additive API contract.
PR2 documents the 20-name tool set, HTTP/auth/parity results and client acceptance.
Merge only under the user's delivery authorization with all required checks green.
After merge align local default branch while preserving unrelated edits. Use the
repository release script; do not hand-edit VERSION.

Production rollback restores the previous image and compose/env configuration,
never rewrites lake data. Revalidate existing REST reads after either deploy or
rollback. Keep the previously accepted image identity and operator procedure in the
private runbook. Worktree cleanup waits for delivery and a dirty/unique-work check.

## 6. Design-review completion versus implementation readiness

This plan is complete when its scope, dependencies, contracts, matrix/gates and
unverified external prerequisites are explicit and independently reviewed. That does
not mean the described runner exists, tests pass, macmini verification ran, or the
upstream PIT dependency is available. Report those separately at execution time.
