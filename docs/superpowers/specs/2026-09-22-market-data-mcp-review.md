# Market-data MCP design review

Date: 2026-09-22. Scope: design and implementation plan only; no business-code changes,
commits, project dependency changes, production writes or deployment.

**Verdict: APPROVE — design/plan only.** Lead Astra, Claude Fable 5 High and
Grok 4.7 High reviewed the final r2 snapshot. No unresolved documentation blocker.
Implementation, tests, PR1 real-lake gate, merge and deployment are not complete.

## Starting evidence

- Worktree HEAD: `2b44666217d0de1865af56e39721d5dbfab901bf`.
- Worktree tracked status clean; original design and handover ignored by `/docs/`.
- Main checkout has unrelated `.serena/project.yml`, `uv.lock`, and documentation deletions;
  all preserved. Local Livewire HEAD: `471e963aac82f6b3590f721160c70b99100d670b`;
  its existing dirty documentation was not changed.

## Pass 1 — lead findings before independent reports

1. **Critical, high confidence — PIT overclaim.** The proposed parser checks a few fields
   and clamps only the end date. Livewire's `clients/pit_silver_revision.py` also binds
   member/security intervals, receipt, input hashes and policy. `as_of` is an aware
   datetime, not a date. Scope and evidence validation must be specified.
2. **Critical, high confidence — pinned intraday is not immutable.** Apex provider
   `fetch_bars` reads current Bronze intraday with pinned Silver factors. Neither a
   numbered Silver revision nor a PIT label freezes those source bars. Restrict proven
   historical reads to immutable supported artifacts and report unsupported requests.
3. **Important, high confidence — request-local pinning.** `pin_snapshot` returns a copy;
   callers must use that copy, including indicator warmup and bulk loops. Pinning does
   not change the provider's default raw price mode. Explicit revision inputs must not
   silently compute raw indicators or mutate the shared service.
4. **Important, high confidence — incompatible parity promise.** Current REST bars use
   `time` plus OHLCV; provider drops Bronze `price_basis` and other raw columns. A mere
   columnar transform cannot satisfy the draft's raw-schema and exact-parity promises.
   Specify one semantic contract, legacy envelope preservation, and deliberate extensions.
5. **Important, high confidence — limit semantics are misstated.** `_resolve_window`
   uses a now-anchored lookback for listed symbols and ignores tail limit with explicit
   start. MCP needs bounded output/read work without silently changing legacy REST.
6. **Important, high confidence — gap overclaim.** Clamping to observed file bounds
   hides missing edges; one intraday row does not establish day completeness. Calendar
   approximations must be labeled and unknown expected lifetimes must not mean healthy.
7. **Important, judgment call — unnecessary performance architecture.** Long-lived
   DuckDB plus LRU is prescribed without baseline timings; only replacing connection
   calls does not wire a result cache. Preserve current single-file readers and measure
   first; optimize only a demonstrated bottleneck with bounded memory and invalidation.
8. **Important, initially unresolved scope — PostgreSQL.** Only regime/signals were exposed despite
   prior recorded request for general read access. Ask the user asynchronously while
   completing independent lake work; do not silently authorize arbitrary DB access.
   Closed by the user's explicit instruction: remove PostgreSQL; separate follow-up.
9. **Important, high confidence — release and auth gaps.** An unauthenticated 405 is
   not service health. Separate liveness, MCP request success, source readiness, caller
   authorization, actual deployed compose, and account-dependent Tunnel acceptance.
10. **Important, high confidence — stale specification.** 22 vs 24 tools, delisted
    unavailable prose, duplicate corporate-action reader in diagram, implemented actions
    described as new, and approved status before this review. Remove contradictions.
11. **Important, high confidence — evidence and portability.** Historical mini facts
    are not this review's live verification. Keep account/host-specific configuration
    outside public documentation. Do not install tools on production to review a plan.

## Review ledger

Pass 1: complete; initial findings above, resolved in revised design/plan.
Pass 2: complete; Fable and Grok initial reports, debate, rebuttal and r2 fix review.
Pass 3: complete; cross-pin state, hidden benchmark, mutable oracle race, corrupt/
missing lineage, route shadowing and default 200-symbol batch checked.
Pass 3b: complete; removed dormant REST auth and unmeasured mandatory cache design;
retained required performance experiment and full verification scope.
Pass 4: complete; lead personally reread cumulative design, plan and handover.
Pass 5: complete for design; installed SDK behavior, full mini inventory and tunnel
permissions remain explicitly unverified implementation prerequisites.
Pass 6: complete for documents; verified 22 tool rows, exclusions, portability,
whitespace and preserved tracked state. No runtime acceptance inferred.
Standing rules: root CLAUDE.md and supplied global instructions read.

## Independent review availability

Herd transport is available. The project's Cursor reviewer role is used in two
fresh independent panes: Fable 5 High and Grok 4.7 High (user explicitly requested
4.7). Model discovery and actual Cursor UI confirmed both serving selections:
claude-fable-5-thinking-high and grok-4.7-high. Both used Ask/read-only assignments.
Fable seat: apex-mcp-fable-review, pane w5:p3, session
55f71674-54f4-4e33-84bb-db513fbcdcb4. Grok seat: apex-mcp-grok47-review,
pane wB:p2, session 5eb61770-f18d-42bc-b97b-61409e71c8df.
Both approved r2 after verifying the actual file hashes. No missing required seat;
each seat including lead has weight 1.0, no optional Gemini seat requested.
The initial 4.6 pane received no assignment and was closed before review.
After saving complete round reports and final snapshot evidence, both task-created
reviewer panes were closed; no pre-existing pane or herd server was stopped.

Reviewed r2 SHA-256 (headers still describe the draft stage; this report records
the completed review without changing the reviewed bytes):

- design: f0ec2a649c464a39798d2895d6bd828c750b4a1151421d6de9f181c3472e2e74
- plan: 1fb0b78a5991615f5a3533bc1df0fb789641910a5496fcd9f55d955203381435
- handover: 1152953bea850a8338a5ef4ed660596449097ba1d421dbe90c0510c3f20fb645

## Panel findings and lead dispositions

Initial reports: Fable 8 findings, Grok 11; these overlap and are not 19 independent
defects. Lead's 11 initial findings are recorded above. Final substantive fixes
were verified by all three seats; no unresolved disputed fix remains.

| Source finding | Lead disposition and evidence |
|---|---|
| F1, G1 PIT/retention | Accepted: exact full typed prefix/core hash contract, bound receipt/evidence, selected-artifact verification scope; replaced old artifacts fail, never satisfy positive live gate. Livewire pit_silver_revision.py is the oracle. |
| F2, G10 routing | Accepted: lake/security before instrument catch-all; test prefix resolution. Existing server registration and instruments comment prove hazard. |
| F3 batch default | Accepted: 60×200 exceeded 10000; MCP default now 50. Existing REST defaults preserved. |
| F4, G8 isolation | Accepted: existing env gates do not suppress all lifecycle work. Candidate runner injects LakeServices into real routes without production lifespan and uses env allowlist. |
| F5 repairs | Accepted: restore actual coverage_report.py/gap_engine.py root filenames and shapes, exclude Shepherd receipts. |
| F6 performance | Accepted PIT workload and same-semantics A/B/C; rejected manifest-only cache after both peers conceded dependency mutation makes it unsafe. |
| F7, G11 REST auth | Accepted deletion: no current caller for dormant optional REST bearer switch; mandatory MCP auth retained. |
| F8 errors | Accepted: typed validation 422 versus semantic 400, explicit unsupported4h. |
| G2–G4 matrix | Accepted explicit residency, process_config and output_policy axes; absent real legal cells block, not N/A. |
| G5 source basis | Accepted raw per-row preservation; rejected proposed mutable Bronze join onto immutable Silver. Both peers conceded the join changes a pinned result. |
| G6 calendar | Rejected alleged 5.2.4 closure defect by direct installed-library reproduction; both peers conceded. Keep named closure regressions and exact separate PIT policy. |
| G7 catalog | Clarified A/B/C affects in-memory Parquet connections; catalog stays per-call/read-only. |
| G9 indicator | Rejected latency denylist. Grok's debate discovered real Yahoo benchmark and R1 fallback; lead verified regime_detector.py:183–209 and :270–290. Fable conceded its pure-function claim. Added dependency audit/lake-input seam and hard blocking statuses, preserving all-indicator requirement. |

Additional lead corrections: current Docker pip install does not enforce copied
uv.lock; require lock-consuming build evidence. Existing carve classifier is a
fixed keepset; require focused lake/MCP import-boundary enforcement. Capture source
identity around candidate/oracle comparisons so concurrent ingest cannot create
false equality/failure. Materialize matrix cardinality/pilot cost without sampling.

Over-engineering sweep: all three seats completed it; r2 clean. No new general cache
service, duplicate corporate-action reader, publisher/recovery import, copied general
calendar or speculative REST auth. Two callers justify shared queries; actual repair
and PIT formats justify their narrow adapters.

## Checks and remaining execution gates

- Installed pandas-market-calendars 5.2.4: XNYS Jan 8–10 2025 returns Jan8/Jan10;
  Dec4–6 2018 returns Dec4/Dec6. Both special closures excluded correctly.
- Read-only macmini presence check: catalog exists; inspected PIT revisions directory
  and unresolved repairs file absent. Positive PIT gate remains BLOCKED_DATA;
  absence behavior is not a successful positive query.
- Source checks covered pin-copy/default mode, prefix serialization/scopes, reference
  replay, price-basis conversion, bulk/window/error routes, lifecycle and Docker.
- 22 explicit tool names checked. No public home paths/host addresses or active
  PostgreSQL work/optional REST key in revised deliverables. Diff whitespace check
  emitted no errors; task docs remain ignored and uncommitted, tracked code unchanged.
- Raw assignments/reports, debate/rebuttal, hashes, original docs and lead evidence
  retained in private review evidence directory `apex-mcp-review.4wM54w`; this report
  preserves durable findings without publishing host details or raw terminal data.
- PR1 must still implement/run the exhaustive real macmini matrix and DuckDB experiment.
  Missing retained PIT/benchmark scopes cannot be repaired or published under this
  read-only task. PR2 must not start until the PR1 gate passes.
- SDK lifecycle/auth probe, candidate tests/build/CI and actual client/tunnel calls
  have not been completed by this documentation review.

## User steering and initial corrections

- PostgreSQL removed entirely; follow-up task owns it. Lake tool set is 22.
- PR1 requires macmini real-data verification across all classes/combinations
  before PR2. Added the exhaustive manifest, result statuses and hard gate to plan §4.
- DuckDB stays central. Initial removal of a mandatory LakeDb implementation is
  revised to a mandatory A/B/C performance experiment: per-call, per-worker cursor,
  bounded cache. No assumption that the simplest variant is fast enough.
- Implementation must use herd; Astra owns bounded task contracts, integration,
  review and final macmini acceptance. Cursor Grok uses 4.7.
- Read-only Terra investigations corroborated PIT scope/known-at and API parity
  findings. A claim that per-thread cursors categorically require a global lock was
  rejected: DuckDB's official multiple-thread guide supports that usage. Installed
  version and cancellation behavior still need implementation-time verification.
- A worker attempted existing tests: collection failed for missing asyncpg/scipy
  in the isolated environment; no tests passed. No dependency files were changed.
  Runtime tests are not the acceptance gate for this documentation-only review.

## 2026-09-23 delta review (r3)

Scope: the 2026-09-23 changes only (indicators removed, 22 → 20 tools; PIT as a
read-only consumer without lineage replay; global PIT `current.json`; rebase to master
c690792b with PG/UW read REST, `_db_auth.py`, `query_timeout` 504 and the route block;
PROVEN dropped as a matrix value; Livewire PIT/Silver notes).

Panel: lead Claude Opus 5.5 (voting); Codex `gpt-6-astra` (observed in its log);
Cursor `grok-4.7-high`. Transport: one-shot CLI, because this lead session was not in
a herdr pane (`HERDR_ENV` unset) and the only live Apex Codex pane is the spec author.
The lead is Opus, not Fable, so the Fable/Astra pairing rule was applied as Astra peer.

Round 1 raised 8 issues; all were checked against Livewire f83e3df and Apex c690792b
and accepted:
- PIT file "own hash" had no expected digest (Codex, Cursor, lead).
- Mixed path roots: security_master/membership are lake-root relative (Codex, Cursor).
- Served-artifact hash source was ambiguous; now `inputs.silver_artifacts[]` (Cursor).
- Scope predicate lacked `session_to: null`, cutoff inclusivity and ambiguity (Cursor).
- Absence wording read as current fact (Cursor).
- `/current` detail text contradicted PIT tool 20 (Cursor).
- Republish cannot reuse revisions 1/2; `publish()` verifies current first (Codex).
- Lead additions: Silver eviction → `pit_unavailable`; 204/461 sp500 members have
  scope from 2026-09-17, so matrix cases must choose members with longer scope.
Rejected: adding a `security_id` argument (narrowing the window suffices); both
reviewers accepted the rejection.
Round 2: all 7 fixes RESOLVED by both; Cursor raised one new issue (two error codes
for PIT cross-security). Fix: `ambiguous_symbol` 409 in both §3.4 and §6.
Round 3: RESOLVED; **both APPROVE**.

Approved r3 snapshot (sha256):
- design `c231496e9d2ff0e2badf521ad02898cdb26116a8155e64a09a04f987132167c3`
- plan `2fbccaae505fa28cf36785e6030d089b6aab1746e5f24ac6ca196bb57f830b3d`
- handover `982c08a3350b614ed2c6e1eda6577e3ae66ea63126d61d554e2e5d4a1ebb1ea8`
- handoff `9de5f4ab4c7079ef750bf8f14d0870659a7262e6dbde3117be2aac06e7086b17`
  (after approval only its review-status note changed to point here)

Still not done: implementation, candidate tests, PR1 macmini matrix, merge, deploy.

