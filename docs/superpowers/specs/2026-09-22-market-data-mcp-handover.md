# Handover — market-data MCP

Date: 2026-09-22. Lead/owner: Astra.
Read [design](2026-09-22-market-data-mcp-design.md), then
[implementation plan](../plans/2026-09-22-market-data-mcp-plan.md) and
[review evidence](2026-09-22-market-data-mcp-review.md).

## Goal and current authority

Build a private read-only lake MCP inside Apex, with shared REST queries.
This turn authorizes review and improvement of spec/plan; no business implementation,
commit, push or deployment has happened. Do not infer production acceptance from
the source design. User's latest corrections override the initial draft:

1. PostgreSQL is entirely excluded; its own follow-up task handles it.
2. PR1 must be verified on macmini against the real lake across every asset class
   and all declared combinations. PR2 cannot start before that gate passes.
3. DuckDB stays. Performance measurement and optimization selection are mandatory,
   comparing current per-call connections, per-worker cursors, and bounded caching.
4. Cursor/Grok review uses 4.7, explicitly requested by the user.
5. Implementation must use herd. Astra assigns bounded tasks with explicit file
   scope and acceptance checks, reviews and integrates each dependency, and owns
   final macmini acceptance before PR2 starts.

## Workspace

Spec worktree: `.worktrees/spec-market-data-mcp`, branch `spec/market-data-mcp`.
Baseline: master `c690792b126eddfb63eb3f53fe98207675e0318b` (rebased 2026-09-23 from v0.1.9
`2b446662`; lake-read code unchanged, PG/UW read REST added — design §1.1).
Documents are uncommitted and ignored by `/docs/`. On delivery authorization,
force-add only this task's named docs; do not change global docs ignore rules.
No renewed choice between force-add and changing ignore is needed.

Main checkout has unrelated staged/config/lockfile edits and documentation
deletions. Preserve them. Do not clean up other worktrees.
Machine paths, tailnet addresses, credentials and live deployment commands stay
in the private overlay/operator runbook; rediscover current deployment read-only.

## Closed decisions — do not reopen

- Reuse reference.py for existing actions/delisting; no duplicate corporate-action reader.
- Preserve listing=listed|delisted|any and dual union behavior.
- No adjusted-to-raw fallback. Keep source price basis visible.
- 20 lake tools; no indicators, regime/signals or PG pool in MCP.
- Historical Silver/PIT pins are daily equity/listed only until immutable upstream
  intraday evidence exists. Current adjusted intraday is still supported.
- PIT serves the manifest's member/session scope, not just end clamping. Apex does not
  re-verify lineage (user, 2026-09-23): it checks served artifact hashes and echoes
  Livewire's status; Livewire's own verify owns lineage.
- Membership known_at is date-granular; history remains current effective history.
- PR1 delivers lake/REST; PR2 adds MCP. Both use the same query logic.
- No guessed SDK API, invented market fixtures, lake repair or publication.
- Indicators are excluded (user, 2026-09-23): no indicator tools, no indicator
  changes, no indicator matrix cells; existing REST indicator route unchanged.
  regime_detector's Yahoo dependency goes to the regime/PG follow-up.
- No optional REST-auth work. MCP authentication remains mandatory.

## Ordered next steps

1. Close remaining review findings/gates listed in the review report.
2. At implementation start execute plan P0: exact code/dependency contracts and
   read-only macmini inventory, sample matrix and performance baseline.
3. Implement PR1 in the order given, with focused regression checks.
4. Run the complete candidate on macmini; persist every matrix cell and independent
   value comparison. Missing real PIT or other legal data is BLOCKED_DATA, not PASS.
5. Only after PR1 acceptance proceed to PR2, then independently authorize and verify
   release/image/runtime/Claude Code/ChatGPT acceptance.

## Open environment checks

- Current deployed image/compose and reference/catalog mounts; the initial author's
  observations are historical, not refreshed evidence.
- Real samples across all class/timeframe/residency combinations; retained revisions,
  PIT PROVEN/PARTIAL and repairs. Upstream unavailability keeps the gate open.
  Read-only check on 2026-09-22 found the catalog present but PIT revisions and
  unresolved repairs absent at the inspected root. This was not a candidate run.
  2026-09-23: first PIT publish done (rev 1 sp500, rev 2 ndx100, both PARTIAL,
  Silver rev 77). No PROVEN manifest exists; the matrix no longer requires one
  (publisher status is echoed, not a dimension). Livewire will republish under new
  revision numbers after PR #144; P0 discovers what exists.
- Installed MCP/DuckDB APIs, supported cancellation and actual lifespan behavior.
- Secure MCP Tunnel account/workspace permissions, installed client/backend auth and
  actual ChatGPT call. Do not install on production merely to review this document.

## Review notes

Read-only Terra workers checked lake/PIT and API contracts. Independent panel and
lead acceptance are recorded separately in the review report. A test collection
attempt lacked asyncpg/scipy in the isolated environment; it is not passing test
evidence. No runtime acceptance has been claimed.
