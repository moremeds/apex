# Market-data MCP and lake read API — design

Date: 2026-09-22. Owner: Astra. Status: revised design under review; not an
implementation, merge or deployment approval. Replaces the initial 828-line draft.
Implementation order and executable acceptance: [plan](../plans/2026-09-22-market-data-mcp-plan.md).
Review evidence: [review](2026-09-22-market-data-mcp-review.md).

## 0. Product and scope

Apex exposes the existing Livewire lake to Claude Code and ChatGPT through a
read-only MCP server. REST and MCP use the same application queries. Users can
discover data, retrieve bars/yields, inspect identity/actions/membership, select
published versions, and investigate coverage/gaps.

User decisions carried forward: inside Apex; all supported lake asset classes;
Bronze and Silver; coverage and PIT; REST parity; private tailnet access with a
static key; ChatGPT through Secure MCP Tunnel; two sequential PRs.
User correction on 2026-09-22: **all PostgreSQL work is excluded and belongs to a
follow-up task**. No signals/regime tools, database discovery, table query, joins,
PG pool, or PG startup dependency in this MCP. Existing REST database features
remain outside this change.
User correction on 2026-09-23: **indicators are excluded**. No indicator MCP tools,
no new indicator REST route, no indicator changes, and no indicator cells in the PR1
matrix. The existing REST indicator route stays unchanged. Reason recorded for the
follow-up: `regime_detector` fetches SPY from Yahoo through yfinance 1.0/curl_cffi
(bypassing Python `socket`), and on the same real SPY 1d input 140 of 1000 regime
labels changed depending on whether Yahoo was reachable, with no error surfaced.
The other 47 registry indicators were verified network/write free.

User acceptance requirement: **PR1 must run on macmini against the real lake,
covering all asset classes and all combinations before PR2 begins**. Unit tests
and laptop HTTP requests alone cannot close that gate; see plan §4.

In scope: six registered asset classes; raw and adjusted where supported;
listed/delisted/any reads under the existing contract; catalog coverage, session
gap diagnosis, corporate actions, security resolution,
membership, Silver/PIT manifests, private MCP operation.

Out of scope: lake writes/repairs/publication, order execution, options, ledger,
raw-provider archives, gold, quarantine, streaming/subscriptions, OAuth/multi-user,
public hosting, frozen backtest changes, and the PostgreSQL follow-up.
There is no promise to reconstruct arbitrary historical knowledge from mutable
files or to prove minute completeness from session presence.

## 1. Evidence baseline

Source first checked at Apex `2b44666217d0de1865af56e39721d5dbfab901bf` (v0.1.9) and
rebased on 2026-09-23 to master `c690792b126eddfb63eb3f53fe98207675e0318b` (after v0.1.11);
Livewire `origin/main` `f83e3df0c4c9ba1ba54d6e5e9abd49be50b278e2`. Between the two Apex
bases no lake-read code changed (`_chart_guards.py`, `chart.py`, `bulk_bars.py`,
`instruments.py`, `adapters/livewire/`, `src/application/`, chart payload and its
schemas: zero diff). What landed is a PostgreSQL/UW read REST surface (§1.1). Neither is a claim about the current
production image. Host paths, addresses and credentials belong in private
operator configuration, not committed documents.

### 1.1 Master changes since v0.1.9 that this design must respect

- **PG/UW read REST exists** (7bbd9735, 60641683, 67ce3127): `/v1/db/catalog`,
  `/v1/db/{database}/{schema}/{table}`, `/v1/uw/{join_name}`. Read pools come from
  `APEX_PG_READ_URLS` via `src/infrastructure/persistence/read_pools.py`, created in the
  REST lifespan into `app.state.pg_read_pools`; unset or failing DSNs degrade those routes
  to 503, not boot failure. This MCP still exposes none of it and `LakeServices` never
  touches `pg_read_pools`; the MCP exposure of PG remains the separate follow-up.
- **Route-scoped REST bearer auth exists for PG routes only**: `src/api/routes/_db_auth.py`
  `require_db_token` (env `APEX_PG_READ_TOKEN`, `Authorization: Bearer`,
  `secrets.compare_digest`, unset → `provider_not_configured` 503, bad → `unauthorized`
  401), attached as a router dependency on the three PG/UW routers. Lake REST routes stay
  unauthenticated as before; do not attach the PG token to them.
- **New error codes exist**: `query_timeout` (504), `unauthorized` (401), `forbidden`
  (403) in `src/api/errors.py`. §6 reuses them rather than adding parallel codes.
- **Route precedence precedent**: `src/api/server.py` registers the PG/UW routers under the
  comment "Literal namespaces must precede the asset-class catch-all", immediately before
  `instruments_router`. New lake/security routers register in that same block.
- **Name collision**: `db_catalog.py` has a PG `CatalogCache` (600s TTL, invalidated on
  undefined table/column). In this design "catalog" always means Livewire's coverage
  catalog (`analytics.duckdb`, `adapters/livewire/coverage.py`), which the stale-catalog
  fix did not touch; it stays read-only and open per call.

Existing source to reuse:

| Source | Existing behavior |
|---|---|
| `src/infrastructure/adapters/livewire/asset_classes.py` | Equity/fx: 1m,5m,30m,1h,1d; volatility: 5m,30m,1h,1d; cmdty/futures/rates: 1d. Per-symbol availability can be narrower. |
| `ohlc_provider.py` | Single-file DuckDB reads; adjusted daily from immutable Silver; adjusted intraday from mutable Bronze plus Silver factors; `pin_snapshot` returns an operation-local copy. |
| `revisions.py` | Current-pointer/numbered-manifest equality, structure/path validation, selected artifact SHA-256 checks; no numbered-revision public reader yet. |
| `coverage.py` | Catalog opened read-only per call, safe across atomic replacement; never register Livewire temporary views. |
| `reference.py` | Corporate actions and identity intervals already implemented; no new corporate-action reader. |
| `membership.py` | Date-based effective and known-at replay, ambiguity, unresolved identity handling and current effective history. |
| `src/api/routes/bulk_bars.py` | Equity batch up to 200, one pinned snapshot, per-symbol missing map; registered before instrument detail route. |
| `src/api/routes/_chart_guards.py` | Existing validation/window/listing behavior; explicit start bypasses tail limit; listed default window is anchored to now. |
| `src/api/payload/chart.py` and `config/verification/schemas/` | Actual wire contract, including basis, OHLCV, futures extras and contract identity. |

Livewire contracts are in `clients/pit_silver_revision.py`,
`clients/silver_revision.py`, `clients/security_master.py`,
`clients/index_membership_store.py`, `clients/source_evidence.py`,
`clients/symbol_paths.py`, and `clients/duckdb_catalog.py`.
Read the pinned definitions and tests before implementing their adapters.

The initial author's mini inspection reported Silver revision 76, absent
`silver/pit-revisions/`, and a running compose different from the repository.
Those are historical observations, **not refreshed production evidence**.
Do not bake those counts, dates or revision numbers into acceptance.
Read-only macmini inspection on 2026-09-22 confirmed the configured catalog exists,
while `silver/pit-revisions/` and `repairs/unresolved.json` are absent at the inspected
root. This is availability evidence, not a candidate run; positive PIT acceptance
is currently blocked and must be rechecked in P0.
Cause checked 2026-09-23: Livewire `origin/main` (f83e3df) has the publisher
(`clients/pit_silver_revision.py`, CLI `shepherd-silver publish --index sp500|ndx100
--membership-revision N --as-of T`), but no macmini launchd job or crontab runs it, so
no manifest was ever published. PIT is per index (sp500/ndx100), daily only, and
`publish` rejects an as-of earlier than the Silver revision's `published_at`, so PIT
history can only accumulate forward. Publishing is a Livewire lake write: it needs
user authorization and runs from the Livewire side, never from this project.
First production publish, 2026-09-23 (user-authorized, run by the Livewire session
from release f83e3df; hashes rechecked read-only by Apex): revision 1 = sp500
(membership_revision 4896, 461 members, manifest sha256 `a338d78c…805d`) and
revision 2 = ndx100 (membership_revision 1407, 98 members, sha256 `76b3daff…3531`).
Both have as_of 2026-09-23T00:00Z, daily_bar_cutoff 2026-09-22, silver_revision 77,
status **PARTIAL** and exit-0 `verify`. PARTIAL comes from the corporate-action
receipt (sp500 43/458 verified, ndx100 12/87); the cause is not yet diagnosed on the
Livewire side. There is **no PROVEN manifest**. Producing one is a Livewire data fix;
the PR1 matrix no longer needs one (plan §4.2: publisher status is echoed, not a dimension).
These are historical observations; P0 re-reads them.
Livewire diagnosis of PARTIAL, 2026-09-23 (read-only, citations checked by Apex):
receipts mark 413/415 sp500 and 75/75 ndx100 symbols `missing-event-evidence` because
the export demands per-row provenance on every historical action row
(`livewire_scripts/shepherd_actions.py:120-130`) while July rows predate provenance and
`reconcile` never backfills unchanged rows (`clients/corporate_action_store.py:255-256`);
PROVEN needs every symbol VERIFIED (`clients/pit_silver_revision.py:93-94`). A proposed
export-only fix (prove each event's head row against the bound verified fetch page)
reaches sp500 435/458 and ndx100 83/87; the remainder needs real data work (yahoo split
heads wrongly cancelled in July, stale Massive heads, rename mapping FLT/JEC, ALB
conversion provenance). No PROVEN manifest exists until Livewire ships that work.
Follow-up, same day: the export fix is Livewire PR #144 (CI green, not merged). It
changes the receipt format, so revisions 1 and 2 must be republished; they already fail
Livewire's own verify because reconcile rewrote a store row's status in place after
publication. Measured on the branch: ndx100 83/87, sp500 440/458; the remainder needs a
yahoo-split repair entrypoint that does not exist yet. Apex does not verify lineage
(§3.4), so it can still serve revisions 1-2 by their echoed status. A republish
creates new revision numbers (revisions are immutable); Livewire must also resolve
that `publish()` first runs `verify` on the current revision, which now fails. P0
discovers whatever revisions exist before the matrix runs.

**Upstream data defect (checked 2026-09-23, Silver revision 77).** Livewire reports 45
symbols whose published adjusted 1d series still contains the unadjusted split drop,
caused by the July cancellation of yahoo split rows. Apex read-only confirms two:
CMCSA close 4.18 on 1999-05-05 → 2.09 on 1999-05-06, HON 43.81 on 1997-09-15 → 22.29
on 1997-09-16. Apex serves this today in adjusted mode. It is a Livewire fix (repair
entrypoint + Silver rebuild), not an Apex workaround. The matrix oracle compares against
the same files, so it will not catch it; do not cite pre-split history of affected
symbols as evidence of value correctness.

The live source-evidence store sits only on the host's internal disk and is not
visible to the container. That is irrelevant to Apex because Apex does not replay
source evidence (§3.4); it needs only manifests and Silver artifacts on the volume.
The draft's `tools/mcp_context_server/server.py` is absent from this worktree;
repairing an untracked/local-only context server is not an acceptance criterion.

## 2. Architecture and lifecycle

```text
Claude Code --------------------> private MCP
ChatGPT -> Secure MCP Tunnel ----> private MCP
                                      |
REST routes ----------------> application lake queries
                                      |
                          existing Livewire adapters
                                      |
                        lake mounts and catalog (read-only)
```

- Thin transport adapters call shared queries. MCP never imports REST routes or
  creates the REST application (which also starts unrelated services).
- A small `LakeServices` container holds existing provider, catalog, membership,
  reference, revision readers and repairs reader only.
  Construct it from env; allow explicit injection in tests.
- REST retains its current startup/teardown and app-state injection seams.
  Initialize lake fields without rebuilding the PG/streaming lifecycle.
  The revision watcher continues to use its existing provider. Historical queries
  always use a local pinned copy; they never mutate the shared provider.
- Do not move unrelated routes merely to make every function pass through a
  container. Extract only query/validation logic shared by the two callers.
- DuckDB remains the Parquet query engine and the Livewire catalog remains the
  discovery accelerator. Improving real-lake read performance is required in PR1.
  Benchmark the existing per-call connections against a process-owned connection
  with a per-worker cursor, then bounded result caching when repeated disk reads
  dominate. Select the simplest option meeting plan §4's measured acceptance.
  Retain read-only/open-per-call catalog access across atomic catalog replacement.
- No request scans an entire lake tree or creates catalog views. Resolve encoded
  symbol paths; bounded directory enumeration is allowed for revision metadata.
  Futures candidates come from catalog rows, then exact-file metadata reads.
- Reject temporary/AppleDouble paths and paths escaping configured roots, including
  symlink escapes. SQL identifiers/paths are code-controlled; values parameterized.
- Each unset source disables only dependent queries. Distinguish unconfigured,
  absent, corrupt and available; corruption must not become a healthy empty result.

### 2.1 DuckDB performance decision (mandatory PR1 work)

The original LakeDb proposal is retained as a candidate, not assumed to be faster.
DuckDB's official multiple-thread guide explicitly supports thread-local cursors
from one parent connection, including in-memory databases; do not reject that
pattern merely because the parent connection is shared.

Compare: (A) current per-call connections, (B) lifespan-owned in-memory parent with
cursor created/queried/closed inside each worker, and (C) B plus bounded cache only
if repeated disk reads remain material. Never share a cursor between workers.
Parent closes after workers drain; interruption must target only the request's
cursor. Record installed-version behavior, memory, concurrent query correctness,
startup/shutdown and cold-process/repeated-query timings on macmini.
These variants apply to in-memory Parquet connections; durable catalog access
stays read-only/open-per-call across atomic replacement.

If C is needed, wire actual reads through it (not merely replace connect calls).
Keys include canonical query parameters, effective basis/listing, revision and all
source identities. Immutable artifacts use manifest/hash identity; mutable files
need before/after stat identity (device/inode/size/mtime_ns), with changed-during-read
results excluded. Bound total bytes as well as entries, exclude failures, and retain
required Silver integrity checks on hits. Test atomic replacement with unchanged
mtime/size in an isolated scratch lake; production lake remains read-only.

The performance report decides A/B/C; absence of a measurement is not evidence to
remove the performance requirement. No material optimization regression may hide
behind average latency. Do not add a shared cache service or new dependency.

[DuckDB multiple-thread guide](https://duckdb.org/docs/current/guides/python/multiple_threads)
is the reference; verify against the locked installed version before implementation.

## 3. Data contract

### 3.1 Identity, time and price basis

Keep asset-class registry and symbol encoding as the source of truth. Follow
Livewire canonicalization for new tools (trim, upper-case wholly lower-case symbols,
preserve mixed case), but retain existing REST normalization where compatibility
requires it. Test equivalence using explicit canonical symbols; document exceptions.

New MCP daily windows use ISO session dates, inclusive start/end. Intraday windows
require offset-aware ISO timestamps and normalize to UTC; reject naive timestamps.
Legacy REST date parsing and status codes stay unchanged. New REST queries use the
same typed arguments as their MCP twin; date-to-timestamp conversion is explicit.

Return canonical bar values using the existing economic fields: time, OHLCV,
futures settlement/open_interest plus contract identity. Rates return tenor_years
and yield_pct, never fabricated OHLC prices. Do not promise the entire physical
Parquet schema as the existing chart contract.

Keep legacy `basis` and `adjustment_revision` fields. Add provenance deliberately:
For raw Bronze equity, per-row `source_price_basis` preserves raw/split_adjusted/unknown,
and unknown when absent. It must survive the provider conversion;
never infer it from the API's raw mode. The legacy `basis=unadjusted` describes
the serving mode and is not proof that every source row was unadjusted.
Adjusted Silver carries `basis=split+dividend`; never silently fall back to raw.
Silver has no source_price_basis column: omit it and return manifest/artifact
provenance. Never join mutable Bronze to enrich a historical Silver pin.
Update serializers and JSON schemas alongside any additive fields.

### 3.2 Bounded queries and legacy compatibility

MCP defaults: bars 250 rows, rates 500; positive limits, at most 5000
rows per single-series request. Bulk: 1..200 unique equity symbols, default 50
rows each, per-symbol limit <=2000, and requested symbols × limit <=10000.
Serialized MCP data budget: 2 MiB per result; reject with `result_too_large`
and instructions to narrow/paginate, never emit malformed/truncated JSON.

For new bounded series queries, apply the cap even with an explicit start.
Return chronological last N rows in the requested window and `truncated`
based on N+1 selection, not guesswork. Echo the effective window. With no start,
use the existing window resolver; do not claim it always ends at the last
available bar. Preserve the archive's from-epoch behavior for old delisted data.

Legacy REST retains explicit-start behavior and limit<=0 meaning all. Shared
queries therefore accept an explicit output policy: legacy REST or bounded MCP.
This is a deliberate compatibility difference, not a failed parity test.
New REST routes use bounded defaults. Add revision parameters to bulk as well
as single bars. Keep deprecated aliases compatible.

Push bounded selection into disk reads rather than loading a full minute file
into Python and slicing it. Disk work runs off the event loop; retain a concrete
query deadline/interrupt strategy verified against the installed DuckDB in P0.
A returned row cap alone is not a bound on I/O or CPU.

All pageable catalog/revision/identity/action/member/history lists have limit
(default 100, maximum 2000), offset>=0, deterministic ordering, returned count,
`truncated` and `next_offset`. Catalog pages are not a multi-call snapshot:
echo catalog identity/mtime; a changed source requires a fresh traversal.
Legacy REST envelopes remain unchanged; new pagination parameters are opt-in.

### 3.3 Listing

Reuse the existing guards and merge logic (unchanged from v0.1.9 through c690792b). Listed uses live Bronze; delisted uses archive;
any resolves residency and dual union, with live winning each shared NY trading
date. This applies per artifact/timeframe, not globally per ticker.
Adjusted with delisted/any is rejected as current guards require. Dual is a
residency label, not evidence of issuer continuity. Do not emit the reserved
`ambiguous_symbol` error for dual residency.

Corporate-action and delisting responses remain ticker-keyed. Delisting returns
identity intervals and `delisting_reason_available=false`, not invented terminal
consideration or cause. Actions may belong to another issuer after ticker reuse.

### 3.4 Silver revision and PIT are different guarantees

`silver_revision` and `pit_revision` are mutually exclusive positive integers.
Explicit raw conflicts with either and is rejected; an omitted mode becomes
adjusted. Both require equity, daily timeframe and listing=listed.
This conservative boundary prevents immutable-history claims over mutable Bronze
intraday. Current adjusted intraday remains supported without a historical pin,
with provenance `immutable_history=false`.

Numbered Silver reads reuse current manifest validation, including filename/payload
revision agreement, path containment and artifact hashes. A current read also
compares the pointer bytes with its numbered manifest. Missing revision is 404;
present but corrupt evidence is unavailable, not 404 and not raw fallback.
A valid Silver pin is reproducible adjusted data, not by itself as-known-at data.

PIT follows Livewire `policy_version=pit-silver-v1`:
`as_of` and `published_at` are aware timestamps; `daily_bar_cutoff` is a
session date. Preserve `members`, policy, index_id, membership_revision,
silver_revision, corporate_actions_as_of, input_hash and inputs, including
silver_artifacts and corporate_action_receipt. Do not rename keys by guessing.

**Apex does not re-verify PIT lineage** (user decision 2026-09-23). Apex is a read-only
consumer; Livewire's publisher computes the lineage and its own `verify` replays it,
including source evidence. Both run on the same host under the same owner, so there is
no trust boundary for Apex to defend, and a second implementation of Livewire's
canonical serialization would only duplicate it and drift from it. Apex therefore does
not recompute input_hash, receipt hashes, append-order prefix hashes or source
evidence, and does not need the host-only evidence store.

A PIT read does only what serving the data correctly requires:
1. Parse `pit-revisions/revision={n}.json` by explicit revision (never via
   `current.json`): schema/policy keys present, filename revision equals payload
   revision. The PIT file has no self-digest; do not check it against anything.
   Malformed → `pit_unavailable`.
2. Path namespaces differ (Livewire `clients/pit_silver_revision.py:408-425`):
   `inputs.silver_manifest`, `inputs.silver_artifacts[]` and
   `inputs.corporate_action_receipt` are relative to the Silver root;
   `inputs.security_master` and `inputs.membership` are relative to the lake root.
   Apex opens only Silver artifacts, so containment is checked only on the
   `silver_artifacts[]` entry it serves. The lake-root references are carried as
   metadata, never opened, never a reason to reject.
3. Serve the requested symbol's daily artifact from `inputs.silver_artifacts[]`
   (entries are `{path, sha256}`), and compare that entry's `sha256` with the file
   bytes before use, the same byte check `revisions.py` applies. Do not call
   `read_current()`, do not compare against `inputs.silver_manifest.sha256`, and
   do not check `input_hash`. Mismatch, missing file or no entry for the symbol →
   `pit_unavailable`, never another revision or raw data. An artifact that Silver
   later evicts makes that PIT revision `pit_unavailable`; that is expected.
4. Apply the manifest's scope (below) and `daily_bar_cutoff`.
5. Echo Livewire's `status` unchanged as `publisher_status` (PROVEN or PARTIAL) with
   revision, index_id, as_of, silver_revision. Apex never upgrades or recomputes it.

Only symbols in the manifest's member scopes may be served. A `trade_date` is in
scope when, for some member entry of that symbol, `session_from <= trade_date` and
(`session_to` is null or `trade_date < session_to`), and `trade_date <=
daily_bar_cutoff`. `session_to: null` means still a member at as_of (the common case;
`pit_silver_revision.py:316-318`). If the requested window intersects scopes of two
different `security_id`s for the symbol, return the existing `ambiguous_symbol` (409,
`src/api/errors.py`, defined but not yet raised) listing each scope's security_id and
dates so the caller can narrow the window; no new argument.
Echo allowed scopes, effective window, revisions, policy and publisher status.
Scope can be short: on the first sp500 publish, 204 of 461 members have
`session_from` 2026-09-17 (Livewire recorded their membership then), so a PIT read
for them returns only a few sessions. Matrix cases pick members whose scope covers
the window under test; this is upstream data, not an Apex defect.
A query wholly outside scope returns a scoped error, not unrelated Silver history.

Livewire contract facts observed on the first publish:
- PIT revision numbers are global across indexes, and `silver/pit-revisions/current.json`
  is a byte copy of the last publish of any index (now ndx100). "Current" is
  therefore not a per-index notion: never resolve an index's PIT via current.json.
  Discovery scans `revision=*.json`, lists each with index_id, and reports the latest
  per index; reads use an explicit `pit_revision`.
- Revisions are immutable and numbered +1 (`pit_silver_revision.py:91`, `:497-500`):
  a republish creates new revision numbers and leaves earlier files in place.
- `membership_revision` is a prefix length over the index's append-order event list
  (`events[:revision]`), not the event `revision` column.
- publish writes no Livewire ledger run, so ledger/digest cannot be used as PIT
  presence evidence; read the pit-revisions directory itself.

When the pit-revisions directory is absent or has no numbered manifests, discovery
returns available=false and an empty list, and an explicit request returns
`unknown_revision`. When numbered manifests exist, that is the populated case. P0
re-reads what is on disk rather than assuming either state.

### 3.5 Membership and security

Reuse current MembershipReader replay. Existing `as_of` and `known_at`
are dates interpreted at UTC end-of-day; do not advertise instant precision.
No known_at means today's reconstruction, not historical knowledge.

`resolve_security` returns the existing SymbolResolution fields plus requested
dates and provenance. Do not attach an interval from today's `fetch_identity`
to a historical known_at result. The separate delisting query supplies explicitly
current identity intervals. Ambiguity is never resolved arbitrarily.

History is today's effective timeline; as_of only selects ticker identity.
Preserve unresolved ticker event union and supersession behavior from REST.
`include_candidates=true` includes non-rejected candidate evidence; it does not
prove complete or survivorship-free membership. Expose status/unresolved counts.
The previously observed pre-2003-09-11 identity limitation is a historical warning,
not a hardcoded data floor that hides future upstream backfills.

## 4. Coverage and gaps

Coverage rows use the durable catalog table (view_name, symbol, n_rows,
first_date, last_date), not transient views. Parse tier/class/timeframe using
the registry; report unknown catalog views instead of silently misclassifying.
Return catalog as-of information and distinguish no catalog row from no disk file.

`find_gaps` takes symbol, asset_class, timeframe, start/end session dates,
listing (same read contract), max_gaps<=2000 and calendar policy.
The returned assessment is **session_presence**, not minute completeness.
For intraday, use the file's verified timestamp convention and session timezone;
do not assume every class trades on NYSE hours.

- Equity: authoritative XNYS session calendar. Other classes: an explicitly labeled
  approximate policy (FX weekdays; volatility/cmdty/futures/rates XNYS approximation)
  unless verified per-instrument metadata supports a precise one.
  Use locked pandas-market-calendars (baseline 5.2.4), expose calendar/version and
  test closures 2018-12-05 and 2025-01-09; no generic weekday/federal substitute.
- Present sessions come from actual selected artifacts, including listing union,
  not the daily catalog snapshot. Daily DATE stays a date.
- Return interior missing sessions, leading/trailing unobserved ranges, file bounds,
  requested window, expected/present counts, calendar and certainty.
- Do not silently clamp away edges. Outside known identity lifetime, mark
  not_expected; with unknown lifetime, mark unknown rather than complete.
  An empty/absent artifact is no_data, never zero gaps/healthy.
- A gap is consecutive expected sessions, not consecutive calendar days.
  `max_gaps` caps output; response states truncation and remaining count when known.
- Repairs JSON is supplementary historical evidence, not current truth. Return
  source filename/report date; filter by symbol/class/timeframe and requested
  sessions, deduplicate exact repeats, preserve conflicting reports.
  Missing/malformed report state is surfaced as unavailable/degraded with warnings;
  it is not indistinguishable from a successfully read empty repairs list.

Lake status summarizes configured/available/degraded sources, catalog freshness,
Silver/PIT summaries and repairs state. Do not expose secret env values or
absolute host paths. No full-tree scan and no full 27000-artifact response.
Status does not initialize a PG connection.

Repairs contract (Livewire coverage_report.py/gap_engine.py): root-level
`tier_a_<date>.json` is an object with repairs[]; `decisions_<date>.json` is a list
with verdicts including terminus/inconclusive. Entries carry symbol, asset_class,
timeframe, gap/sessions, heal_by_days and source where supplied. `unresolved.json`
is a list with symbol, asset_class, timeframe, session, reason, as_of. Validate each
shape and preserve optional fields/provenance. Exclude dated subdirectories and
Shepherd staging/rollback receipts; never create missing reports or trigger repairs.

## 5. MCP tool catalog and REST mapping

Exactly **20 tools**. Common validation/pagination/budget rules above apply.
Parameters below are the semantic set; defaults come from §3, not independent copies.
All tools advertise read-only/idempotent hints; runtime controls enforce that claim.

| # | Tool | Arguments / result | REST twin |
|---|---|---|---|
| 1 | list_asset_classes | registry including payload kind/timeframes/adjusted support | new GET /v1/lake/asset-classes |
| 2 | search_instruments | q, asset_class, limit; catalog matches/provenance | existing GET /v1/instruments |
| 3 | get_instrument | symbol, asset_class; catalog + per-timeframe existence/residency | existing GET /v1/{asset_class}/{symbol} |
| 4 | get_coverage | symbol?, asset_class?, include_silver, limit, offset | new GET /v1/lake/coverage |
| 5 | find_gaps | §4 arguments; session assessment + repair evidence | new GET /v1/{asset_class}/{symbol}/gaps |
| 6 | get_lake_status | source availability/freshness | new GET /v1/lake/status |
| 7 | get_bars | symbol, asset_class, timeframe, start?, end?, limit, price_mode?, listing, silver_revision?, pit_revision? | existing GET /v1/{asset_class}/{symbol}/bars |
| 8 | get_bulk_bars | symbols[], timeframe, start?, end?, limit, price_mode?, listing, silver_revision? | existing GET /v1/equity/bars |
| 9 | get_rate_series | symbol, start?, end?, limit | existing GET /v1/rates/{symbol}/series |
| 10 | list_futures_contracts | root, limit, offset; contract metadata and coverage | new GET /v1/futures/{root}/contracts |
| 11 | get_corporate_actions | symbol, action_type?, start?, end?, limit, offset | existing GET /v1/equity/{symbol}/actions |
| 12 | get_delisting | symbol, limit, offset; current intervals | existing GET /v1/equity/{symbol}/delisting |
| 13 | resolve_security | symbol, as_of, known_at? | new GET /v1/security/{symbol} |
| 14 | list_indices | limit, offset | existing GET /v1/membership/indices |
| 15 | get_index_members | index_id, as_of, known_at?, include_candidates, limit, offset | existing GET /v1/membership/{index_id} |
| 16 | get_membership_history | symbol, as_of, index_id?, limit, offset | existing GET /v1/membership/history |
| 17 | list_silver_revisions | limit, offset; newest first, current flagged | new GET /v1/lake/silver-revisions |
| 18 | get_silver_revision | revision?; summary, paged affected symbols | new GET /v1/lake/silver-revisions/{n}; list identifies current |
| 19 | list_pit_revisions | index_id?, limit, offset; per-revision index_id/status summaries and latest per index | new GET /v1/lake/pit-revisions |
| 20 | get_pit_revision | revision; manifest summary incl. publisher_status, paged scopes | new GET /v1/lake/pit-revisions/{n}; no global-current shortcut (§3.4) |

Revision detail tools also accept limit/offset for their long lists. Never dump
the full artifact manifest. For Silver only, current detail resolves current once;
REST uses the resolved numbered detail after discovery or an explicit /current route
registered before /{n}. Implement and test one documented choice, not two competing
paths. PIT detail always requires a revision and has no /current route (§3.4).
Mount lake/security routes before `/v1/{asset_class}/{symbol}` in instruments;
otherwise lake/status and security/symbol hit the catch-all. Preserve existing
bulk/returns/membership precedence, test new prefixes and update the stale comment.

MCP series use columns+rows with an output schema; REST preserves records and
existing envelopes. Semantic parity compares values, effective window, identity,
basis, source status and revision after a small named projection. It may ignore
generated_at/transport metadata and documented pagination wrappers only.
Do not discard substantive fields to make parity pass. Legacy REST schemas and
consumer-facing docs are updated for explicit additive changes.

Legacy rates REST without new limit keeps its full-history default; a positive
limit opts into the bounded MCP policy. Bars(rates) points to rates series.
Bulk remains equity-only. Tool names are an explicit tested set, not count alone.
No tools_signals module or database tool is registered.

## 6. Errors, auth and transport

Application errors are transport-neutral. REST maps them through existing
ApiError machinery; MCP maps expected failures to ToolError with stable code
and actionable message. Do not import REST exceptions into the application.

Preserve existing REST status mapping: invalid_parameter/unsupported class/
timeframe are 400, not the draft's 422; adjusted_unavailable is 503.
Typed request validation retains its existing 422 envelope; distinguish semantic
400 guards, including syntactically valid unsupported timeframe `4h`.
Reuse existing `query_timeout` (504) for deadline expiry and `unauthorized` (401) for
MCP auth failures. Add only the missing narrowly scoped codes: unknown_revision (404),
pit_unavailable (503), revision_not_supported (400), result_too_large (400).
Out-of-PIT-scope requests use invalid_parameter. A PIT window spanning two security_ids
uses `ambiguous_symbol` (409, §3.4); membership keeps `ambiguous_security` (404).
Corrupt evidence is not unknown_revision. Unexpected errors are logged with a
correlation id; do not return tracebacks, credentials or filesystem paths.

MCP requires non-empty APEX_MCP_API_KEY; missing key refuses boot. This differs
deliberately from `require_db_token` (route dependency, degrades to 503): the MCP app
has no unauthenticated mode. Pure ASGI Bearer middleware uses `secrets.compare_digest`
like `_db_auth.py` and protects all MCP requests
(including discovery). Bad/missing key gives 401. No wildcard CORS; preserve SDK
Host/Origin protections, allowing only configured private hostnames/addresses.
Test Origin rejection and legitimate clients rather than disabling protection.

New REST capabilities are available under existing private deployment assumptions.
Do not add unused optional REST auth or change HTTP/WebSocket access here; the
existing PG-route token stays scoped to `/v1/db/*` and `/v1/uw/*`.
Preserve existing /health; mandatory MCP authentication remains in scope.

MCP uses Streamable HTTP with stateless JSON responses. Confirm actual SDK
constructor/app/lifespan APIs in a local minimal smoke test before wiring.
Official SDK v2 documentation supports MCPServer and Client; pin a tested v2
version in uv.lock and constrain the api extra to <3. No guessed exact latest
version or unused CLI dependency. Low-level SDK APIs changed; migrate only an
existing tracked consumer discovered by rg, not the absent context-server path.

Add a dedicated cheap MCP /healthz (no data/secrets) for process liveness.
Authenticated real list_tools/call_tool is the readiness probe; an arbitrary
405 or any HTTP response does not prove readiness. Readiness and source
availability are separate because absent optional data must not restart the service.

## 7. Configuration and deployment

Reuse existing lake env vars: APEX_LIVEWIRE_ROOT (Bronze),
APEX_LIVEWIRE_SILVER_ROOT, APEX_LIVEWIRE_PRICE_MODE,
APEX_LIVEWIRE_COVERAGE_DB, APEX_LIVEWIRE_LAKE_ROOT,
APEX_LIVEWIRE_DELISTED_ROOT. New: APEX_LIVEWIRE_REPAIRS_ROOT,
APEX_MCP_API_KEY, APEX_MCP_HOST (loopback default), APEX_MCP_PORT (8333 default).

Use a private env file for lake host path, tailnet bind address and image reference.
One read-only lake mount may replace overlapping mounts only after validating the
actual production compose, root layout and reference-reader paths. Do not mount a
parent home directory. Do not inherit the whole API env file into MCP: pass only
its lake/transport variables, no database/provider credentials.

MCP runs a separate process from REST and does not start streaming machinery.
Container listener may be 0.0.0.0 internally; host publication must bind an explicit
tailnet address. Verify effective socket/firewall behavior on macmini, including
whether a reverse tunnel fronts the deployment. Never infer it from compose text.
Use the approved release image digest/version, not a floating latest tag during
acceptance. No production replacement/restart during read-only PR1 verification.
Candidate/release dependency installation must actually consume uv.lock. Merely
copying it into the current pip-install Dockerfile does not enforce versions.
Record image digest and installed SDK/calendar/DuckDB versions.

Claude Code configuration is an operator template with URL/key placeholders.
ChatGPT connects via a host tunnel-client and outbound HTTPS. Official docs confirm
this topology; workspace developer mode and Platform tunnel permissions/association
are separate prerequisites. Do not claim current account access from the handover.
Record actual installed client version, help output, backend Bearer injection and
stateless compatibility in a private runbook before deployment. No invented flags
and no production installation merely to finish a design review.

Public docs contain portable templates. Credentials, host addresses, absolute host
paths and account-specific image references remain in the private overlay/runbook.
The operator must prove both direct tailnet access and actual ChatGPT tools/call;
listing a connector or receiving a non-401 alone is insufficient.

Official references checked during review:
- [MCP Python SDK migration](https://py.sdk.modelcontextprotocol.io/migration/)
- [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)

## 8. Implementation seams

Expected additions, split only on real responsibilities:
- `src/application/lake/`: services, typed results/errors, shared bars,
  coverage/gaps and identity/revision queries. Avoid one trivial wrapper per file.
- `src/infrastructure/adapters/livewire/pit_revisions.py` and `repairs.py`.
- `src/api/routes/lake.py`; mandatory MCP auth stays at its server boundary.
- `src/mcp/server.py`, tool groups for discovery, bars, identity, revisions.
- Focused tests and a reusable real-lake verification runner/report under the
  repository's scripts/test conventions; implementation chooses verified paths.

Expected modifications:
- Existing revisions/coverage/provider adapters.
- Existing relevant REST routes, payload builders and verification JSON schemas.
- API lifespan only at lake initialization seam; preserve injection/lifecycle tests.
- pyproject api extra + uv.lock, compose, Docker image if required, Makefile.
- Import-graph/keepset guards and existing consumer/API documentation.
  Existing carve classification covers a fixed keepset, not arbitrary new packages;
  add a focused AST boundary check for lake/MCP imports, including transitive paths
  into REST application/PG/streaming startup.

LakeDb is added only if the measured B/C option is selected (§2.1). No duplicate
corporate-action model/reader, PG services, frozen subsystem edits, shared cache
service or mandatory arbitrary per-file line quotas.
Respect the repository's responsibility-based size guidance.

## 9. Verification and completion

The plan defines the full PR1 macmini matrix and its hard gate. For both PRs:
- Keep real market-value fixtures dated/provenanced; reuse existing SPY/QQQ/VSCO
  and membership/Silver builders. Derived/corrupt manifests and removed rows are
  explicitly test mutations, not published production facts. Tests never fetch.
- PIT fixtures copy real Livewire manifest serialization and cover scope filtering,
  malformed manifest, escaped path, artifact hash mismatch, PARTIAL echo and absence.
- Assert two concurrent revision reads cannot affect each other or the default
  provider/watcher; include bulk pinned revision.
- Verify every tool/REST mapping, route precedence, defaults, limits, JSON schema,
  price basis and expected errors. Check the explicit 20-name set.
- Check read-only behavior through mount permissions and before/after selected
  artifact hashes; import/grep guards are supplementary, not proof of no writes.
- Run targeted existing tests first, then unit/carve and make quality. CI and live
  matrix are separate gates; no release conclusion from tests alone.

Completion states must be reported separately: design reviewed; implementation
verified locally; PR1 macmini matrix passed; PR merged; exact image deployed;
Claude Code call succeeded; ChatGPT call succeeded. A missing positive real PIT
case or any legal combination with no real artifact remains **BLOCKED_DATA**,
not PASS. Nothing publishes new Livewire evidence as part of this read-only task.

## 10. Open implementation prerequisites

1. Source-check the exact installed MCP/DuckDB APIs and minimal lifespan/auth smoke.
2. Freeze PIT manifest fixtures copied from Livewire's serialization.
3. Refresh macmini roots/catalog/retained revisions/calendars/repairs and actual compose;
   discover fixture candidates without a request-time full-tree scan.
4. Resolve every real-data gap in the PR1 matrix with the upstream owner; this task
   does not authorize lake repair or publication. Keep the gate open if unavailable.
5. Confirm tunnel account permission/association, client version and backend auth;
   this gates ChatGPT acceptance, not the independent lake query implementation.
6. Independent review status and remaining findings live in the review report;
   a missing required review seat cannot be described as a passed tribunal.
