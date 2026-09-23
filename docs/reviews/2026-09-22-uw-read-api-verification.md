# UW read API — review and verification

Baseline: `2b44666217d0de1865af56e39721d5dbfab901bf` (`v0.1.9`).
Branch: `feat/uw-read-api`. Initial tracked tree was clean; the design already
existed locally and was hidden by the repository's `/docs/` ignore rule.

## Design review and resulting behavior

- The original universal “100% match” claim did not hold against current data.
  All 11 requested joins remain, preserve driving rows with LEFT JOIN, and expose
  per-relation matched/total counts for the returned page.
- Signal hits/context flags and trade candidates/analyses aggregate independently;
  two child collections cannot multiply each other. Watchlist/card/quote are
  ticker-keyed current snapshots, not historical run reconstruction.
- Default run selection uses the populated driving table for the requested ticker.
  Strike grid returns both parsed OCC roots; the common root is null if they
  disagree. Multiplier and settlement are not invented.
- Catalog entries require schema USAGE and table SELECT. Catalog identifiers are
  quoted only after allowlist lookup; filter values are bound. Excluded relations
  are also removed from FK metadata. Every query runs read-only, with a 30-second
  statement timeout and a separate pool capped at three connections per database.
- All three new routers require the user-approved dedicated Bearer token.
  An unset token gives 503; missing/wrong credentials give 401 with a Bearer
  challenge before catalog or connection access. Other existing routes retain
  their behavior. The unit checks assert that credentials are absent from errors
  and captured application logs.
- Macro timestamp bounds now compare UTC calendar dates. The previous DATE-to-
  TIMESTAMPTZ comparison lost the end day's rows and depended on session timezone.
- Generic user ordering appends PK tie-breakers for nonunique columns. Offset
  overflow is rejected with a parameter-specific 400. Ordinary offset pagination
  is not a snapshot under concurrent writes.
- DBA activation requires creator-specific default privileges, existing SELECT
  grants and revokes for excluded relations. Those operations are documented,
  not executed by this implementation.

## Verification evidence

| Check | Result | Re-run |
|---|---|---|
| Whole unit suite, existing offline switch | 2,228 passed, 87 skipped | `SKIP_NETWORK_TESTS=1 uv run --frozen pytest tests/unit/ --no-cov -q` |
| Lead independent route/lifecycle/error/payload regression run | 77 passed | `SKIP_NETWORK_TESTS=1 uv run --frozen pytest tests/unit/api/test_db_catalog.py tests/unit/api/test_db_table.py tests/unit/api/test_uw_joins.py tests/unit/api/test_tabular_payload.py tests/unit/api/test_errors.py tests/unit/api/test_server_lifespan.py --no-cov -q` |
| Full source/test type check | 753 files, no issues | `uv run --frozen mypy src/ tests/` |
| Import sorting and flake8 | Passed across source, tests and verification script | `uv run --frozen isort --check-only src/ tests/ scripts/check_pg_read_api.py`; `uv run --frozen flake8 src/ tests/ scripts/check_pg_read_api.py` |
| CI-equivalent Black (Python 3.13, Black 26.5.1) | 745 files unchanged | `uvx --from black==26.5.1 --python 3.13 black --check src/ tests/ scripts/check_pg_read_api.py` |
| Authenticated Mini candidate | Four catalogs, generic reads, rejection checks and all 11 joins passed | `APEX_PG_READ_URLS=... APEX_PG_READ_TOKEN=... uv run python scripts/check_pg_read_api.py --output result.json` |
| Real source fixtures | 11 captured query records, at most two rows each, dated 2026-09-22 | `tests/fixtures/pg_read_api/rows.jsonl` and its README |

The authenticated Mini run was captured at `2026-09-22T08:37:10.306616+00:00`.
It observed `transaction_read_only=on` and `statement_timeout=30s` for all four
connections. Catalog relation counts were core 30, option_chain 7,
option_wizard 164 and apex_signals 9. Counts reflect the verification role at
capture time, not a permanent inventory or production-reader privilege claim.

The candidate verified missing/wrong Bearer credentials, numeric strings, empty
results retaining columns, excluded-table rejection, injected-identifier
rejection, oversized run-ID rejection, and coverage counts for every join.
For `macro_evidence_chain?start=2026-09-21&end=2026-09-21`, the candidate returned
four distinct state IDs; independent SQL returned four in both UTC and
Asia/Hong_Kong sessions. Legacy fundamental provenance and daily-panel matches
remain partially absent and are reported as such.

The first network-enabled unit run had two Yahoo-loading failures in the legacy
validation runner. External market-provider connectivity is not part of the
passing offline claim. An initial sandboxed run also denied an existing test's
temporary-directory write; all tests passed once that test path was writable.

Local locked Black 25.12.0 differs from the existing CI's Black 26.5.1. The older
formatter flags five untouched baseline files, including frozen backtest files.
They were not modified. CI's actual formatter/version/interpreter combination
passes across its source/test scope plus the new verification script. A broader
`black --check .` also flags 11 unchanged files outside that CI scope; those are
not part of this feature. Final PR CI is checked separately at delivery.

## Review and delivery boundary

Independent spec and initial code reviews used pinned Fable 5 and Grok 4.7.
Accepted findings include coverage, child multiplicity, access authentication,
creator-specific grants, macro date semantics, pagination and delivery of the
fixture files. The lead refuted two schema/driver claims with direct evidence:
`company_identity_open_uq` is a valid unique index on ticker where valid_to is
null; out-of-range int2/int4/int8 parameters raise asyncpg DataError, already
mapped to 400. No arbitrary latest-identity workaround or duplicated integer
range table was added. String filter whitespace remains literal by design.

Final review snapshot SHA-256:
`fed4371da7193e598825c7bf92113cc68660cb2270523d598c9ee0d0d62c73d1`.
Both final reports matched that exact snapshot. Grok found no issues. Fable
found no remaining code defects and identified one missing canonical API-reference
update. Herd task 2 added the reference and removed README duplication; the lead
checked it against the source and clarified pagination bounds and serialization.
The lead accepted the code
after personally reviewing the cumulative changes, inspecting Mini evidence and
rerunning the checks above. Python AST hashes remained unchanged after the review
freeze. The bytea conversion branch was retained: it is a small scalar binding
case for the requested generic table reader, not a new abstraction.

Implementation began with native workers. Following the user's correction,
remaining implementation, authentication, confirmed review fixes and verification
were transferred to the persistent herd worker `uw-implementer`, using SWE-2 Max.
The lead owns integration and acceptance. The Fable/Grok workers are independent
reviewers, not implementation substitutes.

Mini verification used an isolated candidate and an existing operator role.
`apex_reader` was absent when inspected. No production role/grant/config changes,
service deployment or restart were performed. Production activation still needs
reader-role provisioning, token/DSN configuration and post-deploy consumer checks.
