# PR #177 self-review — authenticated PostgreSQL + UW read APIs

Date: 2026-09-23. Reviewer input for Fable. PR: https://github.com/moremeds/apex/pull/177
Base `2b446662` (v0.1.9). Pushed head `7bbd9735`. The working tree adds 13 modified files and
1 new file on top of that head, **not yet committed**.

## Scope

Read-only HTTP access to macmini Postgres through three routers, all behind a Bearer token:

| Route | File | Purpose |
|---|---|---|
| `GET /v1/db/catalog` | `src/api/routes/db_catalog.py` | Allowlisted tables and columns per database; ops tables are excluded |
| `GET /v1/db/{database}/{schema}/{table}` | `src/api/routes/db_table.py` | Paged generic read with select, where, order, limit and offset |
| `GET /v1/uw/{join_name}` | `src/api/routes/uw_joins.py` + `src/api/uw_join_registry.py` | 11 predefined option_wizard.uw_scan joins, with page-level coverage |

Supporting files:
- `_db_auth.py` checks the token and uses `secrets.compare_digest`.
- `read_pools.py` builds one pool per database. Each pool is read-only, has a 30s statement timeout and `max_size=3`.
- `payload/tabular.py` builds the response, and `tabular_payload.schema.json` defines its schema.

Out of scope: cross-database joins, arbitrary SQL, view migrations, and the frozen backtest.

## Changes since the pushed head (uncommitted)

**Pass 1: verification and test coverage.** This pass changed no `src/` files.
- 71 new unit tests covering:
  - injection attempts through the column, where, order and paging parameters
  - excluded tables that are unreachable through the table route or any join
  - the full set of auth cases
  - pre-existing routes that still need no token
  - limit and offset bounds
  - payload schema edge cases
- Schema fix. `coverage` used to be a bare `object`. It now requires `scope="returned_rows"` and a `tables → {matched,total}` map.
- Fixture fix. A test used the placeholder tickers AAA/BBB with invented scores. It now uses frozen ZM rows from `tests/fixtures/pg_read_api/rows.jsonl`.

**Pass 2: fixes D3, D4 and D5.**
- **D3.** A new helper, `src/api/routes/_db_errors.py` (`map_driver_error`), replaces a copy-pasted except block in all 3 routes. It maps driver errors as follows:
  - Dropped table or column (the catalog cache is stale) → 400 `invalid_parameter`.
  - `InsufficientPrivilege` → 403 `forbidden`. This is a new `ApiErrorCode`, with its status in `errors.py` and a matching row in the `docs/argon-apex-api.md` error table.
  - `TooManyConnections` and connection errors → 503.
  - Anything else is re-raised and becomes a redacted 500.
  - The response never contains SQL or the driver message.
- **D4.** `create_read_pools` now keeps a database whose pool failed to start as `name → None` instead of dropping it:
  - A configured database that is unavailable returns 503.
  - A database that was never configured returns 400.
  - The unfiltered `/catalog` skips unavailable databases.
- **D5.** The join name is validated before the pool lookup, so an unknown join returns 400 even when the pool is missing.

Each fix was ablated: the fix was reverted, its test failed, the fix was restored and the test passed.

**Pass 3: Fable review follow-ups.** Fable reviewed pass 2 and found two items to fix in this PR.
- **A.** A stale password or a role without LOGIN used to return 500 on every request. `InvalidAuthorizationSpecificationError`, the parent class of `InvalidPasswordError`, is now in `_UNAVAILABLE` and returns 503. The tests cover this in the table, join and catalog routes.
- **C.** The unfiltered `/catalog` used to fail with 503 as a whole when any one database was unreachable. It now lists the databases it can reach and names the others in a new required field, `unavailable` (the schema and the API doc are updated). With `?database=x` an unreachable database is still 503.
- Reverting either fix makes its test fail.

## Evidence

| Check | Result |
|---|---|
| `uv run pytest tests/unit/api/ --no-cov -q` | 289 passed; 211 at base, 282 after pass 1, 285 after pass 2 |
| flake8 / isort | clean |
| mypy `src/ tests/` | clean, 754 files |
| black 26.5.1, the version CI installs | 745 unchanged |
| black 25.12.0, the local version | 5 files outside this PR flagged (version drift). CI's newer black accepts them. Not reformatted, because reformatting would revert them to the old style. |
| CI on `7bbd9735` | 13/13 green. The uncommitted work has not run in CI. |
| Module size | Largest source file is `uw_join_registry.py` at 374 lines. Largest test file is 477. All under 500. |

Verified claims, with file and line in the current tree:
- **Identifiers and values.** Every identifier comes from the catalog and is quoted with `_quote` (`db_table.py`). Values are bound as `$n` parameters. Join SQL is static, and `_TOKEN.sub` emits only `$n::type`.
- **Auth scope.** Auth is a router-level `Depends` on the three new routers only. `server.py` includes them with no app-level dependency. `/health` and `/v1/equity/bars` are tested to not return 401.
- **Latest run.** A join's "latest run" is `max(run_id)` from its driving table, never taken from `scan_runs`. The skew join uses `basis='eod'`. Timestamptz keys are compared as UTC dates.

## Self-review findings: open, for the reviewer to judge

1. **The stale catalog is reported but not refreshed.** D3 now returns 400 for a dropped table or column. The `CatalogCache` entry (`db_catalog.py:129`, `ttl_seconds=600`) is not invalidated, so the 400 repeats until the entry expires. Invalidating the entry for that database when `UndefinedTable/UndefinedColumn` is raised would be a small change. The open question is whether it is worth adding.
2. **Resolved in pass 3 (C).** The unfiltered `/catalog` hid a failed database. A consumer cannot tell "not configured" from "configured but down". Asking for `?database=x` does return 503. One option is to add an `unavailable: [...]` field to the payload, which would be a schema change.
3. **`PROVIDER_NOT_CONFIGURED` now also means "configured but down".** This reuse predates this pass, and D4 extends it. It is semantically loose, but consumers already treat 503 as "retry later".
4. **The 403 reveals that a table exists.** This is acceptable: a table only reaches SQL if it is already in the allowlisted catalog, so a 403 means only that the grant and the catalog have drifted apart.
5. **Documented in pass 3.** `strike_grid` and `trade_insight_thread` pick the latest run before applying date filters (`uw_join_registry.py:117-136`). A date-filtered request can return an empty page even when an older run matches. This is as the spec §5 describes. It should be documented or the filter order changed; it is a product decision.
6. **Unit tests cannot prove database permissions or real SQL.** `scripts/check_pg_read_api.py` must be run as `apex_reader` on the mini after the DBA grants are applied.

7. **The D4 `None` path covers only malformed DSNs (Fable, finding B).** The pools use `min_size=0`, so `create_pool` does not connect at startup. A host that is down or rejects the credentials fails on first acquire instead, and `map_driver_error` turns that into 503. The end result is still 503, but the D4 test covers only the startup path. Kept as is: `min_size=1` would hold one idle connection per database just to probe at boot.
8. **`SELECT x.*` plus named right-hand columns could produce duplicate column names (Fable, finding D)** (`uw_join_registry.py:102, 121, 223, 278`). If two output columns share a name, one silently overwrites the other. **Cleared 2026-09-23:** the live run below found no duplicate column names in any of the 11 joins.

## Live run as apex_reader (2026-09-23)

The DBA grants were applied on the mini, and `apex_reader` got a scram password. `scripts/check_pg_read_api.py` then ran from the worktree over an SSH tunnel to the mini's `127.0.0.1:5432`. Result: **PASS**.
- All four databases connect as `apex_reader`, read-only, with a 30s timeout.
- The catalog lists 9 relations in apex_signals, 30 in core, 6 in option_chain and 164 in option_wizard.
- Missing and wrong tokens both return 401. The excluded table `raw_payloads`, an injected column and an oversized run_id all return 400.
- All 11 joins return rows, and none has a duplicate column name.
- For `macro_evidence_chain`, the UTC-date filter returns the same 4 states whether the session time zone is UTC or Asia/Hong_Kong.

Some joins report coverage of 0 for a table. This comes from sparse data: for example, `fundamental_result_provenance` has 0 rows. It is not a join defect.

## Not done, and needs the user

- Commit and push the uncommitted work to this branch, then wait for CI to go green again.
- Merge PR #177.
- DBA grants run as the superuser: draft `apex_reader_grants.sql`, kept in the session scratchpad and not in the repo.
- Put `APEX_PG_READ_TOKEN` and the four `APEX_PG_READ_URLS` DSNs into `~/apex-deploy/.env` on the mini.
- Release v0.1.10, then deploy through the GHCR digest-pinned compose flow.
- Live checks: 401, 200 and 400 responses; the 11 joins; `check_pg_read_api.py` run as `apex_reader`.

## Suggested reviewer focus

The D3 status choice (400 vs 409/410 for a stale catalog), the D4 `None` sentinel versus a separate "configured" set, and open findings 1, 2 and 5.
