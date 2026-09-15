# livewire membership fixtures

Real livewire lake artifacts, copied verbatim (no filtering, no edits). Tests read
them through DuckDB exactly as the running service reads the live lake.

| Fixture                                  | Origin (host `macmini`, user `moremeds`)                                        | Size   | Rows |
| ---------------------------------------- | ------------------------------------------------------------------------------- | ------ | ---- |
| `index_membership/djia/events.parquet`   | `/Users/moremeds/market-warehouse/data-lake/index_membership/djia/events.parquet` | 8.7 KB | 100  |
| `security_master/events.parquet`         | `/Users/moremeds/market-warehouse/data-lake/security_master/events.parquet`       | 7.2 KB | 1    |

Copied **2026-09-14**. The security master was 7.2 KB on the source host, well under
the 2 MB threshold, so it was copied whole rather than filtered.

## What the real data does and does not contain

Production membership data is still being backfilled, and the fixtures show it:

- All 100 djia events have `status = 'unresolved'` and a `security_id` of the form
  `unresolved:<TICKER>` — **zero verified rows**. 65 `add`, 35 `remove`, 61 distinct
  securities, `effective_at` spanning 1991-05-06 .. 2026-06-29, every row with the same
  `known_at` (2026-09-13T14:32:26Z).
- The security master holds exactly **one** verified row: `MUNJ` →
  `sec_405d12b544ef24fee4a9ef06b721d90e`, `provider=massive`, `exchange_mic=ARCX`,
  `effective_from` 2026-08-26T00:00:00Z, open-ended.

So the verified path over this data is legitimately empty, and the tests assert the
503 fail-closed behaviour for it rather than inventing verified events. The replay,
`known_at` gating and symbol-resolution logic are exercised through the candidate
(non-rejected) path and the one real security-master row. No event in these files was
authored, altered or synthesised.

## Refreshing

```bash
scp macmini:/Users/moremeds/market-warehouse/data-lake/index_membership/djia/events.parquet \
    tests/fixtures/livewire_membership/index_membership/djia/events.parquet
scp macmini:/Users/moremeds/market-warehouse/data-lake/security_master/events.parquet \
    tests/fixtures/livewire_membership/security_master/events.parquet
```

Re-run the counts above after any refresh: several tests assert on them, and they are
assertions about real data, not about invented values.
