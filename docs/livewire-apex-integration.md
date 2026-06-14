# livewire → apex: bronze data-lake integration

How **apex** reads market bars from **livewire**, and the exact data contract apex depends on.
Audience: livewire maintainers (don't break this contract) and apex operators (point apex at
the lake). This is the *read* side; livewire owns *writing* the bronze tree.

> **Source of truth.** The contract below was confirmed against livewire's own writers —
> `clients/bronze_client.py`, `clients/intraday_bronze_client.py`, `clients/symbol_paths.py` —
> and smoke-tested against the real lake on 2026-06-14. apex's side lives in
> `src/infrastructure/adapters/livewire/{paths.py,ohlc_provider.py}`.

---

## 1. Model

- livewire is the **warehouse**: it writes per-symbol, per-timeframe **Parquet** files into a
  Hive-partitioned *bronze* tree.
- apex is a **read-only, on-demand** consumer. It **stores nothing**. For each
  `(symbol, timeframe, time-range)` it needs, it runs one ephemeral **DuckDB**
  `read_parquet(...)` query against a single file and returns the rows. It never scans the
  whole universe and never copies the lake.
- apex reads bronze in two places: the streaming **warmup seed** (`SubscriptionManager`
  preloads history before live ticks) and the **chart read surface** (`GET /bars`,
  `GET /indicators` — see [argon-apex-api.md](argon-apex-api.md)).

```
livewire writers ──▶  data-lake/bronze/ (Parquet)  ◀── apex (DuckDB read_parquet, on demand)
                                                          └─▶ argon (/bars, /indicators)
```

---

## 2. Directory layout (the partition contract)

```
<APEX_LIVEWIRE_ROOT>/
  asset_class=equity/
    symbol=<encode_symbol(TICKER)>/
      1d.parquet        # daily
      1m.parquet        # intraday
      5m.parquet
      30m.parquet
      1h.parquet
```

- `<APEX_LIVEWIRE_ROOT>` is livewire's `data-lake/bronze` directory.
- `asset_class=equity` is the only class apex reads today.
- One Parquet file **per timeframe** per symbol. apex resolves the path with
  `parquet_path(root, symbol, timeframe)` and reads that file directly — **no globbing, no
  manifest**. A missing file is treated as "no data" (apex returns an empty list, never errors).

### Timeframes apex reads

`1m`, `5m`, `30m`, `1h`, `1d` (`SUPPORTED_TIMEFRAMES` in `paths.py`). Any other timeframe is
rejected before touching disk. This is **narrower** than argon's signal-contract timeframe enum
(which also lists `15m`/`4h`/`1w`); apex's chart endpoints return `400` for those because
livewire does not warehouse them.

### Symbol encoding (`encode_symbol`) — must match livewire 1:1

The partition directory name is `symbol=<encode_symbol(TICKER)>`, not the raw ticker. apex
mirrors livewire's `clients/symbol_paths.py` byte-for-byte:

- Keep these characters **literal**: `A–Z`, `0–9`, `.`, `_`, `-`.
- Percent-encode **every other** character as its UTF-8 bytes, uppercase hex: `%XX` per byte.

| Ticker | Partition dir |
|---|---|
| `AAPL` | `symbol=AAPL` |
| `BRK.B` | `symbol=BRK.B` |
| `BF-A` | `symbol=BF-A` |
| `RDS/A` | `symbol=RDS%2FA` (`/` → `%2F`) |
| `T+E` | `symbol=T%2BE` (`+` → `%2B`) |

> Tickers are uppercase, so the literal set is intentionally case-safe (no lowercasing) — safe
> on case-insensitive filesystems. If livewire ever changes this encoding, apex's `encode_symbol`
> must change in lockstep or reads silently miss.

---

## 3. Column schema (per timeframe)

apex reads columns **by name** and **ignores any extras**, so livewire may carry additional
columns freely. The names and types apex requires:

### Daily — `1d.parquet`

| Column | Type | apex uses it? | Notes |
|---|---|---|---|
| `trade_date` | `DATE` (date32) | ✅ **timestamp key** | The bar's calendar date. apex maps it to a UTC midnight instant. |
| `open` `high` `low` `close` | `DOUBLE` | ✅ | OHLC. apex uses **raw `close`**, not `adj_close`. |
| `volume` | `BIGINT` | ✅ | Cast to int; null tolerated. |
| `vwap` | `DOUBLE` | ⚪ optional | Read if present, else `null` in the payload. |
| `adj_close` | `DOUBLE` | ❌ ignored | apex charts use raw OHLC, not split/dividend-adjusted. |
| `symbol_id` | `BIGINT` | ❌ ignored | Symbol comes from the partition, not a column. |
| `asset_class`, `symbol` | `VARCHAR` | ❌ ignored | Redundant with the partition. |

### Intraday — `1m.parquet` / `5m.parquet` / `30m.parquet` / `1h.parquet`

| Column | Type | apex uses it? | Notes |
|---|---|---|---|
| `bar_timestamp` | `TIMESTAMP WITH TIME ZONE` (µs, UTC) | ✅ **timestamp key** | Bar open instant. See timezone note. |
| `open` `high` `low` `close` | `DOUBLE` | ✅ | OHLC. |
| `volume` | `BIGINT` | ✅ | Cast to int; null tolerated. |
| `vwap` | `DOUBLE` | ⚪ optional | Read if present, else `null`. |
| `symbol_id` | `BIGINT` | ❌ ignored | |
| `asset_class`, `symbol` | `VARCHAR` | ❌ ignored | |

> Intraday has **no `adj_close`** (and apex wouldn't use it anyway).

### Timezone (important)

`bar_timestamp` is stored as `TIMESTAMPTZ`. DuckDB returns `TIMESTAMPTZ` values in the **session
timezone**, not UTC — e.g. on an `Asia/Hong_Kong` box DuckDB hands back `+08:00`-labelled
datetimes. The *instant* is correct; the *label* is the machine's locale. **apex normalizes
every bar timestamp to UTC** (`_to_utc_datetime` → `astimezone(timezone.utc)`) so downstream
code, payloads, and the indicator engine see one consistent offset. livewire does not need to do
anything here — just keep writing UTC `TIMESTAMPTZ`.

---

## 4. How apex queries (for reference)

Per request, in a worker thread (off the event loop), against an in-memory DuckDB connection:

```sql
-- intraday: bind the tz-aware datetimes directly
SELECT * FROM read_parquet('<file>')
WHERE bar_timestamp >= ? AND bar_timestamp <= ?
ORDER BY bar_timestamp ASC;          -- params: [start, end]

-- daily: bind CALENDAR DATES so DATE-vs-TIMESTAMPTZ comparison is tz-agnostic
SELECT * FROM read_parquet('<file>')
WHERE trade_date >= ? AND trade_date <= ?
ORDER BY trade_date ASC;             -- params: [start.date(), end.date()]
```

Each returned row becomes a `BarData` with `bar_start = bar_end - <timeframe duration>` derived,
`timestamp = bar_start` (event time, never wall-clock), and `source = "livewire"`.

---

## 5. Configuration

| Env var | Meaning |
|---|---|
| `APEX_LIVEWIRE_ROOT` | Absolute path to livewire's `data-lake/bronze`. **Required** for `/bars`, `/indicators`, and the streaming warmup seed — without it those return `503` / stay silent. |

Example (local external volume, 20,389 symbols):

```bash
export APEX_LIVEWIRE_ROOT=/Volumes/DATA_LAKE/livewire/data-lake/bronze
```

R2 backup: livewire also mirrors bronze to R2 (`R2_*` creds + `sync_to_r2.py --download` pulls
the full tree, no symbol filter). For apex, point `APEX_LIVEWIRE_ROOT` at any local copy with the
layout above.

---

## 6. Contract stability — what livewire must not break

apex reads silently miss (empty results, no error) if any of these change without a matching
apex change:

1. **Partition layout** `asset_class=equity/symbol=<encode_symbol>/<tf>.parquet`.
2. **`encode_symbol` rules** (literal `[A-Z0-9._-]`, else `%XX` UTF-8).
3. **Timestamp column names**: `trade_date` (daily), `bar_timestamp` (intraday).
4. **OHLCV column names**: `open`, `high`, `low`, `close`, `volume`.
5. **Timeframe file names**: `1m/5m/30m/1h/1d.parquet`.

Safe to change freely: add/remove any **other** column (apex ignores extras), row counts, file
internals/compression, partition values for non-equity classes.

---

## 7. Smoke test

With a populated lake and `APEX_LIVEWIRE_ROOT` set, boot apex and pull bars:

```bash
export APEX_LIVEWIRE_ROOT=/Volumes/DATA_LAKE/livewire/data-lake/bronze
uv run python -m src.api.server        # serves on :8322
curl -s 'http://127.0.0.1:8322/bars/AAPL?timeframe=1d' | jq '.count, .bars[-1]'
```

Expect `200` with real OHLC and UTC `time` values. A `503` means `APEX_LIVEWIRE_ROOT` is unset;
an empty `bars: []` means the file for that `(symbol, timeframe)` is missing or the range is
empty. See [argon-apex-api.md](argon-apex-api.md) for the full read-surface reference.
