# Apex Silver-Adjusted Operations Manual

**Last updated:** 2026-07-18 (Asia/Hong_Kong)
**Audience:** an Apex dev/ops session operating the Silver-adjusted price path.
**Supersedes** the 2026-07-16 "healthy Silver-aware shadow" handover — the shadow
state is over; production now serves **adjusted** prices from Silver.

---

## 0. Current status (verified 2026-07-18)

- Apex runs `4df7b62` (PR #150 revision watcher + PR #151 adjusted reads).
- The deployed API (`http://127.0.0.1:8322`) reports **`effective_price_mode:
  "adjusted"`** and **`observed_revision: 5`, `last_fully_applied_revision: 5`,
  `consecutive_failures: 0`, `last_error: null`**. Bar reads come from Silver, not
  raw Bronze.
- Livewire **Silver revision 5** manifests **12,904** of the **13,141** current
  daily-Bronze equity symbols. The remaining **237 are quarantined** (fail-closed,
  never served in adjusted mode) — see §6.
- `current.json` lists **25,808 artifacts** (12,904 symbols × `{1d.parquet,
  factors.parquet}`); every one is sha256-verified on every watcher poll.

This is a real advance over rev-1 (9,207 symbols, raw shadow). Do not re-cite the
rev-1 / 9,207 / raw-mode numbers from the old handover; they are historical.

---

## 1. The two independent gates

"Silver-aware" is **not** "all reads are adjusted." Two env gates, set independently:

| Env var | Effect when set |
|---|---|
| `APEX_LIVEWIRE_SILVER_ROOT` | Enables the revision watcher: poll `current.json`, sha-verify **every** artifact, reseed active subscriptions on a new revision. Does **not** by itself change what bars are served. |
| `APEX_LIVEWIRE_PRICE_MODE=adjusted` | Routes bar reads through the Silver daily artifact + factor intervals. `raw` (or unset) serves Bronze. |

Production currently has **both** enabled (`adjusted`). Watcher health (gate 1)
succeeding tells you nothing about whether reads are adjusted — always confirm
`effective_price_mode` (gate 2) from `/health`.

---

## 2. What each price mode serves

- **`raw`** — Bronze OHLCV verbatim, unadjusted for splits/dividends. A symbol with
  a split in-window shows a discontinuous series and a nonsensical total return.
- **`adjusted`** — back-adjusted daily bars from `<silver>/asset_class=equity/symbol=<enc>/1d.parquet`
  plus factor intervals from `<silver>/adjustments/.../factors.parquet`. Splits
  adjust price and volume; dividends adjust price only. Intraday adjusted reads
  LEFT JOIN Bronze intraday bars onto Silver factor intervals and **hard-fail if
  any bar lacks a covering interval** — which is why Livewire keeps factor
  intervals wider than the trimmed daily window.

---

## 3. Deployment

Canonical Compose project (this host uses the standalone `docker-compose`, **not**
the `docker compose` v2 plugin):

```text
working dir:  /Users/moremeds/apex-deploy
compose file: /Users/moremeds/apex-deploy/compose.yml
container:    apex-deploy-api-1
image:        apex-api:silver-cutover-4df7b62
port:         8322
```

Non-secret config (both lake mounts are **read-only** to Apex):

```yaml
environment:
  APEX_LIVEWIRE_ROOT: /data/livewire
  APEX_LIVEWIRE_SILVER_ROOT: /data/livewire-silver
  APEX_LIVEWIRE_PRICE_MODE: adjusted
  APEX_LIVEWIRE_REVISION_POLL_SECONDS: "30"
volumes:
  - /Volumes/DATA_LAKE/livewire/data-lake/bronze:/data/livewire:ro
  - /Volumes/DATA_LAKE/livewire/data-lake/silver:/data/livewire-silver:ro
```

`APEX_PG_URL` lives in `/Users/moremeds/apex-deploy/.env`. **Never** print, copy,
or commit it, and never dump the full container environment.

---

## 4. How to verify

### 4a. Health (fast, always safe)

```bash
curl -fsS http://127.0.0.1:8322/health | jq '{status,pg_connected,livewire,silver_revision}'
```

Confirm: `livewire.effective_price_mode == "adjusted"`,
`silver_revision.observed_revision == silver_revision.last_fully_applied_revision`,
`consecutive_failures == 0`, `last_error == null`.

### 4b. Raw-vs-adjusted canary (read-only, no mutation)

`scripts/check_silver_canary.py` reads each symbol through both `LivewireOhlcProvider`
modes and compares. It touches no artifacts.

```bash
python scripts/check_silver_canary.py \
  --bronze-root /Volumes/DATA_LAKE/livewire/data-lake/bronze \
  --silver-root /Volumes/DATA_LAKE/livewire/data-lake/silver \
  --tickers NVDA AAPL A --control PLTR
```

Read the result: overall `passed`, plus per-symbol `identity_control`
(adjusted==raw — expected only for a symbol with **no** corporate action in-window),
`split_volume_adjusted`, and `continuity_improved`.

**rev-5 canary evidence (2026-07-18), overall `passed: true`:**

| Symbol | identity_control | split_vol_adj | raw_return → adjusted_return | Meaning |
|---|---|---|---|---|
| NVDA | false | true | −0.136 → **33.763×** | splits correctly applied |
| AAPL | false | true | 0.048 → **3.353×** | splits correctly applied |
| A (Agilent) | false | false | 0.573 → 0.648 | dividend-adjusted; serves correctly |
| PLTR (control) | **true** | false | 13.076 → 13.076 | no action → adjustment is a no-op |

Do **not** run multiple full-lake canaries concurrently — the data lake is on a
slow external exFAT volume.

---

## 5. Fail-closed contract

In `adjusted` mode, a symbol with **no Silver daily artifact** raises
`AdjustedDataUnavailable` → **HTTP 500**. There is **no silent fallback to raw**.
This is intentional: a quarantined/corrupt symbol must fail loudly rather than
serve wrong prices. Confirmed 2026-07-18: `GET /bars/AIG` (quarantined, §6) → 500;
`GET /bars/A` and `/bars/NVDA` (manifested) → 200.

Membership is resolved by **filesystem path**, never by reading the manifest for
membership. Consequences (do not violate):

- Dropping a symbol from the manifest does **nothing** — its stale parquet keeps
  serving. Physically deleting/evicting the file is the only removal Apex perceives.
- Never narrow factor intervals to the trimmed daily window (see §2, intraday).

---

## 6. The 237-symbol quarantine tail (Livewire-side backlog)

These are excluded from Silver rev-5 and fail-closed in adjusted mode. They are a
**Livewire data-quality backlog**, not an Apex bug. Breakdown:

- **172** — `unknown price_basis` on a split-affected row (legacy Bronze basis not
  yet resolved to raw; a split touches the window). Includes large caps: CMCSA,
  HON, MMM, MSI, MSTR, LEN, ECL, DD, GOLD, VLO, WY, AIG, TRI.
- **61** — dividend currency ≠ Bronze currency (mostly Canadian/European ADRs: RY,
  TD, BMO, ENB, SU, TRP, AZN, BCE, TTE, STLA, RACE).
- **4** — dividend ≥ prior close (special/return-of-capital): ARKD, DBRG, ELA, MCHB.

Fixing them is Livewire work (Yahoo-basis resolution for the 172; dividend
currency/magnitude normalization for the 65), after which they publish in a future
revision and Apex serves them automatically on the next poll.

---

## 7. Rollback (configuration-only, touches no lake)

First-line rollback is a serving-mode reversal — do **not** delete Silver or
restore Bronze:

```bash
cd /Users/moremeds/apex-deploy
# set APEX_LIVEWIRE_PRICE_MODE: raw in compose.yml, then:
docker-compose config -q
docker-compose up -d --no-build api
```

Verify `/health` shows both `configured` and `effective` price mode `raw` and raw
AAPL bars load. Both mounts are read-only and Silver is replayable, so the mode
flip is sufficient. The revision watcher rejects a **regression** (a lower revision
than observed) but accepts gaps; restart zeroes in-memory revision state.

---

## 8. Common traps

- Use `docker-compose` (v1); `docker compose` is not installed here.
- Watcher success ≠ adjusted reads active — check `effective_price_mode`.
- rev-5 covers 12,904 of 13,141, **not** all Bronze symbols — 237 fail-closed.
- Never silently fall back to raw when adjusted data is absent.
- Keep both lake mounts read-only; preserve provider-significant symbol case in
  path resolution.
- A local checkout can lag `origin/master` and lack the adjusted code entirely
  (this one did, 7 commits behind, before 2026-07-18). Verify against the deployed
  container or a checkout at `4df7b62`+.

---

## 9. Code map (`origin/master` @ `4df7b62`)

| Path | Responsibility |
|---|---|
| `src/infrastructure/adapters/livewire/ohlc_provider.py` | raw vs adjusted reads; `AdjustedDataUnavailable` |
| `src/infrastructure/adapters/livewire/paths.py` | Bronze / Silver daily / factor path resolution |
| `src/infrastructure/adapters/livewire/revisions.py` | manifest parse + per-artifact sha256 verification |
| `src/application/subscriptions/revision_watcher.py` | polling + revision application state |
| `src/api/server.py` | env wiring of both gates |
| `src/api/routes/health.py` | `livewire` + `silver_revision` health |
| `scripts/check_silver_canary.py` | read-only raw-vs-adjusted canary |
| `docs/livewire-apex-integration.md` | the adjusted-read contract |
