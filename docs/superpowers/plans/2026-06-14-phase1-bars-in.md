# Phase 1 — Bars-in (`LivewireOhlcProvider`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give apex a `LivewireOhlcProvider` that reads livewire's DuckDB-over-parquet bronze directly (subscribed tickers only) and satisfies apex's existing `HistoricalSourcePort` protocol, so the TA seed path gets bars from livewire instead of apex's own IB/Parquet loaders.

**Architecture:** A new infrastructure adapter implements `HistoricalSourcePort` (`src/domain/interfaces/historical_source.py`): `source_name`, `async fetch_bars(symbol, timeframe, start, end) -> List[BarData]`, `supports_timeframe`, `get_supported_timeframes`. It issues a DuckDB query against livewire's per-ticker Hive-partitioned parquet (`asset_class=equity/symbol=<SYM>/<tf>.parquet`); rows map to `BarData` domain events. Reads are on-demand, bounded to one symbol/timeframe/date-range.

**Why `HistoricalSourcePort`, not `BarProvider` (and not `HistoricalDataManager`):** the real batch-download protocol apex uses for historical bars is `HistoricalSourcePort` (its implementations are `YahooHistoricalAdapter`, `IbHistoricalAdapter`). Phase 2's `SubscriptionManager._seed` consumes this provider directly and **injects** the bars into `TASignalService`. The legacy pull-path (`domain/signals` → `HistoricalDataManager.ensure_data/get_bars`) is **not reimplemented in Phases 1–3**: in the streaming model the cores RECEIVE injected bars + (Phase 4) live ticks rather than pulling their own history, so that pull-path is bypassed. Whether the residual `domain/signals → historical_data_manager` import is stubbed or removed is a Phase 0 manifest decision, not Phase 1 work. **Backtest is untouched** (spec §2).

**Tech Stack:** Python 3.13, DuckDB (already a dep), `BarData` domain event, pytest with a fixture parquet.

**Spec:** `docs/superpowers/specs/2026-06-14-apex-adaptation-design.md` §5 (Phase 1). **Depends on:** Phase 0 manifest. **Consumed by:** Phase 2 `SubscriptionManager._seed`.

**Verified facts (2026-06-14):** `HistoricalSourcePort.fetch_bars(self, symbol, timeframe, start, end) -> List[BarData]` (start/end required, no `limit`). `BarData` fields: `symbol, timeframe, open, high, low, close, volume, vwap, trade_count, bar_start, bar_end, timestamp, source`; `timestamp` defaults to `now_utc()` at construction, so it MUST be set to the bar's event time explicitly.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/infrastructure/adapters/livewire/__init__.py` | Package marker. |
| `src/infrastructure/adapters/livewire/ohlc_provider.py` | `LivewireOhlcProvider` — DuckDB reads → `BarData`. |
| `src/infrastructure/adapters/livewire/paths.py` | Resolve `(symbol, timeframe)` → parquet path under the configured bronze root. |
| `tests/unit/infrastructure/livewire/test_paths.py` | Path resolution + timeframe validation. |
| `tests/unit/infrastructure/livewire/test_ohlc_provider.py` | fetch_bars against a fixture parquet. |
| `tests/fixtures/livewire/asset_class=equity/symbol=TEST/1d.parquet` | Tiny known-content fixture. |
| `config/` (existing config model) | Add `livewire_bronze_root` setting. |

---

## Task 1: Verify livewire's real parquet schema (no guessing)

**Files:** none created — this is a discovery task that pins the column mapping.

- [ ] **Step 1: Inspect livewire's read helper**

Run: `sed -n '1,80p' /Users/moremeds/projects/livewire/clients/bronze_client.py`
Expected: the canonical read path + column names livewire writes.

- [ ] **Step 2: Read a real parquet's schema via DuckDB**

Run:
```bash
uv run python -c "import duckdb,glob; f=glob.glob('/Users/moremeds/projects/livewire/**/symbol=*/1d.parquet', recursive=True)[0]; print(f); print(duckdb.sql(f\"DESCRIBE SELECT * FROM read_parquet('{f}')\").df())"
```
Expected: column names + types (e.g. `ts`/`timestamp`, `open/high/low/close`, `volume`, possibly `vwap`).

- [ ] **Step 3: Record the mapping**

Write the verified column→`BarData` mapping into this plan's Task 3 (replace the `COLUMN_MAP` placeholder values with the real names). Do not proceed to Task 3 until the real names are confirmed. `BarData` target fields: `symbol, timeframe, open, high, low, close, volume, vwap, trade_count, bar_start, bar_end, source`.

---

## Task 2: Path resolver

**Files:**
- Create: `src/infrastructure/adapters/livewire/__init__.py`
- Create: `src/infrastructure/adapters/livewire/paths.py`
- Test: `tests/unit/infrastructure/livewire/test_paths.py`

- [ ] **Step 1: Create package marker**

Create `src/infrastructure/adapters/livewire/__init__.py`:

```python
"""Livewire bronze read adapter (DuckDB-over-parquet)."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/infrastructure/livewire/test_paths.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure.adapters.livewire.paths import (
    SUPPORTED_TIMEFRAMES,
    parquet_path,
)


def test_parquet_path_layout() -> None:
    root = Path("/data/bronze")
    p = parquet_path(root, "AAPL", "1d")
    assert p == root / "asset_class=equity" / "symbol=AAPL" / "1d.parquet"


def test_unsupported_timeframe_raises() -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        parquet_path(Path("/data/bronze"), "AAPL", "3m")


def test_supported_timeframes_match_livewire() -> None:
    assert SUPPORTED_TIMEFRAMES == ("1m", "5m", "30m", "1h", "1d")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/infrastructure/livewire/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the resolver**

Create `src/infrastructure/adapters/livewire/paths.py`:

```python
"""Resolve (symbol, timeframe) to a livewire bronze parquet path.

The per-ticker Hive layout IS the read contract (livewire-adaptation.md §3, §5):
  <root>/asset_class=equity/symbol=<SYM>/<tf>.parquet
"""

from __future__ import annotations

from pathlib import Path

# Timeframes livewire warehouses for equities (livewire-adaptation.md §2).
SUPPORTED_TIMEFRAMES = ("1m", "5m", "30m", "1h", "1d")


def parquet_path(bronze_root: Path, symbol: str, timeframe: str) -> Path:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"unsupported timeframe: {timeframe!r} (have {SUPPORTED_TIMEFRAMES})")
    return bronze_root / "asset_class=equity" / f"symbol={symbol}" / f"{timeframe}.parquet"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/infrastructure/livewire/test_paths.py -v`
Expected: PASS (3 passed). If livewire's real timeframe set differs (Task 1), update `SUPPORTED_TIMEFRAMES` and the test together.

- [ ] **Step 6: Commit**

```bash
git add src/infrastructure/adapters/livewire/ tests/unit/infrastructure/livewire/test_paths.py
git commit -m "feat(livewire): bronze parquet path resolver"
```

---

## Task 3: `LivewireOhlcProvider.fetch_bars`

**Files:**
- Create: `src/infrastructure/adapters/livewire/ohlc_provider.py`
- Create: `tests/fixtures/livewire/asset_class=equity/symbol=TEST/1d.parquet` (Step 1)
- Test: `tests/unit/infrastructure/livewire/test_ohlc_provider.py`

- [ ] **Step 1: Build the fixture parquet**

Run:
```bash
uv run python -c "
import pandas as pd, pathlib
d = pathlib.Path('tests/fixtures/livewire/asset_class=equity/symbol=TEST')
d.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame({
    'ts': pd.to_datetime(['2026-01-02','2026-01-03','2026-01-06'], utc=True),
    'open':[10.0,11.0,12.0],'high':[10.5,11.5,12.5],
    'low':[9.5,10.5,11.5],'close':[11.0,12.0,13.0],'volume':[100,200,300],
})
df.to_parquet(d/'1d.parquet', index=False)
print('wrote', d/'1d.parquet')
"
```
Expected: `wrote tests/fixtures/livewire/.../1d.parquet`.
**Adjust the column names** (`ts`, `open`, …) to the real livewire schema from Task 1 if they differ.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/infrastructure/livewire/test_ohlc_provider.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "fixtures" / "livewire"
WIDE_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
WIDE_END = datetime(2026, 1, 31, tzinfo=timezone.utc)


@pytest.fixture
def provider() -> LivewireOhlcProvider:
    return LivewireOhlcProvider(bronze_root=FIXTURE_ROOT)


@pytest.mark.asyncio
async def test_fetch_bars_returns_bardata_sorted(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars("TEST", "1d", WIDE_START, WIDE_END)
    assert len(bars) == 3
    assert [b.close for b in bars] == [11.0, 12.0, 13.0]
    assert bars[0].symbol == "TEST"
    assert bars[0].timeframe == "1d"
    assert bars[0].source == "livewire"
    # ascending by time
    assert bars[0].bar_start < bars[1].bar_start < bars[2].bar_start


@pytest.mark.asyncio
async def test_bar_timestamps_are_event_time_not_now(provider: LivewireOhlcProvider) -> None:
    """Regression: timestamp must be the bar's time, never construction-time now()."""
    bars = await provider.fetch_bars("TEST", "1d", WIDE_START, WIDE_END)
    first = bars[0]
    assert first.timestamp == first.bar_start          # not now()
    assert first.bar_end > first.bar_start             # 1d duration, not zero
    assert first.bar_start.year == 2026 and first.bar_start.month == 1


@pytest.mark.asyncio
async def test_fetch_bars_date_range_filter(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars(
        "TEST", "1d",
        datetime(2026, 1, 3, tzinfo=timezone.utc),
        datetime(2026, 1, 3, 23, 59, tzinfo=timezone.utc),
    )
    assert len(bars) == 1
    assert bars[0].close == 12.0


@pytest.mark.asyncio
async def test_fetch_bars_missing_symbol_returns_empty(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars("NOPE", "1d", WIDE_START, WIDE_END)
    assert bars == []


@pytest.mark.asyncio
async def test_unsupported_timeframe_raises(provider: LivewireOhlcProvider) -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        await provider.fetch_bars("TEST", "3m", WIDE_START, WIDE_END)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/infrastructure/livewire/test_ohlc_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the provider**

Create `src/infrastructure/adapters/livewire/ohlc_provider.py`:

```python
"""DuckDB-over-parquet provider that reads livewire bronze for subscribed tickers.

Implements apex's HistoricalSourcePort (src/domain/interfaces/historical_source.py).
Reads are on-demand, one (symbol, timeframe, date-range) at a time — never the
full universe.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import duckdb

from ....domain.events.domain_events import BarData
from .paths import SUPPORTED_TIMEFRAMES, parquet_path

# livewire column -> BarData field. CONFIRM the `ts`/OHLCV names against Task 1.
COLUMN_MAP = {
    "ts": "bar_start",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

# Bar duration per timeframe — used to derive bar_end (not a zero-width bar).
_TF_DELTAS = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


class LivewireOhlcProvider:
    """Reads historical bars from livewire's bronze parquet via DuckDB.

    Satisfies HistoricalSourcePort (runtime_checkable Protocol).
    """

    def __init__(self, bronze_root: Path) -> None:
        self._root = Path(bronze_root)

    # --- HistoricalSourcePort ---
    @property
    def source_name(self) -> str:
        return "livewire"

    def supports_timeframe(self, timeframe: str) -> bool:
        return timeframe in SUPPORTED_TIMEFRAMES

    def get_supported_timeframes(self) -> List[str]:
        return list(SUPPORTED_TIMEFRAMES)

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        # parquet_path raises ValueError on an unsupported timeframe.
        path = parquet_path(self._root, symbol, timeframe)
        if not path.exists():
            return []
        return await asyncio.to_thread(self._query, path, symbol, timeframe, start, end)

    # --- internals ---
    def _query(
        self,
        path: Path,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        ts_col = next(k for k, v in COLUMN_MAP.items() if v == "bar_start")
        # NOTE: read_parquet path is inlined (NOT a bound parameter) — DuckDB does
        # not accept a prepared-statement parameter for the parquet path. The path
        # is constructed by us from a validated symbol/timeframe, not user SQL.
        sql = (
            f"SELECT * FROM read_parquet('{path.as_posix()}') "
            f"WHERE {ts_col} >= ? AND {ts_col} <= ? ORDER BY {ts_col} ASC"
        )
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(sql, [start, end]).fetch_arrow_table().to_pylist()
        finally:
            con.close()
        return [self._row_to_bar(r, symbol, timeframe) for r in rows]

    @staticmethod
    def _row_to_bar(row: dict, symbol: str, timeframe: str) -> BarData:
        ts = row[next(k for k, v in COLUMN_MAP.items() if v == "bar_start")]
        end = ts + _TF_DELTAS.get(timeframe, timedelta(0))
        vol = row.get("volume")
        return BarData(
            symbol=symbol,
            timeframe=timeframe,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=int(vol) if vol is not None else None,
            vwap=row.get("vwap"),
            bar_start=ts,
            bar_end=end,
            timestamp=ts,   # event time = bar time, NOT construction-time now()
            source="livewire",
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/infrastructure/livewire/test_ohlc_provider.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/infrastructure/adapters/livewire/ohlc_provider.py tests/unit/infrastructure/livewire/test_ohlc_provider.py tests/fixtures/livewire/
git commit -m "feat(livewire): LivewireOhlcProvider.fetch_bars via DuckDB"
```

---

## Task 4: Conformance test against the `HistoricalSourcePort` protocol

**Files:**
- Test: `tests/unit/infrastructure/livewire/test_ohlc_provider.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
from src.domain.interfaces.historical_source import HistoricalSourcePort


def test_provider_satisfies_historical_source_port() -> None:
    p = LivewireOhlcProvider(bronze_root=FIXTURE_ROOT)
    assert isinstance(p, HistoricalSourcePort)  # runtime_checkable Protocol
    assert p.source_name == "livewire"
    assert p.supports_timeframe("1d") is True
    assert p.supports_timeframe("3m") is False
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/unit/infrastructure/livewire/test_ohlc_provider.py::test_provider_satisfies_historical_source_port -v`
Expected: PASS. If FAIL, the error names the missing member — add it (the protocol in `historical_source.py` lists `source_name`, `fetch_bars`, `supports_timeframe`, `get_supported_timeframes`).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/infrastructure/livewire/test_ohlc_provider.py
git commit -m "test(livewire): assert HistoricalSourcePort conformance"
```

---

## Task 5: Configuration + exposure for Phase 2

The provider is consumed by Phase 2's `SubscriptionManager._seed` (not wired into
the legacy `HistoricalDataManager` pull-path — see Architecture). This task adds
the config root and a factory so Phase 2 can construct it. **No backtest changes;
no deletion of old loaders (that is Phase 6).**

**Files:**
- Modify: `config/models.py` (add `livewire_bronze_root`)
- Modify: `config/dev.yaml` (add the path)
- Create: `src/infrastructure/adapters/livewire/factory.py` (build from config)
- Test: nearest existing config test + `tests/unit/infrastructure/livewire/test_factory.py`

- [ ] **Step 1: Add the config field (failing test first)**

Find the config model: `grep -rn "class .*Config" config/models.py | head`. Add a `livewire_bronze_root: str | None = None` field following the file's existing pattern. Write a test in the nearest existing config test that asserts the field loads from `config/dev.yaml`.

- [ ] **Step 2: Run config test**

Run: `uv run pytest tests/unit -k config -v`
Expected: PASS once the field + a `dev.yaml` entry exist.

- [ ] **Step 3: Add a factory**

Create `src/infrastructure/adapters/livewire/factory.py`:

```python
"""Construct a LivewireOhlcProvider from config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ohlc_provider import LivewireOhlcProvider


def build_livewire_provider(config: Any) -> LivewireOhlcProvider:
    root = getattr(config, "livewire_bronze_root", None)
    if not root:
        raise ValueError("livewire_bronze_root is not configured")
    return LivewireOhlcProvider(bronze_root=Path(root))
```

Add `tests/unit/infrastructure/livewire/test_factory.py`:

```python
from types import SimpleNamespace

import pytest

from src.infrastructure.adapters.livewire.factory import build_livewire_provider


def test_factory_builds_provider(tmp_path) -> None:
    p = build_livewire_provider(SimpleNamespace(livewire_bronze_root=str(tmp_path)))
    assert p.source_name == "livewire"


def test_factory_requires_root() -> None:
    with pytest.raises(ValueError, match="livewire_bronze_root"):
        build_livewire_provider(SimpleNamespace(livewire_bronze_root=None))
```

- [ ] **Step 4: Run the factory + provider tests**

Run: `uv run pytest tests/unit/infrastructure/livewire -v`
Expected: PASS (all livewire tests green). The provider is now ready for Phase 2 to inject.

- [ ] **Step 5: Commit**

```bash
git add config/ src/infrastructure/adapters/livewire/factory.py tests/unit/infrastructure/livewire/test_factory.py
git commit -m "feat(livewire): config root + provider factory for Phase 2"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** §5 Phase 1 "DuckDB-over-parquet provider" → Tasks 2–5; "reads subscribed tickers on demand" → Task 3 (per-call symbol/tf/range); "retire own loaders" → deferred to Phase 6 (Task 5 only adds the new source, deletes nothing). ✅
- **Port correction (ISSUE-8):** targets `HistoricalSourcePort` (the real batch-download protocol), not `BarProvider`/`market_data_provider`. The legacy `HistoricalDataManager` pull-path is bypassed (cores receive injected bars), not reimplemented — documented in Architecture. ✅
- **Timestamp fix (ISSUE-12):** `timestamp` set to bar time; `bar_end` derived from timeframe via `_TF_DELTAS`. Regression test added. ✅
- **DuckDB path (Phase-1 watch-item):** `read_parquet('<path>')` inlined, not parameter-bound (DuckDB limitation); path is internally constructed, not user SQL. ✅
- **Backtest (ISSUE-15):** untouched — Task 5 wires only the seed/config path; no `src/backtest` edits. ✅
- **Honest gaps flagged:** livewire column names (Task 1 gate), residual `domain/signals → historical_data_manager` disposition (Phase 0 manifest). No fabricated paths. ✅
- **Type consistency:** `LivewireOhlcProvider`, `parquet_path`, `SUPPORTED_TIMEFRAMES`, `COLUMN_MAP`, `_TF_DELTAS`, `build_livewire_provider`, `BarData` field names consistent. ✅
- **D3 (HK/Asia / Futu loader):** out of this plan; equities only. Futu loader untouched. ✅
