# Phase 1 — Bars-in (`LivewireOhlcProvider`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give apex a `LivewireOhlcProvider` that reads livewire's DuckDB-over-parquet bronze directly (subscribed tickers only) and satisfies apex's existing `BarProvider` protocol, so the TA cores get bars from livewire instead of apex's own IB/Parquet loaders.

**Architecture:** A new infrastructure adapter implements `BarProvider.fetch_bars(...)` by issuing a DuckDB query against livewire's per-ticker Hive-partitioned parquet (`asset_class=equity/symbol=<SYM>/<tf>.parquet`). Rows map to apex's `BarData` domain event. Reads are on-demand and bounded to one symbol/timeframe/date-range. The provider is wired behind the `BarProvider` seam that Phase 0 identified (`domain/signals` → `historical_data_manager`, `src/backtest` → `ib.historical_adapter`).

**Tech Stack:** Python 3.13, DuckDB (already a dep), `BarData` domain event, pytest with a fixture parquet.

**Spec:** `docs/superpowers/specs/2026-06-14-apex-adaptation-design.md` §5 (Phase 1). **Depends on:** Phase 0 manifest (the exact cut-point seam).

**Prerequisite from Phase 0:** the coupling-cut list confirms `BarProvider` is the seam to fill. If Phase 0 instead recommends `HistoricalSourcePort` (`fetch_bars(symbol, tf, start, end)`), implement that protocol's signature — the body is identical; only the method set differs.

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


@pytest.fixture
def provider() -> LivewireOhlcProvider:
    return LivewireOhlcProvider(bronze_root=FIXTURE_ROOT)


@pytest.mark.asyncio
async def test_fetch_bars_returns_bardata_sorted(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars("TEST", "1d")
    assert len(bars) == 3
    assert [b.close for b in bars] == [11.0, 12.0, 13.0]
    assert bars[0].symbol == "TEST"
    assert bars[0].timeframe == "1d"
    assert bars[0].source == "livewire"
    # ascending by time
    assert bars[0].bar_start < bars[1].bar_start < bars[2].bar_start


@pytest.mark.asyncio
async def test_fetch_bars_date_range_filter(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars(
        "TEST", "1d",
        start=datetime(2026, 1, 3, tzinfo=timezone.utc),
        end=datetime(2026, 1, 3, 23, 59, tzinfo=timezone.utc),
    )
    assert len(bars) == 1
    assert bars[0].close == 12.0


@pytest.mark.asyncio
async def test_fetch_bars_missing_symbol_returns_empty(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars("NOPE", "1d")
    assert bars == []


@pytest.mark.asyncio
async def test_fetch_bars_limit(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars("TEST", "1d", limit=2)
    assert len(bars) == 2
    assert [b.close for b in bars] == [12.0, 13.0]  # most-recent N, ascending
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/infrastructure/livewire/test_ohlc_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the provider**

Create `src/infrastructure/adapters/livewire/ohlc_provider.py`:

```python
"""DuckDB-over-parquet provider that reads livewire bronze for subscribed tickers.

Implements apex's BarProvider protocol (src/domain/interfaces/bar_provider.py).
Reads are on-demand, one (symbol, timeframe, date-range) at a time — never the
full universe.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import duckdb

from ....domain.events.domain_events import BarData
from .paths import SUPPORTED_TIMEFRAMES, parquet_path

# livewire column -> BarData field. CONFIRM names against Task 1 before running.
COLUMN_MAP = {
    "ts": "bar_start",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


class LivewireOhlcProvider:
    """Reads historical bars from livewire's bronze parquet via DuckDB."""

    def __init__(self, bronze_root: Path) -> None:
        self._root = Path(bronze_root)
        self._connected = True  # parquet reads need no live connection

    # --- BarProvider protocol ---
    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_supported_timeframes(self) -> List[str]:
        return list(SUPPORTED_TIMEFRAMES)

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[BarData]:
        path = parquet_path(self._root, symbol, timeframe)
        if not path.exists():
            return []
        return await asyncio.to_thread(self._query, path, symbol, timeframe, start, end, limit)

    async def fetch_latest_bar(self, symbol: str, timeframe: str) -> Optional[BarData]:
        bars = await self.fetch_bars(symbol, timeframe, limit=1)
        return bars[-1] if bars else None

    def set_bar_callback(self, callback: Optional[Callable[[BarData], None]]) -> None:
        # Phase 1 is historical-only; live callbacks arrive in Phase 4 (xenon).
        self._callback = callback

    async def subscribe_bars(self, symbol: str, timeframe: str) -> None:  # pragma: no cover
        raise NotImplementedError("live subscriptions arrive in Phase 4 (xenon)")

    async def unsubscribe_bars(self, symbol: str, timeframe: str) -> None:  # pragma: no cover
        raise NotImplementedError("live subscriptions arrive in Phase 4 (xenon)")

    async def fetch_bars_batch(self, requests: List[dict]) -> dict:
        out: dict = {}
        for req in requests:
            out[req["symbol"]] = await self.fetch_bars(
                req["symbol"], req["timeframe"],
                req.get("start"), req.get("end"), req.get("limit"),
            )
        return out

    # --- internals ---
    def _query(
        self,
        path: Path,
        symbol: str,
        timeframe: str,
        start: Optional[datetime],
        end: Optional[datetime],
        limit: Optional[int],
    ) -> List[BarData]:
        ts_col = next(k for k, v in COLUMN_MAP.items() if v == "bar_start")
        where = []
        params: list = []
        if start is not None:
            where.append(f"{ts_col} >= ?")
            params.append(start)
        if end is not None:
            where.append(f"{ts_col} <= ?")
            params.append(end)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        # limit = most-recent N, then re-sort ascending for the caller.
        order = "DESC" if limit else "ASC"
        sql = f"SELECT * FROM read_parquet(?){clause} ORDER BY {ts_col} {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(sql, [str(path), *params]).fetch_arrow_table().to_pylist()
        finally:
            con.close()
        if limit:
            rows = list(reversed(rows))
        return [self._row_to_bar(r, symbol, timeframe) for r in rows]

    @staticmethod
    def _row_to_bar(row: dict, symbol: str, timeframe: str) -> BarData:
        kwargs = {dst: row[src] for src, dst in COLUMN_MAP.items() if src in row}
        ts = kwargs.pop("bar_start", None)
        return BarData(
            symbol=symbol,
            timeframe=timeframe,
            open=kwargs.get("open"),
            high=kwargs.get("high"),
            low=kwargs.get("low"),
            close=kwargs.get("close"),
            volume=int(kwargs["volume"]) if kwargs.get("volume") is not None else None,
            vwap=row.get("vwap"),
            bar_start=ts,
            bar_end=ts,
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

## Task 4: Conformance test against the `BarProvider` protocol

**Files:**
- Test: `tests/unit/infrastructure/livewire/test_ohlc_provider.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
from src.domain.interfaces.bar_provider import BarProvider


def test_provider_satisfies_barprovider_protocol() -> None:
    p = LivewireOhlcProvider(bronze_root=FIXTURE_ROOT)
    assert isinstance(p, BarProvider)  # runtime_checkable Protocol
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/unit/infrastructure/livewire/test_ohlc_provider.py::test_provider_satisfies_barprovider_protocol -v`
Expected: PASS. If FAIL, the error names the missing method — add it to the provider (the protocol in `bar_provider.py` lists the full set).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/infrastructure/livewire/test_ohlc_provider.py
git commit -m "test(livewire): assert BarProvider protocol conformance"
```

---

## Task 5: Configuration + wiring

**Files:**
- Modify: `config/models.py` (add `livewire_bronze_root`)
- Modify: the composition root that constructs the bar provider (identified in Phase 0 manifest §4 — the `historical_data_manager` / `ib.historical_adapter` construction site)

- [ ] **Step 1: Add the config field (failing test first)**

Find the config model: `grep -rn "class .*Config" config/models.py | head`. Add a `livewire_bronze_root: str | None = None` field following the file's existing pattern. Write a test in the nearest existing config test that asserts the field loads from `config/dev.yaml`.

- [ ] **Step 2: Run config test**

Run: `uv run pytest tests/unit -k config -v`
Expected: PASS once the field + a `dev.yaml` entry exist.

- [ ] **Step 3: Wire the provider at the seam**

At the construction site named in Phase 0 manifest §4, instantiate `LivewireOhlcProvider(bronze_root=Path(config.livewire_bronze_root))` and inject it where the IB/Parquet bar source was passed. Do **not** delete the old loaders yet (that is Phase 6) — just stop constructing them on the equities path. Keep behaviour behind the existing `BarProvider`-typed parameter so callers are unchanged.

- [ ] **Step 4: Run the signals/pipeline integration tests**

Run: `uv run pytest tests/unit/domain/signals -v`
Expected: PASS — the cores now receive bars through the same protocol, sourced from livewire.

- [ ] **Step 5: Commit**

```bash
git add config/ src/
git commit -m "feat(livewire): wire LivewireOhlcProvider behind the bar-provider seam"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** §5 Phase 1 "DuckDB-over-parquet provider behind market_data_provider" → Tasks 2–5; "reads subscribed tickers on demand" → Task 3 (per-call symbol/tf/range); "retire own loaders" → Task 5 (stop constructing on equities path; deletion deferred to Phase 6). ✅
- **Honest gaps flagged:** livewire column names (Task 1 gate), exact seam (Phase 0 manifest dependency), `BarProvider` vs `HistoricalSourcePort` (prerequisite note). No fabricated paths. ✅
- **Type consistency:** `LivewireOhlcProvider`, `parquet_path`, `SUPPORTED_TIMEFRAMES`, `COLUMN_MAP`, `BarData` field names used consistently. ✅
- **D3 (HK/Asia / Futu loader):** out of this plan; equities only. Futu loader untouched. ✅
