from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.domain.interfaces.historical_source import HistoricalSourcePort
from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

WIDE_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
WIDE_END = datetime(2026, 1, 31, tzinfo=timezone.utc)


def _write_fixture(root: Path) -> None:
    """Write a 3-bar livewire bronze parquet for symbol TEST, timeframe 1d.

    Generated at runtime (not committed) because *.parquet is gitignored -- a
    static fixture would be absent in CI. Mirrors the real bronze schema: a
    tz-aware `ts` column plus OHLCV and the Hive partition columns.
    """
    sym_dir = root / "asset_class=equity" / "symbol=TEST"
    sym_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", "2026-01-06T00:00:00Z"],
                utc=True,
            ),
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [11.0, 12.0, 13.0],
            "volume": [100, 200, 300],
            "asset_class": ["equity", "equity", "equity"],
            "symbol": ["TEST", "TEST", "TEST"],
        }
    )
    df.to_parquet(sym_dir / "1d.parquet")


@pytest.fixture
def bronze_root(tmp_path: Path) -> Path:
    _write_fixture(tmp_path)
    return tmp_path


@pytest.fixture
def provider(bronze_root: Path) -> LivewireOhlcProvider:
    return LivewireOhlcProvider(bronze_root=bronze_root)


@pytest.mark.asyncio
async def test_fetch_bars_returns_bardata_sorted(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars("TEST", "1d", WIDE_START, WIDE_END)
    assert len(bars) == 3
    assert [b.close for b in bars] == [11.0, 12.0, 13.0]
    assert bars[0].symbol == "TEST"
    assert bars[0].timeframe == "1d"
    assert bars[0].source == "livewire"
    assert bars[0].bar_start < bars[1].bar_start < bars[2].bar_start


@pytest.mark.asyncio
async def test_bar_timestamps_are_event_time_not_now(provider: LivewireOhlcProvider) -> None:
    """Regression: timestamp must be the bar's time, never construction-time now()."""
    bars = await provider.fetch_bars("TEST", "1d", WIDE_START, WIDE_END)
    first = bars[0]
    assert first.timestamp == first.bar_start  # not now()
    assert first.bar_end > first.bar_start  # 1d duration, not zero
    assert first.bar_start.year == 2026 and first.bar_start.month == 1


@pytest.mark.asyncio
async def test_fetch_bars_date_range_filter(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars(
        "TEST",
        "1d",
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


def test_provider_satisfies_historical_source_port(bronze_root: Path) -> None:
    p = LivewireOhlcProvider(bronze_root=bronze_root)
    assert isinstance(p, HistoricalSourcePort)  # runtime_checkable Protocol
    assert p.source_name == "livewire"
    assert p.supports_timeframe("1d") is True
    assert p.supports_timeframe("3m") is False
