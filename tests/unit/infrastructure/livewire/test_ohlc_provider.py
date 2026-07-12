from __future__ import annotations

import datetime as dt
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.domain.interfaces.historical_source import HistoricalSourcePort
from src.infrastructure.adapters.livewire.ohlc_provider import (
    AdjustedDataUnavailable,
    LivewireOhlcProvider,
    _to_utc_datetime,
)

WIDE_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
WIDE_END = datetime(2026, 1, 31, tzinfo=timezone.utc)


def _write_fixture(root: Path) -> None:
    """Write a 3-bar daily livewire bronze parquet for symbol TEST.

    Generated at runtime (*.parquet is gitignored). Mirrors livewire's REAL daily
    bronze schema (clients/bronze_client.py): `trade_date` (date32), `symbol_id`,
    OHLC, `adj_close`, `volume` -- the symbol lives in the partition dir, not a column.
    """
    sym_dir = root / "asset_class=equity" / "symbol=TEST"
    sym_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "trade_date": [dt.date(2026, 1, 2), dt.date(2026, 1, 3), dt.date(2026, 1, 6)],
            "symbol_id": [1, 1, 1],
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [11.0, 12.0, 13.0],
            "adj_close": [11.0, 12.0, 13.0],
            "volume": [100, 200, 300],
        }
    )
    df.to_parquet(sym_dir / "1d.parquet")


def _write_intraday_fixture(root: Path) -> None:
    """Write a 3-bar 1m parquet mirroring livewire's intraday schema:
    `bar_timestamp` (tz-aware UTC) + `symbol_id` + OHLCV."""
    sym_dir = root / "asset_class=equity" / "symbol=TEST"
    sym_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "bar_timestamp": pd.to_datetime(
                ["2026-01-02T14:30:00Z", "2026-01-02T14:31:00Z", "2026-01-02T14:32:00Z"],
                utc=True,
            ),
            "symbol_id": [1, 1, 1],
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [11.0, 12.0, 13.0],
            "volume": [100, 200, 300],
        }
    )
    df.to_parquet(sym_dir / "1m.parquet")


def _write_silver_daily_fixture(root: Path) -> None:
    sym_dir = root / "asset_class=equity" / "symbol=TEST"
    sym_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_date": [dt.date(2026, 1, 2), dt.date(2026, 1, 3), dt.date(2026, 1, 6)],
            "symbol_id": [1, 1, 1],
            "open": [5.0, 5.5, 12.0],
            "high": [5.25, 5.75, 12.5],
            "low": [4.75, 5.25, 11.5],
            "close": [5.5, 6.0, 13.0],
            "adj_close": [5.5, 6.0, 13.0],
            "volume": [200, 400, 300],
            "price_adjustment_factor": [0.5, 0.5, 1.0],
            "split_volume_factor": [2.0, 2.0, 1.0],
            "adjustment_revision": [1, 1, 1],
        }
    ).to_parquet(sym_dir / "1d.parquet")


def _write_factor_fixture(root: Path, *, cover_bar_date: bool = True) -> None:
    sym_dir = root / "adjustments" / "asset_class=equity" / "symbol=TEST"
    sym_dir.mkdir(parents=True, exist_ok=True)
    effective_start = dt.date(2026, 1, 2) if cover_bar_date else dt.date(2026, 1, 3)
    pd.DataFrame(
        {
            "effective_start": [effective_start],
            "effective_end": [dt.date(2026, 1, 2) if cover_bar_date else None],
            "price_adjustment_factor": [0.5],
            "split_volume_factor": [2.0],
            "adjustment_revision": [1],
        }
    ).to_parquet(sym_dir / "factors.parquet")


def _write_action_span_fixtures(bronze_root: Path, silver_root: Path) -> None:
    bronze_dir = bronze_root / "asset_class=equity" / "symbol=TEST"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "bar_timestamp": pd.to_datetime(
                ["2026-01-02T14:30:00Z", "2026-01-05T14:30:00Z", "2026-01-06T14:30:00Z"],
                utc=True,
            ),
            "symbol_id": [1, 1, 1],
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 11.0, 11.0],
            "low": [9.0, 9.0, 9.0],
            "close": [10.0, 10.0, 10.0],
            "volume": [100, 100, 100],
        }
    ).to_parquet(bronze_dir / "1m.parquet")

    factor_dir = silver_root / "adjustments" / "asset_class=equity" / "symbol=TEST"
    factor_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "effective_start": [None, dt.date(2026, 1, 3), dt.date(2026, 1, 6)],
            "effective_end": [dt.date(2026, 1, 2), dt.date(2026, 1, 5), None],
            "price_adjustment_factor": [0.45, 0.9, 1.0],
            "split_volume_factor": [2.0, 1.0, 1.0],
            "adjustment_revision": [1, 1, 1],
        }
    ).to_parquet(factor_dir / "factors.parquet")


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
async def test_fetch_bars_intraday_reads_bar_timestamp(tmp_path: Path) -> None:
    """Intraday bars are keyed by `bar_timestamp` (tz-aware UTC), not `trade_date`."""
    _write_intraday_fixture(tmp_path)
    provider = LivewireOhlcProvider(bronze_root=tmp_path)
    bars = await provider.fetch_bars(
        "TEST",
        "1m",
        datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
    )
    assert [b.close for b in bars] == [11.0, 12.0, 13.0]
    assert bars[0].bar_start == datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)
    assert bars[0].bar_start.tzinfo is not None  # tz-aware UTC, not naive
    assert bars[0].bar_end > bars[0].bar_start  # 1m duration


@pytest.mark.asyncio
async def test_adjusted_daily_reads_materialized_silver(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    _write_fixture(bronze_root)
    _write_silver_daily_fixture(silver_root)
    provider = LivewireOhlcProvider(
        bronze_root=bronze_root,
        silver_root=silver_root,
        price_mode="adjusted",
    )

    bars = await provider.fetch_bars("TEST", "1d", WIDE_START, WIDE_END)

    assert [bar.close for bar in bars] == [5.5, 6.0, 13.0]
    assert [bar.volume for bar in bars] == [200, 400, 300]


@pytest.mark.asyncio
async def test_adjusted_daily_does_not_require_bronze_artifact(tmp_path: Path) -> None:
    silver_root = tmp_path / "silver"
    _write_silver_daily_fixture(silver_root)
    provider = LivewireOhlcProvider(
        bronze_root=tmp_path / "bronze",
        silver_root=silver_root,
        price_mode="adjusted",
    )

    bars = await provider.fetch_bars("TEST", "1d", WIDE_START, WIDE_END)

    assert [bar.close for bar in bars] == [5.5, 6.0, 13.0]


@pytest.mark.asyncio
async def test_adjusted_intraday_applies_price_and_split_volume_factors(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    _write_intraday_fixture(bronze_root)
    _write_factor_fixture(silver_root)
    provider = LivewireOhlcProvider(
        bronze_root=bronze_root,
        silver_root=silver_root,
        price_mode="adjusted",
    )

    bars = await provider.fetch_bars(
        "TEST",
        "1m",
        datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
    )

    assert [bar.open for bar in bars] == [5.0, 5.5, 6.0]
    assert [bar.close for bar in bars] == [5.5, 6.0, 6.5]
    assert [bar.volume for bar in bars] == [200, 400, 600]


@pytest.mark.asyncio
async def test_adjusted_intraday_spans_split_dividend_and_identity_intervals(
    tmp_path: Path,
) -> None:
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    _write_action_span_fixtures(bronze_root, silver_root)
    provider = LivewireOhlcProvider(
        bronze_root=bronze_root,
        silver_root=silver_root,
        price_mode="adjusted",
    )

    bars = await provider.fetch_bars("TEST", "1m", WIDE_START, WIDE_END)

    assert [bar.open for bar in bars] == [4.5, 9.0, 10.0]
    assert [bar.volume for bar in bars] == [200, 100, 100]


@pytest.mark.asyncio
async def test_adjusted_intraday_rejects_missing_factor_artifact(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    _write_intraday_fixture(bronze_root)
    provider = LivewireOhlcProvider(
        bronze_root=bronze_root,
        silver_root=tmp_path / "silver",
        price_mode="adjusted",
    )

    with pytest.raises(AdjustedDataUnavailable, match="factor artifact"):
        await provider.fetch_bars("TEST", "1m", WIDE_START, WIDE_END)


@pytest.mark.asyncio
async def test_adjusted_intraday_rejects_incomplete_factor_coverage(tmp_path: Path) -> None:
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    _write_intraday_fixture(bronze_root)
    _write_factor_fixture(silver_root, cover_bar_date=False)
    provider = LivewireOhlcProvider(
        bronze_root=bronze_root,
        silver_root=silver_root,
        price_mode="adjusted",
    )

    with pytest.raises(AdjustedDataUnavailable, match="factor coverage"):
        await provider.fetch_bars(
            "TEST",
            "1m",
            datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc),
        )


@pytest.mark.asyncio
async def test_fetch_bars_missing_symbol_returns_empty(provider: LivewireOhlcProvider) -> None:
    bars = await provider.fetch_bars("NOPE", "1d", WIDE_START, WIDE_END)
    assert bars == []


@pytest.mark.asyncio
async def test_unsupported_timeframe_raises(provider: LivewireOhlcProvider) -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        await provider.fetch_bars("TEST", "3m", WIDE_START, WIDE_END)


def test_to_utc_datetime_normalizes_session_tz_to_utc() -> None:
    """DuckDB returns TIMESTAMPTZ in the session timezone (e.g. +08:00), not UTC.

    The provider must convert those to UTC so warmup-seeded bars (session tz) and
    live tick bars (UTC) share one offset. A mixed-offset column crashes
    `pd.to_datetime(...)` in the indicator engine (`Tz-aware datetime cannot be
    converted to datetime64 unless utc=True`), killing live indicator compute.
    """
    hk = timezone(timedelta(hours=8))
    aware_hk = datetime(2026, 6, 12, 22, 30, tzinfo=hk)  # == 14:30 UTC, same instant

    out = _to_utc_datetime(aware_hk)

    assert out.utcoffset() == timedelta(0)  # normalized to UTC, not left at +08:00
    assert out.tzinfo == timezone.utc
    assert out == datetime(2026, 6, 12, 14, 30, tzinfo=timezone.utc)  # instant preserved


def test_to_utc_datetime_keeps_naive_and_date_as_utc() -> None:
    """Naive datetimes are tagged UTC; date32 (daily `trade_date`) becomes midnight UTC."""
    naive = _to_utc_datetime(datetime(2026, 1, 2, 9, 30))
    assert naive == datetime(2026, 1, 2, 9, 30, tzinfo=timezone.utc)

    daily = _to_utc_datetime(dt.date(2026, 1, 2))
    assert daily == datetime(2026, 1, 2, tzinfo=timezone.utc)


def test_provider_satisfies_historical_source_port(bronze_root: Path) -> None:
    p = LivewireOhlcProvider(bronze_root=bronze_root)
    assert isinstance(p, HistoricalSourcePort)  # runtime_checkable Protocol
    assert p.source_name == "livewire"
    assert p.supports_timeframe("1d") is True
    assert p.supports_timeframe("3m") is False
