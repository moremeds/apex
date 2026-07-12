from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from scripts.check_silver_canary import _same_values, check_silver_canary
from src.api.server import create_app, lifespan
from src.application.subscriptions.manager import SubscriptionManager
from src.domain.events.domain_events import BarData
from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider


class _RecordingCompute:
    def __init__(self) -> None:
        self.started = False
        self.injected: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.replaced: dict[str, dict[str, list[dict[str, Any]]]] = {}

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def inject_historical_bars(
        self, symbol: str, timeframe: str, bars: list[dict[str, Any]]
    ) -> int:
        self.injected[(symbol, timeframe)] = bars
        return len(bars)

    def begin_symbol_refresh(self, symbol: str) -> None:
        pass

    async def replace_symbol_histories(
        self, symbol: str, histories: dict[str, list[dict[str, Any]]]
    ) -> dict[str, int]:
        self.replaced[symbol] = histories
        return {timeframe: len(rows) for timeframe, rows in histories.items()}

    def commit_symbol_refresh(self, symbol: str) -> None:
        pass

    def abort_symbol_refresh(self, symbol: str) -> None:
        pass


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def _publish_symbol(
    bronze_root: Path,
    silver_root: Path,
    symbol: str,
    first_factor: float,
    split_volume_factor: float,
    first_raw_close: float,
    second_raw_close: float,
    first_date: date,
    second_date: date,
    revision: int,
) -> tuple[Path, Path]:
    bronze_dir = bronze_root / "asset_class=equity" / f"symbol={symbol}"
    silver_dir = silver_root / "asset_class=equity" / f"symbol={symbol}"
    factor_dir = silver_root / "adjustments" / "asset_class=equity" / f"symbol={symbol}"
    timestamps = [
        datetime.combine(first_date, time(14, 30), tzinfo=timezone.utc),
        datetime.combine(second_date, time(14, 30), tzinfo=timezone.utc),
    ]
    raw_volume = [100, int(100 * split_volume_factor)]
    raw_daily = pd.DataFrame(
        {
            "trade_date": [first_date, second_date],
            "symbol_id": [1, 1],
            "open": [first_raw_close, second_raw_close],
            "high": [first_raw_close, second_raw_close],
            "low": [first_raw_close, second_raw_close],
            "close": [first_raw_close, second_raw_close],
            "adj_close": [first_raw_close, second_raw_close],
            "volume": raw_volume,
        }
    )
    _atomic_parquet(bronze_dir / "1d.parquet", raw_daily)
    _atomic_parquet(
        bronze_dir / "1m.parquet",
        pd.DataFrame(
            {
                "bar_timestamp": pd.to_datetime(timestamps, utc=True),
                "symbol_id": [1, 1],
                "open": [first_raw_close, second_raw_close],
                "high": [first_raw_close, second_raw_close],
                "low": [first_raw_close, second_raw_close],
                "close": [first_raw_close, second_raw_close],
                "volume": raw_volume,
            }
        ),
    )
    adjusted_first = first_raw_close * first_factor
    daily_path = silver_dir / "1d.parquet"
    _atomic_parquet(
        daily_path,
        pd.DataFrame(
            {
                "trade_date": [first_date, second_date],
                "symbol_id": [1, 1],
                "open": [adjusted_first, second_raw_close],
                "high": [adjusted_first, second_raw_close],
                "low": [adjusted_first, second_raw_close],
                "close": [adjusted_first, second_raw_close],
                "adj_close": [adjusted_first, second_raw_close],
                "volume": [int(100 * split_volume_factor), raw_volume[1]],
                "price_adjustment_factor": [first_factor, 1.0],
                "split_volume_factor": [split_volume_factor, 1.0],
                "adjustment_revision": [revision, revision],
            }
        ),
    )
    factor_path = factor_dir / "factors.parquet"
    _atomic_parquet(
        factor_path,
        pd.DataFrame(
            {
                "effective_start": [None, second_date],
                "effective_end": [first_date, None],
                "price_adjustment_factor": [first_factor, 1.0],
                "split_volume_factor": [split_volume_factor, 1.0],
                "adjustment_revision": [revision, revision],
            }
        ),
    )
    return daily_path, factor_path


def _publish_manifest(
    silver_root: Path,
    revision: int,
    artifacts: list[Path],
    affected: list[str],
    earliest_date: date,
) -> None:
    payload = {
        "schema_version": 1,
        "revision": revision,
        "generation_id": f"integration-{revision}",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "corporate_actions_as_of": datetime.now(timezone.utc).isoformat(),
        "affected": [
            {
                "symbol": symbol,
                "earliest_date": earliest_date.isoformat(),
                "timeframes": ["1d", "1m"],
            }
            for symbol in affected
        ],
        "artifacts": [
            {
                "path": path.relative_to(silver_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in artifacts
        ],
    }
    revisions = silver_root / "revisions"
    revisions.mkdir(parents=True, exist_ok=True)
    immutable = revisions / f"revision={revision}.json"
    immutable.write_text(json.dumps(payload), encoding="utf-8")
    temporary = revisions / "current.json.tmp"
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, revisions / "current.json")


async def _wait_for_revision(app: Any, revision: int) -> None:
    for _ in range(100):
        if app.state.revision_watcher.last_fully_applied_revision == revision:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"revision {revision} was not applied")


def test_canary_value_comparison_rejects_row_count_mismatch() -> None:
    assert _same_values([BarData(close=1.0)], []) is False


@pytest.mark.asyncio
async def test_adjusted_canary_and_revision_reseed_without_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    first_date = date.today() - timedelta(days=2)
    second_date = date.today() - timedelta(days=1)
    artifacts: list[Path] = []
    fixtures = {
        "NVDA": (0.5, 2.0, 100.0, 50.0),
        "AAPL": (0.9, 1.0, 100.0, 90.0),
        "SPY": (0.9, 1.0, 100.0, 90.0),
        "PLTR": (1.0, 1.0, 100.0, 101.0),
    }
    for symbol, values in fixtures.items():
        artifacts.extend(
            _publish_symbol(
                bronze_root,
                silver_root,
                symbol,
                *values,
                first_date,
                second_date,
                1,
            )
        )
    _publish_manifest(silver_root, 1, artifacts, list(fixtures), first_date)

    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    canary = await check_silver_canary(
        bronze_root=bronze_root,
        silver_root=silver_root,
        symbols=("NVDA", "AAPL", "SPY"),
        control="PLTR",
        start=datetime.combine(first_date, time.min, tzinfo=timezone.utc),
        end=datetime.combine(second_date, time.max, tzinfo=timezone.utc),
    )
    assert canary["passed"] is True
    assert canary["revision"] == 1
    assert canary["symbols"]["NVDA"]["split_volume_adjusted"] is True
    assert canary["symbols"]["AAPL"]["volume_unchanged"] is True
    assert canary["symbols"]["SPY"]["volume_unchanged"] is True
    assert canary["symbols"]["PLTR"]["identity_control"] is True
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}

    provider = LivewireOhlcProvider(bronze_root, silver_root, "adjusted")
    compute = _RecordingCompute()
    manager = SubscriptionManager(provider, compute, ["1d", "1m"], seed_lookback_days=10)
    await manager.subscribe("NVDA")
    app = create_app()
    app.state.ohlc_provider = provider
    app.state.subscription_manager = manager
    monkeypatch.setenv("APEX_LIVEWIRE_SILVER_ROOT", str(silver_root))
    monkeypatch.setenv("APEX_LIVEWIRE_PRICE_MODE", "adjusted")
    monkeypatch.setenv("APEX_LIVEWIRE_REVISION_POLL_SECONDS", "0.01")
    monkeypatch.delenv("APEX_LIVEWIRE_ROOT", raising=False)
    monkeypatch.delenv("APEX_PG_URL", raising=False)

    async with lifespan(app):
        await _wait_for_revision(app, 1)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/bars/NVDA",
                params={
                    "timeframe": "1d",
                    "start": first_date.isoformat(),
                    "end": second_date.isoformat(),
                    "limit": 0,
                },
            )
        assert response.status_code == 200
        assert [bar["close"] for bar in response.json()["bars"]] == [50.0, 50.0]

        revised_artifacts = list(
            _publish_symbol(
                bronze_root,
                silver_root,
                "NVDA",
                0.4,
                2.0,
                100.0,
                50.0,
                first_date,
                second_date,
                2,
            )
        )
        _publish_manifest(silver_root, 2, revised_artifacts, ["NVDA"], first_date)
        await _wait_for_revision(app, 2)

        assert compute.replaced["NVDA"]["1d"][0]["close"] == 40.0
        assert compute.replaced["NVDA"]["1m"][0]["close"] == 40.0
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            health = (await client.get("/health")).json()["silver_revision"]
        assert health["observed_revision"] == 2
        assert health["last_fully_applied_revision"] == 2
        assert health["per_symbol_revision"] == {"NVDA": 2}
