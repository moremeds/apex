from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app
from src.infrastructure.adapters.livewire.ohlc_provider import (
    AdjustedDataUnavailable,
    LivewireOhlcProvider,
)
from src.infrastructure.adapters.livewire.revisions import (
    RevisionManifestError,
    RevisionManifestReader,
)
from tests.support.silver_manifest import publish_manifest, write_bronze_intraday, write_generation

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 31, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_pinned_daily_and_factors_keep_old_prices_after_commit(tmp_path: Path) -> None:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    write_bronze_intraday(bronze)
    old = write_generation(silver, "old", 5.0)
    publish_manifest(silver, 1, old)
    shared = LivewireOhlcProvider(bronze, silver, "adjusted")
    pinned = shared.pin_snapshot()
    new = write_generation(silver, "new", 9.0)
    publish_manifest(silver, 2, new)

    daily, intraday, fresh = await asyncio.gather(
        pinned.fetch_bars("TEST", "1d", START, END),
        pinned.fetch_bars("TEST", "1m", START, END),
        shared.fetch_bars("TEST", "1d", START, END),
    )

    assert (daily[0].close, intraday[0].close, fresh[0].close) == (5.0, 5.0, 9.0)
    assert shared.snapshot is None
    assert pinned.snapshot is not None and pinned.snapshot.revision == 1


@pytest.mark.asyncio
async def test_selected_artifact_is_verified_without_scanning_unrequested_symbols(
    tmp_path: Path,
) -> None:
    silver = tmp_path / "silver"
    good = write_generation(silver, "good", 5.0)
    bad = write_generation(silver, "other", 8.0, "OTHER")
    publish_manifest(silver, 1, good + bad)
    bad[0].write_bytes(b"truncated")
    provider = LivewireOhlcProvider(tmp_path / "bronze", silver, "adjusted")

    assert (await provider.fetch_bars("TEST", "1d", START, END))[0].close == 5.0
    with pytest.raises(AdjustedDataUnavailable, match="checksum mismatch"):
        await provider.fetch_bars("OTHER", "1d", START, END)
    good[0].unlink()
    with pytest.raises(AdjustedDataUnavailable, match="cannot read artifact"):
        await provider.fetch_bars("TEST", "1d", START, END)


@pytest.mark.asyncio
async def test_uncommitted_and_withdrawn_files_never_become_readable(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    old = write_generation(silver, "old", 5.0)
    publish_manifest(silver, 1, old)
    provider = LivewireOhlcProvider(tmp_path / "bronze", silver, "adjusted")
    write_generation(silver, "abandoned", 99.0)
    assert (await provider.fetch_bars("TEST", "1d", START, END))[0].close == 5.0
    replacement = write_generation(silver, "replacement", 7.0, "OTHER")
    publish_manifest(silver, 2, replacement)
    assert await provider.fetch_bars("TEST", "1d", START, END) == []
    assert provider.fetch_recency("TEST")["silver_last_trade_date"] is None
    assert old[0].exists()


def test_pointer_must_match_immutable_manifest(tmp_path: Path) -> None:
    publish_manifest(tmp_path, 1, write_generation(tmp_path, "one", 5.0))
    (tmp_path / "revisions/revision=1.json").write_text("{}")
    with pytest.raises(RevisionManifestError, match="does not match"):
        RevisionManifestReader(tmp_path).read_current()


def test_selected_symlink_escape_is_rejected(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    paths = write_generation(silver, "one", 5.0)
    publish_manifest(silver, 1, paths)
    snapshot = RevisionManifestReader(silver).read_current()
    external = tmp_path / "external.parquet"
    paths[0].rename(external)
    paths[0].symlink_to(external)
    with pytest.raises(RevisionManifestError, match="outside Silver root"):
        snapshot.artifact_path("TEST", "daily")


@pytest.mark.asyncio
async def test_chart_payload_revision_and_guard_use_the_read_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver = tmp_path / "silver"
    old = write_generation(silver, "old", 5.0)
    new = write_generation(silver, "new", 9.0)
    publish_manifest(silver, 1, old)
    provider = LivewireOhlcProvider(tmp_path / "bronze", silver, "adjusted")
    original = provider._bars

    async def flip_then_read(*args, **kwargs):
        # A new revision lands after the request pinned its snapshot, mid-read.
        publish_manifest(silver, 2, new)
        return await original(*args, **kwargs)

    monkeypatch.setattr(provider, "_bars", flip_then_read)
    app = create_app()
    app.state.ohlc_provider = provider
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/v1/equity/TEST/bars",
            params={"timeframe": "1d", "start": START.isoformat(), "end": END.isoformat()},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["adjustment_revision"] == 1
    assert payload["bars"][0]["close"] == 5.0
