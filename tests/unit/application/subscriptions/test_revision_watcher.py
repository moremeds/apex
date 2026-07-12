from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import pytest

from src.application.subscriptions.manager import RefreshResult
from src.application.subscriptions.revision_watcher import RevisionWatcher
from src.infrastructure.adapters.livewire.revisions import (
    AffectedSymbol,
    RevisionManifestError,
    SilverRevision,
)


def _revision(number: int) -> SilverRevision:
    now = datetime(2026, 7, 12, tzinfo=timezone.utc)
    return SilverRevision(
        schema_version=1,
        revision=number,
        generation_id=f"test-{number}",
        published_at=now,
        corporate_actions_as_of=now,
        affected=(AffectedSymbol("NVDA", date(1999, 1, 22), ("1d",)),),
    )


class _Reader:
    def __init__(self, revision: SilverRevision) -> None:
        self.revision = revision
        self.error: Exception | None = None
        self.calls = 0

    def read_current(self) -> SilverRevision:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.revision


class _Manager:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.results: list[RefreshResult] = []

    async def refresh_revision(self, revision: SilverRevision) -> RefreshResult:
        self.calls.append(revision.revision)
        if self.results:
            return self.results.pop(0)
        return RefreshResult(applied={"NVDA": revision.revision}, failed={})


@pytest.mark.asyncio
async def test_watcher_applies_new_revision_and_ignores_duplicate() -> None:
    reader, manager = _Reader(_revision(42)), _Manager()
    watcher = RevisionWatcher(reader, manager, poll_seconds=1)

    await watcher.poll_once()
    await watcher.poll_once()

    assert manager.calls == [42]
    assert watcher.health()["observed_revision"] == 42
    assert watcher.health()["last_fully_applied_revision"] == 42
    assert watcher.health()["per_symbol_revision"] == {"NVDA": 42}


@pytest.mark.asyncio
async def test_watcher_retries_failed_current_revision() -> None:
    reader, manager = _Reader(_revision(43)), _Manager()
    manager.results = [
        RefreshResult(applied={}, failed={"NVDA": "Silver unavailable"}),
        RefreshResult(applied={"NVDA": 43}, failed={}),
    ]
    watcher = RevisionWatcher(reader, manager, poll_seconds=1)

    await watcher.poll_once()
    assert watcher.health()["failed"] == {"NVDA": "Silver unavailable"}
    await watcher.poll_once()

    assert manager.calls == [43, 43]
    assert watcher.health()["failed"] == {}
    assert watcher.health()["last_fully_applied_revision"] == 43


@pytest.mark.asyncio
async def test_watcher_applies_latest_when_revisions_are_skipped() -> None:
    reader, manager = _Reader(_revision(40)), _Manager()
    watcher = RevisionWatcher(reader, manager, poll_seconds=1)
    await watcher.poll_once()
    reader.revision = _revision(45)

    await watcher.poll_once()

    assert manager.calls == [40, 45]
    assert watcher.health()["last_fully_applied_revision"] == 45


@pytest.mark.asyncio
async def test_watcher_retains_state_when_manifest_is_invalid() -> None:
    reader, manager = _Reader(_revision(42)), _Manager()
    watcher = RevisionWatcher(reader, manager, poll_seconds=1)
    await watcher.poll_once()
    reader.error = RevisionManifestError("checksum mismatch")

    await watcher.poll_once()

    health = watcher.health()
    assert health["last_fully_applied_revision"] == 42
    assert "checksum mismatch" in health["last_error"]


@pytest.mark.asyncio
async def test_watcher_start_and_stop_manage_background_task() -> None:
    reader, manager = _Reader(_revision(42)), _Manager()
    watcher = RevisionWatcher(reader, manager, poll_seconds=0.01)

    await watcher.start()
    for _ in range(20):
        if manager.calls:
            break
        await asyncio.sleep(0.01)
    await watcher.stop()

    assert manager.calls == [42]
    assert watcher.is_running is False
