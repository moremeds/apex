"""Component-level proof that a running watcher picks up a real, atomically
swapped Silver manifest on disk and advances — no process/container restart.

Uses the real RevisionManifestReader (real file read + immutable pointer check +
os.replace atomic swap). The subscription manager is faked; the manager's own
reseed logic is covered by test_manager.py."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

from src.application.subscriptions.manager import RefreshResult
from src.application.subscriptions.revision_watcher import RevisionWatcher
from src.infrastructure.adapters.livewire.revisions import (
    RevisionManifestError,
    RevisionManifestReader,
    SilverRevision,
)


def _publish_manifest(root: Path, revision: int, symbol: str = "NVDA") -> None:
    """Write revision={n}.json + artifact, then atomically swap current.json —
    the same temp-write → os.replace commit order Livewire's publisher uses."""
    artifact_paths = [
        f"generations/{revision}/asset_class=equity/symbol={symbol}/1d.parquet",
        (
            f"generations/{revision}/adjustments/asset_class=equity/"
            f"symbol={symbol}/factors.parquet"
        ),
    ]
    artifacts = []
    for artifact_path in artifact_paths:
        artifact = root / artifact_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"silver-{symbol}-rev-{revision}-{artifact.name}".encode())
        artifacts.append(
            {"path": artifact_path, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
        )

    payload = {
        "schema_version": 1,
        "revision": revision,
        "generation_id": f"gen-{revision}",
        "published_at": "2026-07-12T10:00:00Z",
        "corporate_actions_as_of": "2026-07-12T09:58:00Z",
        "affected": [{"symbol": symbol, "earliest_date": "1999-01-22", "timeframes": ["1d"]}],
        "artifacts": artifacts,
    }
    revisions = root / "revisions"
    revisions.mkdir(parents=True, exist_ok=True)
    (revisions / f"revision={revision}.json").write_text(json.dumps(payload))

    tmp = revisions / "current.json.tmp"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, revisions / "current.json")  # atomic commit


class _Manager:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def refresh_revision(self, revision: SilverRevision) -> RefreshResult:
        self.calls.append(revision.revision)
        return RefreshResult(
            applied={item.symbol: revision.revision for item in revision.affected},
            failed={},
        )


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


@pytest.mark.asyncio
async def test_running_watcher_observes_real_atomic_manifest_swap(tmp_path: Path) -> None:
    _publish_manifest(tmp_path, revision=1)
    reader = RevisionManifestReader(tmp_path)  # real reader: real IO + checksum
    manager = _Manager()
    watcher = RevisionWatcher(reader, manager, poll_seconds=0.01)

    await watcher.start()
    try:
        await _wait_until(lambda: watcher.last_fully_applied_revision == 1)

        # Atomically swap to a new revision while the watcher keeps running.
        _publish_manifest(tmp_path, revision=2)
        await _wait_until(lambda: watcher.last_fully_applied_revision == 2)
    finally:
        await watcher.stop()

    assert manager.calls == [1, 2]
    health = watcher.health()
    assert health["observed_revision"] == 2
    assert health["per_symbol_revision"] == {"NVDA": 2}
    assert health["pending"] == []


def test_selected_artifact_verification_rejects_tamper(tmp_path: Path) -> None:
    _publish_manifest(tmp_path, revision=1)
    # Corrupt the artifact after publication so the recorded SHA-256 no longer matches.
    artifact = tmp_path / "generations/1/asset_class=equity/symbol=NVDA/1d.parquet"
    artifact.write_bytes(b"tampered")

    snapshot = RevisionManifestReader(tmp_path).read_current()

    with pytest.raises(RevisionManifestError, match="checksum mismatch"):
        snapshot.artifact_path("NVDA", "daily")
