from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.infrastructure.adapters.livewire.revisions import (
    RevisionManifestError,
    RevisionManifestReader,
)


def _write_manifest(
    root: Path,
    *,
    revision: int = 42,
    artifact_path: str = "asset_class=equity/symbol=NVDA/1d.parquet",
    digest: str | None = None,
    affected: list[dict] | None = None,
    schema_version: int = 1,
    published_at: str = "2026-07-12T10:00:00Z",
) -> None:
    artifact_paths = [
        artifact_path,
        "adjustments/asset_class=equity/symbol=NVDA/factors.parquet",
    ]
    for path in artifact_paths:
        artifact = root / path
        if ".." not in Path(path).parts:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"silver")
    if digest is None:
        digest = hashlib.sha256(b"silver").hexdigest()
    payload = {
        "schema_version": schema_version,
        "revision": revision,
        "generation_id": f"20260712T100000Z-{revision}",
        "published_at": published_at,
        "corporate_actions_as_of": "2026-07-12T09:58:00Z",
        "affected": (
            affected
            if affected is not None
            else [
                {
                    "symbol": "NVDA",
                    "earliest_date": "1999-01-22",
                    "timeframes": ["1d", "1m"],
                }
            ]
        ),
        "artifacts": [
            {
                "path": path,
                "sha256": (
                    digest if path == artifact_path else hashlib.sha256(b"silver").hexdigest()
                ),
            }
            for path in artifact_paths
        ],
    }
    revisions = root / "revisions"
    revisions.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload)
    (revisions / f"revision={revision}.json").write_text(encoded, encoding="utf-8")
    (revisions / "current.json").write_text(encoded, encoding="utf-8")


def test_reader_parses_and_verifies_current_manifest(tmp_path: Path) -> None:
    _write_manifest(tmp_path)

    revision = RevisionManifestReader(tmp_path).read_current()

    assert revision.revision == 42
    assert revision.published_at.isoformat() == "2026-07-12T10:00:00+00:00"
    assert revision.affected[0].symbol == "NVDA"
    assert revision.affected[0].timeframes == ("1d", "1m")


@pytest.mark.parametrize("schema_version", [0, 2])
def test_reader_rejects_unknown_schema_versions(tmp_path: Path, schema_version: int) -> None:
    _write_manifest(tmp_path, schema_version=schema_version)

    with pytest.raises(RevisionManifestError, match="schema_version"):
        RevisionManifestReader(tmp_path).read_current()


def test_reader_rejects_path_escape(tmp_path: Path) -> None:
    _write_manifest(tmp_path, artifact_path="../bronze/secret.parquet")

    with pytest.raises(RevisionManifestError, match="outside Silver root"):
        RevisionManifestReader(tmp_path).read_current()


def test_reader_rejects_checksum_mismatch(tmp_path: Path) -> None:
    _write_manifest(tmp_path, digest="0" * 64)

    with pytest.raises(RevisionManifestError, match="checksum mismatch"):
        RevisionManifestReader(tmp_path).read_current().artifact_path("NVDA", "daily")


def test_reader_rejects_duplicate_affected_symbols(tmp_path: Path) -> None:
    affected = [
        {"symbol": "NVDA", "earliest_date": "1999-01-22", "timeframes": ["1d"]},
        {"symbol": "NVDA", "earliest_date": "2024-06-10", "timeframes": ["1m"]},
    ]
    _write_manifest(tmp_path, affected=affected)

    with pytest.raises(RevisionManifestError, match="duplicate affected symbol"):
        RevisionManifestReader(tmp_path).read_current()


def test_reader_rejects_unsupported_timeframe(tmp_path: Path) -> None:
    affected = [{"symbol": "NVDA", "earliest_date": "1999-01-22", "timeframes": ["4h"]}]
    _write_manifest(tmp_path, affected=affected)

    with pytest.raises(RevisionManifestError, match="unsupported timeframe"):
        RevisionManifestReader(tmp_path).read_current()


def test_reader_rejects_empty_publication(tmp_path: Path) -> None:
    _write_manifest(tmp_path, affected=[])

    with pytest.raises(RevisionManifestError, match="affected must be a non-empty list"):
        RevisionManifestReader(tmp_path).read_current()


def test_reader_rejects_non_utc_timestamp(tmp_path: Path) -> None:
    _write_manifest(tmp_path, published_at="2026-07-12T18:00:00+08:00")

    with pytest.raises(RevisionManifestError, match="published_at must be UTC"):
        RevisionManifestReader(tmp_path).read_current()


@pytest.mark.parametrize(
    "artifacts",
    [
        ["asset_class=equity/symbol=NVDA/1d.parquet"],
        [
            "asset_class=equity/symbol=NVDA/1d.parquet",
            "adjustments/asset_class=equity/symbol=NVDA/factors.parquet",
            "asset_class=equity/symbol=AAPL/1d.parquet",
        ],
    ],
)
def test_reader_rejects_artifacts_that_do_not_exactly_match_affected_membership(
    tmp_path: Path, artifacts: list[str]
) -> None:
    digest = hashlib.sha256(b"silver").hexdigest()
    for path in artifacts:
        artifact = tmp_path / path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"silver")
    payload = {
        "schema_version": 1,
        "revision": 42,
        "generation_id": "test-42",
        "published_at": "2026-07-12T10:00:00Z",
        "corporate_actions_as_of": "2026-07-12T09:58:00Z",
        "affected": [{"symbol": "NVDA", "earliest_date": "1999-01-22", "timeframes": ["1d"]}],
        "artifacts": [{"path": path, "sha256": digest} for path in artifacts],
    }
    revisions = tmp_path / "revisions"
    revisions.mkdir()
    encoded = json.dumps(payload)
    (revisions / "revision=42.json").write_text(encoded, encoding="utf-8")
    (revisions / "current.json").write_text(encoded, encoding="utf-8")

    with pytest.raises(RevisionManifestError, match="exactly one daily and one factors"):
        RevisionManifestReader(tmp_path).read_current()
