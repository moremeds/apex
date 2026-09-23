from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.infrastructure.adapters.livewire.pit_revisions import (
    PitRevisionNotFound,
    PitRevisionReader,
    PitUnavailable,
)
from tests.support.pit_manifest import (
    BIIB_SECURITY,
    FSLR_ROWS,
    FSLR_SECURITY,
    pit_payload,
    publish_pit,
    write_daily,
)


def test_reads_manifest_by_explicit_revision_and_echoes_status(tmp_path: Path) -> None:
    publish_pit(tmp_path, pit_payload(tmp_path, 1, status="PARTIAL"))

    manifest = PitRevisionReader(tmp_path).read(1)

    assert manifest.summary.index_id == "sp500"
    assert manifest.summary.status == "PARTIAL"
    assert manifest.summary.silver_revision == 77
    assert manifest.summary.membership_revision == 4896
    assert manifest.summary.daily_bar_cutoff == date(2026, 9, 22)
    assert manifest.summary.member_count == 3
    # Lake-root references are metadata only; the artifact list is not echoed there.
    assert manifest.input_references["security_master"]["path"] == (
        "security_master/events.parquet"
    )
    assert "silver_artifacts" not in manifest.input_references


def test_scopes_keep_both_membership_spells_of_one_security(tmp_path: Path) -> None:
    publish_pit(tmp_path, pit_payload(tmp_path))
    scopes = PitRevisionReader(tmp_path).read(1).scopes_for("FSLR")

    assert [(s.session_from, s.session_to) for s in scopes] == [
        (date(2009, 10, 16), date(2017, 3, 20)),
        (date(2022, 12, 19), None),
    ]
    assert {s.security_id for s in scopes} == {FSLR_SECURITY}
    # session_to is exclusive; None means still a member at as_of.
    assert scopes[0].contains(date(2017, 3, 17)) and not scopes[0].contains(date(2017, 3, 20))
    assert scopes[1].contains(date(2026, 9, 21))
    assert not any(s.contains(date(2020, 1, 2)) for s in scopes)


def test_serves_artifact_after_hash_check(tmp_path: Path) -> None:
    publish_pit(tmp_path, pit_payload(tmp_path))
    path = PitRevisionReader(tmp_path).read(1).daily_artifact_path("BIIB")
    assert path.name == "1d.parquet" and "symbol=BIIB" in path.as_posix()


def test_hash_mismatch_is_unavailable_not_another_revision(tmp_path: Path) -> None:
    publish_pit(tmp_path, pit_payload(tmp_path))
    # Test mutation: overwrite the served artifact after publication.
    write_daily(tmp_path, "FSLR", FSLR_ROWS[:2])
    manifest = PitRevisionReader(tmp_path).read(1)
    with pytest.raises(PitUnavailable, match="does not match its hash"):
        manifest.daily_artifact_path("FSLR")


def test_evicted_artifact_is_unavailable(tmp_path: Path) -> None:
    publish_pit(tmp_path, pit_payload(tmp_path))
    manifest = PitRevisionReader(tmp_path).read(1)
    (tmp_path / manifest.daily_artifacts["BIIB"].path).unlink()
    with pytest.raises(PitUnavailable, match="evicted or missing"):
        manifest.daily_artifact_path("BIIB")


def test_member_without_daily_entry_is_unavailable(tmp_path: Path) -> None:
    rel = write_daily(tmp_path, "BIIB", [])
    publish_pit(tmp_path, pit_payload(tmp_path, artifacts={"BIIB": rel}))
    with pytest.raises(PitUnavailable, match="no daily artifact entry for FSLR"):
        PitRevisionReader(tmp_path).read(1).daily_artifact_path("FSLR")


def test_symlink_escape_is_unavailable(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    outside = tmp_path / "outside"
    rel = write_daily(outside, "BIIB", [])
    link = silver / rel
    link.parent.mkdir(parents=True)
    link.symlink_to(outside / rel)
    publish_pit(silver, pit_payload(silver, artifacts={"BIIB": rel}))
    with pytest.raises(PitUnavailable, match="escapes the Silver root"):
        PitRevisionReader(silver).read(1).daily_artifact_path("BIIB")


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda p: p.update(policy_version="pit-silver-v2"), "policy_version"),
        (lambda p: p.pop("members"), "missing"),
        (lambda p: p.update(status="VERIFIED"), "unknown status"),
        (lambda p: p.update(revision=2), "disagree"),
        (lambda p: p.update(as_of="2026-09-23T00:00:00"), "offset"),
        (
            lambda p: p["inputs"]["silver_artifacts"].append(
                {
                    "path": "../escape/asset_class=equity/symbol=X/1d.parquet",
                    "sha256": "a" * 64,
                }
            ),
            "invalid Silver artifact reference",
        ),
    ],
)
def test_malformed_manifest_is_unavailable(tmp_path: Path, mutate, reason: str) -> None:
    payload = pit_payload(tmp_path)
    mutate(payload)
    publish_pit(tmp_path, payload, revision=1)
    with pytest.raises(PitUnavailable, match=reason):
        PitRevisionReader(tmp_path).read(1)


def test_not_json_is_unavailable_and_absent_is_not_found(tmp_path: Path) -> None:
    directory = tmp_path / "pit-revisions"
    directory.mkdir()
    (directory / "revision=3.json").write_text("{not json")
    reader = PitRevisionReader(tmp_path)
    with pytest.raises(PitUnavailable):
        reader.read(3)
    with pytest.raises(PitRevisionNotFound):
        reader.read(4)


def test_list_is_newest_first_and_ignores_current_and_appledouble(
    tmp_path: Path,
) -> None:
    publish_pit(tmp_path, pit_payload(tmp_path, 1, index_id="sp500"))
    publish_pit(tmp_path, pit_payload(tmp_path, 2, index_id="ndx100", status="PROVEN"))
    (tmp_path / "pit-revisions" / "._revision=2.json").write_bytes(b"\x00\x05")

    listed = PitRevisionReader(tmp_path).list_revisions()

    assert [(s.revision, s.index_id, s.status) for s in listed] == [
        (2, "ndx100", "PROVEN"),
        (1, "sp500", "PARTIAL"),
    ]


def test_absent_directory_lists_nothing(tmp_path: Path) -> None:
    reader = PitRevisionReader(tmp_path)
    assert reader.list_revisions() == [] and reader.available() is False


def test_current_json_is_never_consulted(tmp_path: Path) -> None:
    publish_pit(tmp_path, pit_payload(tmp_path, 1))
    (tmp_path / "pit-revisions" / "current.json").write_text(json.dumps({"garbage": True}))
    assert PitRevisionReader(tmp_path).read(1).summary.index_id == "sp500"


def test_biib_security_is_carried(tmp_path: Path) -> None:
    publish_pit(tmp_path, pit_payload(tmp_path))
    (scope,) = PitRevisionReader(tmp_path).read(1).scopes_for("BIIB")
    assert scope.security_id == BIIB_SECURITY and scope.session_to is None
