"""Point-in-time membership over shaped tmp_path lakes.

These parquet logs are labelled test doubles of livewire's log *shape* -- real tickers,
plausible dates, invented event ids -- never observed market data. They exist because
the frozen real fixtures hold no verified rows, so the supersedes, identity-PIT and
history-union paths cannot be exercised on them. Builders: ``tests/support/membership_logs``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from tests.support.membership_logs import (
    AAPL_SID,
    MSFT_SID,
    _identity_row,
    _membership_row,
    _shaped_lake,
)


@pytest.fixture()
def client() -> Iterator[TestClient]:
    # No `with`: the lifespan would try to open PG, the lake and the xenon socket.
    yield TestClient(create_app())


def test_later_correction_does_not_leak_backward(tmp_path: Path) -> None:
    """Mirrors livewire's `test_later_membership_correction_does_not_leak_backward`."""
    reader = _shaped_lake(
        tmp_path,
        events=[
            _membership_row("original", AAPL_SID, "add", "2024-02-01", "2024-01-15"),
            _membership_row(
                "correction",
                AAPL_SID,
                "add",
                "2024-03-01",
                "2024-04-01",
                revision=2,
                supersedes="original",
            ),
        ],
        identities=[_identity_row("id-aapl", AAPL_SID, "AAPL", "2024-01-01")],
    )
    assert reader.members_as_of("sp500", date(2024, 2, 15), known_at=date(2024, 2, 15)) == [
        AAPL_SID
    ]
    assert reader.members_as_of("sp500", date(2024, 2, 15), known_at=date(2024, 4, 2)) == []
    assert reader.members_as_of("sp500", date(2024, 3, 1), known_at=date(2024, 4, 2)) == [AAPL_SID]


def test_known_at_hides_a_superseding_correction(tmp_path: Path) -> None:
    """Without a cutoff the correction always applies; with one it applies only once known."""
    reader = _shaped_lake(
        tmp_path,
        events=[
            _membership_row("original", AAPL_SID, "add", "2024-02-01", "2024-01-15"),
            _membership_row(
                "correction",
                AAPL_SID,
                "add",
                "2024-03-01",
                "2024-04-01",
                revision=2,
                supersedes="original",
            ),
        ],
        identities=[_identity_row("id-aapl", AAPL_SID, "AAPL", "2024-01-01")],
    )
    # Current-knowledge reconstruction of 2024-02-15: the correction is on disk.
    assert reader.members_as_of("sp500", date(2024, 2, 15)) == []
    # What was believed on 2024-03-31, before the correction was published.
    assert reader.members_as_of("sp500", date(2024, 2, 15), known_at=date(2024, 3, 31)) == [
        AAPL_SID
    ]


def test_verified_replay_drops_a_disputed_identity(tmp_path: Path) -> None:
    """A member whose master row is superseded by a non-verified one stops being a member."""
    reader = _shaped_lake(
        tmp_path,
        events=[_membership_row("msft-add", MSFT_SID, "add", "2024-01-01", "2024-01-01")],
        identities=[
            _identity_row("id-msft", MSFT_SID, "MSFT", "2024-01-01"),
            _identity_row(
                "id-msft-disputed",
                MSFT_SID,
                "MSFT",
                "2024-04-01",
                revision=2,
                supersedes="id-msft",
                status="unresolved",
            ),
        ],
    )
    assert reader.members_as_of("sp500", date(2024, 2, 1), known_at=date(2024, 2, 1)) == [MSFT_SID]
    assert reader.members_as_of("sp500", date(2024, 2, 1), known_at=date(2024, 4, 2)) == []
    assert reader.members_as_of("sp500", date(2024, 2, 1)) == []
    # The looser reading never consults the master, so the id survives there.
    assert reader.members_as_of("sp500", date(2024, 2, 1), include_candidates=True) == [MSFT_SID]


def test_verified_replay_drops_an_id_absent_from_the_master(tmp_path: Path) -> None:
    reader = _shaped_lake(
        tmp_path,
        events=[_membership_row("aapl-add", AAPL_SID, "add", "2024-01-01", "2024-01-01")],
        identities=[_identity_row("id-msft", MSFT_SID, "MSFT", "2024-01-01")],
    )
    assert reader.members_as_of("sp500", date(2024, 2, 1)) == []


def test_same_instant_order_is_decided_by_revision(tmp_path: Path) -> None:
    """With effective_at and known_at identical, revision -- not action -- decides.

    There is no "remove wins" convention: reverse the revisions and the member stays.
    """
    identities = [_identity_row("id-aapl", AAPL_SID, "AAPL", "2024-01-01")]
    instant = ("2024-02-01", "2024-02-01")
    removed = _shaped_lake(
        tmp_path / "removed",
        events=[
            _membership_row("event-add", AAPL_SID, "add", *instant, revision=1),
            _membership_row("event-remove", AAPL_SID, "remove", *instant, revision=2),
        ],
        identities=identities,
    )
    assert removed.members_as_of("sp500", date(2024, 3, 1)) == []

    readded = _shaped_lake(
        tmp_path / "readded",
        events=[
            _membership_row("event-remove", AAPL_SID, "remove", *instant, revision=1),
            _membership_row("event-add", AAPL_SID, "add", *instant, revision=2),
        ],
        identities=identities,
    )
    assert readded.members_as_of("sp500", date(2024, 3, 1)) == [AAPL_SID]


def test_members_route_over_a_shaped_lake(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _shaped_lake(
        tmp_path,
        events=[
            _membership_row("aapl-add", AAPL_SID, "add", "2024-01-01", "2024-01-01"),
            _membership_row("msft-add", MSFT_SID, "add", "2024-01-01", "2024-01-01"),
        ],
        identities=[
            _identity_row("id-aapl", AAPL_SID, "AAPL", "2024-01-01"),
            _identity_row("id-msft", MSFT_SID, "MSFT", "2024-01-01"),
        ],
    )
    monkeypatch.setenv("APEX_LIVEWIRE_LAKE_ROOT", str(tmp_path))
    response = client.get("/v1/membership/sp500", params={"as_of": "2024-02-01"})
    assert response.status_code == 200
    body = response.json()
    assert body["unresolved_count"] == 0
    assert sorted(m["symbol"] for m in body["members"]) == ["AAPL", "MSFT"]


def test_history_route_drops_the_superseded_row(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """History is the effective timeline: a retracted row is gone, not shown alongside."""
    _shaped_lake(
        tmp_path,
        events=[
            _membership_row("original", AAPL_SID, "add", "2024-02-01", "2024-01-15"),
            _membership_row(
                "correction",
                AAPL_SID,
                "add",
                "2024-03-01",
                "2024-04-01",
                revision=2,
                supersedes="original",
            ),
        ],
        identities=[_identity_row("id-aapl", AAPL_SID, "AAPL", "2024-01-01")],
    )
    monkeypatch.setenv("APEX_LIVEWIRE_LAKE_ROOT", str(tmp_path))
    response = client.get(
        "/v1/membership/history", params={"symbol": "AAPL", "as_of": "2024-05-01"}
    )
    assert response.status_code == 200
    events = response.json()["events"]
    assert [e["event_id"] for e in events] == ["correction"]
    assert events[0]["supersedes"] == "original"


def test_history_unions_the_placeholder_with_the_resolved_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pre-identity-floor events stay under `unresolved:<TICKER>` and must still show."""
    _shaped_lake(
        tmp_path,
        events=[
            _membership_row("old-add", "unresolved:AAPL", "add", "1999-11-01", "1999-11-01"),
            _membership_row("new-remove", AAPL_SID, "remove", "2024-11-08", "2024-11-08"),
        ],
        identities=[_identity_row("id-aapl", AAPL_SID, "AAPL", "2024-01-01")],
    )
    monkeypatch.setenv("APEX_LIVEWIRE_LAKE_ROOT", str(tmp_path))
    response = client.get(
        "/v1/membership/history", params={"symbol": "aapl", "as_of": "2026-01-01"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["security_id"] == AAPL_SID
    assert [e["event_id"] for e in body["events"]] == ["old-add", "new-remove"]
    assert [e["security_id"] for e in body["events"]] == ["unresolved:AAPL", AAPL_SID]
    assert [e["action"] for e in body["events"]] == ["add", "remove"]


def test_history_collapses_the_backfill_triple(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Backfill logs three rows per event; the union must resolve to one live add.

    The rejected revision sits under the placeholder id and the replacement under the
    resolved one, so the retraction only cancels out across the union of both ids.
    """
    _shaped_lake(
        tmp_path,
        events=[
            _membership_row(
                "placeholder-add", "unresolved:AAPL", "add", "1999-11-01", "1999-11-01"
            ),
            _membership_row(
                "placeholder-rejected",
                "unresolved:AAPL",
                "add",
                "1999-11-01",
                "2026-01-02",
                revision=2,
                supersedes="placeholder-add",
                status="rejected",
            ),
            _membership_row(
                "resolved-add",
                AAPL_SID,
                "add",
                "1999-11-01",
                "2026-01-02",
                revision=3,
            ),
        ],
        identities=[_identity_row("id-aapl", AAPL_SID, "AAPL", "2024-01-01")],
    )
    monkeypatch.setenv("APEX_LIVEWIRE_LAKE_ROOT", str(tmp_path))
    response = client.get(
        "/v1/membership/history", params={"symbol": "AAPL", "as_of": "2026-02-01"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["security_id"] == AAPL_SID
    assert [e["event_id"] for e in body["events"]] == ["resolved-add"]
    assert body["events"][0]["security_id"] == AAPL_SID


def test_history_retracts_across_ids(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A replacement under the resolved id can itself supersede the placeholder row.

    Filtering per id would leave both alive; only the union-wide superseded set
    retracts the placeholder add.
    """
    _shaped_lake(
        tmp_path,
        events=[
            _membership_row(
                "placeholder-add", "unresolved:AAPL", "add", "1999-11-01", "1999-11-01"
            ),
            _membership_row(
                "resolved-add",
                AAPL_SID,
                "add",
                "1999-11-01",
                "2026-01-02",
                revision=2,
                supersedes="placeholder-add",
            ),
        ],
        identities=[_identity_row("id-aapl", AAPL_SID, "AAPL", "2024-01-01")],
    )
    monkeypatch.setenv("APEX_LIVEWIRE_LAKE_ROOT", str(tmp_path))
    body = client.get(
        "/v1/membership/history", params={"symbol": "AAPL", "as_of": "2026-02-01"}
    ).json()
    assert [(e["event_id"], e["security_id"]) for e in body["events"]] == [
        ("resolved-add", AAPL_SID)
    ]
