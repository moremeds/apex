"""Point-in-time membership endpoints, over the real livewire fixtures.

Every number asserted here is a property of `tests/fixtures/livewire_membership/`,
which is a verbatim copy of the production lake (see its README). Nothing is
synthesised: the fixture has zero verified membership events, so the verified path is
asserted to fail closed rather than fed invented events to make it look populated.

The second half of this module builds small parquet logs in `tmp_path` for the cases
the real fixture cannot reach (superseding corrections, a knowledge-gated security
master, a ticker collision). Those are labelled test doubles of log *shape* only --
the schemas are livewire's verbatim, the tickers are real and the dates plausible,
but no row there is observed market data and none is presented as such.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.infrastructure.adapters.livewire.membership import (
    MembershipReader,
    resolve_security_ids,
)
from tests.support.membership_logs import (
    AAPL_SID,
    MSFT_SID,
    _identity_row,
    _membership_row,
    _shaped_lake,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "livewire_membership"

# Facts about the real fixture, recorded 2026-09-14. See the fixture README.
REAL_SECURITY_ID = "sec_405d12b544ef24fee4a9ef06b721d90e"
REAL_SYMBOL = "MUNJ"
DJIA_KNOWN_AT = date(2026, 9, 13)
DJIA_FIRST_EFFECTIVE = date(1991, 5, 6)


@pytest.fixture()
def lake_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("APEX_LIVEWIRE_LAKE_ROOT", str(FIXTURE_ROOT))
    yield FIXTURE_ROOT


@pytest.fixture()
def client() -> Iterator[TestClient]:
    # No `with`: the lifespan would try to open PG, the lake and the xenon socket, and
    # none of that is this surface's business.
    yield TestClient(create_app())


@pytest.fixture()
def reader(lake_root: Path) -> MembershipReader:
    built = MembershipReader.from_env()
    assert built is not None
    return built


# -- adapter ---------------------------------------------------------------


def test_from_env_is_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APEX_LIVEWIRE_LAKE_ROOT", raising=False)
    assert MembershipReader.from_env() is None


def test_list_indices_comes_from_disk(reader: MembershipReader) -> None:
    assert reader.list_indices() == ["djia"]


def test_fixture_has_no_verified_events(reader: MembershipReader) -> None:
    """The honest state of the real data: nothing in djia is verified yet."""
    assert reader.members_as_of("djia", date(2026, 9, 14)) == []


def test_candidate_replay_produces_members(reader: MembershipReader) -> None:
    members = reader.members_as_of("djia", date(2026, 9, 14), include_candidates=True)
    assert members, "the non-rejected replay of the real log must not be empty"
    # 65 adds against 35 removes over 61 distinct securities.
    assert len(members) == 30
    assert all(m.startswith("unresolved:") for m in members)


def test_replay_is_a_function_of_as_of(reader: MembershipReader) -> None:
    """Membership is replayed, not read: an earlier as_of sees an earlier index."""
    early = reader.members_as_of("djia", DJIA_FIRST_EFFECTIVE, include_candidates=True)
    late = reader.members_as_of("djia", date(2026, 9, 14), include_candidates=True)
    assert early != late
    assert len(early) == 30
    assert "unresolved:AA" in early


def test_known_at_gates_the_replay(reader: MembershipReader) -> None:
    """Every real event became known on 2026-09-13, so the day before knows nothing."""
    before = reader.members_as_of(
        "djia", date(2026, 9, 14), known_at=date(2026, 9, 12), include_candidates=True
    )
    on_day = reader.members_as_of(
        "djia", date(2026, 9, 14), known_at=DJIA_KNOWN_AT, include_candidates=True
    )
    assert before == []
    assert len(on_day) == 30


def test_history_for_a_real_member(reader: MembershipReader) -> None:
    events = reader.history_for_security("unresolved:AA")
    assert events, "AA is in the real djia log"
    assert {e.index_id for e in events} == {"djia"}
    assert {e.action for e in events} == {"add", "remove"}
    keys = [(e.effective_at, e.known_at, e.revision, e.event_id) for e in events]
    assert keys == sorted(keys)
    assert events[0].effective_at == datetime(1991, 5, 6, 0, 0, tzinfo=timezone.utc)


def test_symbol_resolution_of_the_one_verified_security(
    reader: MembershipReader,
) -> None:
    resolved = reader.resolve_symbol(REAL_SYMBOL, date(2026, 9, 14))
    assert resolved.security_id == REAL_SECURITY_ID
    assert resolved.ambiguous is False


def test_symbol_resolution_respects_the_interval(reader: MembershipReader) -> None:
    """effective_from is 2026-08-26; the day before it, the ticker does not exist."""
    assert reader.resolve_symbol(REAL_SYMBOL, date(2026, 8, 25)).security_id is None


def test_unknown_symbol_resolves_to_nothing(reader: MembershipReader) -> None:
    assert reader.resolve_symbol("ZZZZNOTATICKER", date(2026, 9, 14)).security_id is None


def test_ambiguity_is_reported_not_guessed() -> None:
    """Pure decision logic: the lake holds no ticker collision to read today, and one
    will not be invented, so the reducer is exercised directly over real ids."""
    two = resolve_security_ids([REAL_SECURITY_ID, "unresolved:AA"])
    assert two.ambiguous is True and two.security_id is None
    one = resolve_security_ids([REAL_SECURITY_ID, REAL_SECURITY_ID])
    assert one.ambiguous is False and one.security_id == REAL_SECURITY_ID
    assert resolve_security_ids([]).security_id is None


# -- routes ----------------------------------------------------------------


def test_503_when_lake_root_unset(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APEX_LIVEWIRE_LAKE_ROOT", raising=False)
    for path in ("/v1/membership/indices", "/v1/membership/djia"):
        response = client.get(path)
        assert response.status_code == 503, path
        assert response.json()["error"]["code"] == "provider_not_configured"


def test_indices_endpoint(client: TestClient, lake_root: Path) -> None:
    response = client.get("/v1/membership/indices")
    assert response.status_code == 200
    assert response.json() == {"indices": ["djia"]}


def test_verified_members_fail_closed(client: TestClient, lake_root: Path) -> None:
    """Fail closed on real data: no verified events means 503, never an empty list."""
    response = client.get("/v1/membership/djia", params={"as_of": "2026-09-14"})
    assert response.status_code == 503
    body = response.json()["error"]
    assert body["code"] == "membership_unavailable"
    assert "not yet available" in body["message"]


def test_members_with_candidates(client: TestClient, lake_root: Path) -> None:
    response = client.get(
        "/v1/membership/djia",
        params={"as_of": "2026-09-14", "include_candidates": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["index_id"] == "djia"
    assert body["as_of"] == "2026-09-14"
    assert body["known_at"] is None
    assert len(body["members"]) == 30
    # The whole djia log is unresolved upstream, so no member carries a ticker.
    assert body["unresolved_count"] == 30
    assert all(member["symbol"] is None for member in body["members"])


def test_known_at_gating_over_http(client: TestClient, lake_root: Path) -> None:
    """Before anything was known the answer is an honest empty index, not a 503."""
    gated = client.get(
        "/v1/membership/djia",
        params={
            "as_of": "2026-09-14",
            "known_at": "2026-09-12",
            "include_candidates": "true",
        },
    )
    assert gated.status_code == 200
    body = gated.json()
    assert body["members"] == []
    assert body["as_of"] == "2026-09-14"
    assert body["known_at"] == "2026-09-12"

    # The verified reading still fails closed: the fixture holds no verified row at all.
    unpublished = client.get(
        "/v1/membership/djia", params={"as_of": "2026-09-14", "known_at": "2026-09-12"}
    )
    assert unpublished.status_code == 503
    assert unpublished.json()["error"]["code"] == "membership_unavailable"

    known = client.get(
        "/v1/membership/djia",
        params={
            "as_of": "2026-09-14",
            "known_at": "2026-09-13",
            "include_candidates": "true",
        },
    )
    assert known.status_code == 200
    assert known.json()["known_at"] == "2026-09-13"


def test_unknown_index_is_404(client: TestClient, lake_root: Path) -> None:
    response = client.get("/v1/membership/nosuchindex")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_index"


def test_path_traversal_index_is_404(client: TestClient, lake_root: Path) -> None:
    response = client.get("/v1/membership/..%2Fsecurity_master")
    assert response.status_code == 404


def test_malformed_as_of_is_400(client: TestClient, lake_root: Path) -> None:
    response = client.get("/v1/membership/djia", params={"as_of": "14-09-2026"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_parameter"


def test_history_route_for_the_real_symbol(client: TestClient, lake_root: Path) -> None:
    response = client.get(
        "/v1/membership/history", params={"symbol": "munj", "as_of": "2026-09-14"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == REAL_SYMBOL
    assert body["security_id"] == REAL_SECURITY_ID
    # MUNJ is in the security master but in no index log -- an empty history, not a 404.
    assert body["events"] == []


def test_history_route_is_not_shadowed_by_the_index_route(
    client: TestClient, lake_root: Path
) -> None:
    """`/v1/membership/history` must not be matched as index_id='history'."""
    response = client.get("/v1/membership/history", params={"symbol": REAL_SYMBOL})
    assert response.status_code != 404 or response.json()["error"]["code"] != "unknown_index"


def test_history_unknown_symbol_is_404(client: TestClient, lake_root: Path) -> None:
    response = client.get("/v1/membership/history", params={"symbol": "ZZZZNOTATICKER"})
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "unknown_symbol"
    assert "not in the security master" in error["message"]


def test_history_falls_back_to_the_unresolved_placeholder(
    client: TestClient, lake_root: Path
) -> None:
    """AA is not in the master, but the real djia log carries `unresolved:AA`."""
    response = client.get("/v1/membership/history", params={"symbol": "aa"})
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AA"
    assert body["security_id"] == "unresolved:AA"
    assert body["events"], "the real djia log has AA events"
    assert {event["index_id"] for event in body["events"]} == {"djia"}


def test_history_unknown_index_is_404(client: TestClient, lake_root: Path) -> None:
    response = client.get(
        "/v1/membership/history",
        params={"symbol": REAL_SYMBOL, "index_id": "nosuchindex"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_index"


def test_history_503_when_unset(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APEX_LIVEWIRE_LAKE_ROOT", raising=False)
    response = client.get("/v1/membership/history", params={"symbol": REAL_SYMBOL})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"


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


def test_history_route_reports_supersedes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """History is the audit trail: the superseded row stays, and says what retracted it."""
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
    assert [e["event_id"] for e in events] == ["original", "correction"]
    assert [e["supersedes"] for e in events] == [None, "original"]
