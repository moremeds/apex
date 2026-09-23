"""Shared bars query: revision pins, PIT scopes, output policies, tail exactness.

Real frozen values only: SPY/VSCO rows from the mini lake (2026-09-21, see
tests/unit/api/test_bars_listing.py for their provenance), SPY's price_basis
('raw', source 'massive' for 2026-09 rows, read 2026-09-23), and the FSLR/BIIB PIT
fixtures in tests/support/pit_manifest.py. Mutated manifests are labelled as test
mutations where they appear.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from src.api.payload.chart import bars_payload_from
from src.application.lake.bars import query_bars
from src.application.lake.bulk import query_bulk_bars
from src.application.lake.errors import LakeError
from src.application.lake.services import LakeServices
from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider
from src.infrastructure.adapters.livewire.pit_revisions import PitRevisionReader
from src.infrastructure.adapters.livewire.revisions import RevisionManifestReader
from tests.support.pit_manifest import (
    FSLR_ROWS,
    FSLR_SCOPES,
    GENERATION_R76,
    pit_payload,
    publish_pit,
    write_daily,
)
from tests.support.silver_manifest import publish_manifest

UTC = dt.timezone.utc

_SPY = [
    (dt.date(2026, 9, 16), 759.5, 761.67, 749.6, 754.05, 59_217_653),
    (dt.date(2026, 9, 17), 763.15, 763.57, 759.96, 762.6, 49_652_754),
]
_VSCO_DELISTED = [
    (dt.date(2021, 7, 21), 55.0, 55.0, 39.99, 42.5, 80_637),
    (dt.date(2021, 7, 22), 42.75, 42.75, 39.79, 40.9, 352_595),
    (dt.date(2021, 7, 23), 41.98, 42.2, 40.99, 42.14, 75_030),
    (dt.date(2026, 5, 29), 58.5, 58.5, 55.0, 55.0, 3_956_703),
    (dt.date(2026, 6, 1), 52.26, 55.84, 51.0592, 54.3, 4_238_635),
]
_VSCO_LIVE = [
    (dt.date(2026, 5, 29), 58.5, 58.5, 55.0, 55.0, 3_956_702),
    (dt.date(2026, 6, 1), 52.26, 55.84, 51.0592, 54.3, 4_238_634),
]


def _daily(root: Path, symbol: str, rows: list, price_basis: str | None = None) -> None:
    directory = root / "asset_class=equity" / f"symbol={symbol}"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "trade_date": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        }
    )
    if price_basis is not None:
        frame["price_basis"] = price_basis
    frame.to_parquet(directory / "1d.parquet", index=False)


@pytest.fixture
def lake(tmp_path: Path) -> dict[str, Path]:
    roots = {k: tmp_path / k for k in ("bronze", "delisted", "silver")}
    _daily(roots["bronze"], "SPY", _SPY, price_basis="raw")
    _daily(roots["bronze"], "VSCO", _VSCO_LIVE)
    _daily(roots["delisted"], "VSCO", _VSCO_DELISTED)
    publish_pit(roots["silver"], pit_payload(roots["silver"], 1))
    return roots


def _services(roots: dict[str, Path], price_mode: str = "raw") -> LakeServices:
    provider = LivewireOhlcProvider(
        bronze_root=roots["bronze"],
        silver_root=roots["silver"],
        price_mode=price_mode,  # type: ignore[arg-type]
        delisted_root=roots["delisted"],
    )
    return LakeServices(
        provider=provider,
        silver=RevisionManifestReader(roots["silver"]),
        pit=PitRevisionReader(roots["silver"]),
    )


def _when(day: str) -> dt.datetime:
    return dt.datetime.fromisoformat(day).replace(tzinfo=UTC)


async def test_pit_serves_scope_and_echoes_publisher_status(
    lake: dict[str, Path],
) -> None:
    result = await query_bars(
        _services(lake),
        symbol="FSLR",
        start=_when("2026-09-15"),
        end=_when("2026-09-22"),
        pit_revision=1,
    )
    assert [b.close for b in result.bars] == [202.34, 191.07, 201.16, 195.96, 199.84]
    assert result.price_mode == "adjusted" and result.immutable_history
    assert result.adjustment_revision == 77
    assert result.pit is not None and result.pit.publisher_status == "PARTIAL"
    assert result.pit.index_id == "sp500"
    payload = bars_payload_from(result, generated_at=_when("2026-09-23"))
    assert payload["provenance"]["pit"]["scopes"] == [
        {
            "security_id": FSLR_SCOPES[1]["security_id"],
            "session_from": "2022-12-19",
            "session_to": None,
        }
    ]


async def test_pit_filters_rows_to_the_member_scope_and_cutoff(
    lake: dict[str, Path],
) -> None:
    # Test mutation: FSLR joined on 2026-09-17 and the manifest cut off at 2026-09-18.
    silver = lake["silver"]
    scope = dict(FSLR_SCOPES[1], session_from="2026-09-17")
    payload = pit_payload(silver, 2, members=[scope])
    payload["daily_bar_cutoff"] = "2026-09-18"
    publish_pit(silver, payload)

    result = await query_bars(
        _services(lake), symbol="FSLR", start=_when("2026-09-01"), pit_revision=2
    )

    assert [b.timestamp.date() for b in result.bars] == [
        dt.date(2026, 9, 17),
        dt.date(2026, 9, 18),
    ]
    assert result.window_end.date() == dt.date(2026, 9, 18)


async def test_pit_window_outside_scope_is_a_scoped_error(
    lake: dict[str, Path],
) -> None:
    with pytest.raises(LakeError) as caught:
        await query_bars(
            _services(lake),
            symbol="FSLR",
            start=_when("2018-01-02"),
            end=_when("2021-12-31"),
            pit_revision=1,
        )
    assert caught.value.code == "invalid_parameter"
    assert caught.value.details is not None and len(caught.value.details["scopes"]) == 2


async def test_pit_window_across_two_security_ids_is_ambiguous(
    lake: dict[str, Path],
) -> None:
    # Test mutation: the real FSLR spells share one security_id; give the old spell
    # another id so the window straddles two securities.
    silver = lake["silver"]
    old = dict(FSLR_SCOPES[0], security_id="sec_mutated_other_issuer")
    publish_pit(silver, pit_payload(silver, 3, members=[old, FSLR_SCOPES[1]]))
    with pytest.raises(LakeError) as caught:
        await query_bars(
            _services(lake),
            symbol="FSLR",
            start=_when("2016-01-04"),
            end=_when("2026-09-21"),
            pit_revision=3,
        )
    assert caught.value.code == "ambiguous_symbol"


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"symbol": "FSLR", "pit_revision": 9}, "unknown_revision"),
        ({"symbol": "SPY", "pit_revision": 1}, "invalid_parameter"),  # not a member
        (
            {"symbol": "FSLR", "pit_revision": 1, "silver_revision_pin": 1},
            "invalid_parameter",
        ),
        (
            {"symbol": "FSLR", "pit_revision": 1, "price_mode": "raw"},
            "invalid_parameter",
        ),
        (
            {"symbol": "FSLR", "pit_revision": 1, "timeframe": "1m"},
            "revision_not_supported",
        ),
        (
            {"symbol": "FSLR", "pit_revision": 1, "listing": "any"},
            "revision_not_supported",
        ),
        (
            {"symbol": "EURUSD", "asset_class": "fx", "silver_revision_pin": 1},
            "revision_not_supported",
        ),
        ({"symbol": "FSLR", "silver_revision_pin": 0}, "invalid_parameter"),
    ],
)
async def test_revision_argument_errors(lake: dict[str, Path], kwargs: dict, code: str) -> None:
    with pytest.raises(LakeError) as caught:
        await query_bars(_services(lake), **kwargs)
    assert caught.value.code == code


async def test_malformed_pit_is_unavailable(lake: dict[str, Path]) -> None:
    (lake["silver"] / "pit-revisions" / "revision=5.json").write_text("{")
    with pytest.raises(LakeError) as caught:
        await query_bars(_services(lake), symbol="FSLR", pit_revision=5)
    assert caught.value.code == "pit_unavailable"


async def test_bounded_policy_caps_even_with_explicit_start(
    lake: dict[str, Path],
) -> None:
    services = _services(lake)
    capped = await query_bars(
        services,
        symbol="VSCO",
        listing="delisted",
        start=_when("2021-01-01"),
        limit=2,
        policy="bounded",
    )
    assert [b.timestamp.date() for b in capped.bars] == [
        dt.date(2026, 5, 29),
        dt.date(2026, 6, 1),
    ]
    assert capped.truncated is True
    exact = await query_bars(
        services,
        symbol="VSCO",
        listing="delisted",
        start=_when("2021-01-01"),
        limit=5,
        policy="bounded",
    )
    assert len(exact.bars) == 5 and exact.truncated is False
    legacy = await query_bars(
        services, symbol="VSCO", listing="delisted", start=_when("2021-01-01"), limit=2
    )
    assert len(legacy.bars) == 5 and legacy.truncated is False
    with pytest.raises(LakeError):
        await query_bars(services, symbol="VSCO", limit=0, policy="bounded")


async def test_dual_tail_is_exact_over_the_merged_series(lake: dict[str, Path]) -> None:
    """The archive's two most recent rows sit on dates the live tree also has; a
    per-source tail would return them and drop 2021-07-23."""
    services = _services(lake)
    full = await query_bars(services, symbol="VSCO", listing="any", limit=0)
    tail = await query_bars(
        services,
        symbol="VSCO",
        listing="any",
        limit=3,
        policy="bounded",
        start=_when("2000-01-03"),
    )
    assert full.listing_status == "dual"
    assert [(b.timestamp, b.volume) for b in tail.bars] == [
        (b.timestamp, b.volume) for b in full.bars[-3:]
    ]
    assert tail.bars[0].timestamp.date() == dt.date(2021, 7, 23)
    assert tail.bars[-1].volume == 4_238_634  # the live row won the shared date


async def test_raw_rows_carry_livewire_price_basis(lake: dict[str, Path]) -> None:
    services = _services(lake)
    spy = bars_payload_from(
        await query_bars(services, symbol="SPY", start=_when("2026-09-01")),
        generated_at=_when("2026-09-23"),
    )
    vsco = bars_payload_from(
        await query_bars(services, symbol="VSCO", listing="delisted", limit=0),
        generated_at=_when("2026-09-23"),
    )
    assert {row["source_price_basis"] for row in spy["bars"]} == {"raw"}
    # The archive file has no price_basis column: unknown, never inferred from the mode.
    assert {row["source_price_basis"] for row in vsco["bars"]} == {"unknown"}
    assert spy["basis"] == "unadjusted" and spy["provenance"]["immutable_history"] is False


async def test_bulk_bounded_budget_and_one_pin(lake: dict[str, Path]) -> None:
    services = _services(lake)
    with pytest.raises(LakeError, match="budget"):
        await query_bulk_bars(
            services, symbols=[f"S{i}" for i in range(200)], limit=51, policy="bounded"
        )
    result = await query_bulk_bars(
        services, symbols=["spy", "SPY", "NOPE"], limit=1, policy="bounded"
    )
    assert list(result.series) == ["SPY"] and "NOPE" in result.missing
    assert result.series["SPY"].truncated is True and len(result.series["SPY"].bars) == 1


async def test_silver_pin_reads_the_older_retained_revision(lake: dict[str, Path]) -> None:
    """Real Silver revisions 76 and 77 for FSLR: identical prices, but revision 76 was
    published before the 2026-09-21 session and stamps adjustment_revision 75."""
    silver = lake["silver"]
    r76 = silver / write_daily(silver, "FSLR", FSLR_ROWS[:4], GENERATION_R76, 75)
    publish_manifest(silver, 76, [r76])
    r77 = silver / write_daily(silver, "FSLR", FSLR_ROWS)
    publish_manifest(silver, 77, [r77])
    services = _services(lake, price_mode="adjusted")

    old = await query_bars(
        services, symbol="FSLR", start=_when("2026-09-15"), silver_revision_pin=76
    )
    current = await query_bars(services, symbol="FSLR", start=_when("2026-09-15"))

    assert [b.timestamp.date().day for b in old.bars] == [15, 16, 17, 18]
    assert [b.timestamp.date().day for b in current.bars] == [15, 16, 17, 18, 21]
    assert (old.adjustment_revision, current.adjustment_revision) == (76, 77)
    assert old.pinned_silver_revision == 76 and current.pinned_silver_revision is None
    with pytest.raises(LakeError) as caught:
        await query_bars(services, symbol="FSLR", silver_revision_pin=12)
    assert caught.value.code == "unknown_revision"
