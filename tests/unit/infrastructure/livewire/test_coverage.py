"""Discovery reads livewire's coverage snapshot, not a filesystem walk.

A recursive scan of ~14.7K symbol dirs on the external volume is not viable
per-request -- measured 2026-08-23 on the production lake, one scandir of
bronze/asset_class=equity takes 5.5s and descending into symbols to read date
ranges costs 78ms each (~19 minutes for the full set). The lake also carries
macOS AppleDouble siblings (._symbol=DXY) that break a naive listdir join.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure.adapters.livewire.coverage import CoverageCatalog, CoverageUnavailable


def test_lists_every_asset_class(catalog_db: Path) -> None:
    rows = CoverageCatalog(catalog_db).list_instruments()
    assert {r.asset_class for r in rows} == {"equity", "volatility", "rates"}


def test_silver_available_flags_the_symbols_that_would_503(catalog_db: Path) -> None:
    """HON has bronze but no silver -- adjusted mode 503s on it. Consumers must be
    able to learn that up front instead of by making the request."""
    by_symbol = {r.symbol: r for r in CoverageCatalog(catalog_db).list_instruments()}
    assert by_symbol["AAPL"].silver_available is True
    assert by_symbol["HON"].silver_available is False


def test_non_equity_is_always_raw(catalog_db: Path) -> None:
    by_symbol = {r.symbol: r for r in CoverageCatalog(catalog_db).list_instruments()}
    assert by_symbol["VIX"].price_mode == "raw"


def test_filter_by_asset_class(catalog_db: Path) -> None:
    rows = CoverageCatalog(catalog_db).list_instruments(asset_class="rates")
    assert [r.symbol for r in rows] == ["DGS10"]


def test_query_filters_by_symbol_prefix(catalog_db: Path) -> None:
    rows = CoverageCatalog(catalog_db).list_instruments(query="AAP")
    assert [r.symbol for r in rows] == ["AAPL"]


def test_query_escapes_like_wildcards(catalog_db: Path) -> None:
    """'_' is a LIKE wildcard and futures symbols contain it (BZ_202609).
    Verified against the production catalog: LIKE 'BRK_B' matches 'BRK.B'."""
    assert CoverageCatalog(catalog_db).list_instruments(query="A_PL") == []


def test_missing_database_raises_rather_than_reporting_an_empty_universe() -> None:
    """An absent catalog must NOT look like "apex holds nothing".

    Verified on the mini 2026-08-23: binding a catalog path that is outside colima's
    VM mount set produces an empty directory in the container, not the database. If
    that degraded to an empty list, a broken deployment would be indistinguishable
    from a genuinely empty lake -- and a consumer would cache the wrong answer.
    """
    with pytest.raises(CoverageUnavailable):
        CoverageCatalog(Path("/nonexistent/nope.duckdb")).list_instruments()


def test_a_directory_at_the_db_path_raises(tmp_path: Path) -> None:
    """This is the EXACT production failure mode: Docker fabricates a directory when
    the bind source is unreachable, and Path.exists() returns True for a directory."""
    fake = tmp_path / "analytics.duckdb"
    fake.mkdir()
    with pytest.raises(CoverageUnavailable):
        CoverageCatalog(fake).list_instruments()


def test_empty_but_valid_catalog_is_not_an_error(tmp_path: Path) -> None:
    """A real catalog with no matching rows is a legitimate empty result."""
    import duckdb

    db = tmp_path / "analytics.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE coverage (view_name VARCHAR, symbol VARCHAR, n_rows BIGINT, "
        "first_date DATE, last_date DATE)"
    )
    con.close()
    assert CoverageCatalog(db).list_instruments() == []


def test_opens_read_only(catalog_db: Path) -> None:
    """The catalog is mounted :ro in production; a read-write connect would fail."""
    import os
    import stat

    os.chmod(catalog_db, stat.S_IRUSR)
    try:
        assert CoverageCatalog(catalog_db).list_instruments()
    finally:
        os.chmod(catalog_db, stat.S_IRUSR | stat.S_IWUSR)
