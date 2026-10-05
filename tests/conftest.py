"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture()
def catalog_db(tmp_path: Path) -> Path:
    """Mirror livewire's real coverage table shape, with real measured rows
    (frozen 2026-08-23 from ~/market-warehouse/analytics.duckdb).

    Verified against the production catalog: the table is exactly
    (view_name VARCHAR, symbol VARCHAR, n_rows BIGINT, first_date DATE, last_date DATE).
    """
    import duckdb

    db = tmp_path / "analytics.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE coverage ("
        "view_name VARCHAR, symbol VARCHAR, n_rows BIGINT, "
        "first_date DATE, last_date DATE)"
    )
    con.executemany(
        "INSERT INTO coverage VALUES (?, ?, ?, ?, ?)",
        [
            ("bronze_equity_1d", "AAPL", 11514, "1980-12-12", "2026-08-21"),
            ("silver_equity_1d", "AAPL", 11514, "1980-12-12", "2026-08-21"),
            ("bronze_equity_1d", "HON", 11000, "1980-01-02", "2026-08-21"),
            ("bronze_volatility_1d", "VIX", 9256, "1990-01-02", "2026-08-21"),
            ("bronze_rates_1d", "DGS10", 16144, "1962-01-02", "2026-08-20"),
        ],
    )
    con.close()
    return db
