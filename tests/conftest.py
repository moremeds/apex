"""Pytest configuration and fixtures."""

from datetime import datetime
from pathlib import Path
from typing import Dict

import pytest

from src.models.account import AccountInfo
from src.models.market_data import DataQuality, GreeksSource, MarketData
from src.models.position import AssetType, Position, PositionSource


@pytest.fixture
def sample_stock_position() -> Position:
    """Sample stock position fixture."""
    return Position(
        symbol="AAPL",
        underlying="AAPL",
        asset_type=AssetType.STOCK,
        quantity=100,
        avg_price=175.0,
        multiplier=1,
        source=PositionSource.IB,
    )


@pytest.fixture
def sample_option_position() -> Position:
    """Sample option position fixture."""
    return Position(
        symbol="AAPL 240315C180",
        underlying="AAPL",
        asset_type=AssetType.OPTION,
        quantity=10,
        avg_price=5.5,
        multiplier=100,
        expiry="20240315",
        strike=180.0,
        right="C",
        source=PositionSource.IB,
    )


@pytest.fixture
def sample_market_data() -> MarketData:
    """Sample market data fixture."""
    return MarketData(
        symbol="AAPL",
        last=176.0,
        bid=175.98,
        ask=176.02,
        mid=176.0,
        delta=1.0,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        timestamp=datetime.now(),
        greeks_source=GreeksSource.IBKR,
        quality=DataQuality.GOOD,
    )


@pytest.fixture
def sample_option_market_data() -> MarketData:
    """Sample option market data fixture."""
    return MarketData(
        symbol="AAPL 240315C180",
        last=5.5,
        bid=5.45,
        ask=5.55,
        mid=5.5,
        iv=0.25,
        delta=0.55,
        gamma=0.025,
        vega=12.5,
        theta=-15.0,
        timestamp=datetime.now(),
        greeks_source=GreeksSource.IBKR,
        quality=DataQuality.GOOD,
    )


@pytest.fixture
def sample_account_info() -> AccountInfo:
    """Sample account info fixture."""
    return AccountInfo(
        net_liquidation=1_000_000.0,
        total_cash=500_000.0,
        buying_power=2_000_000.0,
        margin_used=100_000.0,
        margin_available=1_900_000.0,
        maintenance_margin=80_000.0,
        init_margin_req=100_000.0,
        excess_liquidity=900_000.0,
        realized_pnl=10_000.0,
        unrealized_pnl=25_000.0,
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_risk_config() -> Dict:
    """Sample risk configuration fixture."""
    return {
        "risk_limits": {
            "max_total_gross_notional": 5_000_000,
            "max_per_underlying_notional": {
                "default": 1_000_000,
                "AAPL": 1_500_000,
            },
            "portfolio_delta_range": [-50_000, 50_000],
            "portfolio_vega_range": [-15_000, 15_000],
            "portfolio_theta_range": [-5_000, 5_000],
            "max_margin_utilization": 0.60,
            "max_concentration_pct": 0.30,
            "soft_breach_threshold": 0.80,
        }
    }


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
