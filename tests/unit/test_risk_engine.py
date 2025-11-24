"""Unit tests for RiskEngine."""

import pytest
from datetime import datetime

from src.domain.services.risk_engine import RiskEngine
from src.models.position import Position
from src.models.market_data import MarketData
from src.models.account import AccountInfo


def test_build_snapshot_empty(sample_account_info, sample_risk_config):
    """Test building snapshot with no positions."""
    engine = RiskEngine(config=sample_risk_config)
    snapshot = engine.build_snapshot([], {}, sample_account_info)

    assert snapshot.total_positions == 0
    assert snapshot.portfolio_delta == 0.0
    assert snapshot.total_gross_notional == 0.0


def test_build_snapshot_stock_position(
    sample_stock_position,
    sample_market_data,
    sample_account_info,
    sample_risk_config,
):
    """Test building snapshot with single stock position."""
    engine = RiskEngine(config=sample_risk_config)

    positions = [sample_stock_position]
    market_data = {sample_stock_position.symbol: sample_market_data}

    snapshot = engine.build_snapshot(positions, market_data, sample_account_info)

    assert snapshot.total_positions == 1
    assert snapshot.portfolio_delta == 100.0  # 100 shares * delta 1.0
    assert snapshot.total_gross_notional == 17_600.0  # 100 * 176


def test_build_snapshot_missing_market_data(
    sample_stock_position,
    sample_account_info,
    sample_risk_config,
):
    """Test handling of missing market data."""
    engine = RiskEngine(config=sample_risk_config)

    positions = [sample_stock_position]
    market_data = {}  # No market data

    snapshot = engine.build_snapshot(positions, market_data, sample_account_info)

    assert snapshot.total_positions == 1
    assert snapshot.positions_with_missing_md == 1
    assert snapshot.portfolio_delta == 0.0


def test_expiry_bucket_classification(
    sample_option_position,
    sample_option_market_data,
    sample_account_info,
    sample_risk_config,
):
    """Test expiry bucket classification."""
    engine = RiskEngine(config=sample_risk_config)

    positions = [sample_option_position]
    market_data = {sample_option_position.symbol: sample_option_market_data}

    snapshot = engine.build_snapshot(positions, market_data, sample_account_info)

    # Check that position was classified into a bucket
    assert any(bucket["count"] > 0 for bucket in snapshot.expiry_buckets.values())


# TODO: Add more unit tests:
# - Test option position Greeks aggregation
# - Test concentration metrics calculation
# - Test P&L calculation accuracy
# - Test gamma/vega notional calculations
# - Test by-underlying aggregation
