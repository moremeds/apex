"""
Unit tests for account info fetching from IBKR.

Tests the mapping of IBKR account summary tags to AccountInfo model.
"""

from datetime import datetime

import pytest

from src.models.account import AccountInfo


def test_account_info_model():
    """Test AccountInfo model initialization and calculations."""
    account = AccountInfo(
        net_liquidation=100000.0,
        total_cash=50000.0,
        buying_power=200000.0,
        margin_used=25000.0,
        margin_available=75000.0,
        maintenance_margin=20000.0,
        init_margin_req=25000.0,
        excess_liquidity=75000.0,
        realized_pnl=5000.0,
        unrealized_pnl=3000.0,
        timestamp=datetime.now(),
        account_id="U12345678",
    )

    # Test margin utilization calculation
    expected_utilization = 25000.0 / 200000.0  # margin_used / buying_power
    assert account.margin_utilization() == pytest.approx(expected_utilization)

    # Verify all fields are set
    assert account.net_liquidation == 100000.0
    assert account.total_cash == 50000.0
    assert account.buying_power == 200000.0
    assert account.margin_used == 25000.0
    assert account.margin_available == 75000.0
    assert account.realized_pnl == 5000.0
    assert account.unrealized_pnl == 3000.0
    assert account.account_id == "U12345678"


def test_margin_utilization_zero_buying_power():
    """Test margin utilization calculation when buying power is zero."""
    account = AccountInfo(
        net_liquidation=100000.0,
        total_cash=50000.0,
        buying_power=0.0,  # Zero buying power
        margin_used=25000.0,
        margin_available=75000.0,
        maintenance_margin=20000.0,
        init_margin_req=25000.0,
        excess_liquidity=75000.0,
    )

    # Should return 0 to avoid division by zero
    assert account.margin_utilization() == 0.0


def test_account_info_defaults():
    """Test AccountInfo with default values."""
    account = AccountInfo(
        net_liquidation=100000.0,
        total_cash=50000.0,
        buying_power=200000.0,
        margin_used=25000.0,
        margin_available=75000.0,
        maintenance_margin=20000.0,
        init_margin_req=25000.0,
        excess_liquidity=75000.0,
    )

    # Check defaults
    assert account.realized_pnl == 0.0
    assert account.unrealized_pnl == 0.0
    assert account.timestamp is None
    assert account.account_id is None


# IBKR Account Summary Tag Reference
"""
Common IBKR Account Summary Tags:

Balance & Cash:
- NetLiquidation: Total account value (net liquidation value)
- TotalCashValue: Cash balance
- AvailableFunds: Funds available for trading
- ExcessLiquidity: Excess liquidity (cushion above maintenance margin)
- BuyingPower: Available buying power (leverage included)

Margin:
- MaintMarginReq: Maintenance margin requirement
- InitMarginReq: Initial margin requirement
- FullInitMarginReq: Full initial margin requirement
- FullMaintMarginReq: Full maintenance margin requirement

Positions:
- GrossPositionValue: Total value of all positions
- StockMarketValue: Value of stock positions
- OptionMarketValue: Value of option positions

P&L:
- RealizedPnL: Realized profit/loss (daily)
- UnrealizedPnL: Unrealized profit/loss (mark-to-market)
- DailyPnL: Combined daily P&L

Currency:
- Currency: Account currency (e.g., "USD")
- BaseCurrency: Base currency

Note: All monetary values are returned as strings and need to be converted to float.
"""
