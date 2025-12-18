"""
Unit tests for AdminFeeModel and its integration in BacktestEngine.
"""

import pytest
from datetime import datetime, date, timedelta

from src.domain.clock import SimulatedClock
from src.domain.strategy.base import Strategy, StrategyContext
from src.domain.events.domain_events import QuoteTick
from src.infrastructure.backtest.backtest_engine import BacktestEngine, BacktestConfig
from src.infrastructure.backtest.data_feeds import InMemoryDataFeed
from src.domain.reality import RealityModelPack, create_zero_cost_pack, ConstantAdminFeeModel


class TestAdminFeeModel:
    """Tests for AdminFeeModel classes."""

    def test_constant_admin_fee_calculation(self):
        """Test mgmt fee and margin interest calculation."""
        # 2% mgmt fee, 5% margin interest
        model = ConstantAdminFeeModel(mgmt_fee_annual_pct=2.0, margin_interest_annual_pct=5.0)
        
        timestamp = datetime(2024, 1, 1)
        nav = 100000.0
        cash = -10000.0  # Margin loan
        pos_value = 110000.0
        
        fees = model.calculate_daily_fees(timestamp, cash, pos_value, nav)
        
        assert len(fees) == 2
        
        # Mgmt fee: 100,000 * 0.02 / 365 = 5.479
        mgmt_fee = next(f for f in fees if f.fee_type == "mgmt_fee")
        assert pytest.approx(mgmt_fee.amount, 0.001) == 100000.0 * 0.02 / 365.0
        
        # Margin interest: 10,000 * 0.05 / 365 = 1.369
        margin_fee = next(f for f in fees if f.fee_type == "margin_interest")
        assert pytest.approx(margin_fee.amount, 0.001) == 10000.0 * 0.05 / 365.0


class TestBacktestEngineAdminFees:
    """Tests for admin fee integration in BacktestEngine."""

    @pytest.mark.asyncio
    async def test_engine_accrues_admin_fees(self):
        """Test that engine actually deducts fees daily."""
        # 10% mgmt fee (high for testing visibility)
        reality_pack = create_zero_cost_pack()
        reality_pack.admin_fee_model = ConstantAdminFeeModel(mgmt_fee_annual_pct=10.0)
        reality_pack.name = "test_pack"

        config = BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            symbols=["AAPL"],
            initial_capital=100000.0,
            reality_pack=reality_pack
        )

        engine = BacktestEngine(config)
        
        # Use buy and hold strategy
        from src.domain.strategy.examples import BuyAndHoldStrategy
        engine.set_strategy(strategy_name="buy_and_hold")

        # Create data for 3 days
        feed = InMemoryDataFeed()
        for i in range(3):
            feed.add_bar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 1 + i, 16, 0),
                open=150.0,
                high=150.0,
                low=150.0,
                close=150.0,
                volume=1000,
            )
        engine.set_data_feed(feed)

        result = await engine.run()

        # Day 1: 100,000 * 0.10 / 365 = 27.397
        # Day 2: (100,000 - 27.397) * 0.10 / 365 = 27.389
        # Total approx 54.78
        assert result.costs.total_admin_fees > 0
        assert pytest.approx(result.costs.total_admin_fees, 0.1) == (100000.0 * 0.10 / 365.0) * 2
        
        # Final capital should be initial - fees
        assert result.final_capital < result.initial_capital
        assert pytest.approx(result.final_capital, 0.1) == 100000.0 - result.costs.total_admin_fees
