"""
Tests for OPT-013: Incremental RiskEngine Rebuild.

Tests the incremental rebuild optimization that only recalculates
positions affected by changed underlyings.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.domain.services.risk.risk_engine import (
    CachedRiskState,
    PositionMetrics,
    RiskEngine,
)
from src.models.account import AccountInfo
from src.models.market_data import MarketData
from src.models.position import AssetType, Position
from src.models.position_risk import PositionRisk


@pytest.fixture
def config():
    """Create test config."""
    return {
        "risk_engine": {
            "incremental_rebuild": True,
            "near_term_gamma_dte": 7,
            "near_term_vega_dte": 30,
        }
    }


@pytest.fixture
def engine(config):
    """Create a test RiskEngine."""
    return RiskEngine(config=config)


@pytest.fixture
def account_info():
    """Create test account info."""
    return AccountInfo(
        net_liquidation=100000.0,
        total_cash=30000.0,
        buying_power=50000.0,
        margin_used=25000.0,
        margin_available=25000.0,
        maintenance_margin=20000.0,
        init_margin_req=25000.0,
        excess_liquidity=30000.0,
    )


@pytest.fixture
def positions():
    """Create test positions."""
    return [
        Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            asset_type=AssetType.STOCK,
            underlying="AAPL",
            multiplier=1,
            source="test",
        ),
        Position(
            symbol="GOOG",
            quantity=50,
            avg_price=100.0,
            asset_type=AssetType.STOCK,
            underlying="GOOG",
            multiplier=1,
            source="test",
        ),
        Position(
            symbol="AAPL_OPT",
            quantity=10,
            avg_price=5.0,
            asset_type=AssetType.OPTION,
            underlying="AAPL",
            multiplier=100,
            expiry="2025-02-01",
            strike=155.0,
            right="C",
            source="test",
        ),
    ]


@pytest.fixture
def market_data():
    """Create test market data."""
    from src.utils.timezone import now_utc

    return {
        "AAPL": MarketData(
            symbol="AAPL",
            bid=151.0,
            ask=151.10,
            mid=151.05,
            last=151.05,
            timestamp=now_utc(),
        ),
        "GOOG": MarketData(
            symbol="GOOG",
            bid=101.0,
            ask=101.10,
            mid=101.05,
            last=101.05,
            timestamp=now_utc(),
        ),
        "AAPL_OPT": MarketData(
            symbol="AAPL_OPT",
            bid=5.50,
            ask=5.60,
            mid=5.55,
            last=5.55,
            delta=0.55,
            gamma=0.05,
            vega=0.15,
            theta=-0.02,
            timestamp=now_utc(),
        ),
    }


class TestCachedRiskState:
    """Tests for CachedRiskState dataclass."""

    def test_cached_risk_state_creation(self):
        """Test CachedRiskState can be created."""
        state = CachedRiskState(
            position_metrics={},
            position_risks={},
            positions_by_underlying={},
            portfolio_delta=100.0,
            portfolio_gamma=10.0,
            portfolio_vega=50.0,
            portfolio_theta=-5.0,
            total_gross_notional=50000.0,
            total_net_notional=40000.0,
            total_unrealized_pnl=1000.0,
            total_daily_pnl=500.0,
            gamma_notional_near_term=1000.0,
            vega_notional_near_term=500.0,
            tracked_symbols={"AAPL", "GOOG"},
        )

        assert state.portfolio_delta == 100.0
        assert len(state.tracked_symbols) == 2


class TestIncrementalRebuild:
    """Tests for incremental rebuild functionality."""

    def test_first_build_is_full(self, engine, positions, market_data, account_info):
        """Test first build is always a full rebuild."""
        assert engine._cached_state is None

        snapshot = engine.build_snapshot(positions, market_data, account_info)

        assert engine._cached_state is not None
        assert len(snapshot.position_risks) == 3
        assert "AAPL" in engine._cached_state.tracked_symbols
        assert "GOOG" in engine._cached_state.tracked_symbols
        assert "AAPL_OPT" in engine._cached_state.tracked_symbols

    def test_positions_by_underlying_tracked(self, engine, positions, market_data, account_info):
        """Test positions_by_underlying mapping is created."""
        engine.build_snapshot(positions, market_data, account_info)

        mapping = engine._cached_state.positions_by_underlying
        assert "AAPL" in mapping
        assert "GOOG" in mapping
        assert "AAPL" in mapping["AAPL"]
        assert "AAPL_OPT" in mapping["AAPL"]
        assert "GOOG" in mapping["GOOG"]

    def test_incremental_rebuild_on_market_tick(self, engine, positions, market_data, account_info):
        """Test incremental rebuild only recalculates affected positions."""
        from src.domain.events.domain_events import MarketDataTickEvent

        # First full build
        engine.build_snapshot(positions, market_data, account_info)

        # Simulate market tick for AAPL (uses proper event type)
        tick_event = MarketDataTickEvent(symbol="AAPL", bid=151.0, ask=151.10)
        engine._on_market_tick(tick_event)

        # Track which positions are recalculated
        original_calc = engine._calculate_position_metrics
        recalculated = []

        def track_calc(pos, md, status):
            recalculated.append(pos.symbol)
            return original_calc(pos, md, status)

        engine._calculate_position_metrics = track_calc

        # Second build should be incremental
        market_data["AAPL"].last = 152.0  # Price changed
        snapshot = engine.build_snapshot(positions, market_data, account_info)

        # Only AAPL positions should be recalculated
        assert "AAPL" in recalculated
        assert "AAPL_OPT" in recalculated
        assert "GOOG" not in recalculated

    def test_full_rebuild_on_position_change(self, engine, positions, market_data, account_info):
        """Test full rebuild when positions are added or removed."""
        from src.domain.events.domain_events import MarketDataTickEvent

        # First full build
        engine.build_snapshot(positions, market_data, account_info)

        # Simulate market tick
        tick_event = MarketDataTickEvent(symbol="AAPL", bid=151.0, ask=151.10)
        engine._on_market_tick(tick_event)

        # Add new position
        new_position = Position(
            symbol="MSFT",
            quantity=25,
            avg_price=300.0,
            asset_type=AssetType.STOCK,
            underlying="MSFT",
            multiplier=1,
            source="test",
        )
        market_data["MSFT"] = MarketData(symbol="MSFT", bid=301.0, ask=301.10, mid=301.05)
        positions_with_new = positions + [new_position]

        # Track calculation
        original_calc = engine._calculate_position_metrics
        recalculated = []

        def track_calc(pos, md, status):
            recalculated.append(pos.symbol)
            return original_calc(pos, md, status)

        engine._calculate_position_metrics = track_calc

        # Should trigger full rebuild
        snapshot = engine.build_snapshot(positions_with_new, market_data, account_info)

        # All positions should be recalculated
        assert len(recalculated) == 4
        assert "MSFT" in recalculated

    def test_full_rebuild_on_needs_rebuild_flag(self, engine, positions, market_data, account_info):
        """Test full rebuild when _needs_rebuild is set."""
        # First full build
        engine.build_snapshot(positions, market_data, account_info)

        # Trigger batch update (sets _needs_rebuild)
        engine._on_data_changed({"source": "test"})

        # Track calculation
        original_calc = engine._calculate_position_metrics
        recalculated = []

        def track_calc(pos, md, status):
            recalculated.append(pos.symbol)
            return original_calc(pos, md, status)

        engine._calculate_position_metrics = track_calc

        # Should trigger full rebuild
        snapshot = engine.build_snapshot(positions, market_data, account_info)

        # All positions should be recalculated
        assert len(recalculated) == 3

    def test_incremental_rebuild_aggregates_update(self, engine, positions, market_data, account_info):
        """Test incremental rebuild correctly updates aggregate Greeks."""
        # First full build
        snapshot1 = engine.build_snapshot(positions, market_data, account_info)
        original_delta = snapshot1.portfolio_delta

        # Simulate market tick for AAPL
        engine._on_market_tick(MagicMock(symbol="AAPL"))

        # Significant price change for AAPL
        market_data["AAPL"].last = 200.0
        market_data["AAPL"].mid = 200.0

        # Incremental rebuild
        snapshot2 = engine.build_snapshot(positions, market_data, account_info)

        # Delta should have changed due to notional change
        # The stock delta contribution is quantity * multiplier
        # AAPL: 100 * 1 = 100 (unchanged for stocks)
        # But P&L and notional should change
        assert snapshot2.total_unrealized_pnl != snapshot1.total_unrealized_pnl

    def test_invalidate_cache_forces_full_rebuild(self, engine, positions, market_data, account_info):
        """Test invalidate_cache forces full rebuild."""
        # First full build
        engine.build_snapshot(positions, market_data, account_info)
        assert engine._cached_state is not None

        # Invalidate cache
        engine.invalidate_cache()
        assert engine._cached_state is None
        assert engine._needs_rebuild is True

        # Next build should be full
        engine.build_snapshot(positions, market_data, account_info)
        assert engine._cached_state is not None

    def test_can_incremental_rebuild_checks_symbols(self, engine, positions, market_data, account_info):
        """Test _can_incremental_rebuild checks position symbols."""
        # No cache - should return False
        assert engine._can_incremental_rebuild(positions) is False

        # First full build
        engine.build_snapshot(positions, market_data, account_info)

        # Same positions - should return True
        assert engine._can_incremental_rebuild(positions) is True

        # Different positions - should return False
        different_positions = [positions[0]]  # Only first position
        assert engine._can_incremental_rebuild(different_positions) is False

    def test_incremental_rebuild_disabled_by_config(self):
        """Test incremental rebuild can be disabled via config."""
        config = {"risk_engine": {"incremental_rebuild": False}}
        engine = RiskEngine(config=config)

        assert engine._incremental_rebuild_enabled is False

    def test_dirty_underlyings_cleared_after_rebuild(self, engine, positions, market_data, account_info):
        """Test dirty underlyings are cleared after rebuild."""
        from src.domain.events.domain_events import MarketDataTickEvent

        tick_event = MarketDataTickEvent(symbol="AAPL", bid=151.0, ask=151.10)
        engine._on_market_tick(tick_event)
        assert len(engine._dirty_underlyings) == 1

        engine.build_snapshot(positions, market_data, account_info)

        # Dirty underlyings should be cleared
        assert len(engine._dirty_underlyings) == 0


class TestMarketTickHandling:
    """Tests for market tick handling."""

    def test_market_tick_tracks_underlying(self, engine):
        """Test market tick adds underlying to dirty set."""
        from src.domain.events.domain_events import MarketDataTickEvent

        event = MarketDataTickEvent(
            symbol="AAPL",
            bid=150.0,
            ask=150.10,
        )

        engine._on_market_tick(event)

        assert "AAPL" in engine._dirty_underlyings

    def test_market_tick_extracts_underlying_from_option(self, engine):
        """Test market tick extracts underlying from option symbol."""
        from src.domain.events.domain_events import MarketDataTickEvent

        # Option symbol format: "AAPL  240119C..." - uses legacy dict handling
        # MarketDataTickEvent with option symbol
        event = MarketDataTickEvent(
            symbol="AAPL  240119C00150000",
            bid=5.0,
            ask=5.10,
        )

        engine._on_market_tick(event)

        # Should extract "AAPL" as underlying from option symbol
        assert "AAPL" in engine._dirty_underlyings

    def test_multiple_ticks_same_underlying(self, engine):
        """Test multiple ticks for same underlying only add once."""
        from src.domain.events.domain_events import MarketDataTickEvent

        event1 = MarketDataTickEvent(symbol="AAPL", bid=150.0, ask=150.10)
        event2 = MarketDataTickEvent(symbol="AAPL", bid=150.5, ask=150.60)

        engine._on_market_tick(event1)
        engine._on_market_tick(event2)

        assert len(engine._dirty_underlyings) == 1
        assert "AAPL" in engine._dirty_underlyings


class TestPerformanceCharacteristics:
    """Tests verifying performance characteristics of incremental rebuild."""

    def test_incremental_rebuild_recalculates_fewer_positions(
        self, engine, account_info
    ):
        """Test incremental rebuild calculates fewer positions than full."""
        from src.domain.events.domain_events import MarketDataTickEvent
        from src.utils.timezone import now_utc

        # Create 20 positions across 5 underlyings
        positions = []
        market_data = {}
        for i in range(20):
            underlying = f"SYM{i % 5}"
            symbol = f"SYM{i}"
            positions.append(
                Position(
                    symbol=symbol,
                    quantity=100,
                    avg_price=100.0,
                    asset_type=AssetType.STOCK,
                    underlying=underlying,
                    multiplier=1,
                    source="test",
                )
            )
            market_data[symbol] = MarketData(
                symbol=symbol, bid=100.0, ask=100.10, mid=100.05, timestamp=now_utc()
            )

        # First full build
        engine.build_snapshot(positions, market_data, account_info)

        # Simulate tick for one underlying
        tick_event = MarketDataTickEvent(symbol="SYM0", bid=100.0, ask=100.10)
        engine._on_market_tick(tick_event)

        # Track recalculations
        calc_count = [0]
        original_calc = engine._calculate_position_metrics

        def count_calc(pos, md, status):
            calc_count[0] += 1
            return original_calc(pos, md, status)

        engine._calculate_position_metrics = count_calc

        # Incremental rebuild
        engine.build_snapshot(positions, market_data, account_info)

        # Should only recalculate 4 positions (SYM0, SYM5, SYM10, SYM15)
        # All have underlying "SYM0"
        assert calc_count[0] == 4
