"""Unit tests for PortfolioState."""

import threading
import time
from datetime import datetime

import pytest

from src.domain.services.risk.state.portfolio_state import (
    PortfolioState,
)
from src.domain.services.risk.state.position_state import (
    PositionDelta,
    PositionState,
)


class TestPortfolioState:
    """Tests for PortfolioState class."""

    def _create_test_position(
        self,
        symbol: str = "AAPL",
        underlying: str = "AAPL",
        quantity: float = 100,
        unrealized_pnl: float = 500.0,
        daily_pnl: float = 100.0,
        delta: float = 100.0,
        notional: float = 15500.0,
        is_reliable: bool = True,
        has_greeks: bool = False,
    ) -> PositionState:
        """Helper to create test PositionState."""
        return PositionState(
            symbol=symbol,
            underlying=underlying,
            quantity=quantity,
            multiplier=1,
            avg_cost=150.0,
            mark_price=155.0,
            yesterday_close=154.0,
            session_open=153.0,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=daily_pnl,
            delta=delta,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            notional=notional,
            delta_dollars=delta * 155.0,  # delta * underlying_price
            underlying_price=155.0,
            is_reliable=is_reliable,
            has_greeks=has_greeks,
            last_update=datetime.now(),
        )

    def test_add_position(self) -> None:
        """add_position() should update aggregates."""
        state = PortfolioState()
        position = self._create_test_position()

        state.add_position(position)

        assert state.position_count == 1
        assert "AAPL" in state.symbols

        # Check aggregates were updated
        snapshot = state.to_snapshot()
        assert snapshot.total_unrealized_pnl == 500.0
        assert snapshot.total_daily_pnl == 100.0
        assert snapshot.portfolio_delta == 100.0
        assert snapshot.total_gross_notional == 15500.0

    def test_add_multiple_positions(self) -> None:
        """Multiple positions should aggregate correctly."""
        state = PortfolioState()
        state.add_position(self._create_test_position("AAPL", "AAPL"))
        state.add_position(self._create_test_position("TSLA", "TSLA", unrealized_pnl=300.0))

        assert state.position_count == 2

        snapshot = state.to_snapshot()
        assert snapshot.total_unrealized_pnl == 800.0  # 500 + 300

    def test_remove_position(self) -> None:
        """remove_position() should reverse aggregates."""
        state = PortfolioState()
        position = self._create_test_position()
        state.add_position(position)

        removed = state.remove_position("AAPL")

        assert removed is not None
        assert removed.symbol == "AAPL"
        assert state.position_count == 0

        snapshot = state.to_snapshot()
        assert snapshot.total_unrealized_pnl == 0.0

    def test_remove_nonexistent_position(self) -> None:
        """remove_position() should return None for unknown symbol."""
        state = PortfolioState()

        removed = state.remove_position("UNKNOWN")

        assert removed is None

    def test_get_position(self) -> None:
        """get_position() should return the position state."""
        state = PortfolioState()
        position = self._create_test_position()
        state.add_position(position)

        retrieved = state.get_position("AAPL")

        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.unrealized_pnl == 500.0

    def test_get_nonexistent_position(self) -> None:
        """get_position() should return None for unknown symbol."""
        state = PortfolioState()

        retrieved = state.get_position("UNKNOWN")

        assert retrieved is None

    def test_apply_delta(self) -> None:
        """apply_delta() should update position and aggregates."""
        state = PortfolioState()
        position = self._create_test_position()
        state.add_position(position)

        delta = PositionDelta(
            symbol="AAPL",
            underlying="AAPL",
            timestamp=datetime.now(),
            new_mark_price=156.0,
            pnl_change=100.0,
            daily_pnl_change=50.0,
            delta_change=0.0,
            gamma_change=0.0,
            vega_change=0.0,
            theta_change=0.0,
            notional_change=100.0,
            delta_dollars_change=0.0,
            underlying_price=156.0,
            is_reliable=True,
            has_greeks=False,
        )

        state.apply_delta(delta)

        # Check position was updated
        updated = state.get_position("AAPL")
        assert updated is not None
        assert updated.unrealized_pnl == 600.0  # 500 + 100

        # Check aggregates were updated
        snapshot = state.to_snapshot()
        assert snapshot.total_unrealized_pnl == 600.0

    def test_apply_delta_unknown_position(self) -> None:
        """apply_delta() should ignore unknown symbols."""
        state = PortfolioState()

        delta = PositionDelta(
            symbol="UNKNOWN",
            underlying="UNKNOWN",
            timestamp=datetime.now(),
            new_mark_price=100.0,
            pnl_change=50.0,
            daily_pnl_change=10.0,
            delta_change=0.0,
            gamma_change=0.0,
            vega_change=0.0,
            theta_change=0.0,
            notional_change=0.0,
            delta_dollars_change=0.0,
            underlying_price=100.0,
            is_reliable=True,
            has_greeks=False,
        )

        # Should not raise
        state.apply_delta(delta)
        assert state.position_count == 0

    def test_apply_delta_updates_reliability_counts(self) -> None:
        """apply_delta() should update reliability counts when state changes."""
        state = PortfolioState()
        # Start with unreliable position
        position = self._create_test_position(is_reliable=False, has_greeks=False)
        state.add_position(position)

        snapshot = state.to_snapshot()
        assert snapshot.positions_with_missing_md == 1
        assert snapshot.missing_greeks_count == 1

        # Update to reliable with greeks
        delta = PositionDelta(
            symbol="AAPL",
            underlying="AAPL",
            timestamp=datetime.now(),
            new_mark_price=156.0,
            pnl_change=0.0,
            daily_pnl_change=0.0,
            delta_change=0.0,
            gamma_change=0.0,
            vega_change=0.0,
            theta_change=0.0,
            notional_change=0.0,
            delta_dollars_change=0.0,
            underlying_price=156.0,
            is_reliable=True,
            has_greeks=True,
        )

        state.apply_delta(delta)

        snapshot = state.to_snapshot()
        assert snapshot.positions_with_missing_md == 0
        assert snapshot.missing_greeks_count == 0

    def test_clear(self) -> None:
        """clear() should reset all state."""
        state = PortfolioState()
        state.add_position(self._create_test_position("AAPL", "AAPL"))
        state.add_position(self._create_test_position("TSLA", "TSLA"))

        state.clear()

        assert state.position_count == 0
        snapshot = state.to_snapshot()
        assert snapshot.total_unrealized_pnl == 0.0

    def test_to_snapshot_thread_safety(self) -> None:
        """to_snapshot() should be thread-safe."""
        state = PortfolioState()
        state.add_position(self._create_test_position())

        # Take snapshot while another thread updates
        errors = []

        def update_loop() -> None:
            try:
                for i in range(100):
                    delta = PositionDelta(
                        symbol="AAPL",
                        underlying="AAPL",
                        timestamp=datetime.now(),
                        new_mark_price=155.0 + i * 0.01,
                        pnl_change=1.0,
                        daily_pnl_change=0.5,
                        delta_change=0.0,
                        gamma_change=0.0,
                        vega_change=0.0,
                        theta_change=0.0,
                        notional_change=0.0,
                        delta_dollars_change=0.0,
                        underlying_price=155.0 + i * 0.01,
                        is_reliable=True,
                        has_greeks=False,
                    )
                    state.apply_delta(delta)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def snapshot_loop() -> None:
            try:
                for _ in range(100):
                    snapshot = state.to_snapshot()
                    # Verify snapshot is internally consistent
                    assert snapshot.total_positions == 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_loop),
            threading.Thread(target=snapshot_loop),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concentration_calculation(self) -> None:
        """to_snapshot() should calculate concentration correctly."""
        state = PortfolioState()
        # Add positions with different notionals
        state.add_position(self._create_test_position("AAPL", "AAPL", notional=25000.0))
        state.add_position(self._create_test_position("TSLA", "TSLA", notional=15000.0))
        state.add_position(self._create_test_position("SPY", "SPY", notional=10000.0))

        snapshot = state.to_snapshot()

        assert snapshot.max_underlying_symbol == "AAPL"
        assert snapshot.max_underlying_notional == 25000.0
        assert snapshot.concentration_pct == pytest.approx(0.5, rel=0.01)  # 25000/50000
