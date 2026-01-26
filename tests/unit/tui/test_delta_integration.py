"""Unit tests for TUI delta integration (Phase 3).

Tests the streaming delta flow from event bus to position table updates.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from src.domain.events.domain_events import PositionDeltaEvent
from src.tui.event_bus import TUIEventBus
from src.tui.viewmodels.position_vm import PositionViewModel


# Mock position for testing
@dataclass
class MockPosition:
    symbol: str
    underlying: str
    market_value: float
    daily_pnl: float
    unrealized_pnl: float
    delta: float
    gamma: float
    vega: float
    theta: float
    mark_price: float
    delta_dollars: float = 0.0
    beta: float = 1.0
    expiry: str = None
    quantity: int = 100
    source: str = "ib"


class TestTUIEventBusDelta:
    """Tests for TUIEventBus delta handling."""

    def test_push_delta_adds_to_buffer(self) -> None:
        """push_delta() should add delta to buffer."""
        bus = TUIEventBus()
        delta = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", new_mark_price=150.0)

        bus.push_delta(delta)

        with bus._delta_lock:
            assert "AAPL" in bus._delta_buffer
            assert bus._delta_buffer["AAPL"].new_mark_price == 150.0

    def test_push_delta_coalesces_by_symbol(self) -> None:
        """push_delta() should keep only latest delta per symbol."""
        bus = TUIEventBus()
        delta1 = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", new_mark_price=150.0)
        delta2 = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", new_mark_price=151.0)

        bus.push_delta(delta1)
        bus.push_delta(delta2)

        with bus._delta_lock:
            assert len(bus._delta_buffer) == 1
            assert bus._delta_buffer["AAPL"].new_mark_price == 151.0

    def test_push_delta_multiple_symbols(self) -> None:
        """push_delta() should track multiple symbols independently."""
        bus = TUIEventBus()
        delta1 = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", new_mark_price=150.0)
        delta2 = PositionDeltaEvent(symbol="TSLA", underlying="TSLA", new_mark_price=200.0)

        bus.push_delta(delta1)
        bus.push_delta(delta2)

        with bus._delta_lock:
            assert len(bus._delta_buffer) == 2

    def test_poll_returns_deltas(self) -> None:
        """poll() should return accumulated deltas."""
        bus = TUIEventBus()
        delta = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", new_mark_price=150.0)
        bus.push_delta(delta)

        result = bus.poll()

        assert len(result.deltas) == 1
        assert "AAPL" in result.deltas
        assert result.deltas["AAPL"].new_mark_price == 150.0

    def test_poll_clears_delta_buffer(self) -> None:
        """poll() should clear delta buffer after returning."""
        bus = TUIEventBus()
        delta = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", new_mark_price=150.0)
        bus.push_delta(delta)

        bus.poll()
        result2 = bus.poll()

        assert len(result2.deltas) == 0

    def test_poll_has_data_with_deltas(self) -> None:
        """has_data should be True when deltas present."""
        bus = TUIEventBus()
        delta = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", new_mark_price=150.0)
        bus.push_delta(delta)

        result = bus.poll()

        assert result.has_data is True

    def test_poll_has_data_without_deltas(self) -> None:
        """has_data should be False when no events present."""
        bus = TUIEventBus()

        result = bus.poll()

        assert result.has_data is False

    def test_queue_sizes_includes_deltas(self) -> None:
        """queue_sizes() should report delta buffer size."""
        bus = TUIEventBus()
        delta = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", new_mark_price=150.0)
        bus.push_delta(delta)

        sizes = bus.queue_sizes()

        assert sizes["deltas"] == 1


class TestPositionViewModelDelta:
    """Tests for PositionViewModel delta application."""

    @pytest.fixture
    def positions(self):
        """Create test positions."""
        return [
            MockPosition("AAPL", "AAPL", 15000, 100, 500, 100, 5, 10, -5, 150),
            MockPosition("TSLA", "TSLA", 20000, 200, 1000, 200, 10, 20, -10, 200),
        ]

    @pytest.fixture
    def consolidated_vm(self):
        """Create consolidated view model."""
        return PositionViewModel(consolidated=True, show_portfolio_row=True)

    @pytest.fixture
    def detailed_vm(self):
        """Create detailed view model."""
        return PositionViewModel(consolidated=False, show_portfolio_row=True, broker_filter="ib")

    def test_compute_display_data_populates_symbol_mapping(
        self, consolidated_vm: Any, positions: Any
    ) -> None:
        """compute_display_data() should populate symbol to row key mapping."""
        consolidated_vm.compute_display_data(positions)

        assert "AAPL" in consolidated_vm._symbol_to_row_keys
        assert "TSLA" in consolidated_vm._symbol_to_row_keys
        assert "__portfolio__" in consolidated_vm._symbol_to_row_keys["AAPL"]
        assert "underlying-AAPL" in consolidated_vm._symbol_to_row_keys["AAPL"]

    def test_compute_display_data_populates_values_cache(
        self, consolidated_vm: Any, positions: Any
    ) -> None:
        """compute_display_data() should populate values cache."""
        consolidated_vm.compute_display_data(positions)

        assert "__portfolio__" in consolidated_vm._values_cache
        assert "underlying-AAPL" in consolidated_vm._values_cache
        assert "underlying-TSLA" in consolidated_vm._values_cache

    def test_apply_deltas_returns_cell_updates(self, consolidated_vm: Any, positions: Any) -> None:
        """apply_deltas() should return cell updates."""
        consolidated_vm.compute_display_data(positions)
        delta = PositionDeltaEvent(
            symbol="AAPL",
            underlying="AAPL",
            new_mark_price=151.0,
            pnl_change=50.0,
            daily_pnl_change=25.0,
            delta_change=5.0,
            gamma_change=0.5,
            vega_change=1.0,
            theta_change=-0.5,
            notional_change=100.0,
        )

        cell_updates = consolidated_vm.apply_deltas({"AAPL": delta})

        # Should update portfolio row and AAPL underlying row
        assert len(cell_updates) > 0
        # Check that we have updates for both rows
        row_keys = {u.row_key for u in cell_updates}
        assert "__portfolio__" in row_keys
        assert "underlying-AAPL" in row_keys

    def test_apply_deltas_updates_values_cache(self, consolidated_vm: Any, positions: Any) -> None:
        """apply_deltas() should update cached values."""
        consolidated_vm.compute_display_data(positions)
        original_pnl = consolidated_vm._values_cache["underlying-AAPL"]["pnl"]

        delta = PositionDeltaEvent(
            symbol="AAPL",
            underlying="AAPL",
            new_mark_price=151.0,
            daily_pnl_change=25.0,
        )
        consolidated_vm.apply_deltas({"AAPL": delta})

        assert consolidated_vm._values_cache["underlying-AAPL"]["pnl"] == original_pnl + 25.0

    def test_apply_deltas_skips_unknown_symbols(self, consolidated_vm: Any, positions: Any) -> None:
        """apply_deltas() should skip symbols not in view."""
        consolidated_vm.compute_display_data(positions)

        delta = PositionDeltaEvent(
            symbol="UNKNOWN",
            underlying="UNKNOWN",
            new_mark_price=100.0,
        )
        cell_updates = consolidated_vm.apply_deltas({"UNKNOWN": delta})

        assert len(cell_updates) == 0

    def test_apply_deltas_detailed_view(self, detailed_vm: Any, positions: Any) -> None:
        """apply_deltas() should work with detailed view."""
        detailed_vm.compute_display_data(positions)

        delta = PositionDeltaEvent(
            symbol="AAPL",
            underlying="AAPL",
            new_mark_price=151.0,
            pnl_change=50.0,
            daily_pnl_change=25.0,
        )
        cell_updates = detailed_vm.apply_deltas({"AAPL": delta})

        # Should update total row, header row, and position row
        row_keys = {u.row_key for u in cell_updates}
        assert "__total__" in row_keys
        assert "header-AAPL" in row_keys

    def test_apply_deltas_position_row_includes_mark_price(
        self, detailed_vm: Any, positions: Any
    ) -> None:
        """apply_deltas() on position row should update mark price."""
        detailed_vm.compute_display_data(positions)

        delta = PositionDeltaEvent(
            symbol="AAPL",
            underlying="AAPL",
            new_mark_price=151.5,
            pnl_change=50.0,
        )
        cell_updates = detailed_vm.apply_deltas({"AAPL": delta})

        # Find position row updates (pos-AAPL-*)
        pos_updates = [u for u in cell_updates if u.row_key.startswith("pos-")]
        # Position row should have mark price update (column 2)
        spot_updates = [u for u in pos_updates if u.column_index == 2]
        assert len(spot_updates) > 0
        assert "151" in spot_updates[0].value

    def test_apply_deltas_consolidated_column_indices(
        self, consolidated_vm: Any, positions: Any
    ) -> None:
        """apply_deltas() should use correct column indices for consolidated view."""
        consolidated_vm.compute_display_data(positions)

        delta = PositionDeltaEvent(
            symbol="AAPL",
            underlying="AAPL",
            new_mark_price=151.0,
        )
        cell_updates = consolidated_vm.apply_deltas({"AAPL": delta})

        # Consolidated columns: 4=mkt_value, 5=pnl, 6=upnl, 8=delta, 9=gamma, 10=vega, 11=theta
        col_indices = {u.column_index for u in cell_updates}
        assert 4 in col_indices  # mkt_value
        assert 5 in col_indices  # pnl
        assert 6 in col_indices  # upnl

    def test_apply_deltas_detailed_column_indices(self, detailed_vm: Any, positions: Any) -> None:
        """apply_deltas() should use correct column indices for detailed view."""
        detailed_vm.compute_display_data(positions)

        delta = PositionDeltaEvent(
            symbol="AAPL",
            underlying="AAPL",
            new_mark_price=151.0,
        )
        cell_updates = detailed_vm.apply_deltas({"AAPL": delta})

        # Detailed columns: 5=mkt_value, 6=pnl, 7=upnl, 9=delta, 10=gamma, 11=vega, 12=theta
        col_indices = {u.column_index for u in cell_updates if u.row_key.startswith("header-")}
        assert 5 in col_indices  # mkt_value
        assert 6 in col_indices  # pnl
        assert 7 in col_indices  # upnl

    def test_multiple_deltas_accumulate(self, consolidated_vm: Any, positions: Any) -> None:
        """Multiple deltas for same symbol should accumulate."""
        consolidated_vm.compute_display_data(positions)
        original_pnl = consolidated_vm._values_cache["underlying-AAPL"]["pnl"]

        delta1 = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", daily_pnl_change=25.0)
        delta2 = PositionDeltaEvent(symbol="AAPL", underlying="AAPL", daily_pnl_change=30.0)

        consolidated_vm.apply_deltas({"AAPL": delta1})
        consolidated_vm.apply_deltas({"AAPL": delta2})

        assert consolidated_vm._values_cache["underlying-AAPL"]["pnl"] == original_pnl + 55.0
