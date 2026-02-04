"""
Unit tests for the backtest engine.
"""

from datetime import date, datetime, timedelta
from typing import Any

import pytest

# Import to register example strategies (decorator runs on import)
import src.domain.strategy.examples  # noqa: F401
from src.backtest.data.feeds import InMemoryDataFeed
from src.backtest.execution.engines.backtest_engine import BacktestConfig, BacktestEngine
from src.backtest.execution.simulated import FillModel, SimulatedExecution
from src.domain.clock import SimulatedClock
from src.domain.events.domain_events import QuoteTick
from src.domain.interfaces.execution_provider import OrderRequest
from src.domain.strategy.base import Strategy, StrategyContext
from src.domain.strategy.registry import StrategyRegistry, register_strategy


class TestSimulatedClock:
    """Tests for SimulatedClock."""

    def test_initial_time(self) -> None:
        """Test clock starts at specified time."""
        start = datetime(2024, 1, 1, 9, 30)
        clock = SimulatedClock(start)
        assert clock.now() == start

    def test_advance_to(self) -> None:
        """Test advancing clock to future time."""
        start = datetime(2024, 1, 1, 9, 30)
        clock = SimulatedClock(start)

        target = datetime(2024, 1, 1, 10, 0)
        clock.advance_to(target)

        assert clock.now() == target

    def test_advance_by(self) -> None:
        """Test advancing clock by delta."""
        start = datetime(2024, 1, 1, 9, 30)
        clock = SimulatedClock(start)

        clock.advance_by(timedelta(minutes=30))

        assert clock.now() == datetime(2024, 1, 1, 10, 0)

    def test_timer_fires(self) -> None:
        """Test timer fires during advancement."""
        start = datetime(2024, 1, 1, 9, 30)
        clock = SimulatedClock(start)

        fired = []
        clock.set_timer(60, lambda: fired.append(True))  # Fire in 60 seconds

        # Advance past timer
        clock.advance_by(timedelta(minutes=2))

        assert len(fired) == 1

    def test_timer_cancelled(self) -> None:
        """Test cancelled timer doesn't fire."""
        start = datetime(2024, 1, 1, 9, 30)
        clock = SimulatedClock(start)

        fired = []
        timer_id = clock.set_timer(60, lambda: fired.append(True))
        clock.cancel_timer(timer_id)

        clock.advance_by(timedelta(minutes=2))

        assert len(fired) == 0


class TestSimulatedExecution:
    """Tests for SimulatedExecution."""

    @pytest.fixture
    def execution(self):
        """Create execution engine for testing."""
        clock = SimulatedClock(datetime(2024, 1, 1))
        return SimulatedExecution(clock, fill_model=FillModel.IMMEDIATE)

    @pytest.mark.asyncio
    async def test_market_order_fills_immediately(self, execution) -> None:
        """Test market order fills at ask for buys."""
        # Set price
        tick = QuoteTick(symbol="AAPL", bid=150.0, ask=150.05, last=150.02)
        execution.update_price(tick)

        # Submit market buy
        order = OrderRequest(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type="MARKET",
            client_order_id="test-1",
        )

        result = await execution.submit_order(order)
        assert result.success

        # Should be filled
        fills = execution.get_pending_fills()
        assert len(fills) == 1
        assert fills[0].symbol == "AAPL"
        assert fills[0].quantity == 100
        assert fills[0].price == 150.05  # Filled at ask

    @pytest.mark.asyncio
    async def test_position_tracking(self, execution) -> None:
        """Test position is tracked after fill."""
        tick = QuoteTick(symbol="AAPL", bid=150.0, ask=150.05, last=150.02)
        execution.update_price(tick)

        order = OrderRequest(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type="MARKET",
            client_order_id="test-1",
        )

        await execution.submit_order(order)
        execution.get_pending_fills()  # Process fills

        pos = execution.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 100
        assert pos.avg_price == 150.05

    @pytest.mark.asyncio
    async def test_limit_order_fills_when_price_reached(self, execution) -> None:
        """Test limit order fills when price is favorable."""
        # Set initial price (above limit)
        tick1 = QuoteTick(symbol="AAPL", bid=151.0, ask=151.05, last=151.0)
        execution.update_price(tick1)

        # Submit limit buy at 150
        order = OrderRequest(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            order_type="LIMIT",
            limit_price=150.0,
            client_order_id="test-1",
        )

        await execution.submit_order(order)

        # Should not fill yet
        fills = execution.get_pending_fills()
        assert len(fills) == 0

        # Price drops to limit
        tick2 = QuoteTick(symbol="AAPL", bid=149.95, ask=150.0, last=149.98)
        execution.update_price(tick2)

        # Should now be filled
        fills = execution.get_pending_fills()
        assert len(fills) == 1


class TestDataFeeds:
    """Tests for data feeds."""

    @pytest.mark.asyncio
    async def test_in_memory_feed(self) -> None:
        """Test in-memory data feed."""
        feed = InMemoryDataFeed()
        feed.add_bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            volume=1000,
        )
        feed.add_bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, 9, 30),
            open=150.5,
            high=152.0,
            low=150.0,
            close=151.5,
            volume=1200,
        )

        await feed.load()

        bars = []
        async for bar in feed.stream_bars():
            bars.append(bar)

        assert len(bars) == 2
        assert bars[0].close == 150.5
        assert bars[1].close == 151.5


class TestStrategy:
    """Tests for strategy base class."""

    def test_strategy_lifecycle(self) -> None:
        """Test strategy start/stop lifecycle."""

        # Create a simple test strategy
        @register_strategy("test_simple")
        class SimpleStrategy(Strategy):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.started = False
                self.stopped = False
                self.ticks_received = []

            def on_start(self) -> None:
                self.started = True

            def on_stop(self) -> None:
                self.stopped = True

            def on_tick(self, tick: Any) -> None:
                self.ticks_received.append(tick)

        clock = SimulatedClock(datetime(2024, 1, 1))
        context = StrategyContext(clock=clock)

        strategy = SimpleStrategy(
            strategy_id="test-1",
            symbols=["AAPL"],
            context=context,
        )

        assert not strategy.started
        strategy.start()
        assert strategy.started
        assert strategy.is_running

        strategy.stop()
        assert strategy.stopped

        # Cleanup
        StrategyRegistry.unregister("test_simple")


class TestBacktestEngine:
    """Tests for backtest engine."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            symbols=["AAPL"],
            initial_capital=100000.0,
            strategy_name="buy_and_hold",
        )

    @pytest.mark.asyncio
    async def test_engine_runs_with_in_memory_feed(self, config) -> None:
        """Test engine runs successfully with in-memory data."""
        # Import to register strategies

        engine = BacktestEngine(config)
        engine.set_strategy(strategy_name="buy_and_hold")

        # Create in-memory feed with sample data
        feed = InMemoryDataFeed()
        for i in range(20):
            feed.add_bar(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 1 + i, 9, 30),
                open=150.0 + i,
                high=151.0 + i,
                low=149.0 + i,
                close=150.5 + i,
                volume=1000000,
            )

        engine.set_data_feed(feed)

        result = await engine.run()

        assert result is not None
        assert result.strategy_name == "buy_and_hold"
        assert result.initial_capital == 100000.0
        assert len(result.equity_curve) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
