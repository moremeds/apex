"""
Unit tests for SignalCoordinator.

Tests the coordinator that wires the signal pipeline including:
- Start/stop lifecycle
- Options symbol detection
- Event subscriptions
- Pipeline component orchestration
- Bar aggregation
- Persistence callbacks
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.events.event_types import EventType

from .conftest import MockEventBus

# =============================================================================
# Import SignalCoordinator (may need path adjustment)
# =============================================================================


@pytest.fixture
def signal_coordinator_class():
    """Import SignalCoordinator lazily to avoid import-time issues."""
    from src.application.orchestrator.signal_coordinator import SignalCoordinator

    return SignalCoordinator


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestSignalCoordinatorLifecycle:
    """Test start/stop lifecycle behavior."""

    def test_init_defaults(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Constructor should set default values."""
        coordinator = signal_coordinator_class(mock_event_bus)

        assert coordinator._enabled is True
        assert coordinator._started is False
        assert coordinator._timeframes == ["1d"]
        assert coordinator._exclude_options is True

    def test_init_custom_timeframes(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Constructor should accept custom timeframes."""
        coordinator = signal_coordinator_class(mock_event_bus, timeframes=["1h", "4h", "1d"])

        assert coordinator._timeframes == ["1h", "4h", "1d"]

    def test_init_deduplicates_timeframes(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Constructor should deduplicate timeframes."""
        coordinator = signal_coordinator_class(mock_event_bus, timeframes=["1d", "1d", "1h"])

        assert coordinator._timeframes == ["1d", "1h"]

    def test_start_creates_components(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """start() should create bar aggregators and engines."""
        coordinator = signal_coordinator_class(mock_event_bus, timeframes=["1d"])

        coordinator.start()

        assert coordinator._started is True
        assert len(coordinator._bar_aggregators) == 1
        assert "1d" in coordinator._bar_aggregators
        assert coordinator._indicator_engine is not None
        assert coordinator._rule_engine is not None
        assert coordinator._confluence_calculator is not None

    def test_start_subscribes_to_events(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """start() should subscribe to tick and indicator events."""
        coordinator = signal_coordinator_class(mock_event_bus)

        coordinator.start()

        assert EventType.MARKET_DATA_TICK in mock_event_bus.subscriptions
        assert EventType.INDICATOR_UPDATE in mock_event_bus.subscriptions

    def test_start_is_idempotent(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Multiple start() calls should not create duplicates."""
        coordinator = signal_coordinator_class(mock_event_bus)

        coordinator.start()
        coordinator.start()  # Second call

        # Should only have one subscription per event type
        assert len(mock_event_bus.subscriptions[EventType.MARKET_DATA_TICK]) == 1

    def test_start_disabled(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """start() with enabled=False should be a no-op."""
        coordinator = signal_coordinator_class(mock_event_bus, enabled=False)

        coordinator.start()

        assert coordinator._started is False
        assert len(coordinator._bar_aggregators) == 0

    def test_stop_clears_components(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """stop() should clean up all components."""
        coordinator = signal_coordinator_class(mock_event_bus)
        coordinator.start()

        coordinator.stop()

        assert coordinator._started is False
        assert len(coordinator._bar_aggregators) == 0
        assert coordinator._indicator_engine is None
        assert coordinator._rule_engine is None
        assert coordinator._confluence_calculator is None

    def test_stop_before_start(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """stop() before start() should be safe."""
        coordinator = signal_coordinator_class(mock_event_bus)

        coordinator.stop()  # Should not raise

        assert coordinator._started is False


# =============================================================================
# Properties Tests
# =============================================================================


class TestSignalCoordinatorProperties:
    """Test property accessors."""

    def test_is_started_property(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """is_started should reflect _started state."""
        coordinator = signal_coordinator_class(mock_event_bus)

        assert coordinator.is_started is False

        coordinator.start()
        assert coordinator.is_started is True

        coordinator.stop()
        assert coordinator.is_started is False

    def test_timeframes_property(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """timeframes property should return a copy."""
        coordinator = signal_coordinator_class(mock_event_bus, timeframes=["1d", "1h"])

        tf = coordinator.timeframes
        tf.append("5m")  # Modify the returned list

        # Original should be unchanged
        assert coordinator._timeframes == ["1d", "1h"]


# =============================================================================
# Options Symbol Detection Tests
# =============================================================================


class TestOptionsSymbolDetection:
    """Test options symbol pattern matching."""

    def test_stock_symbols_not_detected(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Short stock symbols should not be detected as options."""
        coordinator = signal_coordinator_class(mock_event_bus)

        # Common stock symbols
        assert coordinator._is_options_symbol("AAPL") is False
        assert coordinator._is_options_symbol("TSLA") is False
        assert coordinator._is_options_symbol("SPY") is False
        assert coordinator._is_options_symbol("QQQ") is False
        assert coordinator._is_options_symbol("VIX") is False
        assert coordinator._is_options_symbol("GOOGL") is False

    def test_long_options_symbols_detected(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Long options symbols should be detected."""
        coordinator = signal_coordinator_class(mock_event_bus)

        # Options symbols > 10 characters
        assert coordinator._is_options_symbol("AAPL  250117C00250000") is True
        assert coordinator._is_options_symbol("AAPL250117C00250000") is True

    def test_ib_style_options_detected(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """IB-style options symbols should be detected."""
        coordinator = signal_coordinator_class(mock_event_bus)

        # Pattern with 6+ digits (YYMMDD strike)
        assert coordinator._is_options_symbol("SPY240315C500") is True
        assert coordinator._is_options_symbol("AAPL240120P175") is True

    def test_colon_separated_options_detected(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Colon-separated options notation should be detected."""
        coordinator = signal_coordinator_class(mock_event_bus)

        assert coordinator._is_options_symbol("AAPL:OPT:240117:175:C") is True
        assert coordinator._is_options_symbol("TSLA:opt:240315:200:P") is True

    def test_strike_pattern_detected(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """C or P followed by digits should be detected."""
        coordinator = signal_coordinator_class(mock_event_bus)

        # C/P followed by 5+ digits (strike price)
        assert coordinator._is_options_symbol("AAPLC00250") is True
        assert coordinator._is_options_symbol("TSLAP00175") is True

    def test_empty_symbol(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Empty symbol should not be detected as options."""
        coordinator = signal_coordinator_class(mock_event_bus)

        assert coordinator._is_options_symbol("") is False
        assert coordinator._is_options_symbol(None) is False  # type: ignore


# =============================================================================
# Market Data Tick Handling Tests
# =============================================================================


class TestMarketDataTickHandling:
    """Test MARKET_DATA_TICK event handling."""

    def test_tick_ignored_after_stop(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Ticks should be ignored after coordinator stops."""
        coordinator = signal_coordinator_class(mock_event_bus)
        coordinator.start()
        coordinator.stop()

        # Create a tick-like payload
        tick = MagicMock()
        tick.symbol = "AAPL"

        # Should not raise
        coordinator._on_market_data_tick(tick)

    def test_options_tick_filtered(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Options ticks should be filtered when exclude_options=True."""
        coordinator = signal_coordinator_class(mock_event_bus, exclude_options=True)
        coordinator.start()

        # Create a tick with options symbol
        tick = MagicMock()
        tick.symbol = "AAPL  250117C00250000"

        # Track aggregator calls
        aggregator = coordinator._bar_aggregators["1d"]
        aggregator.on_tick = MagicMock()

        coordinator._on_market_data_tick(tick)

        # Aggregator should not be called for options
        aggregator.on_tick.assert_not_called()

    def test_stock_tick_passed_to_aggregators(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Stock ticks should be passed to all aggregators."""
        coordinator = signal_coordinator_class(
            mock_event_bus, timeframes=["1d", "1h"], exclude_options=True
        )
        coordinator.start()

        # Mock the aggregators
        for tf, agg in coordinator._bar_aggregators.items():
            agg.on_tick = MagicMock()

        tick = MagicMock()
        tick.symbol = "AAPL"

        coordinator._on_market_data_tick(tick)

        # All aggregators should receive the tick
        for tf, agg in coordinator._bar_aggregators.items():
            agg.on_tick.assert_called_once_with(tick)


# =============================================================================
# Stats Tests
# =============================================================================


class TestGetStats:
    """Test get_stats() functionality."""

    def test_stats_before_start(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Stats should reflect stopped state before start."""
        coordinator = signal_coordinator_class(mock_event_bus, timeframes=["1d"])

        stats = coordinator.get_stats()

        assert stats["started"] is False
        assert stats["timeframes"] == ["1d"]
        assert stats["enabled"] is True

    def test_stats_after_start(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Stats should include component stats after start."""
        coordinator = signal_coordinator_class(mock_event_bus, timeframes=["1d"])
        coordinator.start()

        stats = coordinator.get_stats()

        assert stats["started"] is True
        assert "bars_emitted" in stats
        assert "indicator_count" in stats


# =============================================================================
# Clear Cooldowns Tests
# =============================================================================


class TestClearCooldowns:
    """Test clear_cooldowns() delegation."""

    def test_clear_cooldowns_before_start(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """clear_cooldowns() before start should return 0."""
        coordinator = signal_coordinator_class(mock_event_bus)

        cleared = coordinator.clear_cooldowns()
        assert cleared == 0

    def test_clear_cooldowns_delegates_to_rule_engine(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """clear_cooldowns() should delegate to rule engine."""
        coordinator = signal_coordinator_class(mock_event_bus)
        coordinator.start()

        # Mock rule engine
        coordinator._rule_engine.clear_cooldowns = MagicMock(return_value=5)

        cleared = coordinator.clear_cooldowns()

        coordinator._rule_engine.clear_cooldowns.assert_called_once()
        assert cleared == 5


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Test persistence callback functionality."""

    def test_trading_signal_subscription_with_persistence(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Should subscribe to TRADING_SIGNAL when persistence is provided."""
        persistence = MagicMock()
        coordinator = signal_coordinator_class(mock_event_bus, persistence=persistence)

        coordinator.start()

        assert EventType.TRADING_SIGNAL in mock_event_bus.subscriptions

    def test_no_trading_signal_subscription_without_persistence(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Should not subscribe to TRADING_SIGNAL without persistence."""
        coordinator = signal_coordinator_class(mock_event_bus, persistence=None)

        coordinator.start()

        # Should not have TRADING_SIGNAL subscription
        assert len(mock_event_bus.subscriptions.get(EventType.TRADING_SIGNAL, [])) == 0


# =============================================================================
# Indicator Update Handler Tests
# =============================================================================


class TestIndicatorUpdateHandler:
    """Test _on_indicator_update handler."""

    def test_indicator_update_ignored_after_stop(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Indicator updates should be ignored after stop."""
        coordinator = signal_coordinator_class(mock_event_bus)
        coordinator.start()
        coordinator.stop()

        payload = MagicMock()
        payload.symbol = "AAPL"
        payload.timeframe = "1d"
        payload.indicator = "rsi"
        payload.state = {"value": 50}

        # Should not raise
        coordinator._on_indicator_update(payload)

    def test_indicator_update_delegates_to_confluence(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """Indicator updates should be delegated to confluence calculator."""
        coordinator = signal_coordinator_class(mock_event_bus)
        coordinator.start()

        # Mock confluence calculator
        coordinator._confluence_calculator.on_indicator_update = MagicMock()

        payload = MagicMock()
        payload.symbol = "AAPL"
        payload.timeframe = "1d"
        payload.indicator = "rsi"
        payload.state = {"value": 50}

        coordinator._on_indicator_update(payload)

        coordinator._confluence_calculator.on_indicator_update.assert_called_once()


# =============================================================================
# Preload Bar History Tests
# =============================================================================


class TestPreloadBarHistory:
    """Test preload_bar_history async method."""

    @pytest.mark.asyncio
    async def test_preload_without_preloader(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """preload_bar_history without preloader should return empty dict."""
        coordinator = signal_coordinator_class(mock_event_bus)
        coordinator.start()

        result = await coordinator.preload_bar_history(["AAPL", "TSLA"])

        assert result == {}

    @pytest.mark.asyncio
    async def test_preload_delegates_to_preloader(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """preload_bar_history should delegate to bar preloader."""
        historical_data_manager = MagicMock()
        coordinator = signal_coordinator_class(
            mock_event_bus, historical_data_manager=historical_data_manager
        )
        coordinator.start()

        # Mock the preloader
        coordinator._bar_preloader = MagicMock()
        coordinator._bar_preloader.preload_startup = AsyncMock(return_value={"AAPL": 100})

        result = await coordinator.preload_bar_history(["AAPL"])

        coordinator._bar_preloader.preload_startup.assert_called_once_with(["AAPL"])
        assert result == {"AAPL": 100}


# =============================================================================
# Refresh Disk Cache Tests
# =============================================================================


class TestRefreshDiskCache:
    """Test refresh_disk_cache async method."""

    @pytest.mark.asyncio
    async def test_refresh_without_preloader(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """refresh_disk_cache without preloader should return False."""
        coordinator = signal_coordinator_class(mock_event_bus)
        coordinator.start()

        result = await coordinator.refresh_disk_cache(["AAPL"])

        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_delegates_to_preloader(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """refresh_disk_cache should delegate to bar preloader."""
        historical_data_manager = MagicMock()
        coordinator = signal_coordinator_class(
            mock_event_bus, historical_data_manager=historical_data_manager
        )
        coordinator.start()

        # Mock the preloader
        coordinator._bar_preloader = MagicMock()
        coordinator._bar_preloader.refresh_disk_cache = AsyncMock(return_value=True)

        result = await coordinator.refresh_disk_cache(["AAPL"])

        coordinator._bar_preloader.refresh_disk_cache.assert_called_once_with(["AAPL"])
        assert result is True


# =============================================================================
# Persistence Handler Tests
# =============================================================================


class TestPersistenceHandlers:
    """Test async persistence handlers."""

    @pytest.mark.asyncio
    async def test_persist_signal_without_persistence(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """_persist_signal without persistence should be a no-op."""
        coordinator = signal_coordinator_class(mock_event_bus, persistence=None)
        coordinator.start()

        signal = MagicMock()

        # Should not raise
        await coordinator._persist_signal(signal)

        assert coordinator._signals_persisted == 0

    @pytest.mark.asyncio
    async def test_persist_signal_unwraps_event(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """_persist_signal should unwrap TradingSignalEvent."""
        persistence = MagicMock()
        persistence.save_signal = AsyncMock()

        coordinator = signal_coordinator_class(mock_event_bus, persistence=persistence)
        coordinator.start()

        # Create a wrapped signal event
        inner_signal = MagicMock()
        wrapped = MagicMock()
        wrapped.signal = inner_signal

        await coordinator._persist_signal(wrapped)

        # Should save the unwrapped signal
        persistence.save_signal.assert_called_once_with(inner_signal)
        assert coordinator._signals_persisted == 1

    @pytest.mark.asyncio
    async def test_persist_indicator(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """_persist_indicator should save indicator state."""
        persistence = MagicMock()
        persistence.save_indicator = AsyncMock()

        coordinator = signal_coordinator_class(mock_event_bus, persistence=persistence)
        coordinator.start()

        await coordinator._persist_indicator(
            symbol="AAPL",
            timeframe="1d",
            indicator="rsi",
            timestamp=datetime.now(timezone.utc),
            state={"value": 50},
        )

        persistence.save_indicator.assert_called_once()
        assert coordinator._indicators_persisted == 1

    @pytest.mark.asyncio
    async def test_persist_confluence(
        self, mock_event_bus: MockEventBus, signal_coordinator_class: type
    ) -> None:
        """_persist_confluence should save confluence score."""
        persistence = MagicMock()
        persistence.save_confluence = AsyncMock()

        coordinator = signal_coordinator_class(mock_event_bus, persistence=persistence)
        coordinator.start()

        await coordinator._persist_confluence(
            symbol="AAPL",
            timeframe="1d",
            alignment_score=0.75,
            bullish_count=5,
            bearish_count=2,
            neutral_count=3,
            total_indicators=10,
        )

        persistence.save_confluence.assert_called_once()
        assert coordinator._confluence_persisted == 1
