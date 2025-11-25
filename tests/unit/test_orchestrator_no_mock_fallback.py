"""
Unit tests for orchestrator behavior when market data fetch fails.

Tests that the orchestrator properly marks the system as UNHEALTHY
instead of silently falling back to mock data.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.application.orchestrator import Orchestrator
from src.infrastructure.monitoring.health_monitor import HealthStatus


@pytest.mark.asyncio
async def test_orchestrator_unhealthy_when_ib_disconnected():
    """Test that orchestrator marks system UNHEALTHY when IB is not connected."""
    # Create mock dependencies
    ib_adapter = MagicMock()
    ib_adapter.is_connected = MagicMock(return_value=False)
    ib_adapter.fetch_positions = AsyncMock(return_value=[])
    ib_adapter.fetch_account_info = AsyncMock(return_value=MagicMock())

    file_loader = MagicMock()
    file_loader.fetch_positions = AsyncMock(return_value=[])

    health_monitor = MagicMock()

    # Create mock stores
    position_store = MagicMock()
    position_store.get_all = MagicMock(return_value=[])
    position_store.upsert_positions = MagicMock()

    market_data_store = MagicMock()
    market_data_store.get_stale_symbols = MagicMock(return_value=['AAPL'])
    market_data_store.get_symbols = MagicMock(return_value=[])
    market_data_store.get_all = MagicMock(return_value=[])

    account_store = MagicMock()

    # Create other mocks
    risk_engine = MagicMock()
    risk_engine.build_snapshot = MagicMock()

    reconciler = MagicMock()
    reconciler.reconcile = MagicMock(return_value=[])

    mdqc = MagicMock()
    rule_engine = MagicMock()
    rule_engine.evaluate = MagicMock(return_value=[])

    watchdog = MagicMock()
    event_bus = MagicMock()

    config = {"dashboard": {"refresh_interval_sec": 2}}

    # Create orchestrator
    orchestrator = Orchestrator(
        ib_adapter=ib_adapter,
        file_loader=file_loader,
        position_store=position_store,
        market_data_store=market_data_store,
        account_store=account_store,
        risk_engine=risk_engine,
        reconciler=reconciler,
        mdqc=mdqc,
        rule_engine=rule_engine,
        health_monitor=health_monitor,
        watchdog=watchdog,
        event_bus=event_bus,
        config=config,
    )

    # Create a mock position that needs market data
    from src.models.position import Position, AssetType, PositionSource
    mock_position = Position(
        symbol="AAPL",
        underlying="AAPL",
        asset_type=AssetType.STOCK,
        quantity=100,
        avg_price=150.0,
        source=PositionSource.IB,
    )

    # Mock merge positions to return our test position
    reconciler.merge_positions = MagicMock(return_value=[mock_position])
    await orchestrator._run_cycle()

    # Verify health monitor was updated with UNHEALTHY status
    health_monitor.update_component_health.assert_any_call(
        "market_data_feed",
        HealthStatus.UNHEALTHY,
        "IB adapter not connected"
    )


@pytest.mark.asyncio
async def test_orchestrator_unhealthy_when_fetch_fails():
    """Test that orchestrator marks system UNHEALTHY when market data fetch fails."""
    # Create mock dependencies
    ib_adapter = MagicMock()
    ib_adapter.is_connected = MagicMock(return_value=True)
    ib_adapter.fetch_positions = AsyncMock(return_value=[])
    ib_adapter.fetch_market_data = AsyncMock(side_effect=Exception("Network error"))
    ib_adapter.fetch_account_info = AsyncMock(return_value=MagicMock())

    file_loader = MagicMock()
    file_loader.fetch_positions = AsyncMock(return_value=[])

    health_monitor = MagicMock()

    # Create mock stores
    position_store = MagicMock()
    position_store.get_all = MagicMock(return_value=[])
    position_store.upsert_positions = MagicMock()

    market_data_store = MagicMock()
    market_data_store.get_stale_symbols = MagicMock(return_value=['AAPL'])
    market_data_store.get_symbols = MagicMock(return_value=[])
    market_data_store.get_all = MagicMock(return_value=[])

    account_store = MagicMock()

    # Create other mocks
    risk_engine = MagicMock()
    risk_engine.build_snapshot = MagicMock()

    reconciler = MagicMock()
    reconciler.reconcile = MagicMock(return_value=[])

    mdqc = MagicMock()
    rule_engine = MagicMock()
    rule_engine.evaluate = MagicMock(return_value=[])

    watchdog = MagicMock()
    event_bus = MagicMock()

    config = {"dashboard": {"refresh_interval_sec": 2}}

    # Create orchestrator
    orchestrator = Orchestrator(
        ib_adapter=ib_adapter,
        file_loader=file_loader,
        position_store=position_store,
        market_data_store=market_data_store,
        account_store=account_store,
        risk_engine=risk_engine,
        reconciler=reconciler,
        mdqc=mdqc,
        rule_engine=rule_engine,
        health_monitor=health_monitor,
        watchdog=watchdog,
        event_bus=event_bus,
        config=config,
    )

    # Create a mock position that needs market data
    from src.models.position import Position, AssetType, PositionSource
    mock_position = Position(
        symbol="AAPL",
        underlying="AAPL",
        asset_type=AssetType.STOCK,
        quantity=100,
        avg_price=150.0,
        source=PositionSource.IB,
    )

    # Mock merge positions to return our test position
    reconciler.merge_positions = MagicMock(return_value=[mock_position])
    await orchestrator._run_cycle()

    # Verify health monitor was updated with UNHEALTHY status
    health_monitor.update_component_health.assert_any_call(
        "market_data_feed",
        HealthStatus.UNHEALTHY,
        "Fetch failed: Network error"
    )


@pytest.mark.asyncio
async def test_orchestrator_no_mock_fallback():
    """Test that orchestrator does NOT use mock data fallback."""
    # Create mock dependencies
    ib_adapter = MagicMock()
    ib_adapter.is_connected = MagicMock(return_value=True)
    ib_adapter.fetch_positions = AsyncMock(return_value=[])
    ib_adapter.fetch_market_data = AsyncMock(side_effect=Exception("Fetch failed"))
    ib_adapter.fetch_account_info = AsyncMock(return_value=MagicMock())

    file_loader = MagicMock()
    file_loader.fetch_positions = AsyncMock(return_value=[])

    health_monitor = MagicMock()

    # Create mock stores
    position_store = MagicMock()
    position_store.get_all = MagicMock(return_value=[])
    position_store.upsert_positions = MagicMock()

    market_data_store = MagicMock()
    market_data_store.get_stale_symbols = MagicMock(return_value=['AAPL'])
    market_data_store.get_symbols = MagicMock(return_value=[])
    market_data_store.get_all = MagicMock(return_value=[])
    market_data_store.upsert = MagicMock()  # Should NOT be called with mock data

    account_store = MagicMock()

    # Create other mocks
    risk_engine = MagicMock()
    risk_engine.build_snapshot = MagicMock()

    reconciler = MagicMock()
    reconciler.reconcile = MagicMock(return_value=[])

    mdqc = MagicMock()
    rule_engine = MagicMock()
    rule_engine.evaluate = MagicMock(return_value=[])

    watchdog = MagicMock()
    event_bus = MagicMock()

    config = {"dashboard": {"refresh_interval_sec": 2}}

    # Create orchestrator
    orchestrator = Orchestrator(
        ib_adapter=ib_adapter,
        file_loader=file_loader,
        position_store=position_store,
        market_data_store=market_data_store,
        account_store=account_store,
        risk_engine=risk_engine,
        reconciler=reconciler,
        mdqc=mdqc,
        rule_engine=rule_engine,
        health_monitor=health_monitor,
        watchdog=watchdog,
        event_bus=event_bus,
        config=config,
    )

    # Create a mock position that needs market data
    from src.models.position import Position, AssetType, PositionSource
    mock_position = Position(
        symbol="AAPL",
        underlying="AAPL",
        asset_type=AssetType.STOCK,
        quantity=100,
        avg_price=150.0,
        source=PositionSource.IB,
    )

    # Mock merge positions to return our test position
    reconciler.merge_positions = MagicMock(return_value=[mock_position])
    await orchestrator._run_cycle()

    # Verify market data store upsert was NOT called with mock data
    # (it should only be called if real data was fetched successfully)
    # Since fetch failed, upsert should not be called
    market_data_store.upsert.assert_not_called()

    # Verify system was marked UNHEALTHY
    health_monitor.update_component_health.assert_any_call(
        "market_data_feed",
        HealthStatus.UNHEALTHY,
        "Fetch failed: Fetch failed"
    )
