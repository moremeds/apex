"""Unit tests for AdapterManager."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.infrastructure.adapters.adapter_manager import AdapterManager, AdapterStatus


class MockAdapter:
    """Mock adapter for testing."""

    def __init__(self, should_fail: bool = False):
        self._connected = False
        self._should_fail = should_fail

    async def connect(self):
        if self._should_fail:
            raise ConnectionError("Mock connection failure")
        self._connected = True

    async def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected


class TestAdapterStatus:
    """Tests for AdapterStatus dataclass."""

    def test_create_adapter_status(self):
        """Test creating AdapterStatus."""
        status = AdapterStatus(
            name="ib_live",
            adapter_type="live",
            broker="ib",
        )
        assert status.name == "ib_live"
        assert status.adapter_type == "live"
        assert status.broker == "ib"
        assert status.connected is False
        assert status.last_error is None
        assert status.reconnect_count == 0


class TestAdapterManagerRegistration:
    """Tests for adapter registration."""

    def test_register_live_adapter(self):
        """Test registering a live adapter."""
        manager = AdapterManager()
        adapter = MockAdapter()

        manager.register_live_adapter("ib_live", adapter, "ib")

        assert "ib_live" in manager.get_all_live_adapters()
        status = manager.get_status("ib_live")
        assert status is not None
        assert status.adapter_type == "live"
        assert status.broker == "ib"

    def test_register_historical_adapter(self):
        """Test registering a historical adapter."""
        manager = AdapterManager()
        adapter = MockAdapter()

        manager.register_historical_adapter("ib_historical", adapter, "ib")

        assert "ib_historical" in manager.get_all_historical_adapters()
        status = manager.get_status("ib_historical")
        assert status is not None
        assert status.adapter_type == "historical"

    def test_register_execution_adapter(self):
        """Test registering an execution adapter."""
        manager = AdapterManager()
        adapter = MockAdapter()

        manager.register_execution_adapter("ib_execution", adapter, "ib")

        assert "ib_execution" in manager.get_all_execution_adapters()
        status = manager.get_status("ib_execution")
        assert status is not None
        assert status.adapter_type == "execution"


class TestAdapterManagerLifecycle:
    """Tests for adapter lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_all_success(self):
        """Test starting all adapters successfully."""
        manager = AdapterManager()
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()

        manager.register_live_adapter("ib_live", adapter1, "ib")
        manager.register_historical_adapter("ib_historical", adapter2, "ib")

        await manager.start_all()

        assert adapter1._connected is True
        assert adapter2._connected is True
        assert manager.get_status("ib_live").connected is True
        assert manager.get_status("ib_historical").connected is True

    @pytest.mark.asyncio
    async def test_start_all_partial_failure(self):
        """Test starting adapters with one failure."""
        manager = AdapterManager()
        good_adapter = MockAdapter()
        bad_adapter = MockAdapter(should_fail=True)

        manager.register_live_adapter("good", good_adapter, "ib")
        manager.register_live_adapter("bad", bad_adapter, "ib")

        await manager.start_all()

        assert good_adapter._connected is True
        assert bad_adapter._connected is False
        assert manager.get_status("good").connected is True
        assert manager.get_status("bad").connected is False
        assert "Mock connection failure" in manager.get_status("bad").last_error

    @pytest.mark.asyncio
    async def test_stop_all(self):
        """Test stopping all adapters."""
        manager = AdapterManager()
        adapter = MockAdapter()

        manager.register_live_adapter("ib_live", adapter, "ib")
        await manager.start_all()
        assert adapter._connected is True

        await manager.stop_all()
        assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_restart_adapter(self):
        """Test restarting a specific adapter."""
        manager = AdapterManager()
        adapter = MockAdapter()

        manager.register_live_adapter("ib_live", adapter, "ib")
        await manager.start_all()

        initial_reconnect_count = manager.get_status("ib_live").reconnect_count

        success = await manager.restart_adapter("ib_live")

        assert success is True
        assert adapter._connected is True
        assert manager.get_status("ib_live").reconnect_count == initial_reconnect_count + 1

    @pytest.mark.asyncio
    async def test_restart_adapter_not_found(self):
        """Test restarting non-existent adapter."""
        manager = AdapterManager()

        success = await manager.restart_adapter("nonexistent")

        assert success is False

    @pytest.mark.asyncio
    async def test_start_all_idempotent(self):
        """Test that start_all is idempotent."""
        manager = AdapterManager()
        adapter = MockAdapter()
        manager.register_live_adapter("ib_live", adapter, "ib")

        await manager.start_all()
        await manager.start_all()  # Should not raise

        assert adapter._connected is True


class TestAdapterManagerAccess:
    """Tests for adapter access methods."""

    def test_get_live_adapter(self):
        """Test getting a live adapter by name."""
        manager = AdapterManager()
        adapter = MockAdapter()
        manager.register_live_adapter("ib_live", adapter, "ib")

        result = manager.get_live_adapter("ib_live")

        assert result is adapter

    def test_get_live_adapter_not_found(self):
        """Test getting non-existent live adapter."""
        manager = AdapterManager()

        result = manager.get_live_adapter("nonexistent")

        assert result is None

    def test_get_historical_adapter(self):
        """Test getting a historical adapter by name."""
        manager = AdapterManager()
        adapter = MockAdapter()
        manager.register_historical_adapter("ib_historical", adapter, "ib")

        result = manager.get_historical_adapter("ib_historical")

        assert result is adapter

    def test_get_execution_adapter(self):
        """Test getting an execution adapter by name."""
        manager = AdapterManager()
        adapter = MockAdapter()
        manager.register_execution_adapter("ib_execution", adapter, "ib")

        result = manager.get_execution_adapter("ib_execution")

        assert result is adapter


class TestAdapterManagerStatus:
    """Tests for status and health methods."""

    def test_get_all_status(self):
        """Test getting status for all adapters."""
        manager = AdapterManager()
        manager.register_live_adapter("ib_live", MockAdapter(), "ib")
        manager.register_historical_adapter("ib_historical", MockAdapter(), "ib")

        statuses = manager.get_all_status()

        assert len(statuses) == 2
        assert "ib_live" in statuses
        assert "ib_historical" in statuses

    def test_get_connected_adapters(self):
        """Test getting list of connected adapters."""
        manager = AdapterManager()
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()
        manager.register_live_adapter("a1", adapter1, "ib")
        manager.register_live_adapter("a2", adapter2, "ib")

        # Mark one as connected manually for test
        manager.get_status("a1").connected = True

        connected = manager.get_connected_adapters()

        assert "a1" in connected
        assert "a2" not in connected

    def test_is_any_connected(self):
        """Test checking if any adapter is connected."""
        manager = AdapterManager()
        manager.register_live_adapter("a1", MockAdapter(), "ib")

        assert manager.is_any_connected() is False

        manager.get_status("a1").connected = True

        assert manager.is_any_connected() is True

    @pytest.mark.asyncio
    async def test_check_health(self):
        """Test health check updates status."""
        manager = AdapterManager()
        adapter = MockAdapter()
        manager.register_live_adapter("ib_live", adapter, "ib")

        # Manually set connected to True
        adapter._connected = True

        statuses = await manager.check_health()

        assert statuses["ib_live"].connected is True


class TestAdapterManagerMetrics:
    """Tests for metrics functionality."""

    def test_get_metrics(self):
        """Test getting metrics dictionary."""
        manager = AdapterManager()
        manager.register_live_adapter("ib_live", MockAdapter(), "ib")
        manager.register_historical_adapter("ib_historical", MockAdapter(), "ib")
        manager.register_execution_adapter("futu_exec", MockAdapter(), "futu")

        metrics = manager.get_metrics()

        assert metrics["total_adapters"] == 3
        assert metrics["connected_adapters"] == 0
        assert metrics["adapters_by_type"]["live"] == 1
        assert metrics["adapters_by_type"]["historical"] == 1
        assert metrics["adapters_by_type"]["execution"] == 1
        assert metrics["adapters_by_broker"]["ib"] == 2
        assert metrics["adapters_by_broker"]["futu"] == 1

    def test_set_adapter_metrics(self):
        """Test setting adapter metrics instance."""
        manager = AdapterManager()
        mock_metrics = MagicMock()

        manager.set_adapter_metrics(mock_metrics)

        assert manager._adapter_metrics is mock_metrics

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_connect(self):
        """Test that metrics are recorded on connect."""
        mock_metrics = MagicMock()
        manager = AdapterManager(adapter_metrics=mock_metrics)
        adapter = MockAdapter()

        manager.register_live_adapter("ib_live", adapter, "ib")
        await manager.start_all()

        mock_metrics.record_connection_status.assert_called()

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_error(self):
        """Test that error metrics are recorded on connection failure."""
        mock_metrics = MagicMock()
        manager = AdapterManager(adapter_metrics=mock_metrics)
        adapter = MockAdapter(should_fail=True)

        manager.register_live_adapter("ib_live", adapter, "ib")
        await manager.start_all()

        mock_metrics.record_error.assert_called()


class TestAdapterManagerHealthMonitor:
    """Tests for health monitor integration."""

    def test_set_health_monitor(self):
        """Test setting health monitor instance."""
        manager = AdapterManager()
        mock_monitor = MagicMock()

        manager.set_health_monitor(mock_monitor)

        assert manager._health_monitor is mock_monitor
