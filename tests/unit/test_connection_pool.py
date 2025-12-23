"""
Tests for IbConnectionPool (A4: Execution Connection).

Verifies:
- Pool configuration
- Connection status tracking
- Execution connection (A4 feature)
- Status reporting
- Disconnect behavior
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.infrastructure.adapters.ib.connection_pool import (
    IbConnectionPool,
    ConnectionPoolConfig,
)
from config.models import IbClientIdsConfig


class TestConnectionPoolConfig:
    """Configuration behavior."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = ConnectionPoolConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 7497
        assert config.connect_timeout == 10
        assert config.client_ids is not None

    def test_custom_config(self):
        """Custom config values are preserved."""
        client_ids = IbClientIdsConfig(
            monitoring=10,
            execution=11,
            historical_pool=[20, 21, 22],
        )
        config = ConnectionPoolConfig(
            host="192.168.1.100",
            port=4001,
            client_ids=client_ids,
            connect_timeout=30,
        )

        assert config.host == "192.168.1.100"
        assert config.port == 4001
        assert config.connect_timeout == 30
        assert config.client_ids.monitoring == 10
        assert config.client_ids.execution == 11


class TestPoolInitialization:
    """Pool initialization."""

    def test_pool_starts_disconnected(self):
        """Pool starts in disconnected state."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        assert pool.is_connected() is False
        assert pool.is_monitoring_connected() is False
        assert pool.is_historical_connected() is False
        assert pool.is_execution_connected() is False

    def test_properties_return_none_when_disconnected(self):
        """Connection properties return None when not connected."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        assert pool.monitoring is None
        assert pool.historical is None
        assert pool.execution is None


class TestConnectionStatus:
    """Connection status tracking."""

    def test_is_connected_requires_all_connections(self):
        """is_connected() returns True only when all connections are alive."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        # Create mock IB connections
        mock_mon = MagicMock()
        mock_hist = MagicMock()
        mock_exec = MagicMock()

        mock_mon.isConnected.return_value = True
        mock_hist.isConnected.return_value = True
        mock_exec.isConnected.return_value = True

        pool._monitoring = mock_mon
        pool._historical = mock_hist
        pool._execution = mock_exec
        pool._connected = True

        assert pool.is_connected() is True

    def test_is_connected_false_if_any_disconnected(self):
        """is_connected() returns False if any connection is down."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        mock_mon = MagicMock()
        mock_hist = MagicMock()
        mock_exec = MagicMock()

        mock_mon.isConnected.return_value = True
        mock_hist.isConnected.return_value = False  # Disconnected
        mock_exec.isConnected.return_value = True

        pool._monitoring = mock_mon
        pool._historical = mock_hist
        pool._execution = mock_exec
        pool._connected = True

        assert pool.is_connected() is False

    def test_individual_connection_status_checks(self):
        """Individual connection status methods work correctly."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        mock_mon = MagicMock()
        mock_hist = MagicMock()
        mock_exec = MagicMock()

        mock_mon.isConnected.return_value = True
        mock_hist.isConnected.return_value = False
        mock_exec.isConnected.return_value = True

        pool._monitoring = mock_mon
        pool._historical = mock_hist
        pool._execution = mock_exec

        assert pool.is_monitoring_connected() is True
        assert pool.is_historical_connected() is False
        assert pool.is_execution_connected() is True


class TestExecutionConnection:
    """A4: Execution connection tests."""

    def test_execution_property_exists(self):
        """Execution property is available (A4)."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        # Should have execution property
        assert hasattr(pool, "execution")
        assert hasattr(pool, "is_execution_connected")
        assert hasattr(pool, "connect_execution")
        assert hasattr(pool, "ensure_execution_connected")

    def test_execution_connection_isolation(self):
        """Execution connection is separate from data connections."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        mock_mon = MagicMock()
        mock_hist = MagicMock()
        mock_exec = MagicMock()

        pool._monitoring = mock_mon
        pool._historical = mock_hist
        pool._execution = mock_exec

        # All three should be distinct objects
        assert pool.monitoring is not pool.historical
        assert pool.monitoring is not pool.execution
        assert pool.historical is not pool.execution


class TestGetStatus:
    """Status reporting."""

    def test_get_status_returns_all_connection_info(self):
        """get_status() includes all connection details."""
        client_ids = IbClientIdsConfig(
            monitoring=1,
            execution=2,
            historical_pool=[3],
        )
        config = ConnectionPoolConfig(
            host="localhost",
            port=7497,
            client_ids=client_ids,
        )
        pool = IbConnectionPool(config)

        status = pool.get_status()

        assert "connected" in status
        assert "monitoring" in status
        assert "historical" in status
        assert "execution" in status  # A4
        assert "host" in status
        assert "port" in status

        assert status["host"] == "localhost"
        assert status["port"] == 7497

    def test_get_status_includes_client_ids(self):
        """Status includes client IDs for each connection."""
        client_ids = IbClientIdsConfig(
            monitoring=10,
            execution=11,
            historical_pool=[20],
        )
        config = ConnectionPoolConfig(client_ids=client_ids)
        pool = IbConnectionPool(config)

        status = pool.get_status()

        assert status["monitoring"]["client_id"] == 10
        assert status["execution"]["client_id"] == 11
        assert status["historical"]["client_id"] == 20

    def test_get_status_shows_connected_state(self):
        """Status shows connection state for each connection type."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        mock_mon = MagicMock()
        mock_mon.isConnected.return_value = True
        pool._monitoring = mock_mon

        mock_exec = MagicMock()
        mock_exec.isConnected.return_value = False
        pool._execution = mock_exec

        status = pool.get_status()

        assert status["monitoring"]["connected"] is True
        assert status["historical"]["connected"] is False
        assert status["execution"]["connected"] is False


class TestDisconnect:
    """Disconnect behavior."""

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_all_connections(self):
        """disconnect() cleans up all IB instances."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        mock_mon = MagicMock()
        mock_hist = MagicMock()
        mock_exec = MagicMock()

        pool._monitoring = mock_mon
        pool._historical = mock_hist
        pool._execution = mock_exec
        pool._connected = True

        await pool.disconnect()

        mock_mon.disconnect.assert_called_once()
        mock_hist.disconnect.assert_called_once()
        mock_exec.disconnect.assert_called_once()

        assert pool._monitoring is None
        assert pool._historical is None
        assert pool._execution is None
        assert pool._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_handles_missing_connections(self):
        """disconnect() handles None connections gracefully."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        # Connections are None
        await pool.disconnect()  # Should not raise

        assert pool._connected is False


class TestConnectMethods:
    """Connect method structure (without actual IB)."""

    @pytest.mark.asyncio
    async def test_connect_execution_does_nothing_when_already_connected(self):
        """connect_execution() no-ops if already connected."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        mock_exec = MagicMock()
        mock_exec.isConnected.return_value = True
        pool._execution = mock_exec

        # Should return immediately without calling connectAsync
        await pool.connect_execution()

        # Should not have tried to create new connection
        assert pool._execution is mock_exec

    @pytest.mark.asyncio
    async def test_ensure_execution_connected_calls_connect_when_needed(self):
        """ensure_execution_connected() connects if not connected."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        # Mock connect_execution
        pool.connect_execution = AsyncMock()

        await pool.ensure_execution_connected()

        pool.connect_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_execution_connected_skips_when_connected(self):
        """ensure_execution_connected() skips if already connected."""
        config = ConnectionPoolConfig()
        pool = IbConnectionPool(config)

        mock_exec = MagicMock()
        mock_exec.isConnected.return_value = True
        pool._execution = mock_exec

        pool.connect_execution = AsyncMock()

        await pool.ensure_execution_connected()

        pool.connect_execution.assert_not_called()
