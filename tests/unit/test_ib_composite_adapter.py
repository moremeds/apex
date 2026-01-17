"""
Tests for OPT-016: IB Composite Adapter.

Tests the IbCompositeAdapter which wraps split adapters via connection pool.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.models import IbClientIdsConfig
from src.infrastructure.adapters.ib.composite_adapter import IbCompositeAdapter
from src.infrastructure.adapters.ib.connection_pool import ConnectionPoolConfig


@dataclass
class MockPosition:
    """Mock position for testing."""

    symbol: str
    quantity: float
    avg_cost: float
    asset_type: str = "STK"
    underlying: str = ""
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None
    multiplier: int = 100
    source: str = "test"


@dataclass
class MockPositionSnapshot:
    """Mock position snapshot for testing."""

    symbol: str
    quantity: float
    avg_cost: float
    asset_type: str = "STK"
    underlying: str = ""
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None
    multiplier: int = 100
    source: str = "test"


@dataclass
class MockAccountSnapshot:
    """Mock account snapshot for testing."""

    account_id: str = "U12345"
    net_liquidation: float = 100000.0
    cash_balance: float = 50000.0
    buying_power: float = 200000.0
    initial_margin: float = 25000.0
    maintenance_margin: float = 20000.0
    currency: str = "USD"
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MockMarketData:
    """Mock market data for testing."""

    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    mid: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TestIbCompositeAdapter:
    """Tests for IbCompositeAdapter."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return ConnectionPoolConfig(
            host="127.0.0.1",
            port=7497,
            client_ids=IbClientIdsConfig(
                monitoring=1,
                execution=2,
                historical_pool=[3, 4, 5],
            ),
        )

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        pool = MagicMock()
        pool.connect = AsyncMock()
        pool.disconnect = AsyncMock()
        pool.is_connected.return_value = True
        pool.get_status.return_value = {
            "connected": True,
            "monitoring": {"connected": True},
            "historical": {"connected": True},
            "execution": {"connected": True},
        }

        # Mock IB instances
        pool.monitoring = MagicMock()
        pool.monitoring.isConnected.return_value = True

        pool.historical = MagicMock()
        pool.historical.isConnected.return_value = True

        pool.execution = MagicMock()
        pool.execution.isConnected.return_value = True

        return pool

    @pytest.fixture
    def adapter(self, pool_config):
        """Create an IbCompositeAdapter instance."""
        return IbCompositeAdapter(pool_config)

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization(self, pool_config):
        """Test adapter initializes correctly."""
        adapter = IbCompositeAdapter(pool_config)

        assert adapter._pool_config == pool_config
        assert adapter._pool is None
        assert adapter._live_adapter is None
        assert adapter._historical_adapter is None
        assert adapter._execution_adapter is None
        assert not adapter._connected

    def test_initialization_with_event_bus(self, pool_config):
        """Test adapter initializes with event bus."""
        event_bus = MagicMock()
        adapter = IbCompositeAdapter(pool_config, event_bus=event_bus)

        assert adapter._event_bus == event_bus

    # -------------------------------------------------------------------------
    # Connection Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_is_connected_before_connect(self, adapter):
        """Test is_connected returns False before connect."""
        assert not adapter.is_connected()

    @pytest.mark.asyncio
    async def test_disconnect_before_connect(self, adapter):
        """Test disconnect is safe to call before connect."""
        await adapter.disconnect()
        assert not adapter.is_connected()

    # -------------------------------------------------------------------------
    # Interface Tests (MarketDataProvider)
    # -------------------------------------------------------------------------

    def test_supports_streaming(self, adapter):
        """Test supports_streaming returns True."""
        assert adapter.supports_streaming() is True

    def test_supports_greeks(self, adapter):
        """Test supports_greeks returns True."""
        assert adapter.supports_greeks() is True

    # -------------------------------------------------------------------------
    # Cache Tests
    # -------------------------------------------------------------------------

    def test_update_cache_basic(self, adapter):
        """Test cache update works."""
        md = MockMarketData(symbol="AAPL", last=150.0)
        adapter._update_cache("AAPL", md)

        assert "AAPL" in adapter._market_data_cache
        assert adapter._market_data_cache["AAPL"].last == 150.0

    def test_update_cache_lru_eviction(self, adapter):
        """Test cache evicts old entries when full."""
        adapter._market_data_cache_max_size = 3

        # Add 4 items
        for i in range(4):
            md = MockMarketData(symbol=f"SYM{i}", last=float(i))
            adapter._update_cache(f"SYM{i}", md)

        # Should have evicted first entry
        assert len(adapter._market_data_cache) == 3
        assert "SYM0" not in adapter._market_data_cache
        assert "SYM3" in adapter._market_data_cache

    def test_update_cache_moves_to_end(self, adapter):
        """Test cache moves accessed items to end (LRU)."""
        adapter._market_data_cache_max_size = 3

        # Add 3 items
        for i in range(3):
            md = MockMarketData(symbol=f"SYM{i}", last=float(i))
            adapter._update_cache(f"SYM{i}", md)

        # Access first item
        md = MockMarketData(symbol="SYM0", last=100.0)
        adapter._update_cache("SYM0", md)

        # Add new item - should evict SYM1 (now oldest)
        md = MockMarketData(symbol="SYM3", last=3.0)
        adapter._update_cache("SYM3", md)

        assert "SYM0" in adapter._market_data_cache  # Was accessed
        assert "SYM1" not in adapter._market_data_cache  # Evicted
        assert "SYM2" in adapter._market_data_cache
        assert "SYM3" in adapter._market_data_cache

    # -------------------------------------------------------------------------
    # Empty Response Tests (when adapters not connected)
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fetch_positions_no_adapter(self, adapter):
        """Test fetch_positions returns empty when adapter not connected."""
        positions = await adapter.fetch_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_fetch_market_data_no_adapter(self, adapter):
        """Test fetch_market_data returns empty when adapter not connected."""
        positions = [MockPosition("AAPL", 100, 150.0)]
        market_data = await adapter.fetch_market_data(positions)
        assert market_data == []

    @pytest.mark.asyncio
    async def test_fetch_quotes_no_adapter(self, adapter):
        """Test fetch_quotes returns empty when adapter not connected."""
        quotes = await adapter.fetch_quotes(["AAPL", "GOOG"])
        assert quotes == {}

    @pytest.mark.asyncio
    async def test_fetch_orders_no_adapter(self, adapter):
        """Test fetch_orders returns empty when adapter not connected."""
        orders = await adapter.fetch_orders()
        assert orders == []

    @pytest.mark.asyncio
    async def test_fetch_trades_no_adapter(self, adapter):
        """Test fetch_trades returns empty when adapter not connected."""
        trades = await adapter.fetch_trades()
        assert trades == []

    @pytest.mark.asyncio
    async def test_fetch_historical_bars_no_adapter(self, adapter):
        """Test fetch_historical_bars returns empty when adapter not connected."""
        bars = await adapter.fetch_historical_bars("AAPL", "1d")
        assert bars == []

    # -------------------------------------------------------------------------
    # Streaming Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_subscribe_no_adapter(self, adapter):
        """Test subscribe is safe when adapter not connected."""
        await adapter.subscribe(["AAPL", "GOOG"])

    @pytest.mark.asyncio
    async def test_unsubscribe_no_adapter(self, adapter):
        """Test unsubscribe is safe when adapter not connected."""
        await adapter.unsubscribe(["AAPL", "GOOG"])

    def test_enable_streaming_no_adapter(self, adapter):
        """Test enable_streaming is safe when adapter not connected."""
        adapter.enable_streaming()

    def test_disable_streaming_no_adapter(self, adapter):
        """Test disable_streaming is safe when adapter not connected."""
        adapter.disable_streaming()

    def test_get_latest_no_adapter(self, adapter):
        """Test get_latest returns None when adapter not connected."""
        result = adapter.get_latest("AAPL")
        assert result is None

    def test_get_latest_from_cache(self, adapter):
        """Test get_latest returns cached data."""
        md = MockMarketData(symbol="AAPL", last=150.0)
        adapter._market_data_cache["AAPL"] = md

        result = adapter.get_latest("AAPL")
        assert result is not None
        assert result.last == 150.0

    # -------------------------------------------------------------------------
    # Status Tests
    # -------------------------------------------------------------------------

    def test_get_connection_info_disconnected(self, adapter):
        """Test get_connection_info when disconnected."""
        info = adapter.get_connection_info()

        assert info["adapter_type"] == "composite"
        assert info["connected"] is False
        assert info["pool_status"] is None
        assert info["live_connected"] is False
        assert info["historical_connected"] is False
        assert info["execution_connected"] is False
