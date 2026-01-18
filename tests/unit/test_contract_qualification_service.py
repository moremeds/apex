"""
Tests for OPT-012: Contract Qualification Service.

Tests batching, caching, debouncing, and fallback behavior.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.adapters.ib.contract_qualification_service import (
    ContractQualificationService,
    QualifiedContract,
)


@dataclass
class MockContract:
    """Mock IB contract for testing."""

    symbol: str
    secType: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    conId: int = 0
    lastTradeDateOrContractMonth: str = ""
    strike: float = 0.0
    right: str = ""


class TestQualifiedContract:
    """Tests for the QualifiedContract dataclass."""

    def test_is_expired_fresh_contract(self):
        """Fresh contracts should not be expired."""
        qc = QualifiedContract(
            contract=MockContract("AAPL"),
            qualified_at=datetime.now(),
            ttl_hours=24,
        )
        assert not qc.is_expired

    def test_is_expired_old_contract(self):
        """Contracts older than TTL should be expired."""
        qc = QualifiedContract(
            contract=MockContract("AAPL"),
            qualified_at=datetime.now() - timedelta(hours=25),
            ttl_hours=24,
        )
        assert qc.is_expired

    def test_is_expired_boundary(self):
        """Contracts just under TTL should not be expired."""
        # Use slightly less than TTL to avoid timing edge cases
        qc = QualifiedContract(
            contract=MockContract("AAPL"),
            qualified_at=datetime.now() - timedelta(hours=23, minutes=59),
            ttl_hours=24,
        )
        assert not qc.is_expired


class TestContractQualificationService:
    """Tests for the ContractQualificationService."""

    @pytest.fixture
    def mock_ib(self):
        """Create a mock IB instance."""
        ib = MagicMock()
        ib.qualifyContractsAsync = AsyncMock()
        return ib

    @pytest.fixture
    def service(self, mock_ib):
        """Create a ContractQualificationService instance."""
        return ContractQualificationService(
            ib=mock_ib,
            debounce_ms=50,  # Short debounce for fast tests
            max_batch_size=5,
            cache_ttl_hours=24,
        )

    # -------------------------------------------------------------------------
    # Cache Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cache_hit(self, service, mock_ib):
        """Cached contracts should be returned without API call."""
        contract = MockContract("AAPL", conId=12345)

        # Pre-populate cache
        service._update_cache(contract)

        # Qualify should return from cache
        result = await service.qualify(contract)

        assert result is not None
        assert result.symbol == "AAPL"
        assert service._stats["cache_hits"] == 1
        mock_ib.qualifyContractsAsync.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss(self, service, mock_ib):
        """Uncached contracts should be queued."""
        contract = MockContract("AAPL")

        # Qualify returns None (queued)
        result = await service.qualify(contract)

        assert result is None
        assert service._stats["cache_misses"] == 1

    @pytest.mark.asyncio
    async def test_cache_expired(self, service, mock_ib):
        """Expired cache entries should be treated as misses."""
        contract = MockContract("AAPL", conId=12345)

        # Pre-populate with expired entry
        service._cache["AAPL:STK:SMART:USD"] = QualifiedContract(
            contract=contract,
            qualified_at=datetime.now() - timedelta(hours=25),
            ttl_hours=24,
        )

        # Should be treated as a miss
        result = await service.qualify(contract)

        assert result is None
        assert service._stats["cache_misses"] == 1

    def test_clear_cache(self, service, mock_ib):
        """Cache should be clearable."""
        contract = MockContract("AAPL", conId=12345)
        service._update_cache(contract)
        assert service.get_cache_size() == 1

        service.clear_cache()
        assert service.get_cache_size() == 0

    def test_clear_expired(self, service, mock_ib):
        """clear_expired should remove only expired entries."""
        # Add fresh entry
        fresh = MockContract("AAPL", conId=1)
        service._cache["AAPL:STK:SMART:USD"] = QualifiedContract(
            contract=fresh,
            qualified_at=datetime.now(),
            ttl_hours=24,
        )

        # Add expired entry
        expired = MockContract("GOOG", conId=2)
        service._cache["GOOG:STK:SMART:USD"] = QualifiedContract(
            contract=expired,
            qualified_at=datetime.now() - timedelta(hours=25),
            ttl_hours=24,
        )

        assert service.get_cache_size() == 2
        removed = service.clear_expired()
        assert removed == 1
        assert service.get_cache_size() == 1

    # -------------------------------------------------------------------------
    # Batching Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_qualify_batch_success(self, service, mock_ib):
        """qualify_batch should batch multiple contracts."""
        contracts = [
            MockContract("AAPL", conId=1),
            MockContract("GOOG", conId=2),
            MockContract("MSFT", conId=3),
        ]

        # Mock qualification
        mock_ib.qualifyContractsAsync.return_value = contracts

        result = await service.qualify_batch(contracts)

        assert len(result) == 3
        mock_ib.qualifyContractsAsync.assert_called_once()
        assert service._stats["batches_sent"] == 1
        assert service._stats["contracts_qualified"] == 3

    @pytest.mark.asyncio
    async def test_qualify_batch_partial_failure(self, service, mock_ib):
        """qualify_batch should handle partial qualification failures."""
        contracts = [
            MockContract("AAPL"),
            MockContract("INVALID"),
            MockContract("GOOG"),
        ]

        # Only 2 of 3 qualify (INVALID returns None-like)
        mock_ib.qualifyContractsAsync.return_value = [
            MockContract("AAPL", conId=1),
            None,  # Failed
            MockContract("GOOG", conId=2),
        ]

        result = await service.qualify_batch(contracts)

        assert len(result) == 2
        assert all(c.conId for c in result)

    @pytest.mark.asyncio
    async def test_qualify_batch_uses_cache(self, service, mock_ib):
        """qualify_batch should use cache for known contracts."""
        cached = MockContract("AAPL", conId=1)
        service._update_cache(cached)

        contracts = [
            MockContract("AAPL"),  # Should hit cache
            MockContract("GOOG"),  # Should need qualification
        ]

        mock_ib.qualifyContractsAsync.return_value = [MockContract("GOOG", conId=2)]

        result = await service.qualify_batch(contracts)

        assert len(result) == 2
        # Only GOOG should be in the API call (1 contract, not 2)
        args = mock_ib.qualifyContractsAsync.call_args[0]
        assert len(args) == 1
        assert args[0].symbol == "GOOG"

    @pytest.mark.asyncio
    async def test_qualify_batch_empty(self, service, mock_ib):
        """qualify_batch should handle empty input."""
        result = await service.qualify_batch([])
        assert result == []
        mock_ib.qualifyContractsAsync.assert_not_called()

    # -------------------------------------------------------------------------
    # Debouncing Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_debounce_collects_requests(self, service, mock_ib):
        """Multiple qualify calls within debounce window should batch."""
        contracts = [
            MockContract("AAPL"),
            MockContract("GOOG"),
            MockContract("MSFT"),
        ]

        qualified = [
            MockContract("AAPL", conId=1),
            MockContract("GOOG", conId=2),
            MockContract("MSFT", conId=3),
        ]
        mock_ib.qualifyContractsAsync.return_value = qualified

        # Queue multiple contracts rapidly
        for c in contracts:
            await service.qualify(c)

        # Wait for debounce
        await asyncio.sleep(0.1)

        # Should result in single batch call
        assert mock_ib.qualifyContractsAsync.call_count == 1

    # -------------------------------------------------------------------------
    # Callback Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_callback_on_cache_hit(self, service, mock_ib):
        """Callback should be invoked for cached contracts."""
        contract = MockContract("AAPL", conId=12345)
        service._update_cache(contract)

        callback_result = []

        async def callback(c):
            callback_result.append(c)

        await service.qualify(contract, callback=callback)

        assert len(callback_result) == 1
        assert callback_result[0].symbol == "AAPL"

    # -------------------------------------------------------------------------
    # Contract Key Tests
    # -------------------------------------------------------------------------

    def test_contract_key_stock(self, service):
        """Stock contract key should include symbol, type, exchange, currency."""
        contract = MockContract("AAPL", secType="STK", exchange="SMART", currency="USD")
        key = service._contract_key(contract)
        assert key == "AAPL:STK:SMART:USD"

    def test_contract_key_option(self, service):
        """Option contract key should include symbol, expiry, strike, right."""
        contract = MockContract(
            symbol="AAPL",
            secType="OPT",
            lastTradeDateOrContractMonth="20240315",
            strike=180.0,
            right="C",
        )
        key = service._contract_key(contract)
        assert key == "AAPL:OPT:20240315:180.0:C"

    # -------------------------------------------------------------------------
    # Fallback Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self, service, mock_ib):
        """Should fall back to sequential on batch timeout."""
        contracts = [
            MockContract("AAPL"),
            MockContract("GOOG"),
        ]

        # First call times out
        mock_ib.qualifyContractsAsync.side_effect = [
            asyncio.TimeoutError(),
            # Sequential fallback calls
            [MockContract("AAPL", conId=1)],
            [MockContract("GOOG", conId=2)],
        ]

        result = await service.qualify_batch(contracts)

        # Should have qualified both via sequential fallback
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self, service, mock_ib):
        """Should fall back to sequential on batch exception."""
        contracts = [MockContract("AAPL")]

        mock_ib.qualifyContractsAsync.side_effect = [
            Exception("Batch failed"),
            [MockContract("AAPL", conId=1)],
        ]

        result = await service.qualify_batch(contracts)
        assert len(result) == 1

    # -------------------------------------------------------------------------
    # Stats Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stats_tracking(self, service, mock_ib):
        """Stats should be tracked correctly."""
        # Cache hit
        cached = MockContract("AAPL", conId=1)
        service._update_cache(cached)
        await service.qualify(MockContract("AAPL"))

        # Cache miss -> batch
        mock_ib.qualifyContractsAsync.return_value = [MockContract("GOOG", conId=2)]
        await service.qualify_batch([MockContract("GOOG")])

        stats = service.get_stats()
        assert stats["cache_hits"] >= 1
        assert stats["cache_misses"] >= 1
        assert stats["batches_sent"] >= 1
        assert stats["contracts_qualified"] >= 1

    def test_get_cached(self, service, mock_ib):
        """get_cached should return cached contract without triggering qualification."""
        contract = MockContract("AAPL", conId=12345)

        # Not cached
        assert service.get_cached(contract) is None

        # Cache it
        service._update_cache(contract)

        # Now should be available
        cached = service.get_cached(contract)
        assert cached is not None
        assert cached.symbol == "AAPL"
