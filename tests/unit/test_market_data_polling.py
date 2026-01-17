"""
Unit tests for market data polling mechanism.

Tests the robust data population waiting with timeout and early exit.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.infrastructure.adapters.market_data_fetcher import MarketDataFetcher


@pytest.mark.asyncio
async def test_wait_for_data_population_all_ready():
    """Test that polling exits early when all data is ready."""
    # Create mock IB instance
    ib = MagicMock()

    # Create fetcher with short intervals for testing
    fetcher = MarketDataFetcher(ib, data_timeout=5.0, poll_interval=0.05)

    # Create mock tickers with data already populated
    ticker1 = MagicMock()
    ticker1.bid = 100.0
    ticker1.ask = 101.0
    ticker1.last = 100.5

    ticker2 = MagicMock()
    ticker2.bid = 200.0
    ticker2.ask = 201.0
    ticker2.last = 200.5

    tickers = [ticker1, ticker2]

    # Measure time
    start = asyncio.get_event_loop().time()

    # Wait for data - should exit immediately since data is ready
    populated = await fetcher._wait_for_data_population(tickers, timeout=5.0)

    elapsed = asyncio.get_event_loop().time() - start

    # Verify all tickers reported as populated
    assert populated == 2

    # Verify it exited quickly (< 0.5s) instead of waiting full timeout
    assert elapsed < 0.5, f"Should exit early, but took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_wait_for_data_population_timeout():
    """Test that polling times out when data doesn't arrive."""
    # Create mock IB instance
    ib = MagicMock()

    # Create fetcher with short timeout for testing
    fetcher = MarketDataFetcher(ib, data_timeout=0.3, poll_interval=0.05)

    # Create mock tickers with NO data (NaN values)
    ticker1 = MagicMock()
    ticker1.bid = float("nan")
    ticker1.ask = float("nan")
    ticker1.last = float("nan")
    ticker1.close = None  # Also set close to prevent MagicMock comparison issues

    ticker2 = MagicMock()
    ticker2.bid = None
    ticker2.ask = None
    ticker2.last = None
    ticker2.close = None

    tickers = [ticker1, ticker2]

    # Measure time
    start = asyncio.get_event_loop().time()

    # Wait for data - should timeout
    populated = await fetcher._wait_for_data_population(tickers, timeout=0.3)

    elapsed = asyncio.get_event_loop().time() - start

    # Verify no tickers populated
    assert populated == 0

    # Verify it waited approximately the timeout duration
    assert 0.3 <= elapsed < 0.5, f"Expected ~0.3s timeout, got {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_wait_for_data_population_partial():
    """Test polling with partial data availability."""
    # Create mock IB instance
    ib = MagicMock()

    # Create fetcher
    fetcher = MarketDataFetcher(ib, data_timeout=0.5, poll_interval=0.05)

    # Create mock tickers - one with data, one without
    ticker1 = MagicMock()
    ticker1.bid = 100.0
    ticker1.ask = 101.0
    ticker1.last = 100.5
    ticker1.close = None

    ticker2 = MagicMock()
    ticker2.bid = None
    ticker2.ask = None
    ticker2.last = None
    ticker2.close = None

    tickers = [ticker1, ticker2]

    # Wait for data - should timeout with partial data
    populated = await fetcher._wait_for_data_population(tickers, timeout=0.5)

    # Verify partial population
    assert populated == 1


@pytest.mark.asyncio
async def test_wait_for_data_population_delayed():
    """Test polling that waits for delayed data arrival."""
    # Create mock IB instance
    ib = MagicMock()

    # Create fetcher
    fetcher = MarketDataFetcher(ib, data_timeout=2.0, poll_interval=0.05)

    # Create mock ticker with initially no data
    ticker = MagicMock()
    ticker.bid = None
    ticker.ask = None
    ticker.last = None
    ticker.close = None

    tickers = [ticker]

    # Schedule data to arrive after 0.2 seconds
    async def populate_data_later():
        await asyncio.sleep(0.2)
        ticker.bid = 100.0
        ticker.ask = 101.0

    # Start background task to populate data
    asyncio.create_task(populate_data_later())

    # Measure time
    start = asyncio.get_event_loop().time()

    # Wait for data - should detect when it arrives
    populated = await fetcher._wait_for_data_population(tickers, timeout=2.0)

    elapsed = asyncio.get_event_loop().time() - start

    # Verify ticker populated
    assert populated == 1

    # Verify it waited for data but exited early (not full timeout)
    assert 0.2 <= elapsed < 1.0, f"Expected ~0.2s wait, got {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_wait_for_data_zero_timeout():
    """Test polling with zero timeout."""
    # Create mock IB instance
    ib = MagicMock()

    # Create fetcher with zero timeout
    fetcher = MarketDataFetcher(ib, data_timeout=0.0, poll_interval=0.05)

    # Create mock ticker with no data
    ticker = MagicMock()
    ticker.bid = None
    ticker.ask = None
    ticker.last = None
    ticker.close = None

    tickers = [ticker]

    # Wait with zero timeout - should return immediately
    start = asyncio.get_event_loop().time()
    populated = await fetcher._wait_for_data_population(tickers, timeout=0.0)
    elapsed = asyncio.get_event_loop().time() - start

    # Verify no data and quick exit
    assert populated == 0
    assert elapsed < 0.1, f"Should exit immediately, took {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_wait_for_data_empty_ticker_list():
    """Test polling with empty ticker list."""
    # Create mock IB instance
    ib = MagicMock()

    # Create fetcher
    fetcher = MarketDataFetcher(ib, data_timeout=1.0, poll_interval=0.05)

    # Empty ticker list
    tickers = []

    # Wait - should return immediately with 0 populated
    start = asyncio.get_event_loop().time()
    populated = await fetcher._wait_for_data_population(tickers, timeout=1.0)
    elapsed = asyncio.get_event_loop().time() - start

    # Verify
    assert populated == 0
    assert elapsed < 0.1, "Should exit immediately for empty list"
