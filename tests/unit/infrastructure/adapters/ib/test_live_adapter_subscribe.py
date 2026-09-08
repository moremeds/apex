"""
Tests for IbLiveAdapter.subscribe_quotes / unsubscribe_quotes real IB wiring.

Verifies that:
- qualifyContractsAsync and reqMktData are called per symbol on subscribe
- cancelMktData is called per symbol on unsubscribe
- _quote_callback fires when a fake pendingTickersEvent is dispatched
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from math import nan
from threading import Lock
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy transitive imports before importing live_adapter
# ---------------------------------------------------------------------------

for _mod in [
    "scipy",
    "scipy.stats",
    "futu",
    "futu.api",
    "openssl",
    "redis",
    "aiohttp",
    "talib",
    "pandas_market_calendars",
    "exchange_calendars",
    "psycopg2",
    "asyncpg",
    "sqlalchemy",
    "alembic",
    "boto3",
    "botocore",
    "anthropic",
    "openai",
    "matplotlib",
    "seaborn",
    "plotly",
    "sklearn",
    "ta",
    "mplfinance",
    "rich",
    "textual",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Stub out entire backtest package to avoid scipy/pandas_market_calendars deps
_backtest_stub = MagicMock()
for _mod in [
    "src.backtest",
    "src.backtest.data",
    "src.backtest.data.calendar",
    "src.backtest.analysis",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = _backtest_stub

# Stub out futu adapter package
for _mod in [
    "src.infrastructure.adapters.futu",
    "src.infrastructure.adapters.futu.adapter",
    "src.infrastructure.adapters.futu.order_fetcher",
    "src.infrastructure.adapters.futu.converters",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Now we can safely import the module under test
from src.infrastructure.adapters.ib.live_adapter import IbLiveAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


@dataclass
class FakeTicker:
    """Simulates an ib_async Ticker returned by reqMktData."""

    contract: Any = None
    bid: float = 100.0
    ask: float = 101.0
    last: float = 100.5


@dataclass
class FakeContract:
    """Simulates a qualified ib_async Contract."""

    symbol: str
    conId: int
    secType: str = "STK"


def _make_adapter() -> IbLiveAdapter:
    """Return an IbLiveAdapter with minimal state, bypassing real network init."""
    adapter = IbLiveAdapter.__new__(IbLiveAdapter)
    adapter._connected = True
    adapter._event_bus = None
    # Replicate __init__ state
    adapter._market_data_fetcher = None
    adapter._market_data_cache = OrderedDict()
    adapter._market_data_cache_max_size = 1000
    adapter._market_data_cache_lock = Lock()
    adapter._subscribed_symbols = []
    adapter._quote_callback = None
    adapter._active_tickers = {}
    adapter._ticker_to_symbol = {}
    adapter._streaming_lock = Lock()
    adapter._streaming_active = False
    adapter._position_cache = None
    adapter._position_cache_time = None
    adapter._position_cache_ttl_sec = 10
    adapter._position_callback = None
    adapter._position_subscription_active = False
    adapter._account_cache = None
    adapter._account_cache_time = None
    adapter._account_cache_ttl_sec = 10
    adapter._account_callback = None
    adapter._previous_close_cache = {}
    adapter._previous_close_cache_ttl_sec = 3600
    adapter._qual_service = None
    adapter._qualified_contract_cache = {}
    adapter._contract_cache_lock = Lock()
    return adapter


def _attach_mock_ib(adapter: IbLiveAdapter, symbols: List[str]) -> MagicMock:
    """Attach a mock IB that returns fake qualified contracts and tickers."""
    mock_ib = MagicMock()
    mock_ib.isConnected.return_value = True

    qualified = [FakeContract(symbol=sym, conId=i + 1) for i, sym in enumerate(symbols)]
    mock_ib.qualifyContractsAsync = AsyncMock(return_value=qualified)

    def _req_mkt_data(contract, *args, **kwargs):
        return FakeTicker(contract=contract)

    mock_ib.reqMktData.side_effect = _req_mkt_data
    mock_ib.pendingTickersEvent = MagicMock()

    adapter.ib = mock_ib
    return mock_ib


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_calls_qualify_and_reqmktdata():
    """subscribe_quotes must qualify each symbol and call reqMktData once per symbol."""
    adapter = _make_adapter()
    symbols = ["AAPL", "MSFT"]
    mock_ib = _attach_mock_ib(adapter, symbols)

    await adapter.subscribe_quotes(symbols)

    mock_ib.qualifyContractsAsync.assert_called_once()
    call_args = mock_ib.qualifyContractsAsync.call_args[0]
    assert len(call_args) == 2
    assert {c.symbol for c in call_args} == {"AAPL", "MSFT"}

    assert mock_ib.reqMktData.call_count == 2
    assert set(adapter._active_tickers.keys()) == {"AAPL", "MSFT"}
    assert set(adapter._subscribed_symbols) == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_subscribe_skips_already_subscribed():
    """subscribe_quotes must not re-qualify or re-subscribe an already-active symbol."""
    adapter = _make_adapter()
    mock_ib = _attach_mock_ib(adapter, ["AAPL"])

    await adapter.subscribe_quotes(["AAPL"])
    first_count = mock_ib.reqMktData.call_count

    await adapter.subscribe_quotes(["AAPL"])

    assert mock_ib.reqMktData.call_count == first_count


@pytest.mark.asyncio
async def test_subscribe_registers_pending_tickers_event_once():
    """pendingTickersEvent handler must be registered exactly once across multiple subscribes."""
    adapter = _make_adapter()
    mock_ib = _attach_mock_ib(adapter, ["AAPL", "TSLA"])

    # Capture the original mock before += reassigns the attribute
    original_event = mock_ib.pendingTickersEvent

    await adapter.subscribe_quotes(["AAPL"])
    # Re-attach returns new qualified list for TSLA
    mock_ib.qualifyContractsAsync = AsyncMock(
        return_value=[FakeContract(symbol="TSLA", conId=99)]
    )
    await adapter.subscribe_quotes(["TSLA"])

    # += calls __iadd__ on the original event object exactly once
    assert original_event.__iadd__.call_count == 1


@pytest.mark.asyncio
async def test_unsubscribe_calls_cancel_mkt_data():
    """unsubscribe_quotes must call cancelMktData for each unsubscribed symbol."""
    adapter = _make_adapter()
    mock_ib = _attach_mock_ib(adapter, ["AAPL", "MSFT"])

    await adapter.subscribe_quotes(["AAPL", "MSFT"])
    await adapter.unsubscribe_quotes(["AAPL"])

    mock_ib.cancelMktData.assert_called_once()
    assert "AAPL" not in adapter._active_tickers
    assert "AAPL" not in adapter._subscribed_symbols
    assert "MSFT" in adapter._active_tickers


@pytest.mark.asyncio
async def test_quote_callback_fires_on_pending_tickers_event():
    """When _on_pending_tickers fires with a subscribed ticker, _quote_callback must be called."""
    adapter = _make_adapter()
    _attach_mock_ib(adapter, ["AAPL"])
    await adapter.subscribe_quotes(["AAPL"])

    received: List[Any] = []
    adapter.set_quote_callback(received.append)

    aapl_ticker = adapter._active_tickers["AAPL"]
    adapter._on_pending_tickers([aapl_ticker])

    assert len(received) == 1
    tick = received[0]
    assert tick.symbol == "AAPL"
    assert tick.bid == pytest.approx(100.0)
    assert tick.ask == pytest.approx(101.0)
    assert tick.last == pytest.approx(100.5)


@pytest.mark.asyncio
async def test_unknown_ticker_does_not_raise():
    """_on_pending_tickers must silently ignore tickers not in _ticker_to_symbol."""
    adapter = _make_adapter()
    adapter.ib = MagicMock()

    received: List[Any] = []
    adapter.set_quote_callback(received.append)

    alien_ticker = FakeTicker()
    adapter._on_pending_tickers([alien_ticker])

    assert received == []


def test_nan_fields_coerced_to_none():
    """_ticker_to_market_data must convert NaN float fields to None."""
    adapter = _make_adapter()

    nan_ticker = FakeTicker(bid=nan, ask=nan, last=nan)
    md = adapter._ticker_to_market_data("AAPL", nan_ticker)

    assert md.bid is None
    assert md.ask is None
    assert md.last is None
    assert md.mid is None
