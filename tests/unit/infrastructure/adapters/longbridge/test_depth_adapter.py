"""Tests for Longbridge DepthAdapter — mock-based tests."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.domain.interfaces.depth_provider import DepthProvider, DepthSnapshot
from src.infrastructure.adapters.longbridge.depth_adapter import LongbridgeDepthAdapter


@pytest.fixture
def adapter():
    return LongbridgeDepthAdapter()


def test_implements_depth_provider(adapter):
    assert isinstance(adapter, DepthProvider)


def test_get_latest_depth_empty(adapter):
    assert adapter.get_latest_depth("AAPL") is None


def test_set_depth_callback_and_push(adapter):
    cb = MagicMock()
    adapter.set_depth_callback(cb)

    # Simulate SDK depth push
    mock_event = MagicMock()
    bid = MagicMock()
    bid.price = 185.40
    bid.volume = 100
    bid.order_num = 3
    ask = MagicMock()
    ask.price = 185.50
    ask.volume = 200
    ask.order_num = 5
    mock_event.bids = [bid]
    mock_event.asks = [ask]

    adapter._on_sdk_depth("AAPL.US", mock_event)
    cb.assert_called_once()

    snap = cb.call_args[0][0]
    assert isinstance(snap, DepthSnapshot)
    assert snap.symbol == "AAPL"
    assert snap.source == "longbridge"
    assert len(snap.bids) == 1
    assert len(snap.asks) == 1
    assert snap.bids[0].price == 185.40
    assert snap.asks[0].price == 185.50
    assert snap.asks[0].order_count == 5


def test_depth_cached_after_push(adapter):
    mock_event = MagicMock()
    mock_event.bids = []
    mock_event.asks = []

    adapter._on_sdk_depth("SPY.US", mock_event)
    snap = adapter.get_latest_depth("SPY")
    assert snap is not None
    assert snap.symbol == "SPY"


def test_attach_context(adapter):
    mock_ctx = MagicMock()
    adapter.attach_context(mock_ctx)
    assert adapter._connected is True
    mock_ctx.set_on_depth.assert_called_once()


def test_set_depth_callback_none(adapter):
    adapter.set_depth_callback(None)
    # Pushing with no callback should not crash
    mock_event = MagicMock()
    mock_event.bids = []
    mock_event.asks = []
    adapter._on_sdk_depth("AAPL.US", mock_event)
