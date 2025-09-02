"""Tests for Yahoo Finance data provider."""

import pytest
from datetime import datetime, date
from unittest.mock import Mock, patch, AsyncMock
import polars as pl
import pandas as pd

from apex.core.types import ProviderConfig, MarketDataFrame
from apex.data.providers.yahoo import YahooDataProvider


@pytest.fixture
def provider_config():
    """Create a test provider configuration."""
    return ProviderConfig(name="test_yahoo", quality_threshold=0.7)


@pytest.fixture
def yahoo_provider(provider_config):
    """Create a Yahoo data provider for testing."""
    return YahooDataProvider(provider_config)


@pytest.fixture
def sample_yahoo_data():
    """Create sample Yahoo Finance data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    data = {
        'Date': dates,
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'High': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        'Volume': [1000000] * 10,
    }
    return pd.DataFrame(data)


class TestYahooDataProvider:
    """Test cases for Yahoo Finance data provider."""

    def test_initialization_default_config(self):
        """Test provider initialization with default config."""
        provider = YahooDataProvider()
        assert provider.config.name == "yahoo_finance"
        assert provider.config.quality_threshold == 0.7

    def test_initialization_custom_config(self, provider_config):
        """Test provider initialization with custom config."""
        provider = YahooDataProvider(provider_config)
        assert provider.config.name == "test_yahoo"
        assert provider.config.quality_threshold == 0.7

    @patch('yfinance.Ticker')
    @pytest.mark.asyncio
    async def test_fetch_raw_data_success(self, mock_ticker, yahoo_provider, sample_yahoo_data):
        """Test successful data fetching from Yahoo Finance."""
        # Mock the ticker and history method
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_yahoo_data
        mock_ticker.return_value = mock_ticker_instance

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        result = await yahoo_provider._fetch_raw_data("AAPL", start_date, end_date)
        
        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10
        expected_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in result.columns for col in expected_columns)
        
        # Verify data types
        assert result.schema['datetime'] == pl.Datetime
        assert result.schema['open'] == pl.Float64
        assert result.schema['volume'] == pl.Int64

    @patch('yfinance.Ticker')
    @pytest.mark.asyncio
    async def test_fetch_raw_data_empty_response(self, mock_ticker, yahoo_provider):
        """Test handling of empty data response."""
        # Mock empty response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        with pytest.raises(ValueError, match="No data available"):
            await yahoo_provider._fetch_raw_data("INVALID", start_date, end_date)

    @patch('yfinance.Ticker')
    @pytest.mark.asyncio
    async def test_fetch_raw_data_api_error(self, mock_ticker, yahoo_provider):
        """Test handling of API errors."""
        # Mock API error
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_ticker_instance

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        with pytest.raises(Exception, match="API Error"):
            await yahoo_provider._fetch_raw_data("AAPL", start_date, end_date)

    @patch('yfinance.Ticker')
    @pytest.mark.asyncio
    async def test_fetch_data_with_caching(self, mock_ticker, yahoo_provider, sample_yahoo_data, tmp_path):
        """Test data fetching with caching enabled."""
        # Set cache directory
        yahoo_provider.cache_dir = tmp_path / "cache"
        yahoo_provider.cache_dir.mkdir(parents=True)
        
        # Mock the ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_yahoo_data
        mock_ticker.return_value = mock_ticker_instance

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        # First fetch should hit the API
        result1 = await yahoo_provider.fetch_data("AAPL", start_date, end_date)
        assert isinstance(result1, MarketDataFrame)
        assert len(result1.data) == 10
        assert mock_ticker_instance.history.call_count == 1
        
        # Second fetch should use cache
        result2 = await yahoo_provider.fetch_data("AAPL", start_date, end_date)
        assert isinstance(result2, MarketDataFrame)
        assert len(result2.data) == 10
        # Should not call API again
        assert mock_ticker_instance.history.call_count == 1

    def test_get_cache_key(self, yahoo_provider):
        """Test cache key generation."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        key1 = yahoo_provider.get_cache_key("AAPL", start_date, end_date)
        key2 = yahoo_provider.get_cache_key("AAPL", start_date, end_date)
        key3 = yahoo_provider.get_cache_key("GOOGL", start_date, end_date)
        
        # Same parameters should generate same key
        assert key1 == key2
        # Different parameters should generate different keys
        assert key1 != key3

    @patch('yfinance.Ticker')
    def test_validate_symbol(self, mock_ticker, yahoo_provider):
        """Test symbol validation."""
        # Mock valid symbol
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"symbol": "AAPL", "shortName": "Apple Inc."}
        mock_ticker.return_value = mock_ticker_instance
        
        assert yahoo_provider._validate_symbol("AAPL") is True
        
        # Mock invalid symbol
        mock_ticker_instance.info = {}
        assert yahoo_provider._validate_symbol("INVALID") is False

    def test_ensure_required_columns_valid(self, yahoo_provider):
        """Test column validation with valid data."""
        df = pl.DataFrame({
            "datetime": [datetime(2023, 1, 1)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000000],
            "extra_col": ["extra"],  # Should be filtered out
        })
        
        result = yahoo_provider._ensure_required_columns(df)
        expected_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        assert result.columns == expected_columns

    def test_ensure_required_columns_missing(self, yahoo_provider):
        """Test column validation with missing columns."""
        df = pl.DataFrame({
            "datetime": [datetime(2023, 1, 1)],
            "open": [100.0],
            "high": [101.0],
            # Missing 'low', 'close', 'volume'
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            yahoo_provider._ensure_required_columns(df)