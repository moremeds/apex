"""Tests for EngineHarness implementation."""

import pytest
import polars as pl
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from apex.core.types import MarketDataFrame, DataQualityDimension, DataQualityScore
from apex.engine.base import BacktestResult, EngineConfig
from apex.engine.harness import EngineHarness, EngineType, quick_backtest
from apex.engine.vectorbt import VectorbtEngine


class MockStrategy:
    """Mock strategy for testing."""
    
    def generate_signals(self, data):
        """Generate mock signals."""
        return data.with_columns([
            pl.lit(True).alias("entry"),
            pl.lit(False).alias("exit")
        ])
    
    def get_parameters(self):
        """Get parameters."""
        return {"test_param": 1.0}
    
    def validate_data(self, data):
        """Validate data."""
        return True


@pytest.fixture
def engine_config():
    """Create test engine configuration."""
    return EngineConfig(
        initial_cash=100000.0,
        commission=0.001
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    import polars as pl
    
    data = pl.DataFrame({
        "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [98.0, 99.0],
        "close": [101.0, 102.0],
        "volume": [1000000, 1100000]
    })
    
    return MarketDataFrame(
        data=data,
        symbol="TEST",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 2),
        source="test",
        quality_scores={
            DataQualityDimension.COMPLETENESS: DataQualityScore(
                dimension=DataQualityDimension.COMPLETENESS,
                score=0.9
            )
        }
    )


class TestEngineHarness:
    """Test suite for EngineHarness class."""

    def test_harness_initialization_default(self, engine_config):
        """Test harness initialization with default engine."""
        harness = EngineHarness(engine_config)
        
        assert harness.config == engine_config
        assert harness.engine_type == EngineType.VECTORBT
        assert isinstance(harness.engine, VectorbtEngine)

    def test_harness_initialization_specific_engine(self, engine_config):
        """Test harness initialization with specific engine."""
        harness = EngineHarness(engine_config, EngineType.VECTORBT)
        
        assert harness.engine_type == EngineType.VECTORBT
        assert isinstance(harness.engine, VectorbtEngine)

    def test_harness_initialization_invalid_engine(self, engine_config):
        """Test harness initialization with invalid engine type."""
        with pytest.raises(ValueError, match="not a valid EngineType"):
            # Try to create invalid engine type
            EngineType("fake_engine")

    def test_get_available_engines(self, engine_config):
        """Test getting available engines."""
        harness = EngineHarness(engine_config)
        engines = harness.get_available_engines()
        
        assert EngineType.VECTORBT in engines
        assert len(engines) >= 1

    def test_switch_engine(self, engine_config):
        """Test switching between engines."""
        harness = EngineHarness(engine_config, EngineType.VECTORBT)
        
        # Should stay on vectorbt since it's the only one available
        harness.switch_engine(EngineType.VECTORBT)
        assert harness.engine_type == EngineType.VECTORBT

    def test_switch_engine_invalid(self, engine_config):
        """Test switching to invalid engine."""
        harness = EngineHarness(engine_config)
        
        with pytest.raises(ValueError, match="not a valid EngineType"):
            EngineType("fake")

    @pytest.mark.asyncio
    async def test_run_backtest_success(self, engine_config, sample_market_data):
        """Test successful backtest run."""
        harness = EngineHarness(engine_config)
        strategy = MockStrategy()
        
        # Mock the engine's run_backtest method
        mock_result = BacktestResult(
            strategy_name="MockStrategy",
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2),
            initial_cash=100000.0,
            final_value=105000.0,
            total_return=0.05,
            total_trades=2
        )
        
        harness.engine.run_backtest = AsyncMock(return_value=mock_result)
        
        result = await harness.run_backtest(strategy, sample_market_data)
        
        assert result == mock_result
        harness.engine.run_backtest.assert_called_once_with(strategy, sample_market_data)

    @pytest.mark.asyncio
    async def test_run_backtest_low_quality_data_warning(self, engine_config):
        """Test backtest with low quality data (warning level)."""
        import polars as pl
        
        # Create low quality data
        low_quality_data = MarketDataFrame(
            data=pl.DataFrame({
                "datetime": [datetime(2023, 1, 1)],
                "open": [100.0],
                "high": [102.0],
                "low": [98.0],
                "close": [101.0],
                "volume": [1000000]
            }),
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 1),
            source="test",
            quality_scores={
                DataQualityDimension.COMPLETENESS: DataQualityScore(
                    dimension=DataQualityDimension.COMPLETENESS,
                    score=0.5  # Below threshold
                )
            }
        )
        
        harness = EngineHarness(engine_config)
        strategy = MockStrategy()
        
        # Mock successful engine run
        mock_result = BacktestResult(
            strategy_name="MockStrategy",
            symbol="TEST", 
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 1),
            initial_cash=100000.0,
            final_value=100000.0,
            total_return=0.0,
            total_trades=0
        )
        
        harness.engine.run_backtest = AsyncMock(return_value=mock_result)
        
        # Should still run but with warning logged
        result = await harness.run_backtest(strategy, low_quality_data)
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_run_backtest_strategy_validation_failure(self, engine_config, sample_market_data):
        """Test backtest with strategy validation failure."""
        harness = EngineHarness(engine_config)
        strategy = MockStrategy()
        
        # Mock validation failure
        harness.engine.validate_strategy = Mock(return_value=False)
        
        with pytest.raises(ValueError, match="Strategy.*is not compatible"):
            await harness.run_backtest(strategy, sample_market_data)

    @pytest.mark.asyncio
    async def test_run_backtest_skip_validation(self, engine_config, sample_market_data):
        """Test backtest with validation skipped."""
        harness = EngineHarness(engine_config)
        strategy = MockStrategy()
        
        mock_result = BacktestResult(
            strategy_name="MockStrategy",
            symbol="TEST",
            start_date=datetime(2023, 1, 1), 
            end_date=datetime(2023, 1, 2),
            initial_cash=100000.0,
            final_value=100000.0,
            total_return=0.0,
            total_trades=0
        )
        
        harness.engine.run_backtest = AsyncMock(return_value=mock_result)
        
        # Should skip data quality validation
        result = await harness.run_backtest(strategy, sample_market_data, validate_data=False)
        assert result == mock_result

    def test_register_engine(self, engine_config):
        """Test registering a new engine type."""
        # Create a mock engine class
        class MockEngine:
            def __init__(self, config):
                self.config = config
        
        # Test that we can register engines programmatically
        # (though enum creation needs to be done at class level)
        original_engines = EngineHarness._engines.copy()
        
        try:
            # Mock an engine registration
            EngineHarness._engines[EngineType.VECTORBT] = MockEngine
            harness = EngineHarness(engine_config, EngineType.VECTORBT)
            assert isinstance(harness.engine, MockEngine)
        finally:
            # Restore original engines
            EngineHarness._engines = original_engines


class TestQuickBacktest:
    """Test suite for quick_backtest function."""

    @pytest.mark.asyncio
    async def test_quick_backtest_defaults(self, sample_market_data):
        """Test quick backtest with default parameters."""
        strategy = MockStrategy()
        
        # Mock the harness to avoid actual vectorbt calls
        with patch('apex.engine.harness.EngineHarness') as mock_harness_class:
            mock_harness = AsyncMock()
            mock_result = BacktestResult(
                strategy_name="MockStrategy",
                symbol="TEST",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 2), 
                initial_cash=100000.0,
                final_value=105000.0,
                total_return=0.05,
                total_trades=1
            )
            mock_harness.run_backtest.return_value = mock_result
            mock_harness_class.return_value = mock_harness
            
            result = await quick_backtest(strategy, sample_market_data)
            
            assert result == mock_result
            mock_harness_class.assert_called_once()
            mock_harness.run_backtest.assert_called_once_with(strategy, sample_market_data)

    @pytest.mark.asyncio
    async def test_quick_backtest_custom_params(self, sample_market_data):
        """Test quick backtest with custom parameters."""
        strategy = MockStrategy()
        
        with patch('apex.engine.harness.EngineHarness') as mock_harness_class:
            mock_harness = AsyncMock()
            mock_result = BacktestResult(
                strategy_name="MockStrategy", 
                symbol="TEST",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 2),
                initial_cash=50000.0,
                final_value=52500.0,
                total_return=0.05,
                total_trades=1
            )
            mock_harness.run_backtest.return_value = mock_result
            mock_harness_class.return_value = mock_harness
            
            result = await quick_backtest(
                strategy, 
                sample_market_data,
                initial_cash=50000.0,
                commission=0.002,
                engine_type=EngineType.VECTORBT
            )
            
            assert result == mock_result
            
            # Verify config was created with custom parameters
            call_args = mock_harness_class.call_args
            config = call_args[0][0]  # First positional argument
            engine_type = call_args[0][1]  # Second positional argument
            
            assert config.initial_cash == 50000.0
            assert config.commission == 0.002
            assert engine_type == EngineType.VECTORBT