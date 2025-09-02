"""Tests for VectorbtEngine implementation."""

import pandas as pd
import polars as pl
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from apex.core.types import MarketDataFrame
from apex.engine.base import EngineConfig
from apex.engine.vectorbt import VectorbtEngine


class MockStrategy:
    """Mock strategy for testing purposes."""
    
    def __init__(self, signals_data: dict = None):
        """Initialize mock strategy with optional signals data."""
        self.signals_data = signals_data or {"entry": [True, False, True, False], "exit": [False, True, False, True]}
    
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate mock signals."""
        if len(self.signals_data.get("entry", [])) != len(data):
            # Adjust signal length to match data
            n_rows = len(data)
            entry_signals = [True if i % 4 == 0 else False for i in range(n_rows)]
            exit_signals = [True if i % 4 == 2 else False for i in range(n_rows)]
        else:
            entry_signals = self.signals_data.get("entry", [])
            exit_signals = self.signals_data.get("exit", [])
        
        return pl.DataFrame({
            "datetime": data["datetime"],
            "entry": entry_signals,
            "exit": exit_signals
        })
    
    def get_parameters(self) -> dict:
        """Get strategy parameters."""
        return {"test_param": 1.0}
    
    def validate_data(self, data: pl.DataFrame) -> bool:
        """Validate data requirements."""
        required_columns = ["datetime", "open", "high", "low", "close", "volume"]
        return all(col in data.columns for col in required_columns)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)
    
    dates = [start_date + timedelta(days=i) for i in range(10)]
    
    # Create realistic OHLCV data
    base_price = 100.0
    data = []
    
    for i, date in enumerate(dates):
        price_change = (i - 5) * 0.02  # Some price movement
        open_price = base_price + price_change
        high_price = open_price + abs(price_change) + 1
        low_price = open_price - abs(price_change) - 1
        close_price = open_price + price_change * 0.5
        volume = 1000000 + i * 100000
        
        data.append({
            "datetime": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        })
    
    df = pl.DataFrame(data)
    
    return MarketDataFrame(
        data=df,
        symbol="TEST",
        start_date=start_date,
        end_date=end_date,
        source="test",
        quality_scores={}
    )


@pytest.fixture
def engine_config():
    """Create test engine configuration."""
    return EngineConfig(
        initial_cash=100000.0,
        commission=0.001,
        slippage=0.0001,
        max_position_size=0.1
    )


@pytest.fixture 
def vectorbt_engine(engine_config):
    """Create VectorbtEngine instance for testing."""
    return VectorbtEngine(engine_config)


class TestVectorbtEngine:
    """Test suite for VectorbtEngine class."""

    def test_engine_initialization(self, engine_config):
        """Test engine initialization."""
        engine = VectorbtEngine(engine_config)
        
        assert engine.config == engine_config
        assert engine.config.initial_cash == 100000.0
        assert engine.config.commission == 0.001

    def test_validate_strategy_success(self, vectorbt_engine, sample_market_data):
        """Test successful strategy validation."""
        strategy = MockStrategy()
        
        result = vectorbt_engine.validate_strategy(strategy, sample_market_data)
        assert result is True

    def test_validate_strategy_failure(self, vectorbt_engine):
        """Test strategy validation failure."""
        # Create market data missing required columns
        incomplete_data = MarketDataFrame(
            data=pl.DataFrame({"datetime": [datetime.now()], "price": [100.0]}),
            symbol="TEST",
            start_date=datetime.now(),
            end_date=datetime.now(),
            source="test"
        )
        
        strategy = MockStrategy()
        result = vectorbt_engine.validate_strategy(strategy, incomplete_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_run_backtest_basic(self, vectorbt_engine, sample_market_data):
        """Test basic backtest execution."""
        strategy = MockStrategy()
        
        result = await vectorbt_engine.run_backtest(strategy, sample_market_data)
        
        assert result is not None
        assert result.strategy_name == "MockStrategy"
        assert result.symbol == "TEST"
        assert result.initial_cash == 100000.0
        assert isinstance(result.final_value, float)
        assert isinstance(result.total_return, float)

    @pytest.mark.asyncio  
    async def test_run_backtest_with_invalid_strategy(self, vectorbt_engine):
        """Test backtest with invalid strategy."""
        # Create invalid market data
        invalid_data = MarketDataFrame(
            data=pl.DataFrame({"datetime": [datetime.now()], "price": [100.0]}),
            symbol="TEST",
            start_date=datetime.now(),
            end_date=datetime.now(),
            source="test"
        )
        
        strategy = MockStrategy()
        
        with pytest.raises(ValueError, match="Strategy validation failed"):
            await vectorbt_engine.run_backtest(strategy, invalid_data)

    def test_calculate_position_sizes(self, vectorbt_engine, sample_market_data):
        """Test position size calculation."""
        close_prices = pd.Series([100.0, 101.0, 99.0, 102.0], 
                                index=pd.date_range('2023-01-01', periods=4))
        entries = pd.Series([True, False, True, False],
                           index=close_prices.index)
        
        # Method moved to SignalProcessor
        sizes = vectorbt_engine.signal_processor._calculate_position_sizes(close_prices, entries)
        
        # Should have position sizes only where entries are True
        assert not pd.isna(sizes.iloc[0])  # First entry
        assert pd.isna(sizes.iloc[1])      # No entry
        assert not pd.isna(sizes.iloc[2])  # Third entry
        assert pd.isna(sizes.iloc[3])      # No entry

    def test_calculate_metrics_empty_data(self, vectorbt_engine):
        """Test metrics calculation with empty data."""
        from apex.engine.base import BacktestResult
        
        result = BacktestResult(
            strategy_name="Test",
            symbol="TEST",
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_cash=100000.0,
            final_value=100000.0,
            total_return=0.0,
            portfolio_value=pl.DataFrame({"datetime": [], "portfolio_value": []}),
            total_trades=0
        )
        
        updated_result = vectorbt_engine.calculate_metrics(result)
        assert updated_result.sharpe_ratio is None
        assert updated_result.max_drawdown is None

    def test_calculate_metrics_with_data(self, vectorbt_engine):
        """Test metrics calculation with valid data."""
        from apex.engine.base import BacktestResult
        
        # Create sample portfolio value data
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        values = [100000 + i * 1000 for i in range(10)]  # Growing portfolio
        
        portfolio_df = pl.DataFrame({
            "datetime": dates,
            "portfolio_value": values
        })
        
        # Create sample trades data
        trades_df = pl.DataFrame({
            "entry_time": dates[:5],
            "exit_time": dates[1:6],
            "entry_price": [100.0] * 5,
            "exit_price": [105.0, 95.0, 110.0, 98.0, 107.0],
            "size": [100.0] * 5,
            "pnl": [500.0, -500.0, 1000.0, -200.0, 700.0],
            "return": [0.05, -0.05, 0.10, -0.02, 0.07]
        })
        
        result = BacktestResult(
            strategy_name="Test",
            symbol="TEST", 
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            initial_cash=100000.0,
            final_value=110000.0,
            total_return=0.1,
            portfolio_value=portfolio_df,
            trades=trades_df,
            total_trades=5
        )
        
        updated_result = vectorbt_engine.calculate_metrics(result)
        
        assert updated_result.sharpe_ratio is not None
        assert updated_result.max_drawdown is not None
        assert updated_result.winning_trades == 3
        assert updated_result.losing_trades == 2
        assert updated_result.win_rate == 0.6
        assert updated_result.avg_win > 0
        assert updated_result.avg_loss < 0
        assert updated_result.profit_factor is not None

    def test_convert_to_polars(self, vectorbt_engine):
        """Test pandas to polars conversion."""
        pandas_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        # Method moved to VectorbtDataExtractor
        polars_df = vectorbt_engine.data_extractor.convert_to_polars(pandas_df, ['col1', 'col2'])
        
        assert isinstance(polars_df, pl.DataFrame)
        assert polars_df.columns == ['col1', 'col2']
        assert len(polars_df) == 3

    def test_convert_empty_pandas_to_polars(self, vectorbt_engine):
        """Test conversion of empty pandas DataFrame."""
        empty_df = pd.DataFrame()
        column_names = ['datetime', 'value']
        
        # Method moved to VectorbtDataExtractor
        polars_df = vectorbt_engine.data_extractor.convert_to_polars(empty_df, column_names)
        
        assert isinstance(polars_df, pl.DataFrame)
        assert polars_df.columns == column_names
        assert len(polars_df) == 0

    @pytest.mark.asyncio
    async def test_backtest_end_to_end(self, sample_market_data):
        """Test complete end-to-end backtest workflow."""
        config = EngineConfig(
            initial_cash=50000.0,
            commission=0.002,
            max_position_size=0.2
        )
        
        engine = VectorbtEngine(config)
        strategy = MockStrategy()
        
        result = await engine.run_backtest(strategy, sample_market_data)
        
        # Validate result structure
        assert result.strategy_name == "MockStrategy"
        assert result.symbol == "TEST"
        assert result.initial_cash == 50000.0
        assert isinstance(result.final_value, float)
        assert isinstance(result.total_return, float)
        
        # Should have portfolio value data
        assert result.portfolio_value is not None
        assert len(result.portfolio_value) > 0