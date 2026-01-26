"""Tests for VectorBT backtest engine."""

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.core import RunResult, RunSpec, RunStatus, TimeWindow
from src.backtest.execution.engines import (
    BacktestEngine,
    EngineType,
    VectorBTConfig,
    VectorBTEngine,
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )


@pytest.fixture
def run_spec():
    """Create sample run specification."""
    window = TimeWindow(
        window_id="fold_1",
        fold_index=0,
        train_start=date(2023, 1, 1),
        train_end=date(2023, 2, 28),
        test_start=date(2023, 3, 1),
        test_end=date(2023, 4, 10),
        is_train=False,
        is_oos=True,
    )
    return RunSpec(
        trial_id="trial_test123",
        symbol="AAPL",
        window=window,
        profile_version="v1",
        data_version="test",
        params={"fast_period": 10, "slow_period": 30, "strategy_type": "ma_cross"},
        initial_capital=100000.0,
        experiment_id="exp_test",
    )


class TestVectorBTEngineInterface:
    """Test VectorBT engine interface compliance."""

    def test_implements_protocol(self) -> None:
        """VectorBTEngine should implement BacktestEngine protocol."""
        engine = VectorBTEngine()
        assert isinstance(engine, BacktestEngine)

    def test_engine_type(self) -> None:
        """Engine type should be VECTORBT."""
        engine = VectorBTEngine()
        assert engine.engine_type == EngineType.VECTORBT

    def test_supports_vectorization(self) -> None:
        """VectorBT should support vectorization."""
        engine = VectorBTEngine()
        assert engine.supports_vectorization is True

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        engine = VectorBTEngine()
        assert engine.config.engine_type == EngineType.VECTORBT
        assert engine.config.enable_caching is True


class TestVectorBTEngineRun:
    """Test single run execution."""

    def test_run_with_data(self, sample_data: Any, run_spec: Any) -> None:
        """Run should succeed with provided data."""
        engine = VectorBTEngine()
        result = engine.run(run_spec, sample_data)

        assert isinstance(result, RunResult)
        assert result.run_id == run_spec.run_id
        assert result.trial_id == run_spec.trial_id
        assert result.symbol == "AAPL"

    def test_run_returns_metrics(self, sample_data: Any, run_spec: Any) -> None:
        """Run should calculate metrics."""
        engine = VectorBTEngine()
        result = engine.run(run_spec, sample_data)

        # Metrics should be populated
        assert result.metrics is not None
        # Values should be reasonable (not NaN)
        assert not np.isnan(result.metrics.sharpe) or result.metrics.total_trades == 0

    def test_run_with_empty_data(self, run_spec: Any) -> None:
        """Run should fail gracefully with empty data."""
        engine = VectorBTEngine()
        empty_data = pd.DataFrame()
        result = engine.run(run_spec, empty_data)

        assert result.status == RunStatus.FAIL_DATA
        assert result.error is not None

    def test_run_with_none_data_no_fetch(self, run_spec: Any) -> None:
        """Run without data should attempt to load."""
        engine = VectorBTEngine()
        # Mock fetch to return None
        engine._fetch_data = MagicMock(return_value=None)

        result = engine.run(run_spec, data=None)

        assert result.status == RunStatus.FAIL_DATA

    def test_run_tracks_timing(self, sample_data: Any, run_spec: Any) -> None:
        """Run should track execution timing."""
        engine = VectorBTEngine()
        result = engine.run(run_spec, sample_data)

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_seconds >= 0

    def test_run_preserves_spec_metadata(self, sample_data: Any, run_spec: Any) -> None:
        """Run should preserve spec metadata in result."""
        engine = VectorBTEngine()
        result = engine.run(run_spec, sample_data)

        assert result.experiment_id == run_spec.experiment_id
        assert result.profile_version == run_spec.profile_version
        assert result.data_version == run_spec.data_version
        assert result.is_oos == run_spec.window.is_oos


class TestSignalGenerators:
    """Test SignalGenerator implementations used by VectorBT."""

    def test_ma_cross_generates_signals(self, sample_data: Any) -> None:
        """MA crossover SignalGenerator should generate valid signals."""
        from src.domain.strategy.signals import MACrossSignalGenerator

        generator = MACrossSignalGenerator()
        params = {"short_window": 5, "long_window": 20}

        entries, exits = generator.generate(sample_data, params)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_data)
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_rsi_generates_signals(self, sample_data: Any) -> None:
        """RSI SignalGenerator should generate signals."""
        from src.domain.strategy.signals import RSIMeanReversionSignalGenerator

        generator = RSIMeanReversionSignalGenerator()
        params = {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70}

        entries, exits = generator.generate(sample_data, params)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)

    def test_momentum_generates_signals(self, sample_data: Any) -> None:
        """Momentum SignalGenerator should generate signals."""
        from src.domain.strategy.signals import MomentumBreakoutSignalGenerator

        generator = MomentumBreakoutSignalGenerator()
        params = {"lookback_days": 20, "momentum_threshold": 0.0}

        entries, exits = generator.generate(sample_data, params)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)

    def test_ta_metrics_generates_signals(self, sample_data: Any) -> None:
        """TA Metrics SignalGenerator should generate signals."""
        from src.domain.strategy.signals import TAMetricsSignalGenerator

        generator = TAMetricsSignalGenerator()
        params = {"min_score": 3}

        entries, exits = generator.generate(sample_data, params)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)

    def test_buy_and_hold_generates_signals(self, sample_data: Any) -> None:
        """Buy and hold SignalGenerator should generate entry on first bar."""
        from src.domain.strategy.signals import BuyAndHoldSignalGenerator

        generator = BuyAndHoldSignalGenerator()
        params = {}

        entries, exits = generator.generate(sample_data, params)

        assert entries.iloc[0] is True or entries.iloc[0] == True
        assert exits.sum() == 0  # No exits

    def test_unknown_strategy_fails(self, sample_data: Any, run_spec: Any) -> None:
        """Unknown strategy should return error from manifest lookup."""
        engine = VectorBTEngine()
        run_spec.params["strategy_type"] = "unknown_strategy"

        result = engine.run(run_spec, sample_data)

        assert result.status == RunStatus.FAIL_STRATEGY
        assert "unknown_strategy" in result.error

    def test_apex_only_strategy_fails(self, sample_data: Any, run_spec: Any) -> None:
        """Apex-only strategies should fail in VectorBT."""
        engine = VectorBTEngine()

        # Test scheduled_rebalance (portfolio strategy)
        run_spec.params["strategy_type"] = "scheduled_rebalance"
        result = engine.run(run_spec, sample_data)
        assert result.status == RunStatus.FAIL_STRATEGY
        assert "apex_only" in result.error.lower()

        # Test pairs_trading (multi-symbol strategy)
        run_spec.params["strategy_type"] = "pairs_trading"
        result = engine.run(run_spec, sample_data)
        assert result.status == RunStatus.FAIL_STRATEGY
        assert "apex_only" in result.error.lower()


class TestVectorBTBatch:
    """Test batch execution with vectorization."""

    def test_batch_single_symbol(self, sample_data: Any, run_spec: Any) -> None:
        """Batch should handle multiple specs for same symbol."""
        engine = VectorBTEngine()

        specs = []
        for i, fast in enumerate([5, 10, 15]):
            spec = RunSpec(
                trial_id=f"trial_{i}",
                symbol="AAPL",
                window=run_spec.window,
                profile_version="v1",
                data_version="test",
                params={"fast_period": fast, "slow_period": 30, "strategy_type": "ma_cross"},
                experiment_id="exp_test",
            )
            specs.append(spec)

        results = engine.run_batch(specs, {"AAPL": sample_data})

        assert len(results) == 3
        assert all(isinstance(r, RunResult) for r in results)

    def test_batch_preserves_order(self, sample_data: Any, run_spec: Any) -> None:
        """Batch results should match input order."""
        engine = VectorBTEngine()

        specs = []
        for i in range(5):
            spec = RunSpec(
                trial_id=f"trial_{i}",
                symbol="AAPL",
                window=run_spec.window,
                profile_version="v1",
                data_version="test",
                params={"fast_period": 5 + i * 5, "slow_period": 30, "strategy_type": "ma_cross"},
                experiment_id="exp_test",
            )
            specs.append(spec)

        results = engine.run_batch(specs, {"AAPL": sample_data})

        for i, result in enumerate(results):
            assert result.trial_id == f"trial_{i}"

    def test_batch_empty_specs(self) -> None:
        """Batch with empty specs should return empty list."""
        engine = VectorBTEngine()
        results = engine.run_batch([])
        assert results == []


class TestVectorBTDataLoading:
    """Test data loading and caching."""

    def test_cache_enabled_by_default(self) -> None:
        """Data caching should be enabled by default."""
        config = VectorBTConfig()
        assert config.enable_caching is True

    def test_clear_cache(self, sample_data: Any) -> None:
        """Cache should be clearable."""
        engine = VectorBTEngine()
        engine._data_cache["test_key"] = sample_data

        engine.clear_cache()

        assert len(engine._data_cache) == 0

    def test_date_range_filtering(self, sample_data: Any) -> None:
        """Date range filtering should work correctly."""
        engine = VectorBTEngine()

        filtered = engine._filter_date_range(
            sample_data,
            date(2023, 1, 15),
            date(2023, 2, 15),
        )

        assert len(filtered) < len(sample_data)
        assert filtered.index.min().date() >= date(2023, 1, 15)
        assert filtered.index.max().date() <= date(2023, 2, 15)
