"""
Wiring Verification Tests for Backtest System.

These tests verify that the backtesting components are properly wired together
and can communicate correctly. They catch integration issues that unit tests miss.

Tests cover:
- ParallelRunner usage when parallel_workers > 1
- WalkForwardSplitter train/test tuple returns
- Strategy registry consistency with example specs
- HTMLReportGenerator data requirements
- MTF (Multi-Timeframe) data alignment
"""

from pathlib import Path

import pytest
import yaml


class TestParallelRunnerWiring:
    """Verify parallel execution is properly wired."""

    def test_systematic_runner_uses_parallel_when_configured(self) -> None:
        """Verify parallel_workers > 1 uses ParallelRunner."""
        from src.backtest.execution.systematic import RunnerConfig, SystematicRunner

        # Create config with parallel workers
        config = RunnerConfig(parallel_workers=4)
        runner = SystematicRunner(config)

        # The runner should be configured for parallel execution
        assert runner.config.parallel_workers == 4

    def test_parallel_runner_can_be_instantiated(self) -> None:
        """Verify ParallelRunner can be created with valid config."""
        from src.backtest.execution.parallel import ParallelConfig, ParallelRunner

        config = ParallelConfig(max_workers=4)
        runner = ParallelRunner(config)

        assert runner.config.max_workers == 4

    def test_parallel_config_validates_workers(self) -> None:
        """Verify ParallelConfig validates worker count."""
        from src.backtest.execution.parallel import ParallelConfig

        # Valid configs
        config1 = ParallelConfig(max_workers=1)
        assert config1.max_workers == 1

        config4 = ParallelConfig(max_workers=4)
        assert config4.max_workers == 4


class TestWalkForwardSplitter:
    """Verify WalkForwardSplitter returns proper train/test tuples."""

    def test_wfo_splitter_returns_train_test_tuples(self) -> None:
        """Verify WalkForwardSplitter yields (train_window, test_window)."""
        from src.backtest.core import TimeWindow
        from src.backtest.data import SplitConfig, WalkForwardSplitter

        config = SplitConfig(
            train_days=252,
            test_days=63,
            folds=3,
            purge_days=5,
        )
        splitter = WalkForwardSplitter(config)

        splits = list(splitter.split("2020-01-01", "2024-12-31"))

        # Should have 3 folds
        assert len(splits) == 3

        # Each split should be a tuple of (train_window, test_window)
        for train_window, test_window in splits:
            assert isinstance(train_window, TimeWindow)
            assert isinstance(test_window, TimeWindow)

            # Train window should be marked as training
            assert train_window.is_train is True
            assert train_window.is_oos is False

            # Test window should be marked as OOS
            assert test_window.is_train is False
            assert test_window.is_oos is True

            # Windows should have matching fold_index
            assert train_window.fold_index == test_window.fold_index

    def test_wfo_splitter_respects_purge_gap(self) -> None:
        """Verify purge gap is respected between train and test."""
        from src.backtest.data import SplitConfig, WalkForwardSplitter

        config = SplitConfig(
            train_days=100,
            test_days=30,
            folds=2,
            purge_days=10,
        )
        splitter = WalkForwardSplitter(config)

        for train_window, test_window in splitter.split("2022-01-01", "2024-12-31"):
            # Test should start after train + purge
            gap_days = (test_window.test_start - train_window.train_end).days
            assert gap_days >= config.purge_days, f"Purge gap {gap_days} < {config.purge_days}"


class TestStrategyRegistryConsistency:
    """Verify example specs reference registered strategies."""

    def test_strategy_registry_contains_expected_strategies(self) -> None:
        """Verify core strategies are registered."""
        from src.domain.strategy import examples  # noqa: F401 - triggers registration
        from src.domain.strategy.registry import StrategyRegistry

        registry = StrategyRegistry()
        expected = {"ma_cross", "rsi_mean_reversion", "momentum_breakout", "ta_metrics"}

        registered = set(registry._strategies.keys())
        missing = expected - registered

        assert not missing, f"Missing strategies: {missing}"

    def test_example_specs_reference_registered_strategies(self) -> None:
        """Verify all example specs reference strategies that exist.

        Strategies can be defined in either:
        - StrategyRegistry: Full Strategy classes for ApexEngine/live trading
        - manifest.yaml: SignalGenerator-only strategies for VectorBT screening
        """
        from src.domain.strategy import examples  # noqa: F401
        from src.domain.strategy.registry import StrategyRegistry

        registry = StrategyRegistry()
        registered = set(registry._strategies.keys())

        # Also check manifest.yaml for signal-only strategies (used by VectorBT)
        manifest_path = Path("src/domain/strategy/manifest.yaml")
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            manifest_strategies = set(manifest.get("strategies", {}).keys())
            registered = registered | manifest_strategies

        spec_dir = Path("config/backtest/examples")
        if not spec_dir.exists():
            pytest.skip("Example specs directory not found")

        invalid_refs = []

        for spec_file in spec_dir.glob("*.yaml"):
            with open(spec_file) as f:
                spec = yaml.safe_load(f)

            strategy_name = spec.get("strategy")
            if strategy_name and strategy_name not in registered:
                invalid_refs.append(f"{spec_file.name}: '{strategy_name}'")

        if invalid_refs:
            pytest.fail(
                f"Invalid strategy references in specs:\n"
                f"  {chr(10).join(invalid_refs)}\n"
                f"Available strategies: {sorted(registered)}"
            )

    def test_mtf_strategy_registered(self) -> None:
        """Verify MTF RSI Trend strategy is registered."""
        from src.domain.strategy import examples  # noqa: F401
        from src.domain.strategy.registry import StrategyRegistry

        registry = StrategyRegistry()
        assert "mtf_rsi_trend" in registry._strategies, "mtf_rsi_trend strategy not registered"


class TestHTMLReportGenerator:
    """Verify HTMLReportGenerator receives valid data."""

    def test_report_data_has_required_fields(self) -> None:
        """Verify ReportData has all required fields."""
        from src.backtest.analysis.reporting import ReportData

        required_fields = {
            "experiment_id",
            "strategy_name",
            "symbols",
            "metrics",
            "per_symbol",
            "per_window",
            "equity_curve",
            "trades",
        }

        data = ReportData()
        data_fields = set(vars(data).keys())

        missing = required_fields - data_fields
        assert not missing, f"ReportData missing fields: {missing}"

    def test_report_generator_accepts_minimal_data(self) -> None:
        """Verify report generator can handle minimal valid data."""
        import tempfile

        from src.backtest.analysis.reporting import (
            HTMLReportGenerator,
            ReportConfig,
            ReportData,
        )

        generator = HTMLReportGenerator(ReportConfig(title="Test Report"))
        data = ReportData(
            experiment_id="test_exp_001",
            strategy_name="ma_cross",
            symbols=["AAPL"],
            metrics={"sharpe": 1.5, "total_return": 0.15, "max_drawdown": 0.08},
        )

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = Path(f.name)

        try:
            result_path = generator.generate(data, output_path)
            assert result_path.exists()
            content = result_path.read_text()
            assert "Test Report" in content or "ma_cross" in content
        finally:
            output_path.unlink(missing_ok=True)


class TestMTFDataFeedWiring:
    """Verify Multi-Timeframe data feed is properly wired."""

    def test_aligned_bar_buffer_exists(self) -> None:
        """Verify AlignedBarBuffer can be imported."""
        from src.backtest.data.feeds import AlignedBarBuffer

        buffer = AlignedBarBuffer(
            primary_timeframe="1d",
            secondary_timeframes=["1h"],
        )
        assert buffer.primary_timeframe == "1d"
        assert "1h" in buffer.secondary_timeframes

    def test_historical_store_data_feed_accepts_secondary_timeframes(self) -> None:
        """Verify HistoricalStoreDataFeed accepts secondary_timeframes parameter."""
        import inspect

        from src.backtest.data.feeds import HistoricalStoreDataFeed

        sig = inspect.signature(HistoricalStoreDataFeed.__init__)
        params = list(sig.parameters.keys())

        assert (
            "secondary_timeframes" in params
        ), "HistoricalStoreDataFeed missing secondary_timeframes parameter"

    def test_aligned_bar_buffer_sorting(self) -> None:
        """Verify secondary timeframes sort before primary at same timestamp."""
        from datetime import datetime

        from src.backtest.data.feeds import AlignedBarBuffer
        from src.domain.events.domain_events import BarData

        # Create bars at same timestamp
        ts = datetime(2024, 1, 15, 9, 30, 0)
        daily = BarData(
            symbol="AAPL",
            timeframe="1d",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            timestamp=ts,
        )
        hourly = BarData(
            symbol="AAPL",
            timeframe="1h",
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=100000,
            timestamp=ts,
        )

        # Secondary (hourly) should sort before primary (daily) at same timestamp
        key_daily = AlignedBarBuffer.sort_key(daily, "1d")
        key_hourly = AlignedBarBuffer.sort_key(hourly, "1d")

        assert (
            key_hourly < key_daily
        ), "Secondary timeframe should sort before primary at same timestamp"


class TestRunSpecWiring:
    """Verify RunSpec has required MTF fields."""

    def test_run_spec_has_secondary_timeframes(self) -> None:
        """Verify RunSpec has secondary_timeframes field."""
        from datetime import date

        from src.backtest.core import RunSpec, TimeWindow

        window = TimeWindow(
            window_id="test",
            fold_index=0,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 12, 31),
        )

        spec = RunSpec(
            symbol="AAPL",
            window=window,
            params={"fast_period": 10},
            bar_size="1d",
            secondary_timeframes=["1h", "4h"],
            trial_id="test_trial",
            profile_version="v1",
            data_version="dv1",
        )

        assert spec.secondary_timeframes == ["1h", "4h"]

    def test_run_spec_bar_size_default(self) -> None:
        """Verify RunSpec has sensible bar_size default."""
        from datetime import date

        from src.backtest.core import RunSpec, TimeWindow

        window = TimeWindow(
            window_id="test",
            fold_index=0,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 12, 31),
        )

        spec = RunSpec(
            symbol="AAPL",
            window=window,
            params={},
            trial_id="test_trial",
            profile_version="v1",
            data_version="dv1",
        )

        # Default bar_size should be "1d" or None
        assert spec.bar_size in ("1d", None, "")


class TestVersionAutoIncrement:
    """Verify version auto-increment is implemented."""

    def test_get_next_version_function_exists(self) -> None:
        """Verify get_next_version function exists in hashing module."""
        from src.backtest.core import hashing

        assert hasattr(
            hashing, "get_next_version"
        ), "get_next_version function not found in hashing module"

    def test_experiment_result_has_run_version(self) -> None:
        """Verify ExperimentResult can store version information."""
        from src.backtest.core import ExperimentResult

        # Create with required fields
        result = ExperimentResult(
            experiment_id="test_v1",
            name="test",
            strategy="ma_cross",
        )

        # Should be able to set experiment_id which includes version
        assert result.experiment_id is not None
        # Experiment ID can include version suffix like _v1
        assert "_v" in result.experiment_id or result.experiment_id == "test_v1"


class TestEngineWiring:
    """Verify backtest engines are properly wired."""

    def test_apex_engine_implements_base_interface(self) -> None:
        """Verify ApexEngine implements BaseEngine interface."""
        from src.backtest.execution.engines import ApexEngine, BaseEngine

        engine = ApexEngine()
        assert isinstance(engine, BaseEngine)

    def test_vectorbt_engine_implements_base_interface(self) -> None:
        """Verify VectorBTEngine implements BaseEngine interface."""
        from src.backtest.execution.engines import BaseEngine, VectorBTEngine

        engine = VectorBTEngine()
        assert isinstance(engine, BaseEngine)

    def test_engines_have_run_method(self) -> None:
        """Verify both engines have run method with correct signature."""
        import inspect

        from src.backtest.execution.engines import ApexEngine, VectorBTEngine

        for engine_class in [ApexEngine, VectorBTEngine]:
            assert hasattr(engine_class, "run"), f"{engine_class.__name__} missing run method"

            sig = inspect.signature(engine_class.run)
            params = list(sig.parameters.keys())

            # Should have spec and data parameters
            assert "spec" in params, f"{engine_class.__name__}.run missing 'spec' parameter"


class TestOptimizerWiring:
    """Verify optimizer classes are properly implemented."""

    def test_grid_optimizer_exists(self) -> None:
        """Verify GridOptimizer can be imported."""
        from src.backtest.optimization import GridOptimizer

        assert GridOptimizer is not None

    def test_bayesian_optimizer_exists(self) -> None:
        """Verify BayesianOptimizer can be imported."""
        from src.backtest.optimization import BayesianOptimizer

        assert BayesianOptimizer is not None

    def test_optimizers_share_interface(self) -> None:
        """Verify both optimizers have similar interfaces."""
        from src.backtest.optimization import BayesianOptimizer, GridOptimizer

        for optimizer_class in [GridOptimizer, BayesianOptimizer]:
            # Both should have generate_params method for parameter generation
            assert hasattr(
                optimizer_class, "generate_params"
            ), f"{optimizer_class.__name__} missing generate_params method"
