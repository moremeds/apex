"""Shared fixtures for backtest tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from src.backtest import DatabaseManager, ExperimentSpec, RunMetrics, RunResult, RunStatus


@pytest.fixture
def temp_db() -> Generator[DatabaseManager, None, None]:
    """Temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()
        yield db
        db.close()


@pytest.fixture
def sample_experiment_spec() -> ExperimentSpec:
    """Sample experiment specification."""
    return ExperimentSpec(
        name="Test_Experiment",
        strategy="ma_cross",
        parameters={
            "fast_period": {"type": "range", "min": 10, "max": 30, "step": 10},
            "slow_period": {"type": "range", "min": 50, "max": 70, "step": 10},
        },
        universe={"type": "static", "symbols": ["AAPL", "MSFT"]},
        temporal={
            "primary_method": "walk_forward",
            "train_days": 252,
            "test_days": 63,
            "folds": 3,
            "purge_days": 5,
            "embargo_days": 2,
        },
        optimization={
            "method": "grid",
            "metric": "sharpe",
            "direction": "maximize",
        },
        reproducibility={"random_seed": 42, "data_version": "test_v1"},
    )


@pytest.fixture
def sample_run_metrics() -> RunMetrics:
    """Sample run metrics for testing."""
    return RunMetrics(
        total_return=0.15,
        cagr=0.12,
        annualized_return=0.12,
        sharpe=1.5,
        sortino=2.0,
        calmar=1.2,
        max_drawdown=-0.10,
        avg_drawdown=-0.05,
        max_dd_duration_days=30,
        total_trades=50,
        win_rate=0.55,
        profit_factor=1.8,
        expectancy=0.02,
        sqn=2.5,
        best_trade_pct=0.08,
        worst_trade_pct=-0.05,
        avg_win_pct=0.03,
        avg_loss_pct=-0.02,
        exposure_pct=0.7,
        avg_trade_duration_days=5.0,
    )


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Sample OHLCV data for backtesting."""
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    np.random.seed(42)

    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.cumprod(1 + returns)

    return pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            "low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, len(dates)),
        }
    )


@pytest.fixture
def sample_run_result(sample_run_metrics: RunMetrics) -> RunResult:
    """Sample run result for testing."""
    return RunResult(
        run_id="run_test123",
        trial_id="trial_test456",
        experiment_id="exp_test789",
        symbol="AAPL",
        window_id="window_1",
        profile_version="v1",
        data_version="test_v1",
        status=RunStatus.SUCCESS,
        is_train=False,
        is_oos=True,
        metrics=sample_run_metrics,
    )
