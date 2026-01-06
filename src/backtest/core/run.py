"""
Run specification - single backtest execution.

A run is the atomic unit of backtesting:
- One symbol
- One time window (train/test period)
- One execution profile
- One set of parameters (from parent trial)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .hashing import generate_run_id


class TimeWindow(BaseModel):
    """
    Time window specification for a single backtest run.

    Includes train and test periods with purge/embargo gaps.

    Train Period: [train_start, train_end]
    Purge Gap:    (train_end, test_start)  <- prevents look-ahead bias
    Test Period:  [test_start, test_end]
    Embargo:      (test_end, next_train)   <- allows model decay
    """

    window_id: str = Field(description="Unique window identifier (e.g., 'fold_1')")
    fold_index: int = Field(description="Fold number (0-indexed)")

    # Training period
    train_start: date = Field(description="Training period start")
    train_end: date = Field(description="Training period end")

    # Test period
    test_start: date = Field(description="Test period start")
    test_end: date = Field(description="Test period end")

    # Gaps
    purge_days: int = Field(default=0, description="Purge gap between train and test")
    embargo_days: int = Field(default=0, description="Embargo gap after test")

    # Flags
    is_train: bool = Field(default=True, description="Whether this is training window")
    is_oos: bool = Field(default=False, description="Whether this is out-of-sample")

    @property
    def train_days(self) -> int:
        """Number of calendar days in training period."""
        return (self.train_end - self.train_start).days + 1

    @property
    def test_days(self) -> int:
        """Number of calendar days in test period."""
        return (self.test_end - self.test_start).days + 1

    def get_train_range(self) -> tuple[date, date]:
        """Get training date range as tuple."""
        return (self.train_start, self.train_end)

    def get_test_range(self) -> tuple[date, date]:
        """Get test date range as tuple."""
        return (self.test_start, self.test_end)


class RunSpec(BaseModel):
    """
    Specification for a single backtest run.

    A run is the atomic execution unit - one symbol, one time window,
    one profile. Multiple runs make up a trial.

    Run ID is deterministically generated from:
    - trial_id (includes experiment_id and parameters)
    - symbol
    - window_id
    - profile_version
    - data_version
    - is_train (disambiguates IS vs OOS runs on same window)
    """

    trial_id: str = Field(description="Parent trial ID")
    symbol: str = Field(description="Symbol to backtest")
    window: TimeWindow = Field(description="Time window specification")
    profile_version: str = Field(description="Execution profile version")
    data_version: str = Field(description="Data version")

    # Strategy parameters (inherited from trial)
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")

    # Execution settings
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    slippage_bps: float = Field(default=5.0, description="Slippage in bps")
    commission_per_share: float = Field(default=0.005, description="Commission per share")

    # Timeframe settings
    bar_size: str = Field(default="1d", description="Primary bar timeframe")
    secondary_timeframes: List[str] = Field(
        default_factory=list,
        description="Secondary timeframes for multi-timeframe strategies",
    )

    @property
    def is_multi_timeframe(self) -> bool:
        """Check if this run uses multiple timeframes."""
        return len(self.secondary_timeframes) > 0

    @property
    def all_timeframes(self) -> List[str]:
        """Get all timeframes (primary + secondary)."""
        return [self.bar_size] + self.secondary_timeframes

    # Metadata
    run_index: Optional[int] = Field(default=None, description="Run index in trial")
    experiment_id: Optional[str] = Field(default=None, description="Experiment ID for reference")

    # Computed
    run_id: Optional[str] = Field(default=None, description="Generated run ID")

    def model_post_init(self, __context) -> None:
        """Generate run ID after initialization."""
        if self.run_id is None:
            self.run_id = generate_run_id(
                trial_id=self.trial_id,
                symbol=self.symbol,
                window_id=self.window.window_id,
                profile_version=self.profile_version,
                data_version=self.data_version,
                is_train=self.window.is_train,  # Disambiguate IS vs OOS
            )

    @property
    def start_date(self) -> date:
        """Get start date for this run (train or test based on window type)."""
        return self.window.train_start if self.window.is_train else self.window.test_start

    @property
    def end_date(self) -> date:
        """Get end date for this run."""
        return self.window.train_end if self.window.is_train else self.window.test_end

    def to_backtest_config(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for BacktestConfig."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "symbols": [self.symbol],
            "initial_capital": self.initial_capital,
            "slippage_bps": self.slippage_bps,
            "commission_per_share": self.commission_per_share,
            "strategy_params": self.params,
            "bar_size": self.bar_size,
            "secondary_timeframes": self.secondary_timeframes,
        }
