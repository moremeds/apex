"""
ApexEngine worker functions for multiprocessing backtests.

Used for MTF (multi-timeframe) strategies and apex_only strategies
that cannot run in vectorized VectorBT mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import yaml

if TYPE_CHECKING:
    from ...core import RunResult, RunSpec

logger = logging.getLogger(__name__)

# Module-level globals for ApexEngine worker state
_apex_cached_data: Optional[Dict[str, Dict[str, Any]]] = None  # {symbol: {timeframe: DataFrame}}
_apex_engine: Optional[Any] = None


def is_apex_required(strategy_name: str) -> bool:
    """
    Check if a strategy requires ApexEngine (cannot run in VectorBT).

    A strategy requires ApexEngine if:
    - It's marked apex_only: true in manifest.yaml
    - It has no vectorized signal generator (signals: null)

    Note: multi_timeframe strategies CAN run in VectorBT if they have
    a SignalGenerator that accepts secondary_data parameter.

    Args:
        strategy_name: Strategy name to check

    Returns:
        True if strategy requires ApexEngine
    """
    manifest_path = Path(__file__).parent.parent.parent.parent / "domain/strategy/manifest.yaml"
    if not manifest_path.exists():
        return False

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    strategies = manifest.get("strategies", {})
    if strategy_name not in strategies:
        return False

    entry = strategies[strategy_name]
    return entry.get("apex_only", False) or entry.get("signals") is None


def init_apex_worker(
    cached_data: Optional[Dict[str, Dict[str, Any]]],
    config_dict: Dict[str, Any],
) -> None:
    """
    Worker initializer for ApexEngine backtests.

    Called once per worker process to set up the engine and cached data.
    """
    global _apex_cached_data, _apex_engine
    from . import ApexEngine, ApexEngineConfig

    _apex_cached_data = cached_data
    _apex_engine = ApexEngine(ApexEngineConfig(**config_dict))


def run_apex_backtest(spec: "RunSpec") -> "RunResult":
    """
    Top-level backtest function for ApexEngine multiprocessing.

    This runs bar-by-bar event-driven backtests via ApexEngine.
    """
    if _apex_engine is None:
        raise RuntimeError("ApexEngine not initialized. Call init_apex_worker first.")

    # ApexEngine uses HistoricalStoreDataFeed internally, not cached data
    # The cached_data is kept for potential future use (pre-loaded DataFrames)
    result: RunResult = _apex_engine.run(spec, data=None)
    return result


def create_apex_backtest_fn(
    cached_data: Optional[Dict[str, Dict[str, Any]]] = None,
    data_source: str = "historical",
) -> Callable[["RunSpec"], "RunResult"]:
    """
    Create a backtest function using ApexEngine.

    Used for MTF (multi-timeframe) strategies and apex_only strategies
    that cannot run in vectorized VectorBT mode.

    Args:
        cached_data: Optional pre-loaded data {symbol: {timeframe: DataFrame}}
        data_source: Data source type ("historical" for parquet files)

    Returns:
        Backtest function with multiprocessing metadata
    """
    from dataclasses import asdict

    from . import ApexEngineConfig

    config = ApexEngineConfig(
        data_source=data_source,
        bar_size="1d",  # Primary timeframe; secondary set per-spec
    )
    config_dict = asdict(config)

    # Initialize in main process for sequential execution
    init_apex_worker(cached_data, config_dict)

    # Attach multiprocessing metadata
    run_apex_backtest._mp_initializer = init_apex_worker  # type: ignore[attr-defined]
    run_apex_backtest._mp_initargs = (cached_data, config_dict)  # type: ignore[attr-defined]
    run_apex_backtest._mp_context = "spawn"  # type: ignore[attr-defined]

    return run_apex_backtest
