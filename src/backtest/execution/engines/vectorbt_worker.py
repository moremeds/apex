"""
VectorBT worker functions for multiprocessing backtests.

Module-level globals are used to maintain worker state in multiprocessing
environments, where each worker process has its own copy (no shared state).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from ...core import RunResult, RunSpec

# Module-level globals for multiprocessing worker state
# Each worker process has its own copy (no shared state issues)
_vectorbt_cached_data: Optional[Dict[str, Any]] = None
_vectorbt_secondary_data: Optional[Dict[str, Dict[str, Any]]] = None  # {symbol: {timeframe: df}}
_vectorbt_engine: Optional[Any] = None  # VectorBTEngine, typed as Any to avoid circular import


def init_vectorbt_worker(
    cached_data: Optional[Dict[str, Any]],
    config_dict: Dict[str, Any],
    secondary_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Worker initializer for VectorBT backtests.

    Called once per worker process to set up the engine and cached data.
    This avoids pickling issues with closures by using module-level globals.

    Args:
        cached_data: Primary timeframe data {symbol: DataFrame}
        config_dict: VectorBTConfig as dict
        secondary_data: Secondary timeframe data {symbol: {timeframe: DataFrame}}
    """
    global _vectorbt_cached_data, _vectorbt_secondary_data, _vectorbt_engine
    from . import VectorBTConfig, VectorBTEngine

    _vectorbt_cached_data = cached_data
    _vectorbt_secondary_data = secondary_data
    _vectorbt_engine = VectorBTEngine(VectorBTConfig(**config_dict))


def run_vectorbt_backtest(spec: "RunSpec") -> "RunResult":
    """
    Top-level backtest function for multiprocessing.

    This function runs in worker processes after init_vectorbt_worker has
    set up the engine. It's a plain function (not a closure) so it pickles correctly.
    """
    if _vectorbt_engine is None:
        raise RuntimeError("VectorBT engine not initialized. Call init_vectorbt_worker first.")

    symbol_data = _vectorbt_cached_data.get(spec.symbol) if _vectorbt_cached_data else None

    # Get secondary timeframe data for this symbol
    symbol_secondary = None
    if _vectorbt_secondary_data and spec.symbol in _vectorbt_secondary_data:
        symbol_secondary = _vectorbt_secondary_data[spec.symbol]

    result: RunResult = _vectorbt_engine.run(
        spec, data=symbol_data, secondary_data=symbol_secondary
    )
    return result


def create_vectorbt_backtest_fn(
    cached_data: Optional[Dict[str, Any]] = None,
    secondary_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Callable[["RunSpec"], "RunResult"]:
    """
    Create a backtest function using VectorBT engine.

    Returns a top-level function with multiprocessing metadata attached
    so ParallelRunner can properly initialize worker processes.

    Args:
        cached_data: Primary timeframe data {symbol: DataFrame}
        secondary_data: Secondary timeframe data {symbol: {timeframe: DataFrame}}
    """
    from dataclasses import asdict

    from . import VectorBTConfig

    if cached_data:
        config = VectorBTConfig(data_source="local")
    else:
        config = VectorBTConfig(data_source="ib", ib_port=4001)

    config_dict = asdict(config)

    # Initialize in the main process for sequential execution
    init_vectorbt_worker(cached_data, config_dict, secondary_data)

    # Attach multiprocessing metadata for ParallelRunner to use
    # These attributes are dynamically added for the parallel runner
    run_vectorbt_backtest._mp_initializer = init_vectorbt_worker  # type: ignore[attr-defined]
    run_vectorbt_backtest._mp_initargs = (cached_data, config_dict, secondary_data)  # type: ignore[attr-defined]
    run_vectorbt_backtest._mp_context = "spawn"  # type: ignore[attr-defined]

    return run_vectorbt_backtest
