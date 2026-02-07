"""
Systematic experiment runner for VectorBT-based parameter optimization.

Pre-fetches data and runs systematic backtests with VectorBT engine.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa

from ..config import load_historical_data_config

logger = logging.getLogger(__name__)


def _normalize_parquet_filter_dt(value: datetime, ts_type: "pa.TimestampType") -> datetime:
    """Normalize datetime for Parquet filter based on schema timestamp type."""
    from datetime import timezone

    if ts_type.tz is None:
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    if ts_type.tz == "UTC":
        return value.astimezone(timezone.utc)

    try:
        from zoneinfo import ZoneInfo

        return value.astimezone(ZoneInfo(ts_type.tz))
    except Exception:
        return value.astimezone(timezone.utc)


def _build_parquet_timestamp_filters(
    schema: "pa.Schema",
    start_dt: datetime,
    end_dt: datetime,
) -> Optional[list[tuple[str, str, Any]]]:
    """Build Parquet predicate pushdown filters for timestamp column."""
    import pyarrow as pa

    if "timestamp" not in schema.names:
        return None

    ts_type = schema.field("timestamp").type
    if not pa.types.is_timestamp(ts_type):
        return None

    start_value = pa.scalar(_normalize_parquet_filter_dt(start_dt, ts_type), type=ts_type)
    end_value = pa.scalar(_normalize_parquet_filter_dt(end_dt, ts_type), type=ts_type)

    return [("timestamp", ">=", start_value), ("timestamp", "<=", end_value)]


def _to_utc_timestamp(value: datetime) -> "pd.Timestamp":
    """Convert datetime to UTC pandas Timestamp."""
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _read_parquet_cached_data(
    parquet_path: Path,
    start_dt: datetime,
    end_dt: datetime,
) -> "pd.DataFrame":
    """Read Parquet file with efficient filtering."""
    import pandas as pd
    import pyarrow.parquet as pq

    filters = None
    try:
        schema = pq.read_schema(parquet_path)
        filters = _build_parquet_timestamp_filters(schema, start_dt, end_dt)
    except Exception:
        filters = None

    table = None
    if filters:
        try:
            table = pq.read_table(parquet_path, filters=filters)
        except Exception:
            logger.debug(
                "Filtered Parquet read failed for %s; retrying without filters",
                parquet_path,
            )
            table = None

    if table is None:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        if "timestamp" in df.columns and not df.empty:
            timestamps = pd.to_datetime(df["timestamp"], utc=True)
            start_ts = _to_utc_timestamp(start_dt)
            end_ts = _to_utc_timestamp(end_dt)
            df = df.loc[(timestamps >= start_ts) & (timestamps <= end_ts)].copy()
            df["timestamp"] = timestamps
        return df

    return table.to_pandas()


async def prefetch_data(
    symbols: List[str],
    start_date: Any,
    end_date: Any,
    max_retries: int = 3,
    timeframe: str = "1d",
    historical_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Pre-fetch data for systematic experiments, using local cache when available.

    First checks the local Parquet store for cached data, then fetches from IB
    only for symbols that are missing or have incomplete data.

    Args:
        symbols: List of symbols to fetch
        start_date: Start date for data
        end_date: End date for data
        max_retries: Number of retry attempts for IB fetching
        timeframe: Bar timeframe (default: 1d)
        historical_dir: Base directory for historical data (default: data/historical)

    Returns:
        Dict[symbol, DataFrame] with OHLCV data indexed by timestamp
    """
    from datetime import timezone

    import pandas as pd

    logger.info(f"Pre-fetching data for {len(symbols)} symbols...")

    # Parse dates - make timezone-aware (UTC) to match Parquet storage format
    if hasattr(start_date, "isoformat"):
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    else:
        start_dt = datetime.fromisoformat(str(start_date)) if start_date else datetime(2020, 1, 1)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

    if hasattr(end_date, "isoformat"):
        end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
    else:
        end_dt = datetime.fromisoformat(str(end_date)) if end_date else datetime.now(timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

    # Resolve historical data directory
    historical_cfg = load_historical_data_config()
    base_dir = Path(historical_dir or historical_cfg.get("base_dir", "data/historical"))

    results: Dict[str, Any] = {}
    symbols_to_fetch: List[str] = []

    # Step 1: Check local Parquet cache for each symbol
    logger.info(f"Checking local cache in {base_dir}...")
    for symbol in symbols:
        parquet_path = base_dir / symbol.upper() / f"{timeframe}.parquet"
        if parquet_path.exists():
            try:
                df = _read_parquet_cached_data(parquet_path, start_dt, end_dt)

                if not df.empty:
                    # Convert timestamp column to index
                    if "timestamp" in df.columns:
                        df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)

                    # Check coverage - we need at least 80% of expected trading days
                    expected_days = (end_dt - start_dt).days * 252 // 365  # Rough estimate
                    actual_days = len(df)
                    coverage = actual_days / expected_days if expected_days > 0 else 0

                    if coverage >= 0.8:
                        results[symbol] = df
                        logger.info(f"  {symbol}: loaded {len(df)} bars from cache")
                        continue
                    else:
                        logger.info(
                            f"  {symbol}: cache has {actual_days} bars (coverage {coverage:.0%}), will refresh"
                        )
            except Exception as e:
                logger.warning(f"  {symbol}: cache read failed ({e}), will fetch")

        symbols_to_fetch.append(symbol)

    # Step 2: Return early if all symbols are cached
    if not symbols_to_fetch:
        logger.info(f"All {len(symbols)} symbols loaded from cache")
        return results

    logger.info(f"Fetching {len(symbols_to_fetch)} symbols from IB: {', '.join(symbols_to_fetch)}")

    # Step 3: Fetch missing symbols from IB
    from ..data.providers import IbBacktestDataProvider

    logger.info("Waiting 5s for IB connection readiness...")
    await asyncio.sleep(5)

    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 15 * attempt
                logger.info(f"Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(wait_time)

            provider = IbBacktestDataProvider(
                host="127.0.0.1",
                port=4001,
                client_id=4,
                rate_limit=True,
            )

            await provider.connect()
            try:
                fetched_data = await provider.fetch_bars(
                    symbols=symbols_to_fetch,
                    start=start_dt,
                    end=end_dt,
                    timeframe=timeframe,
                )
            finally:
                await provider.disconnect()

            # Step 4: Save fetched data to Parquet cache for future use
            for symbol, df in fetched_data.items():
                if not df.empty:
                    results[symbol] = df
                    # Save to cache
                    try:
                        parquet_path = base_dir / symbol.upper() / f"{timeframe}.parquet"
                        parquet_path.parent.mkdir(parents=True, exist_ok=True)

                        # Reset index for storage
                        df_to_save = df.reset_index() if df.index.name else df.copy()
                        if "timestamp" not in df_to_save.columns and df.index.name == "timestamp":
                            df_to_save = df.reset_index()

                        df_to_save.to_parquet(parquet_path, compression="snappy")
                        logger.info(f"  {symbol}: cached {len(df)} bars to {parquet_path}")
                    except Exception as e:
                        logger.warning(f"  {symbol}: failed to cache ({e})")
                else:
                    results[symbol] = pd.DataFrame()

            break  # Success, exit retry loop

        except Exception as e:
            last_error = e
            logger.warning(f"Pre-fetch attempt {attempt + 1} failed: {e}")
    else:
        # All retries failed
        if symbols_to_fetch and not any(s in results for s in symbols_to_fetch):
            raise RuntimeError(
                f"Failed to pre-fetch data after {max_retries} attempts: {last_error}"
            )

    successful = sum(1 for df in results.values() if not df.empty)
    cached_count = len(symbols) - len(symbols_to_fetch)
    fetched_count = successful - cached_count

    logger.info(
        f"Pre-fetch complete: {successful}/{len(symbols)} symbols with data "
        f"({cached_count} cached, {fetched_count} fetched)"
    )
    return results


async def run_systematic_experiment(
    spec_path: str,
    output_dir: str = "results/experiments",
    parallel: int = 0,  # 0 = auto-scale based on workload
    dry_run: bool = False,
    generate_report: bool = True,  # Default ON
) -> Any:
    """Run a systematic backtest experiment."""
    from .. import ExperimentSpec, RunnerConfig, SystematicRunner
    from ..analysis.reporting import generate_experiment_report
    from .engines import create_vectorbt_backtest_fn

    spec = ExperimentSpec.from_yaml(spec_path)
    logger.info(f"Loaded experiment: {spec.name}")
    logger.info(f"  Strategy: {spec.strategy}")
    logger.info(f"  Experiment ID: {spec.experiment_id}")

    param_combinations = spec.expand_parameter_grid()
    symbols = spec.get_symbols()

    logger.info(f"  Parameter combinations: {len(param_combinations)}")
    logger.info(f"  Symbols: {len(symbols)} ({', '.join(symbols)})")
    logger.info(f"  Folds: {spec.temporal.folds}")

    total_runs = len(param_combinations) * len(symbols) * spec.temporal.folds
    logger.info(f"  Total runs: {total_runs}")

    if dry_run:
        logger.info("Dry run - would execute the above")
        logger.info("First 5 parameter combinations:")
        for i, params in enumerate(param_combinations[:5]):
            logger.info(f"  {i+1}: {params}")
        if len(param_combinations) > 5:
            logger.info(f"  ... and {len(param_combinations) - 5} more")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    db_path = output_path / f"{spec.experiment_id}.db"

    logger.info(f"  Output: {db_path}")

    config = RunnerConfig(
        db_path=str(db_path),
        parallel_workers=parallel,
        skip_existing=True,
    )
    runner = SystematicRunner(config=config)

    # Prefetch data for all timeframes (primary + secondary)
    primary_tf = spec.data.primary_timeframe or "1d"
    secondary_tfs = spec.data.secondary_timeframes or []

    logger.info(f"  Primary timeframe: {primary_tf}")
    if secondary_tfs:
        logger.info(f"  Secondary timeframes: {secondary_tfs}")

    # Load primary timeframe data
    cached_data = await prefetch_data(
        symbols=symbols,
        start_date=spec.temporal.start_date,
        end_date=spec.temporal.end_date,
        timeframe=primary_tf,
    )

    # Load secondary timeframe data (MTF support)
    secondary_data: Dict[str, Dict[str, Any]] = {}
    for tf in secondary_tfs:
        tf_data = await prefetch_data(
            symbols=symbols,
            start_date=spec.temporal.start_date,
            end_date=spec.temporal.end_date,
            timeframe=tf,
        )
        for symbol, df in tf_data.items():
            if symbol not in secondary_data:
                secondary_data[symbol] = {}
            secondary_data[symbol][tf] = df

    # Create backtest function with MTF data
    backtest_fn = create_vectorbt_backtest_fn(
        cached_data=cached_data,
        secondary_data=secondary_data if secondary_tfs else None,
    )

    start_time = datetime.now()
    logger.info("Starting experiment execution...")

    try:
        experiment_id = runner.run(
            spec,
            backtest_fn=backtest_fn,
            on_trial_complete=lambda t: logger.debug(
                f"  Trial {t.trial_index}: score={t.trial_score:.3f}"
            ),
        )

        duration = (datetime.now() - start_time).total_seconds()

        result = runner.get_experiment_result(experiment_id)
        result.total_duration_seconds = duration
        result.print_summary()

        top_trials = runner.get_top_trials(experiment_id, limit=5)
        logger.info("Top 5 trials by score:")
        for i, trial in enumerate(top_trials, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in trial["params"].items())
            logger.info(
                f"  {i}. Score={trial['trial_score']:.3f} "
                f"Sharpe={trial['median_sharpe']:.2f} "
                f"MaxDD={trial['median_max_dd']:.1%} "
                f"Params=[{params_str}]"
            )

        logger.info(f"Experiment complete! Results saved to: {db_path}")

        if generate_report:
            try:
                report_path = generate_experiment_report(runner, experiment_id, spec, output_path)
                print(f"\nHTML Report: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to generate HTML report: {e}")

    finally:
        runner.close()
