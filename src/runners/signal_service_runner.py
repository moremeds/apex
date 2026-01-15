"""
Standalone runner for SignalService with HTML report generation.

Loads historical data from Parquet cache (auto-downloads if missing)
and generates an interactive HTML report with charts and indicators.

Usage:
    # Default: load symbols from universe.yaml, generate HTML report
    python -m src.runners.signal_service_runner --config config/signals/dev.yaml

    # Override symbols
    python -m src.runners.signal_service_runner --symbols AAPL TSLA NVDA

    # Skip HTML report
    python -m src.runners.signal_service_runner --no-html

    # Custom output path
    python -m src.runners.signal_service_runner --output results/my_report.html
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.application.services.signal_service import SignalService, DEFAULT_TIMEFRAMES

logger = logging.getLogger(__name__)


# Timeframe to calendar days multiplier
# For intraday timeframes, we need more calendar days because markets
# only trade ~6.5 hours/day. Multiplier converts bars to calendar days.
TIMEFRAME_CALENDAR_DAYS = {
    "1m": 1 / (60 * 6.5),      # ~390 1m bars per day -> need ~1/390 days per bar
    "5m": 5 / (60 * 6.5),      # ~78 5m bars per day
    "15m": 15 / (60 * 6.5),    # ~26 15m bars per day
    "30m": 30 / (60 * 6.5),    # ~13 30m bars per day
    "1h": 1 / 6.5,             # ~6.5 1h bars per day
    "4h": 4 / 6.5,             # ~1.6 4h bars per day
    "1d": 1.4,                 # 1 daily bar per trading day (~5/7 ratio for weekends)
    "1w": 7,                   # 1 weekly bar per week
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        return yaml.safe_load(f)


def load_symbols_from_universe(universe_path: str = "config/signals/universe.yaml") -> List[str]:
    """
    Load symbols from universe configuration.

    Args:
        universe_path: Path to universe YAML file

    Returns:
        List of symbol strings
    """
    from src.domain.signals.universe.yaml_provider import YamlUniverseProvider

    path = Path(universe_path)
    if not path.exists():
        logger.warning(f"Universe file not found: {path}, using default [AAPL]")
        return ["AAPL"]

    provider = YamlUniverseProvider(str(path))
    symbols = provider.get_symbols()
    logger.info(f"Loaded {len(symbols)} symbols from {path}")
    return symbols


def calculate_start_date(end: datetime, timeframe: str, periods: int) -> datetime:
    """
    Calculate start date for N periods back from end.

    Accounts for trading hours - intraday bars only exist during market hours,
    so we need more calendar days to get N bars.

    Args:
        end: End datetime
        timeframe: Timeframe string (e.g., "1h", "1d")
        periods: Number of periods to go back

    Returns:
        Start datetime
    """
    days_per_bar = TIMEFRAME_CALENDAR_DAYS.get(timeframe, 1.0)
    calendar_days = int(periods * days_per_bar) + 5  # +5 buffer for holidays
    return end - timedelta(days=calendar_days)


async def load_historical_data(
    symbols: List[str],
    timeframes: List[str],
    periods: int = 60,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Load historical bars from parquet, auto-download if missing.

    Args:
        symbols: List of symbols (e.g., ["AAPL", "TSLA"])
        timeframes: List of timeframes (e.g., ["1h", "4h", "1d"])
        periods: Number of bars to load per timeframe (default 60)

    Returns:
        Dict mapping (symbol, timeframe) to DataFrame with OHLCV
    """
    from src.services.historical_data_manager import HistoricalDataManager

    manager = HistoricalDataManager(base_dir=Path("data/historical"))
    result: Dict[Tuple[str, str], pd.DataFrame] = {}

    # Use naive UTC datetime (historical data manager expects naive)
    end = datetime.utcnow()

    for symbol in symbols:
        for tf in timeframes:
            # Request extra periods to ensure we get enough after warmup
            # Many indicators need 35+ bars for warmup
            extra_periods = periods + 50
            start = calculate_start_date(end, tf, extra_periods)

            try:
                # ensure_data: auto-downloads if missing
                bars = await manager.ensure_data(symbol, tf, start, end)

                if bars:
                    # Convert to DataFrame
                    records = [
                        {
                            "timestamp": b.bar_start or b.timestamp,
                            "open": b.open,
                            "high": b.high,
                            "low": b.low,
                            "close": b.close,
                            "volume": b.volume,
                        }
                        for b in bars
                    ]
                    df = pd.DataFrame(records)
                    df.set_index("timestamp", inplace=True)
                    df = df.tail(periods)  # Take last N periods

                    result[(symbol, tf)] = df
                    actual = len(df)
                    status = "OK" if actual >= periods else f"WARN: only {actual}/{periods}"
                    logger.info(f"  {symbol}/{tf}: {actual} bars loaded ({status})")
                else:
                    logger.warning(f"  {symbol}/{tf}: No data returned")

            except Exception as e:
                logger.error(f"  {symbol}/{tf}: Failed to load data - {e}")

    manager.close()
    return result


def compute_indicators_on_df(
    df: pd.DataFrame,
    indicators: List[Any],
) -> pd.DataFrame:
    """
    Compute all indicators on a DataFrame and merge columns.

    Each indicator's columns are prefixed with indicator name to avoid conflicts.
    E.g., MACD produces macd_macd, macd_signal, macd_histogram.
    """
    # Collect all indicator DataFrames first, then concat once (avoids fragmentation)
    indicator_dfs = []

    for indicator in indicators:
        if len(df) < indicator.warmup_periods:
            logger.debug(f"Skipping {indicator.name}: insufficient warmup ({len(df)} < {indicator.warmup_periods})")
            continue

        try:
            ind_df = indicator.calculate(df, indicator.default_params)
            prefixed = ind_df.add_prefix(f"{indicator.name}_")
            indicator_dfs.append(prefixed)
            logger.debug(f"Computed {indicator.name}: {list(ind_df.columns)}")
        except Exception as e:
            logger.warning(f"Failed to compute {indicator.name}: {e}")

    if indicator_dfs:
        return pd.concat([df] + indicator_dfs, axis=1)
    return df.copy()


async def run_with_historical(
    service: SignalService,
    historical_data: Dict[Tuple[str, str], pd.DataFrame],
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Inject historical data into service and compute indicators.

    Returns the data with indicator columns added.
    """
    result: Dict[Tuple[str, str], pd.DataFrame] = {}
    indicator_engine = service._indicator_engine
    indicators = list(indicator_engine._indicators) if indicator_engine else []
    ohlcv_cols = {"open", "high", "low", "close", "volume"}

    for (symbol, tf), df in historical_data.items():
        bars = df.reset_index().to_dict("records")
        injected = await service.inject_bars(symbol, tf, bars)

        df_with_indicators = compute_indicators_on_df(df, indicators)
        result[(symbol, tf)] = df_with_indicators

        indicator_cols = [c for c in df_with_indicators.columns if c not in ohlcv_cols]
        logger.info(f"  {symbol}/{tf}: {injected} bars, {len(indicator_cols)} indicator columns")

    return result


def generate_html_report(
    data: Dict[Tuple[str, str], pd.DataFrame],
    service: SignalService,
    output_path: Path,
) -> Path:
    """
    Generate HTML signal analysis report.

    Args:
        data: Historical data with indicators
        service: SignalService with indicators and rules
        output_path: Where to save HTML

    Returns:
        Path to generated report
    """
    from src.domain.signals.reporting import SignalReportGenerator
    from src.domain.signals.rules import ALL_RULES

    # Get indicators and rules from service
    indicators = []
    rules = []

    if service._indicator_engine:
        indicators = list(service._indicator_engine._indicators)

    if service._rule_engine and service._rule_engine.registry:
        rules = service._rule_engine.registry.get_all_rules()

    # If no rules configured, use all pre-built rules
    if not rules:
        rules = ALL_RULES

    # Generate report
    generator = SignalReportGenerator(theme="dark")
    return generator.generate(
        data=data,
        indicators=indicators,
        rules=rules,
        output_path=output_path,
    )


async def async_main(args: argparse.Namespace) -> int:
    """Async main entry point."""
    # Load config
    config: Dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1

    # Get timeframes from config or defaults
    timeframes = config.get("timeframes", DEFAULT_TIMEFRAMES)

    # Override timeframes if specified on command line
    if args.timeframes:
        timeframes = args.timeframes

    # Load symbols from universe (default) or --symbols override
    if args.symbols:
        symbols = args.symbols
        print(f"Using {len(symbols)} symbols from command line")
    else:
        symbols = load_symbols_from_universe()

    print("=" * 55)
    print("SIGNAL SERVICE RUNNER")
    print("=" * 55)
    print(f"Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Periods: {args.periods}")
    print("-" * 55)

    # Load historical data (auto-download if missing)
    print(f"\nLoading historical data...")
    historical_data = await load_historical_data(symbols, timeframes, args.periods)

    if not historical_data:
        print("Error: No historical data loaded")
        return 1

    print(f"\nTotal: {len(historical_data)} symbol/timeframe combinations loaded")

    # Create and start service
    service = SignalService.from_config(config)
    service.start()

    try:
        # Inject bars and compute indicators
        print("\nProcessing data through signal pipeline...")
        processed_data = await run_with_historical(service, historical_data)

        # Stop service and print summary
        service.stop()
        print("\n")
        service.stats.log_summary()

        # Generate HTML report (by default)
        if not args.no_html:
            output_path = Path(args.output or "results/signals/signal_report.html")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\nGenerating HTML report...")
            report_path = generate_html_report(processed_data, service, output_path)
            print(f"Generated: {report_path}")

    except Exception as e:
        logger.exception("Runner failed")
        service.stop()
        return 1

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Signal service runner with HTML report generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: load from universe.yaml, generate HTML report
    python -m src.runners.signal_service_runner --config config/signals/dev.yaml

    # Override symbols
    python -m src.runners.signal_service_runner --symbols AAPL TSLA NVDA

    # Skip HTML report
    python -m src.runners.signal_service_runner --no-html --symbols AAPL

    # Custom periods and output
    python -m src.runners.signal_service_runner --periods 100 --output my_report.html
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to signal configuration YAML file",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override symbols (default: load from config/signals/universe.yaml)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=60,
        help="Number of bars per timeframe (default: 60)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Override timeframes (default from config or: 30m 1h 4h 1d 1w)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="HTML output path (default: results/signals/signal_report.html)",
    )

    args = parser.parse_args()

    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
