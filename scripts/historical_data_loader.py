#!/usr/bin/env python3
"""
Historical Data Loader CLI.

Download and manage historical market data from Yahoo Finance and IB.

Usage:
    # Download data for symbols
    python scripts/historical_data_loader.py download --symbols AAPL,MSFT --timeframe 1d \
        --start 2024-01-01 --end 2024-12-31 --source yahoo

    # Backfill max history for all configured symbols
    python scripts/historical_data_loader.py backfill
    python scripts/historical_data_loader.py backfill --symbols AAPL,SPY --timeframes 1d,1h
    python scripts/historical_data_loader.py backfill --dry-run

    # Generate HTML coverage report
    python scripts/historical_data_loader.py report
    python scripts/historical_data_loader.py report --output reports/coverage.html

    # Show coverage for a symbol
    python scripts/historical_data_loader.py coverage AAPL

    # List all symbols with data
    python scripts/historical_data_loader.py list

    # Fill gaps in existing data
    python scripts/historical_data_loader.py fill-gaps --timeframe 1d

    # Delete data for a symbol
    python scripts/historical_data_loader.py delete AAPL --timeframe 1d
"""

import asyncio
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set
import sys
import time
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.historical_data_manager import HistoricalDataManager, DownloadResult
from src.services.coverage_visualizer import CoverageVisualizer
from src.infrastructure.adapters.ib.historical_adapter import (
    IbHistoricalAdapter,
    MAX_HISTORY_DAYS as IB_MAX_HISTORY_DAYS,
)
from src.domain.services.bar_count_calculator import BarCountCalculator

# Yahoo max history days - request all available data
# Yahoo will return from IPO (stocks) or inception (ETFs) automatically
# Intraday limits are hard Yahoo API limits
YAHOO_MAX_HISTORY_DAYS = {
    "1m": 7,       # Yahoo hard limit
    "5m": 60,      # Yahoo hard limit
    "15m": 60,     # Yahoo hard limit
    "30m": 60,     # Yahoo hard limit
    "1h": 730,     # ~2 years (Yahoo limit)
    "4h": 730,     # ~2 years
    "1d": 18250,   # ~50 years (covers most stocks since 1976)
    "1w": 18250,   # ~50 years
    "1M": 18250,   # ~50 years
}

# Use standard logging for CLI tool
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def connect_ib_source(host: str, port: int, client_id: int) -> IbHistoricalAdapter:
    """Connect to IB for historical data."""
    adapter = IbHistoricalAdapter(host=host, port=port, client_id=client_id)
    await adapter.connect()
    return adapter


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


async def cmd_download(args: argparse.Namespace) -> int:
    """Download historical data."""
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start = parse_date(args.start)
    end = parse_date(args.end)

    print(f"Downloading {args.timeframe} data for {len(symbols)} symbols")
    print(f"Date range: {start.date()} to {end.date()}")
    print(f"Source: {args.source or 'auto (ib > yahoo)'}")
    print()

    manager = HistoricalDataManager(base_dir=Path(args.data_dir))

    # Connect to IB if needed
    ib_adapter = None
    if args.source in (None, "ib"):
        try:
            print(f"Connecting to IB Gateway at {args.ib_host}:{args.ib_port}...")
            ib_adapter = await connect_ib_source(
                host=args.ib_host,
                port=args.ib_port,
                client_id=args.ib_client_id,
            )
            manager.set_ib_source(ib_adapter)
            print("IB connected successfully")
        except Exception as e:
            print(f"IB connection failed: {e}")
            if args.source == "ib":
                print("ERROR: IB source requested but connection failed")
                return 1
            print("Falling back to Yahoo only")

    try:
        results = await manager.download_symbols(
            symbols=symbols,
            timeframe=args.timeframe,
            start=start,
            end=end,
            source=args.source,
        )

        # Print results
        success_count = 0
        total_bars = 0

        for result in results:
            status = "OK" if result.success else "FAILED"
            print(f"  {result.symbol}: {status} - {result.bars_downloaded} bars from {result.source}")
            if result.success:
                success_count += 1
                total_bars += result.bars_downloaded

        print()
        print(f"Downloaded {total_bars} bars for {success_count}/{len(symbols)} symbols")

        return 0 if success_count == len(symbols) else 1

    finally:
        # Cleanup
        if ib_adapter:
            await ib_adapter.disconnect()
        manager.close()


async def cmd_coverage(args: argparse.Namespace) -> int:
    """Show coverage for a symbol."""
    manager = HistoricalDataManager(base_dir=Path(args.data_dir))

    symbol = args.symbol.upper()
    summary = manager.get_coverage_summary(symbol)

    if not summary:
        print(f"No data found for {symbol}")
        manager.close()
        return 1

    print(f"Coverage for {symbol}:")
    print("-" * 60)

    for entry in summary:
        tf = entry["timeframe"]
        src = entry["source"]
        earliest = entry["earliest"]
        latest = entry["latest"]
        bars = entry["total_bars"]

        earliest_str = earliest.strftime("%Y-%m-%d") if earliest else "N/A"
        latest_str = latest.strftime("%Y-%m-%d") if latest else "N/A"

        print(f"  {tf:8} | {src:8} | {earliest_str} to {latest_str} | {bars:,} bars")

    # Also show file sizes
    print()
    print("Storage:")
    timeframes = manager.list_timeframes(symbol)
    for tf in timeframes:
        bar_count = manager.get_bar_count(symbol, tf)
        file_path = manager._bar_store.get_file_path(symbol, tf)
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  {tf:8} | {bar_count:,} bars | {size_mb:.2f} MB")

    manager.close()
    return 0


async def cmd_list(args: argparse.Namespace) -> int:
    """List all symbols with data."""
    manager = HistoricalDataManager(base_dir=Path(args.data_dir))

    symbols = manager.list_symbols()

    if not symbols:
        print("No data found")
        manager.close()
        return 0

    print(f"Found {len(symbols)} symbols:")
    print("-" * 40)

    for symbol in symbols:
        timeframes = manager.list_timeframes(symbol)
        tf_str = ", ".join(timeframes)
        print(f"  {symbol:8} | {tf_str}")

    manager.close()
    return 0


async def cmd_fill_gaps(args: argparse.Namespace) -> int:
    """Fill gaps in data coverage."""
    manager = HistoricalDataManager(base_dir=Path(args.data_dir))

    print(f"Filling gaps (limit: {args.limit})...")

    results = await manager.fill_gaps(
        symbol=args.symbol.upper() if args.symbol else None,
        timeframe=args.timeframe,
        limit=args.limit,
    )

    if not results:
        print("No gaps found to fill")
        manager.close()
        return 0

    success_count = sum(1 for r in results if r.success)
    total_bars = sum(r.bars_downloaded for r in results)

    for result in results:
        status = "OK" if result.success else "FAILED"
        print(f"  {result.symbol}/{result.timeframe}: {status} - {result.bars_downloaded} bars")

    print()
    print(f"Filled {success_count}/{len(results)} gaps, {total_bars} bars total")

    manager.close()
    return 0


async def cmd_delete(args: argparse.Namespace) -> int:
    """Delete data for a symbol."""
    manager = HistoricalDataManager(base_dir=Path(args.data_dir))

    symbol = args.symbol.upper()

    if not args.force:
        summary = manager.get_coverage_summary(symbol)
        if summary:
            print(f"About to delete data for {symbol}:")
            for entry in summary:
                print(f"  {entry['timeframe']}: {entry['total_bars']} bars")
            print()
            confirm = input("Are you sure? (y/N): ")
            if confirm.lower() != "y":
                print("Aborted")
                manager.close()
                return 1

    deleted = manager.delete_data(symbol, args.timeframe)

    if deleted:
        print(f"Deleted data for {symbol}")
    else:
        print(f"No data found for {symbol}")

    manager.close()
    return 0


async def cmd_rebuild_coverage(args: argparse.Namespace) -> int:
    """Rebuild coverage metadata from stored Parquet data."""
    import pyarrow.parquet as pq

    base_dir = Path(args.data_dir)
    manager = HistoricalDataManager(base_dir=base_dir)

    if not base_dir.exists():
        print(f"Data directory not found: {base_dir}")
        manager.close()
        return 1

    print(f"Rebuilding coverage from Parquet files in {base_dir}...")

    # Clear existing coverage to avoid stale entries
    for symbol in manager._coverage_store.get_all_symbols():
        manager._coverage_store.delete_coverage(symbol)

    parquet_files = sorted(base_dir.glob("*/*.parquet"))
    if not parquet_files:
        print("No Parquet files found")
        manager.close()
        return 0

    total_files = 0
    total_ranges = 0
    total_bars = 0

    for file_path in parquet_files:
        symbol = file_path.parent.name.upper()
        timeframe = file_path.stem
        total_files += 1

        try:
            table = pq.read_table(file_path, columns=["timestamp", "source"])
            df = table.to_pandas()
        except Exception as e:
            print(f"  Failed to read {file_path}: {e}")
            continue

        if df.empty or "timestamp" not in df.columns:
            continue

        # Normalize source names to lowercase
        if "source" not in df.columns:
            df["source"] = "unknown"
        df["source"] = df["source"].fillna("unknown").astype(str).str.strip().str.lower()

        # Group by source and update coverage
        for source, group in df.groupby("source"):
            if group.empty:
                continue

            start_ts = group["timestamp"].min()
            end_ts = group["timestamp"].max()
            bar_count = len(group)

            # Convert to datetime if needed
            start_dt = start_ts.to_pydatetime() if hasattr(start_ts, "to_pydatetime") else start_ts
            end_dt = end_ts.to_pydatetime() if hasattr(end_ts, "to_pydatetime") else end_ts

            manager._coverage_store.update_coverage(
                symbol=symbol,
                timeframe=timeframe,
                source=source,
                start=start_dt,
                end=end_dt,
                bar_count=bar_count,
                quality="complete",
            )
            total_ranges += 1
            total_bars += bar_count
            print(f"  {symbol}/{timeframe}/{source}: {bar_count} bars")

    print()
    print(f"Rebuilt coverage for {total_files} files, {total_ranges} ranges, {total_bars:,} bars")

    manager.close()
    return 0


# Supported timeframes for backfill (largest to smallest for efficiency)
BACKFILL_TIMEFRAMES = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]

# Rate limiting delays (seconds)
IB_DELAY_SECONDS = 11.0   # ~6 req/min to stay under 50/10min limit
YAHOO_DELAY_SECONDS = 1.0  # 1 req/sec


@dataclass
class BackfillWorkItem:
    """A single backfill work item."""
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    max_days: int


def load_universe_symbols(config_path: Path) -> Set[str]:
    """Load symbols from universe.yaml config."""
    symbols: Set[str] = set()

    if not config_path.exists():
        logger.warning(f"Universe config not found: {config_path}")
        return symbols

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract from groups
    groups = config.get("groups", {})
    for group_name, group_config in groups.items():
        if group_config.get("enabled", True):
            group_symbols = group_config.get("symbols", [])
            symbols.update(group_symbols)

    # Extract from overrides
    overrides = config.get("overrides", {})
    symbols.update(overrides.keys())

    return symbols


def get_max_history_days(timeframe: str, source_priority: List[str]) -> int:
    """Get max history days for timeframe based on source priority."""
    for source in source_priority:
        if source == "ib" and timeframe in IB_MAX_HISTORY_DAYS:
            return IB_MAX_HISTORY_DAYS[timeframe]
        elif source == "yahoo" and timeframe in YAHOO_MAX_HISTORY_DAYS:
            return YAHOO_MAX_HISTORY_DAYS[timeframe]
    return 7  # Conservative fallback


async def cmd_backfill(args: argparse.Namespace) -> int:
    """Batch download max history for all configured symbols and timeframes."""
    print()
    print("Historical Data Backfill")
    print("=" * 50)
    print()

    # Determine source priority
    if args.source == "yahoo":
        source_priority = ["yahoo"]
    elif args.source == "ib":
        source_priority = ["ib", "yahoo"]
    else:
        source_priority = ["ib", "yahoo"]  # Default: IB first

    # Discover symbols
    print("Discovering symbols...")
    symbols: Set[str] = set()

    # From config
    if args.symbols:
        config_symbols = {s.strip().upper() for s in args.symbols.split(",")}
        symbols.update(config_symbols)
        print(f"  CLI: {len(config_symbols)} symbols specified")
    else:
        universe_path = Path("config/signals/universe.yaml")
        config_symbols = load_universe_symbols(universe_path)
        symbols.update(config_symbols)
        print(f"  Config: {len(config_symbols)} symbols from universe.yaml")

    # TODO: Optionally add IB positions (requires live connection)
    # For now, we rely on config symbols

    if not symbols:
        print("No symbols found. Specify --symbols or check universe.yaml")
        return 1

    symbols_list = sorted(symbols)
    print(f"  Total: {len(symbols_list)} unique symbols")
    print()

    # Determine timeframes
    if args.timeframes:
        timeframes = [t.strip() for t in args.timeframes.split(",")]
    else:
        timeframes = BACKFILL_TIMEFRAMES

    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Source priority: {' > '.join(source_priority)}")
    print()

    # Initialize manager
    manager = HistoricalDataManager(base_dir=Path(args.data_dir))

    # Connect to IB if needed
    ib_adapter = None
    if "ib" in source_priority and args.source != "yahoo":
        try:
            print(f"Connecting to IB Gateway at {args.ib_host}:{args.ib_port}...")
            ib_adapter = await connect_ib_source(
                host=args.ib_host,
                port=args.ib_port,
                client_id=args.ib_client_id,
            )
            manager.set_ib_source(ib_adapter)
            print("IB connected successfully")
        except Exception as e:
            print(f"IB connection failed: {e}")
            if args.source == "ib":
                print("ERROR: IB source requested but connection failed")
                manager.close()
                return 1
            print("Falling back to Yahoo only")
            source_priority = ["yahoo"]

    print()

    # Generate work items
    print("Generating work items...")
    work_items: List[BackfillWorkItem] = []
    skipped_count = 0

    # Calculate proper end date (previous completed trading day)
    # Avoids requesting data for weekends/holidays or incomplete trading days
    calc = BarCountCalculator()
    previous_trading_day = calc.get_previous_trading_day()
    end_date = datetime.combine(previous_trading_day, datetime.min.time())
    print(f"End date: {end_date.date()} (last completed trading day)")

    for symbol in symbols_list:
        for tf in timeframes:
            max_days = get_max_history_days(tf, source_priority)
            raw_start = end_date - timedelta(days=max_days)
            # Ensure start_date is a trading day (not weekend/holiday)
            start_date = datetime.combine(
                calc.get_next_trading_day(raw_start), datetime.min.time()
            )

            # Check existing coverage
            gaps = manager.find_missing_ranges(symbol, tf, start_date, end_date)

            if not gaps:
                skipped_count += 1
                continue

            # Create work item for each gap
            for gap in gaps:
                work_items.append(BackfillWorkItem(
                    symbol=symbol,
                    timeframe=tf,
                    start=gap.start,
                    end=gap.end,
                    max_days=max_days,
                ))

    total_possible = len(symbols_list) * len(timeframes)
    print(f"  Total possible: {total_possible} (symbol x timeframe combinations)")
    print(f"  Already covered: {skipped_count}")
    print(f"  Pending: {len(work_items)} work items")
    print()

    if not work_items:
        print("All data is already up to date!")
        manager.close()
        if ib_adapter:
            await ib_adapter.disconnect()

        # Generate report if requested
        if args.report:
            return await _generate_report(Path(args.data_dir), args.report)
        return 0

    # Dry run mode
    if args.dry_run:
        print("DRY RUN - Would download:")
        for item in work_items[:20]:
            print(f"  {item.symbol}/{item.timeframe}: {item.start.date()} to {item.end.date()} ({item.max_days}d max)")
        if len(work_items) > 20:
            print(f"  ... and {len(work_items) - 20} more")
        manager.close()
        if ib_adapter:
            await ib_adapter.disconnect()
        return 0

    # Execute backfill with rate limiting
    print(f"Backfilling {len(work_items)} items...")
    print()

    start_time = time.time()
    completed = 0
    failed = 0
    total_bars = 0
    last_source = "yahoo"
    ib_fallbacks = 0  # Track when IB was requested but Yahoo was used
    source_counts = {"ib": 0, "yahoo": 0}

    try:
        for i, item in enumerate(work_items, 1):
            # Rate limiting based on expected source
            if last_source == "ib":
                await asyncio.sleep(IB_DELAY_SECONDS)
            else:
                await asyncio.sleep(YAHOO_DELAY_SECONDS)

            # Download
            result = await manager._download_range(
                symbol=item.symbol,
                timeframe=item.timeframe,
                start=item.start,
                end=item.end,
                priority=source_priority,
            )

            # Update last source for rate limiting
            last_source = result.source if result.source in ("ib", "yahoo") else "yahoo"

            # Print progress with source label
            if result.success:
                completed += 1
                total_bars += result.bars_downloaded
                source_counts[result.source] = source_counts.get(result.source, 0) + 1

                # Source label: [IB] or [Yahoo]
                source_label = f"[{result.source.upper():5}]"

                # Warn if IB was preferred but fell back to Yahoo
                if "ib" in source_priority and source_priority.index("ib") == 0 and result.source == "yahoo":
                    ib_fallbacks += 1
                    source_label += " (IB→Yahoo fallback)"

                status = f"✓ {source_label} {result.bars_downloaded:,} bars"
            else:
                failed += 1
                status = f"✗ [FAILED] {result.error or 'unknown'}"

            # Progress line
            print(f"[{i:4}/{len(work_items)}] {item.symbol:6}/{item.timeframe:4} {status}")

    finally:
        # Cleanup
        if ib_adapter:
            await ib_adapter.disconnect()

    elapsed = time.time() - start_time
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    print()
    print("=" * 50)
    print("Download Summary:")
    print(f"  Completed: {completed}/{len(work_items)}")
    print(f"  Failed: {failed}")
    print(f"  Total bars: {total_bars:,}")
    print(f"  Sources: IB={source_counts.get('ib', 0)}, Yahoo={source_counts.get('yahoo', 0)}")
    if ib_fallbacks > 0:
        print(f"  ⚠ IB fallbacks to Yahoo: {ib_fallbacks}")
    print(f"  Elapsed: {elapsed_str}")

    # Full Range Validation (if enabled)
    if args.validate:
        print()
        print("=" * 50)
        print("Full Range Validation")
        print("=" * 50)
        print("Using NYSE calendar (pandas-market-calendars)")
        print()

        from src.domain.services.data_validator import DataValidator, ValidationStatus
        validator = DataValidator(bar_store=manager._bar_store)

        validation_results = {"PASS": 0, "WARN": 0, "CAUTION": 0, "FAIL": 0}
        total_gaps = 0

        for symbol in symbols_list:
            for tf in timeframes:
                result = validator.validate(symbol, tf)

                if result.actual_bars == 0:
                    continue  # Skip if no data

                validation_results[result.status.name] = validation_results.get(result.status.name, 0) + 1
                total_gaps += len(result.gaps)

                # Status icon
                status_icon = {
                    ValidationStatus.PASS: "✓",
                    ValidationStatus.WARN: "⚠",
                    ValidationStatus.CAUTION: "!",
                    ValidationStatus.FAIL: "✗",
                }.get(result.status, "?")

                print(f"{symbol}/{tf}: [{status_icon} {result.status.name:7}] {result.coverage_pct:.1f}% | {result.actual_bars:,}/{result.expected_bars:,} bars")

                # Show gaps if any (first 2)
                if result.gaps and result.status in (ValidationStatus.FAIL, ValidationStatus.CAUTION):
                    for gap in result.gaps[:2]:
                        gap_start = gap.start.strftime("%Y-%m-%d")
                        gap_end = gap.end.strftime("%Y-%m-%d")
                        print(f"  └─ Gap: {gap_start} to {gap_end} ({gap.expected_bars} bars)")
                    if len(result.gaps) > 2:
                        print(f"  └─ ... and {len(result.gaps) - 2} more gaps")

        print()
        print("Validation Summary:")
        print(f"  PASS:    {validation_results['PASS']}")
        print(f"  WARN:    {validation_results['WARN']}")
        print(f"  CAUTION: {validation_results['CAUTION']}")
        print(f"  FAIL:    {validation_results['FAIL']}")
        print(f"  Total gaps: {total_gaps}")

        if validation_results['FAIL'] > 0:
            print()
            print("⚠ Some data failed validation. Run with --symbols flag to re-download specific symbols.")

    # Generate report
    if args.report:
        print()
        await _generate_report(Path(args.data_dir), args.report)

    manager.close()
    return 0 if failed == 0 else 1


async def _generate_report(base_dir: Path, output_path: str) -> int:
    """Generate HTML coverage report with line charts."""
    visualizer = CoverageVisualizer(base_dir=base_dir)
    visualizer.generate_html(output_path=Path(output_path))
    print(f"Coverage report written to: {output_path}")
    return 0


async def cmd_report(args: argparse.Namespace) -> int:
    """Generate HTML coverage report with line charts."""
    # Filter symbols if specified
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Filter timeframes if specified
    timeframes = None
    if args.timeframes:
        timeframes = [t.strip() for t in args.timeframes.split(",")]

    # CoverageVisualizer reads actual Parquet data (not DuckDB metadata)
    visualizer = CoverageVisualizer(base_dir=Path(args.data_dir))
    visualizer.generate_html(
        symbols=symbols,
        timeframes=timeframes,
        output_path=Path(args.output),
    )

    print(f"Coverage report written to: {args.output}")
    return 0


async def cmd_check_ib(args: argparse.Namespace) -> int:
    """Test IB Gateway/TWS connectivity with diagnostics."""
    import socket

    print()
    print("IB Gateway/TWS Connectivity Check")
    print("=" * 50)
    print()
    print(f"Target: {args.ib_host}:{args.ib_port}")
    print()

    # Step 1: TCP connectivity check
    print("1. TCP Connection Test...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((args.ib_host, args.ib_port))
        sock.close()

        if result == 0:
            print("   ✓ Port is open and accepting connections")
        else:
            print(f"   ✗ Cannot connect to port (error code: {result})")
            print()
            print("Troubleshooting:")
            print("  - Ensure IB Gateway or TWS is running")
            print("  - Check that API is enabled in configuration")
            print("  - Verify the port number (4001 for Gateway, 7497 for TWS paper)")
            return 1
    except socket.timeout:
        print("   ✗ Connection timed out")
        return 1
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return 1

    # Step 2: IB API connection
    print()
    print("2. IB API Connection Test...")
    try:
        adapter = IbHistoricalAdapter(
            host=args.ib_host,
            port=args.ib_port,
            client_id=args.ib_client_id,
        )
        await adapter.connect()
        print("   ✓ IB API connection established")
    except Exception as e:
        print(f"   ✗ IB API connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("  - Check that 'Enable ActiveX and Socket Clients' is enabled")
        print("  - Ensure client ID is not already in use")
        print("  - Verify 'Read-Only API' setting matches your needs")
        return 1

    # Step 3: Test historical data request
    print()
    print("3. Historical Data Request Test (SPY 1d)...")
    try:
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        bars = await adapter.fetch_historical("SPY", "1d", start_date, end_date)
        if bars:
            print(f"   ✓ Received {len(bars)} bars")
            print(f"     Range: {bars[0].bar_start.date() if bars[0].bar_start else 'N/A'} to {bars[-1].bar_start.date() if bars[-1].bar_start else 'N/A'}")
        else:
            print("   ⚠ Request succeeded but no bars returned (market may be closed)")
    except Exception as e:
        print(f"   ✗ Historical data request failed: {e}")
        print()
        print("Troubleshooting:")
        print("  - Check that you have market data subscriptions")
        print("  - Verify API permissions for historical data")
    finally:
        await adapter.disconnect()

    print()
    print("=" * 50)
    print("IB connectivity check complete")
    return 0


async def cmd_validate(args: argparse.Namespace) -> int:
    """Validate downloaded data coverage and quality."""
    from src.domain.services.data_validator import DataValidator, ValidationStatus
    from src.infrastructure.stores.parquet_historical_store import ParquetHistoricalStore
    import json

    print()
    print("Historical Data Validation")
    print("=" * 50)
    print()

    bar_store = ParquetHistoricalStore(base_dir=Path(args.data_dir))
    validator = DataValidator(bar_store=bar_store)

    # Discover symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = sorted(bar_store.list_symbols())

    if not symbols:
        print("No symbols found. Run backfill first or specify --symbols")
        return 1

    # Determine timeframes
    if args.timeframes:
        timeframes = [t.strip() for t in args.timeframes.split(",")]
    else:
        timeframes = BACKFILL_TIMEFRAMES

    # Parse dates
    start_date = parse_date(args.start) if args.start else None
    end_date = parse_date(args.end) if args.end else None

    print(f"Symbols: {len(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    if start_date:
        print(f"Start: {start_date.date()}")
    if end_date:
        print(f"End: {end_date.date()}")
    print()

    # Validate each combination
    results = []
    status_counts = {"PASS": 0, "WARN": 0, "CAUTION": 0, "FAIL": 0, "NO_DATA": 0}

    for symbol in symbols:
        for tf in timeframes:
            result = validator.validate(symbol, tf, start_date, end_date)

            # Handle no data case
            if result.actual_bars == 0:
                status_name = "NO_DATA"
            else:
                status_name = result.status.name
            status_counts[status_name] = status_counts.get(status_name, 0) + 1

            # Collect for JSON output
            results.append({
                "symbol": symbol,
                "timeframe": tf,
                "status": status_name,
                "coverage_pct": result.coverage_pct,
                "expected_bars": result.expected_bars,
                "actual_bars": result.actual_bars,
                "gaps": len(result.gaps),
                "anomalies": len(result.anomalies),
            })

            # Console output
            if result.actual_bars == 0:
                print(f"{symbol}/{tf}: NO DATA")
                continue

            # Status indicator
            status_icon = {
                ValidationStatus.PASS: "✓",
                ValidationStatus.WARN: "⚠",
                ValidationStatus.CAUTION: "!",
                ValidationStatus.FAIL: "✗",
            }.get(result.status, "?")

            coverage_str = f"{result.coverage_pct:.1f}%"
            bars_str = f"{result.actual_bars:,}/{result.expected_bars:,} bars"

            print(f"{symbol}/{tf}: [{status_icon} {status_name:7}] {coverage_str:7} | {bars_str}")

            # Show gaps if any
            if result.gaps:
                for gap in result.gaps[:3]:  # Show first 3 gaps
                    gap_start = gap.start.strftime("%Y-%m-%d")
                    gap_end = gap.end.strftime("%Y-%m-%d")
                    print(f"  └─ Gap: {gap_start} to {gap_end} ({gap.expected_bars} bars missing)")
                if len(result.gaps) > 3:
                    print(f"  └─ ... and {len(result.gaps) - 3} more gaps")

    # Summary
    print()
    print("=" * 50)
    print("Summary:")
    print(f"  PASS:    {status_counts['PASS']}")
    print(f"  WARN:    {status_counts['WARN']}")
    print(f"  CAUTION: {status_counts['CAUTION']}")
    print(f"  FAIL:    {status_counts['FAIL']}")
    print(f"  NO_DATA: {status_counts['NO_DATA']}")

    # JSON output
    if args.json:
        print()
        print("JSON Output:")
        print(json.dumps(results, indent=2))

    # Return non-zero if any failures
    return 1 if status_counts["FAIL"] > 0 else 0


async def cmd_replay(args: argparse.Namespace) -> int:
    """Replay historical bars through signal pipeline."""
    from src.domain.services.bar_replay_service import BarReplayService, ReplaySpeed, ReplayProgress, ReplayGapEvent
    from src.infrastructure.stores.parquet_historical_store import ParquetHistoricalStore

    print()
    print("Historical Bar Replay")
    print("=" * 50)
    print()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    timeframe = args.timeframe

    # Parse dates
    start_date = parse_date(args.start) if args.start else None
    end_date = parse_date(args.end) if args.end else None

    # Map speed string to enum
    speed_map = {
        "realtime": ReplaySpeed.REALTIME,
        "fast": ReplaySpeed.FAST_FORWARD,
        "max": ReplaySpeed.MAX_SPEED,
    }
    speed = speed_map.get(args.speed, ReplaySpeed.MAX_SPEED)

    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {timeframe}")
    print(f"Speed: {args.speed}" + (f" ({args.multiplier}x)" if args.speed == "fast" else ""))
    if start_date:
        print(f"Start: {start_date.date()}")
    if end_date:
        print(f"End: {end_date.date()}")
    print()

    # Initialize replay service (no event bus for CLI - just progress tracking)
    bar_store = ParquetHistoricalStore(base_dir=Path(args.data_dir))
    replay_service = BarReplayService(bar_store=bar_store)

    # Gap tracking
    gaps_detected = []

    def on_gap(gap: ReplayGapEvent):
        gaps_detected.append(gap)
        gap_start = gap.before_timestamp.strftime("%Y-%m-%d %H:%M")
        gap_end = gap.after_timestamp.strftime("%Y-%m-%d %H:%M")
        print(f"  Gap detected: {gap_start} -> {gap_end} ({gap.gap_hours:.1f}h)")

    def on_progress(progress: ReplayProgress):
        pct = progress.progress_pct
        rate = progress.bars_per_second
        eta = progress.eta_seconds
        eta_str = f"{int(eta)}s" if eta and eta < 600 else f"{int(eta/60)}m" if eta else "?"

        # Simple progress bar
        bar_width = 40
        filled = int(bar_width * pct / 100)
        bar = "=" * filled + "-" * (bar_width - filled)

        print(f"\r[{bar}] {pct:5.1f}% | {progress.replayed_bars:,} bars | {rate:.0f}/s | ETA: {eta_str}", end="", flush=True)

    # Replay each symbol
    total_bars = 0
    total_gaps = 0

    for symbol in symbols:
        print(f"Replaying {symbol}/{timeframe}...")
        gaps_detected.clear()

        progress = await replay_service.replay(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            speed=speed,
            speed_multiplier=args.multiplier,
            on_progress=on_progress,
            on_gap=on_gap,
            progress_interval=100,
        )

        print()  # Newline after progress bar

        if progress.total_bars == 0:
            print(f"  No bars found for {symbol}/{timeframe}")
            continue

        total_bars += progress.replayed_bars
        total_gaps += progress.gaps_detected

        print(f"  Completed: {progress.replayed_bars:,} bars in {progress.elapsed_seconds:.1f}s")
        if progress.gaps_detected > 0:
            print(f"  Gaps: {progress.gaps_detected}")
        print()

    # Summary
    print("=" * 50)
    print("Replay Summary:")
    print(f"  Total bars: {total_bars:,}")
    print(f"  Gaps detected: {total_gaps}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Historical Data Loader CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="data/historical",
        help="Base directory for historical data",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download historical data")
    download_parser.add_argument(
        "--symbols", "-s",
        required=True,
        help="Comma-separated list of symbols",
    )
    download_parser.add_argument(
        "--timeframe", "-t",
        default="1d",
        choices=["5min", "15min", "30min", "1h", "4h", "1d", "1w"],
        help="Bar timeframe (default: 1d)",
    )
    download_parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    download_parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD, default: today)",
    )
    download_parser.add_argument(
        "--source",
        choices=["yahoo", "ib"],
        help="Data source (default: auto)",
    )
    download_parser.add_argument(
        "--ib-host",
        default="127.0.0.1",
        help="IB Gateway/TWS host (default: 127.0.0.1)",
    )
    download_parser.add_argument(
        "--ib-port",
        type=int,
        default=4001,
        help="IB Gateway/TWS port (default: 4001 for Gateway, 7497 for TWS)",
    )
    download_parser.add_argument(
        "--ib-client-id",
        type=int,
        default=99,
        help="IB client ID for historical data (default: 99)",
    )

    # Coverage command
    coverage_parser = subparsers.add_parser("coverage", help="Show coverage for a symbol")
    coverage_parser.add_argument("symbol", help="Symbol to check")

    # List command
    subparsers.add_parser("list", help="List all symbols with data")

    # Fill gaps command
    fill_parser = subparsers.add_parser("fill-gaps", help="Fill gaps in data")
    fill_parser.add_argument(
        "--symbol", "-s",
        help="Specific symbol to fill",
    )
    fill_parser.add_argument(
        "--timeframe", "-t",
        help="Specific timeframe to fill",
    )
    fill_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum gaps to fill (default: 100)",
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete data for a symbol")
    delete_parser.add_argument("symbol", help="Symbol to delete")
    delete_parser.add_argument(
        "--timeframe", "-t",
        help="Specific timeframe to delete",
    )
    delete_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation",
    )

    # Rebuild coverage command
    subparsers.add_parser(
        "rebuild-coverage",
        help="Rebuild coverage metadata from stored Parquet files",
    )

    # Backfill command
    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Batch download max history for all configured symbols",
    )
    backfill_parser.add_argument(
        "--symbols", "-s",
        help="Comma-separated symbols (default: from config/signals/universe.yaml)",
    )
    backfill_parser.add_argument(
        "--timeframes", "-t",
        help="Comma-separated timeframes (default: 1d,4h,1h,30m,15m,5m,1m)",
    )
    backfill_parser.add_argument(
        "--source",
        choices=["yahoo", "ib", "auto"],
        default="auto",
        help="Data source (default: auto = ib > yahoo)",
    )
    backfill_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without downloading",
    )
    backfill_parser.add_argument(
        "--report",
        default="data/coverage_report.html",
        help="Generate HTML report after backfill (default: data/coverage_report.html)",
    )
    backfill_parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation after backfill",
    )
    backfill_parser.add_argument(
        "--ib-host",
        default="127.0.0.1",
        help="IB Gateway/TWS host",
    )
    backfill_parser.add_argument(
        "--ib-port",
        type=int,
        default=4001,
        help="IB Gateway/TWS port",
    )
    backfill_parser.add_argument(
        "--ib-client-id",
        type=int,
        default=99,
        help="IB client ID for historical data",
    )
    backfill_parser.add_argument(
        "--validate",
        action="store_true",
        help="Run Full Range Validation after backfill completes",
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate HTML coverage report",
    )
    report_parser.add_argument(
        "--output", "-o",
        default="data/coverage_report.html",
        help="Output HTML file path",
    )
    report_parser.add_argument(
        "--symbols", "-s",
        help="Filter to specific symbols (comma-separated)",
    )
    report_parser.add_argument(
        "--timeframes", "-t",
        help="Filter to specific timeframes (comma-separated)",
    )

    # Check-IB command
    check_ib_parser = subparsers.add_parser(
        "check-ib",
        help="Test IB Gateway/TWS connectivity",
    )
    check_ib_parser.add_argument(
        "--ib-host",
        default="127.0.0.1",
        help="IB Gateway/TWS host (default: 127.0.0.1)",
    )
    check_ib_parser.add_argument(
        "--ib-port",
        type=int,
        default=4001,
        help="IB Gateway/TWS port (default: 4001)",
    )
    check_ib_parser.add_argument(
        "--ib-client-id",
        type=int,
        default=99,
        help="IB client ID (default: 99)",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate downloaded data coverage and quality",
    )
    validate_parser.add_argument(
        "--symbols", "-s",
        help="Comma-separated symbols (default: all)",
    )
    validate_parser.add_argument(
        "--timeframes", "-t",
        help="Comma-separated timeframes (default: 1d,4h,1h,30m,15m,5m,1m)",
    )
    validate_parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD, default: from data)",
    )
    validate_parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD, default: from data)",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # Replay command
    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay historical bars through signal pipeline",
    )
    replay_parser.add_argument(
        "--symbols", "-s",
        required=True,
        help="Comma-separated symbols",
    )
    replay_parser.add_argument(
        "--timeframe", "-t",
        default="1d",
        help="Bar timeframe (default: 1d)",
    )
    replay_parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD)",
    )
    replay_parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD)",
    )
    replay_parser.add_argument(
        "--speed",
        choices=["realtime", "fast", "max"],
        default="max",
        help="Replay speed (default: max)",
    )
    replay_parser.add_argument(
        "--multiplier",
        type=float,
        default=10.0,
        help="Speed multiplier for 'fast' mode (default: 10.0)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return 1

    # Handle no-report flag for backfill
    if hasattr(args, "no_report") and args.no_report:
        args.report = None

    # Run command
    cmd_map = {
        "download": cmd_download,
        "coverage": cmd_coverage,
        "list": cmd_list,
        "fill-gaps": cmd_fill_gaps,
        "delete": cmd_delete,
        "rebuild-coverage": cmd_rebuild_coverage,
        "backfill": cmd_backfill,
        "report": cmd_report,
        "check-ib": cmd_check_ib,
        "validate": cmd_validate,
        "replay": cmd_replay,
    }

    cmd_func = cmd_map.get(args.command)
    if not cmd_func:
        parser.print_help()
        return 1

    return asyncio.run(cmd_func(args))


if __name__ == "__main__":
    sys.exit(main())
