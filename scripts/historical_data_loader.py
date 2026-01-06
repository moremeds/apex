#!/usr/bin/env python3
"""
Historical Data Loader CLI.

Download and manage historical market data from Yahoo Finance and IB.

Usage:
    # Download data for symbols
    python scripts/historical_data_loader.py download --symbols AAPL,MSFT --timeframe 1d \
        --start 2024-01-01 --end 2024-12-31 --source yahoo

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
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.historical_data_manager import HistoricalDataManager
from src.infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

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

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return 1

    # Run command
    cmd_map = {
        "download": cmd_download,
        "coverage": cmd_coverage,
        "list": cmd_list,
        "fill-gaps": cmd_fill_gaps,
        "delete": cmd_delete,
        "rebuild-coverage": cmd_rebuild_coverage,
    }

    cmd_func = cmd_map.get(args.command)
    if not cmd_func:
        parser.print_help()
        return 1

    return asyncio.run(cmd_func(args))


if __name__ == "__main__":
    sys.exit(main())
