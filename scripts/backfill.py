#!/usr/bin/env python3
"""
Standalone backfill script for the persistent layer.

Usage:
    python scripts/backfill.py --full-reload
    python scripts/backfill.py --days 30
    python scripts/backfill.py --config config/persistent.yaml

This script can be run independently of the main risk engine to:
- Backfill historical order/trade data from Futu and/or IB
- Run strategy classification on historical trades
- Validate data consistency via reconciliation
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.persistent.orchestrator import PersistenceOrchestrator, load_config


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the backfill script."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from libraries
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def load_risk_config() -> dict:
    """Load broker configuration from risk_config.yaml."""
    risk_config_paths = [
        PROJECT_ROOT / "config" / "risk_config.yaml",
        Path("config/risk_config.yaml"),
    ]

    for path in risk_config_paths:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)

    logging.warning("risk_config.yaml not found, using defaults")
    return {}


async def create_futu_adapter(config: dict, risk_config: dict):
    """Create and connect Futu adapter if configured."""
    # Get Futu settings from risk_config.yaml
    brokers_config = risk_config.get("brokers", {})
    futu_broker_config = brokers_config.get("futu", {})

    if not futu_broker_config.get("enabled", False):
        logging.info("Futu adapter disabled in risk_config.yaml")
        return None

    try:
        from src.infrastructure.adapters.futu.adapter import FutuAdapter

        # Use risk_config.yaml values, with env overrides
        host = os.environ.get("FUTU_HOST", futu_broker_config.get("host", "127.0.0.1"))
        port = int(os.environ.get("FUTU_PORT", futu_broker_config.get("port", 11111)))
        trd_env = os.environ.get("FUTU_TRD_ENV", futu_broker_config.get("trd_env", "REAL"))
        security_firm = futu_broker_config.get("security_firm", "FUTUSECURITIES")
        filter_trdmarket = futu_broker_config.get("filter_trdmarket", "US")

        adapter = FutuAdapter(
            host=host,
            port=port,
            trd_env=trd_env,
            security_firm=security_firm,
            filter_trdmarket=filter_trdmarket,
        )

        await adapter.connect()
        logging.info(f"Connected to Futu OpenD at {host}:{port}")
        return adapter

    except ImportError:
        logging.warning("futu-api not installed, skipping Futu")
        return None
    except Exception as e:
        logging.error(f"Failed to connect to Futu: {e}")
        return None


async def create_ib_adapter(config: dict, risk_config: dict):
    """Create and connect IB adapter if configured."""
    # Get IB settings from risk_config.yaml
    brokers_config = risk_config.get("brokers", {})
    ib_broker_config = brokers_config.get("ibkr", {})

    if not ib_broker_config.get("enabled", False):
        logging.info("IB adapter disabled in risk_config.yaml")
        return None

    try:
        from src.infrastructure.adapters.ib.adapter import IbAdapter

        # Use risk_config.yaml values, with env overrides
        host = os.environ.get("IB_HOST", ib_broker_config.get("host", "127.0.0.1"))
        port = int(os.environ.get("IB_PORT", ib_broker_config.get("port", 4001)))
        client_id = int(os.environ.get("IB_CLIENT_ID", ib_broker_config.get("client_id", 1)))

        adapter = IbAdapter(
            host=host,
            port=port,
            client_id=client_id,
        )

        await adapter.connect()
        logging.info(f"Connected to IB Gateway at {host}:{port}")
        return adapter

    except ImportError:
        logging.warning("ib-async not installed, skipping IB real-time")
        return None
    except Exception as e:
        logging.warning(f"Failed to connect to IB Gateway: {e}")
        # IB Flex can still work without real-time connection
        return None


async def run_backfill(args: argparse.Namespace) -> int:
    """Run the backfill process."""
    logger = logging.getLogger("backfill")

    # Load persistence configuration
    config_path = args.config
    if not os.path.exists(config_path):
        # Try relative to project root
        config_path = PROJECT_ROOT / config_path
        if not config_path.exists():
            logger.error(f"Config file not found: {args.config}")
            return 1

    logger.info(f"Loading configuration from {config_path}")
    config = load_config(str(config_path))

    # Load broker configuration from risk_config.yaml
    risk_config = load_risk_config()

    # Override DSN from environment if set
    if os.environ.get("POSTGRES_DSN"):
        config["storage"]["dsn"] = os.environ["POSTGRES_DSN"]

    # Override IB Flex credentials from environment
    if os.environ.get("IB_FLEX_TOKEN"):
        config.setdefault("ib", {})["flex_token"] = os.environ["IB_FLEX_TOKEN"]
    if os.environ.get("IB_FLEX_QUERY_ID"):
        config.setdefault("ib", {})["flex_query_id"] = os.environ["IB_FLEX_QUERY_ID"]

    # Create broker adapters using risk_config.yaml settings
    futu_adapter = None
    ib_adapter = None

    if not args.skip_futu:
        futu_adapter = await create_futu_adapter(config, risk_config)

    if not args.skip_ib:
        ib_adapter = await create_ib_adapter(config, risk_config)

    # Create orchestrator
    orchestrator = PersistenceOrchestrator(
        config=config,
        futu_adapter=futu_adapter,
        ib_adapter=ib_adapter,
    )

    try:
        # Drop tables if requested (for schema changes)
        if args.drop_tables:
            logger.warning("Dropping all tables due to --drop-tables flag")
            await orchestrator.connect()
            await orchestrator.store.drop_all_tables()
            await orchestrator.close()
            # Reconnect to recreate schema
            logger.info("Reconnecting to recreate schema...")

        # Parse start date
        start_date = None
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

        # Run backfill
        summary = await orchestrator.run(
            full_reload=args.full_reload or args.drop_tables,  # Force full reload after drop
            days=args.days,
            start_date=start_date,
            skip_classify=args.skip_classify,
            skip_reconcile=args.skip_reconcile,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("BACKFILL SUMMARY")
        print("=" * 60)
        print(f"Mode: {summary['mode'].upper()}")
        print(f"Date range: {summary['start_date'][:10]} to {summary['end_date'][:10]}")
        print()
        print("Raw Data Fetched:")
        print(f"  Futu:  {summary['futu']['orders']} orders, {summary['futu']['trades']} trades, {summary['futu']['fees']} fees")
        print(f"  IB:    {summary['ib']['orders']} orders, {summary['ib']['trades']} trades, {summary['ib']['fees']} fees")
        print()
        print("Normalized Data:")
        print(f"  Orders: {summary['normalized']['orders']}")
        print(f"  Trades: {summary['normalized']['trades']}")
        print()
        print(f"Strategies classified: {summary['strategies']}")
        print(f"Reconciliation anomalies: {summary['anomalies']}")

        if summary["errors"]:
            print()
            print("Errors:")
            for error in summary["errors"]:
                print(f"  - {error}")

        print("=" * 60)

        # Save summary to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Summary saved to {output_path}")

        return 0 if not summary["errors"] else 1

    finally:
        # Cleanup adapters
        if futu_adapter:
            await futu_adapter.disconnect()
        if ib_adapter:
            await ib_adapter.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill historical order/trade data to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full reload (truncate and reload all data)
    python scripts/backfill.py --full-reload

    # Incremental load for last 30 days
    python scripts/backfill.py --days 30

    # Futu only, skip IB
    python scripts/backfill.py --skip-ib --days 90

    # Custom config file
    python scripts/backfill.py --config config/persistent_prod.yaml

Environment Variables:
    POSTGRES_DSN       PostgreSQL connection string
    FUTU_HOST          Futu OpenD host (default: 127.0.0.1)
    FUTU_PORT          Futu OpenD port (default: 11111)
    FUTU_TRD_ENV       Futu trading environment (REAL or SIMULATE)
    IB_HOST            IB Gateway host (default: 127.0.0.1)
    IB_PORT            IB Gateway port (default: 7497)
    IB_FLEX_TOKEN      IB Flex Web Service token
    IB_FLEX_QUERY_ID   IB Flex Query ID
        """,
    )

    parser.add_argument(
        "--config",
        default="config/persistent.yaml",
        help="Path to configuration file (default: config/persistent.yaml)",
    )

    parser.add_argument(
        "--full-reload",
        action="store_true",
        help="Truncate normalized tables and reload all data",
    )

    parser.add_argument(
        "--drop-tables",
        action="store_true",
        help="Drop and recreate all tables (use when schema has changed)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to look back (default: from config)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-08-01",
        help="Start date for full reload in YYYY-MM-DD format (default: 2022-08-01)",
    )

    parser.add_argument(
        "--skip-futu",
        action="store_true",
        help="Skip Futu data processing",
    )

    parser.add_argument(
        "--skip-ib",
        action="store_true",
        help="Skip IB data processing",
    )

    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Skip strategy classification",
    )

    parser.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="Skip reconciliation checks",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Save summary to JSON file",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Run async main
    try:
        exit_code = asyncio.run(run_backfill(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBackfill interrupted")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Backfill failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
