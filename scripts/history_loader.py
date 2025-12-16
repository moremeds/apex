#!/usr/bin/env python3
"""
History Loader CLI - Load historical order/execution data from brokers.

Usage:
    python scripts/history_loader.py --broker futu --account ACC123 --market US --days 30
    python scripts/history_loader.py --broker ib --account U1234567 --days 7
    python scripts/history_loader.py --broker all --days 30

Commands:
    --broker futu   Load Futu orders, deals, and fees
    --broker ib     Load IB executions and commissions
    --broker all    Load from all configured brokers

Options:
    --account       Specific account ID (optional, loads all if not specified)
    --market        Market filter for Futu (US, HK, CN)
    --days          Lookback period in days (default: 30)
    --from-date     Start date (YYYY-MM-DD)
    --to-date       End date (YYYY-MM-DD)
    --dry-run       Show what would be loaded without writing
    --force         Force full reload (ignore last sync time)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config_manager import ConfigManager
from config.models import DatabaseConfig
from src.infrastructure.persistence.database import Database
from src.services.history_loader_service import HistoryLoaderService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("history_loader")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load historical order/execution data from brokers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--broker",
        choices=["futu", "ib", "all"],
        required=True,
        help="Broker to load data from",
    )

    parser.add_argument(
        "--account",
        type=str,
        help="Specific account ID to load (optional)",
    )

    parser.add_argument(
        "--market",
        choices=["US", "HK", "CN", "SG", "JP", "AU"],
        default="US",
        help="Market filter for Futu (default: US)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback period in days (default: 30)",
    )

    parser.add_argument(
        "--from-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--to-date",
        type=str,
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be loaded without writing",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full reload (ignore last sync time)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Config file path (default: config/base.yaml)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def parse_date(date_str: str) -> date:
    """Parse date string to date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine date range
    if args.from_date:
        from_date = parse_date(args.from_date)
    else:
        from_date = date.today() - timedelta(days=args.days)

    if args.to_date:
        to_date = parse_date(args.to_date)
    else:
        to_date = date.today()

    logger.info(f"History Loader starting")
    logger.info(f"  Broker: {args.broker}")
    logger.info(f"  Date range: {from_date} to {to_date}")
    if args.account:
        logger.info(f"  Account: {args.account}")
    if args.market and args.broker in ("futu", "all"):
        logger.info(f"  Market: {args.market}")
    if args.dry_run:
        logger.info("  Mode: DRY RUN (no writes)")
    if args.force:
        logger.info("  Mode: FORCE (full reload)")

    # Load configuration
    try:
        config_manager = ConfigManager(config_dir="config", env="dev")
        config = config_manager.load()
        db_config = config.database if config.database else DatabaseConfig()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Connect to database
    db = Database(db_config)

    try:
        await db.connect()
        logger.info("Database connected")

        # Create history loader service
        service = HistoryLoaderService(
            db=db,
            config=config,
            dry_run=args.dry_run,
        )

        # Run the loader
        if args.broker == "futu":
            result = await service.load_futu_history(
                account_id=args.account,
                market=args.market,
                from_date=from_date,
                to_date=to_date,
                force=args.force,
            )
        elif args.broker == "ib":
            result = await service.load_ib_history(
                account_id=args.account,
                from_date=from_date,
                to_date=to_date,
                force=args.force,
            )
        else:  # all
            result = await service.load_all_history(
                from_date=from_date,
                to_date=to_date,
                force=args.force,
            )

        # Print summary
        logger.info("=" * 60)
        logger.info("Load Summary:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Orders loaded: {result.orders_loaded}")
        logger.info(f"  Deals/Executions loaded: {result.deals_loaded}")
        logger.info(f"  Fees/Commissions loaded: {result.fees_loaded}")
        logger.info(f"  Duration: {result.duration_seconds:.2f}s")

        if result.errors:
            logger.warning(f"  Errors: {len(result.errors)}")
            for error in result.errors[:5]:  # Show first 5 errors
                logger.warning(f"    - {error}")

        return 0 if result.status == "SUCCESS" else 1

    except Exception as e:
        logger.error(f"History loader failed: {e}", exc_info=True)
        return 1

    finally:
        await db.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
