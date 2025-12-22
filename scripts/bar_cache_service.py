#!/usr/bin/env python3
"""
Bar Cache Service - IB historical data cache daemon.

Usage:
  python scripts/bar_cache_service.py --env dev
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from config.config_manager import ConfigManager
from src.services.bar_cache_service import BarCacheService, BarCacheSettings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bar_cache_service")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bar cache service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--env", type=str, default="dev", help="Config environment (default: dev)")
    parser.add_argument("--config-dir", type=str, default="config", help="Config directory")
    parser.add_argument("--host", type=str, help="Override bar cache host")
    parser.add_argument("--port", type=int, help="Override bar cache port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = ConfigManager(config_dir=args.config_dir, env=args.env).load()
    except Exception as exc:
        logger.error(f"Failed to load config: {exc}")
        return 1

    bar_cache_cfg = config.historical_data.bar_cache
    settings = BarCacheSettings(
        host=args.host or bar_cache_cfg.host,
        port=args.port or bar_cache_cfg.port,
        max_cache_entries=bar_cache_cfg.max_cache_entries,
    )

    service = BarCacheService(
        settings=settings,
        ib_host=config.ibkr.host,
        ib_port=config.ibkr.port,
        historical_client_ids=config.ibkr.client_ids.historical_pool,
    )

    await service.start()
    logger.info(
        "Bar cache service started",
        extra={"host": settings.host, "port": settings.port},
    )

    try:
        await service.serve_forever()
    except asyncio.CancelledError:
        logger.info("Bar cache service cancelled")
    finally:
        await service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
