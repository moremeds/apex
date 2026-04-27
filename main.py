"""
APEX Signal Server — Main Entry Point.

Services:
    python main.py --service signal   # Signal service daemon (IB → PG)
    python main.py --service api      # REST API server (:8322)
    python main.py --service all      # Both services (dev convenience)

Legacy modes (backward compatible):
    python main.py --mode backtest --strategy trend_pulse --symbols SPY \
        --start 2025-01-01 --end 2025-06-30
    python main.py --mode trading --strategy trend_pulse --symbols SPY
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="APEX Signal Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Services:
  python main.py --service signal     Signal daemon (IB ticks → PG)
  python main.py --service api        REST API server (:8322)
  python main.py --service all        Both services (dev)

Legacy:
  python main.py --mode backtest --spec config/backtest/trend_pulse_validate.yaml
  python main.py --mode trading --strategy trend_pulse --symbols SPY
        """,
    )

    parser.add_argument(
        "--service",
        type=str,
        default="all",
        choices=["signal", "api", "all"],
        help="Service to run: signal, api, or all (default: all)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["backtest", "trading"],
        help="Legacy mode (backtest or trading) — bypasses --service",
    )

    parser.add_argument("--env", type=str, default="dev", choices=["dev", "prod"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")

    bt = parser.add_argument_group("Backtest")
    bt.add_argument("--engine", type=str, default="apex", choices=["apex", "backtrader"])
    bt.add_argument("--spec", type=str)
    bt.add_argument("--strategy", type=str)
    bt.add_argument("--symbols", type=str)
    bt.add_argument("--start", type=str)
    bt.add_argument("--end", type=str)
    bt.add_argument("--capital", type=float, default=100_000.0)

    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


async def run_services(args: argparse.Namespace) -> None:
    """Run the selected services concurrently."""
    level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s: %(message)s")
    logger = logging.getLogger("apex")

    tasks = []

    if args.service in ("signal", "all"):
        from src.services.signal_service import run_signal_service

        logger.info("Starting signal service...")
        tasks.append(asyncio.create_task(run_signal_service()))

    if args.service in ("api", "all"):
        import uvicorn

        from src.api.server import create_app

        port = int(os.environ.get("APEX_API_PORT", "8322"))
        logger.info("Starting API server on port %d...", port)

        config = uvicorn.Config(
            create_app,
            host="0.0.0.0",
            port=port,
            factory=True,
            log_level="info",
        )
        server = uvicorn.Server(config)
        tasks.append(asyncio.create_task(server.serve()))

    if tasks:
        await asyncio.gather(*tasks)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.mode == "backtest":
        from src.backtest.execution import BACKTRADER_AVAILABLE
        from src.backtest.runner import BacktraderRunner, SingleBacktestRunner

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        if args.engine == "backtrader" and BACKTRADER_AVAILABLE:
            runner = BacktraderRunner.from_args(args)
        else:
            runner = SingleBacktestRunner.from_args(args)
        result = asyncio.run(runner.run())
        sys.exit(0 if getattr(result, "is_profitable", True) else 1)

    elif args.mode == "trading":
        from src.runners.trading_runner import TradingRunner

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        runner = TradingRunner(
            strategy_name=args.strategy,
            symbols=[s.strip() for s in args.symbols.split(",")],
            broker="ib",
            dry_run=args.dry_run,
        )
        sys.exit(asyncio.run(runner.run()))

    else:
        try:
            asyncio.run(run_services(args))
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
