"""
APEX Signal Server — Main Entry Point (the production image runs `python main.py`).

    python main.py                    # REST + WS API server (:8322)

Legacy (frozen backtest, removed with the Phase 6 strip-down):
    python main.py --mode backtest --strategy trend_pulse --symbols SPY \
        --start 2025-01-01 --end 2025-06-30
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
  python main.py                      REST + WS API server (:8322)

Legacy:
  python main.py --mode backtest --spec config/backtest/trend_pulse_validate.yaml
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["backtest"],
        help="Legacy frozen backtest mode — runs instead of the API server",
    )

    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")

    bt = parser.add_argument_group("Backtest")
    bt.add_argument("--spec", type=str)
    bt.add_argument("--strategy", type=str)
    bt.add_argument("--symbols", type=str)
    bt.add_argument("--start", type=str)
    bt.add_argument("--end", type=str)
    bt.add_argument("--capital", type=float, default=100_000.0)

    return parser.parse_args()


async def run_api(args: argparse.Namespace) -> None:
    """Run the REST + WS API server."""
    level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s: %(message)s")
    import uvicorn

    from src.api.server import create_app

    port = int(os.environ.get("APEX_API_PORT", "8322"))
    logging.getLogger("apex").info("Starting API server on port %d...", port)
    config = uvicorn.Config(
        create_app,
        host="0.0.0.0",  # nosec B104 - backend service intentionally listens on all interfaces for Xenon consumers
        port=port,
        factory=True,
        log_level="info",
    )
    await uvicorn.Server(config).serve()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.mode == "backtest":
        from src.backtest.runner import SingleBacktestRunner

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        runner = SingleBacktestRunner.from_args(args)
        result = asyncio.run(runner.run())
        sys.exit(0 if getattr(result, "is_profitable", True) else 1)

    else:
        try:
            asyncio.run(run_api(args))
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
