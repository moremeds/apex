"""
Backtest CLI Commands.

Main entry points for the backtest runner:
- main_async: Async command handler
- main: Sync wrapper for CLI entry
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Union

from .parser import create_parser

if TYPE_CHECKING:
    from ..runner import (
        BacktraderRunner as _BacktraderRunner,
        SingleBacktestRunner as _SingleBacktestRunner,
    )

logger = logging.getLogger(__name__)


async def main_async() -> None:
    """Async main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # List strategies command
    if args.list_strategies:
        from ...domain.strategy.registry import get_strategy_info, list_strategies

        print("\nAvailable strategies:")
        for name in sorted(list_strategies()):
            info = get_strategy_info(name)
            desc = info.get("description", "No description") if info else "No description"
            print(f"  {name:20s} - {desc}")
        return

    # Systematic experiment mode
    if args.spec and not args.strategy:
        from ..runner import run_systematic_experiment

        await run_systematic_experiment(
            spec_path=args.spec,
            output_dir=args.output,
            parallel=args.parallel,
            dry_run=args.dry_run,
            generate_report=not args.no_report,
        )
        return

    # Single backtest mode - validate args
    if not args.strategy:
        parser.error("--strategy or --spec required")
    if not args.symbols:
        parser.error("--symbols required")
    if not args.start or not args.end:
        parser.error("--start and --end required")

    # Parse symbols once after validation
    symbols = [s.strip() for s in args.symbols.split(",")]

    try:
        engine_type = args.engine or "apex"
        runner: Union[_BacktraderRunner, _SingleBacktestRunner]

        if engine_type == "backtrader":
            from ..runner import BacktraderRunner

            runner = BacktraderRunner(
                strategy_name=args.strategy,
                symbols=symbols,
                start_date=args.start,
                end_date=args.end,
                initial_capital=args.capital,
                data_source=args.data_source,
                data_dir=args.data_dir,
                bar_size=args.bar_size,
                commission=args.commission,
            )
            backtest_result = await runner.run()

        elif engine_type == "vectorbt":
            from ..core import RunSpec, TimeWindow
            from ..execution.engines import VectorBTConfig, VectorBTEngine

            config = VectorBTConfig(data_source="ib", ib_port=4001)
            engine = VectorBTEngine(config)

            for symbol in symbols:
                window = TimeWindow(
                    window_id="single_run",
                    fold_index=0,
                    train_start=args.start,
                    train_end=args.end,
                    test_start=args.start,
                    test_end=args.end,
                )
                spec = RunSpec(
                    trial_id="cli_trial",
                    symbol=symbol,
                    window=window,
                    profile_version="1.0",
                    data_version="1.0",
                    params={"strategy_type": args.strategy},
                )
                result = engine.run(spec)
                print(
                    f"\n{symbol}: Return={result.metrics.total_return:.2%}, "
                    f"Sharpe={result.metrics.sharpe:.2f}"
                )
            return

        else:
            # Default: ApexEngine
            from ..runner import SingleBacktestRunner

            runner = SingleBacktestRunner.from_args(args)
            backtest_result = await runner.run()

        sys.exit(0 if backtest_result.is_profitable else 1)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
