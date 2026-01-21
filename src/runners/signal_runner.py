"""
Signal Runner - Standalone TA signal pipeline for live and backfill processing.

Enables running the signal pipeline independently from the TUI for:
- Live signal generation without TUI overhead
- Historical bar backfill for indicator warmup
- Post-generation validation via --validate flag

Usage:
    # Run pipeline on live market data (headless mode)
    python -m src.runners.signal_runner --live --symbols AAPL,TSLA

    # Backfill signals from historical bars
    python -m src.runners.signal_runner --backfill --symbols AAPL --days 365

    # Connect to database for persistence
    python -m src.runners.signal_runner --live --symbols AAPL --with-persistence

    # Generate report and run quality gates
    python -m src.runners.signal_runner --live --symbols AAPL --format package --validate
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from ..domain.signals.pipeline import (
    SignalPipelineConfig,
    SignalPipelineProcessor,
    BarValidator,
    TurningPointTrainer,
    create_argument_parser,
)
from ..domain.signals.pipeline.config import parse_config
from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


# Re-export for backward compatibility
SignalRunnerConfig = SignalPipelineConfig
create_parser = create_argument_parser


class SignalRunner:
    """
    Standalone signal pipeline runner.

    Thin orchestrator that delegates to specialized components:
    - SignalPipelineProcessor: Live/backfill processing and report generation
    - BarValidator: Bar count validation (PR-01)
    - TurningPointTrainer: Model training

    Can run TASignalService independently for:
    - Live execution (real market data, optional DB)
    - Backfill (historical bars from data provider)
    """

    def __init__(self, config: SignalPipelineConfig) -> None:
        """
        Initialize signal runner.

        Args:
            config: Runner configuration.
        """
        self.config = config
        self._processor: Optional[SignalPipelineProcessor] = None

    async def run(self) -> int:
        """
        Run the signal pipeline based on configuration.

        Returns:
            Exit code (0 for success, non-zero for errors).
        """
        try:
            # Create processor and initialize
            self._processor = SignalPipelineProcessor(self.config)
            await self._processor.initialize()

            # Run training phase if requested
            if self.config.train_models or self.config.retrain_models:
                print("\n" + "=" * 60)
                print("=== TRAINING PHASE ===")
                print("=" * 60)

                trainer = TurningPointTrainer(self.config)
                training_result = await trainer.train()
                if training_result != 0:
                    logger.warning("Training phase completed with issues")
                    # Continue to signal phase even if training had issues

            # Run signal phase
            exit_code = 0
            if self.config.validate_bars:
                # Bar validation mode (PR-01): output BarValidationReport
                validator = BarValidator(self.config)
                exit_code = await validator.validate()
            elif self.config.backfill:
                print("\n" + "=" * 60)
                print("=== SIGNAL PHASE (Backfill) ===")
                print("=" * 60)
                exit_code = await self._processor.run_backfill()
            elif self.config.live:
                print("\n" + "=" * 60)
                print("=== SIGNAL PHASE (Live) ===")
                print("=" * 60)
                exit_code = await self._processor.run_live()
            else:
                # Training-only mode is valid
                if self.config.train_models or self.config.retrain_models:
                    return 0
                print("No mode specified. Use --live, --backfill, or --validate-bars")
                return 1

            # Run post-generation validation if requested (M3 integration)
            if self.config.validate and exit_code == 0:
                exit_code = self._run_validation_gates()

            return exit_code

        except KeyboardInterrupt:
            logger.info("Signal runner interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"Signal runner error: {e}", exc_info=True)
            return 1
        finally:
            if self._processor:
                await self._processor.shutdown()

    def _run_validation_gates(self) -> int:
        """
        Run quality gates (G1-G10) on the generated report package.

        Returns:
            Exit code (0 for all gates passed, 1 for failures).
        """
        from scripts.validate_gates import run_all_gates

        print("\n" + "=" * 60)
        print("=== VALIDATION PHASE (Quality Gates G1-G10) ===")
        print("=" * 60)

        # Determine package path from html_output
        if not self.config.html_output:
            logger.error("--validate requires --html-output to specify package path")
            return 1

        package_path = Path(self.config.html_output)
        if not package_path.exists():
            logger.error(f"Package path not found: {package_path}")
            return 1

        # Run all gates
        report = run_all_gates(package_path)

        # Print summary
        print()
        for gate in report.gates:
            status = "PASS" if gate.passed else gate.severity
            symbol = "+" if gate.passed else "X" if gate.severity == "FAIL" else "~"
            print(f"  [{symbol}] {gate.gate_id} {gate.gate_name}: {status}")
            if not gate.passed:
                print(f"      {gate.message}")

        print()
        print("-" * 60)
        print(f"FAILS: {report.fail_count} | WARNS: {report.warn_count}")
        print(f"Overall: {'PASS' if report.all_passed else 'FAIL'}")

        return 0 if report.all_passed else 1

    @property
    def signal_count(self) -> int:
        """Get the count of signals received."""
        return self._processor.signal_count if self._processor else 0


async def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate: at least one mode must be specified
    if not args.live and not args.backfill and not args.train_models and not args.retrain_models and not args.validate_bars:
        parser.error(
            "At least one mode required: --live, --backfill, --validate-bars, --train-models, or --retrain-models"
        )

    # Validate: --deploy requires --format package
    if args.deploy and args.format != "package":
        parser.error("--deploy requires --format package")

    # Validate: --validate requires --format package
    if args.validate and args.format != "package":
        parser.error("--validate requires --format package")

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse configuration
    config = parse_config(args)

    runner = SignalRunner(config)
    exit_code = await runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
