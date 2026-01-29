"""
CaseStudyRunner — runs predefined market episodes through the behavioral gate.

Each case study compares baseline (ungated) vs gated performance on a specific
market regime, generating trade decisions and behavioral metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.domain.strategy.signals.dual_macd_gate import DualMACDGateSignalGenerator

from .behavioral_metrics import BehavioralMetricsCalculator
from .behavioral_models import BehavioralMetrics, BehavioralRunConfig, GatePolicy
from .trade_decision_logger import TradeDecisionLogger

logger = logging.getLogger(__name__)


@dataclass
class CaseStudyResult:
    """Result of a single case study run."""

    name: str
    config: BehavioralRunConfig
    metrics: BehavioralMetrics
    decisions: TradeDecisionLogger
    baseline_entries: pd.Series
    gated_entries: pd.Series
    base_exits: pd.Series
    close_prices: pd.Series


# Predefined case studies
PREDEFINED_CASES: List[Dict[str, Any]] = [
    {
        "name": "SPY Trend Top (2021-10 to 2022-01)",
        "symbol": "SPY",
        "start": date(2021, 10, 1),
        "end": date(2022, 1, 31),
        "description": "Market top with deteriorating trend before 2022 selloff",
    },
    {
        "name": "SPY V-Rebound (2020-03 to 2020-07)",
        "symbol": "SPY",
        "start": date(2020, 3, 1),
        "end": date(2020, 7, 31),
        "description": "COVID crash and rapid V-shaped recovery",
    },
    {
        "name": "SPY False Breakout (2023-07 to 2023-11)",
        "symbol": "SPY",
        "start": date(2023, 7, 1),
        "end": date(2023, 11, 30),
        "description": "Summer rally followed by Q3 correction, then recovery",
    },
]


class CaseStudyRunner:
    """
    Runs predefined case studies comparing baseline vs gated strategies.

    Usage:
        runner = CaseStudyRunner(base_generator=MACrossSignalGenerator())
        results = runner.run_all(data_loader=load_fn)
    """

    def __init__(
        self,
        base_generator: Any,
        slope_lookback: int = 3,
        hist_norm_window: int = 252,
        direction: str = "LONG",
        gate_policies: Optional[Dict[str, GatePolicy]] = None,
        symbol_to_sector: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_generator = base_generator
        self.slope_lookback = slope_lookback
        self.hist_norm_window = hist_norm_window
        self.direction = direction
        self._gate_policies = gate_policies
        self._symbol_to_sector = symbol_to_sector
        self._calculator = BehavioralMetricsCalculator()

    def run_case(
        self,
        case: Dict[str, Any],
        data: pd.DataFrame,
        base_params: Dict[str, Any],
    ) -> CaseStudyResult:
        """
        Run a single case study.

        Args:
            case: Case definition dict with name, symbol, start, end
            data: OHLCV DataFrame covering the case period
            base_params: Parameters for the base strategy
        """
        symbol = case["symbol"]
        config = BehavioralRunConfig(
            symbol=symbol,
            start_date=case["start"],
            end_date=case["end"],
            slope_lookback=self.slope_lookback,
            hist_norm_window=self.hist_norm_window,
            direction=self.direction,
        )

        # Create gated generator
        gate = DualMACDGateSignalGenerator(
            base_generator=self.base_generator,
            slope_lookback=self.slope_lookback,
            hist_norm_window=self.hist_norm_window,
            direction=self.direction,
            gate_policies=self._gate_policies,
            symbol_to_sector=self._symbol_to_sector,
        )

        # Run baseline
        params_with_symbol = {**base_params, "symbol": symbol}
        baseline_entries, base_exits = self.base_generator.generate(data, params_with_symbol)

        # Run gated with decisions
        gated_entries, _, decision_logger = gate.generate_with_decisions(data, params_with_symbol)

        # Get warmup end date for G4
        warmup_end = gate.get_warmup_end_date(data)
        post_warmup_decisions = decision_logger.get_post_warmup(warmup_end)

        # Calculate metrics (using placeholder Sharpe/DD — caller provides real values)
        metrics = self._calculator.calculate(
            decisions=post_warmup_decisions,
            gated_sharpe=0.0,
            baseline_sharpe=0.0,
            gated_max_dd=0.0,
            baseline_max_dd=0.0,
        )

        return CaseStudyResult(
            name=case["name"],
            config=config,
            metrics=metrics,
            decisions=decision_logger,
            baseline_entries=baseline_entries,
            gated_entries=gated_entries,
            base_exits=base_exits,
            close_prices=data["close"],
        )

    def run_all(
        self,
        data_loader: Any,
        base_params: Dict[str, Any],
        cases: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CaseStudyResult]:
        """
        Run all predefined case studies.

        Args:
            data_loader: Callable(symbol, start, end) -> pd.DataFrame
            base_params: Parameters for the base strategy
            cases: Override cases list (default: PREDEFINED_CASES)
        """
        cases = cases or PREDEFINED_CASES
        results = []

        for case in cases:
            logger.info(f"Running case: {case['name']}")
            data = data_loader(case["symbol"], case["start"], case["end"])

            if data.empty:
                logger.warning(f"No data for case: {case['name']}, skipping")
                continue

            result = self.run_case(case, data, base_params)
            results.append(result)

            logger.info(
                f"  Blocked: {result.metrics.blocked_trade_count}, "
                f"Allowed: {result.metrics.allowed_trade_count}, "
                f"Loss ratio: {result.metrics.blocked_trade_loss_ratio:.2f}"
            )

        return results

    def export_results(self, results: List[CaseStudyResult], output_dir: Path) -> None:
        """Export all case study results to JSONL files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            safe_name = result.name.replace(" ", "_").replace("/", "-")
            path = output_dir / f"{safe_name}.jsonl"
            result.decisions.to_jsonl(path)
