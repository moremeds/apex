"""
Behavioral gate validation runner.

Orchestrates DualMACD behavioral gate validation:
- Single run (make behavioral): default params, all symbols, HTML report
- Full optimization (make behavioral-full): Optuna grid search across
  slope_lookback × hist_norm_window, per-symbol reports + optimization summary
- Case studies (make behavioral-cases): predefined market episodes
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.backtest.analysis.behavioral_models import GatePolicy

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("out/behavioral")
UNIVERSE_PATH = Path("config/universe.yaml")
BEHAVIORAL_SPEC_PATH = Path("config/backtest/dual_macd_behavioral.yaml")


def _resolve_symbols(args: argparse.Namespace, subset: str = "quick_test") -> list[str]:
    """Resolve symbols from CLI args or universe.yaml subset."""
    if getattr(args, "symbols", None):
        return [s.strip() for s in args.symbols.split(",")]

    try:
        from src.domain.services.regime.universe_loader import load_universe

        universe = load_universe(UNIVERSE_PATH)
        if subset == "all":
            symbols = universe.all_symbols
            logger.info(f"Loaded {len(symbols)} symbols from universe.yaml (all)")
        else:
            symbols = getattr(universe, subset, universe.quick_test)
            logger.info(f"Loaded {len(symbols)} symbols from universe.yaml {subset} subset")
        return symbols
    except Exception as e:
        logger.warning(f"Failed to load universe.yaml: {e}, falling back to SPY")
        return ["SPY"]


def _load_spec_config() -> Dict[str, Any]:
    """Load behavioral YAML spec for gate policies and universe subset."""
    if BEHAVIORAL_SPEC_PATH.exists():
        return yaml.safe_load(BEHAVIORAL_SPEC_PATH.read_text()) or {}
    return {}


def _load_gate_policies(spec: Dict[str, Any]) -> Dict[str, GatePolicy]:
    """Parse gate_policies from YAML spec into GatePolicy objects."""
    raw = spec.get("gate_policies", {})
    policies: Dict[str, GatePolicy] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            policies[key] = GatePolicy(
                action_on_block=value.get("action_on_block", "BLOCK"),
                size_factor=value.get("size_factor", 0.5),
            )
        else:
            policies[key] = GatePolicy(action_on_block="BLOCK")
    return policies


def _load_symbol_to_sector() -> Dict[str, str]:
    """Build symbol → sector name mapping from universe.yaml."""
    try:
        from src.domain.services.regime.universe_loader import load_universe

        universe = load_universe(UNIVERSE_PATH)
        mapping: Dict[str, str] = {}
        for sector_name, sector_config in universe.sectors.items():
            for stock in sector_config.stocks:
                mapping[stock.upper()] = sector_name
            mapping[sector_config.etf.upper()] = sector_name
        return mapping
    except Exception as e:
        logger.warning(f"Failed to load sector mapping: {e}")
        return {}


async def run_behavioral_validation(args: argparse.Namespace) -> None:
    """Main entry point for behavioral validation from CLI."""
    # Case studies mode
    if getattr(args, "behavioral_cases", False):
        slope_lookback = getattr(args, "slope_lookback", 3)
        hist_norm_window = getattr(args, "hist_norm_window", 252)
        logger.info("Running predefined behavioral case studies...")
        spec = _load_spec_config()
        gate_policies = _load_gate_policies(spec)
        symbol_to_sector = _load_symbol_to_sector()
        await _run_case_studies(
            args,
            slope_lookback,
            hist_norm_window,
            gate_policies=gate_policies,
            symbol_to_sector=symbol_to_sector,
        )
        return

    # Full optimization mode (--spec provided)
    spec_path = getattr(args, "spec", None)
    if spec_path:
        logger.info(f"Running full optimization from {spec_path}...")
        await _run_optimization(args)
        return

    # Single run mode (default params)
    slope_lookback = getattr(args, "slope_lookback", 3)
    hist_norm_window = getattr(args, "hist_norm_window", 252)
    await _run_single(args, slope_lookback, hist_norm_window)


# ── Single run (default params) ──────────────────────────────


async def _run_single(
    args: argparse.Namespace,
    slope_lookback: int,
    hist_norm_window: int,
) -> None:
    """Run behavioral gate with fixed params across all symbols."""
    from src.domain.strategy.signals.dual_macd_gate import DualMACDGateSignalGenerator
    from src.domain.strategy.signals.ma_cross import MACrossSignalGenerator

    from .analysis.behavioral_metrics import BehavioralMetricsCalculator
    from .analysis.behavioral_report import (
        SymbolResult,
        generate_behavioral_report,
        generate_summary_report,
    )
    from .runner import prefetch_data

    spec = _load_spec_config()
    gate_policies = _load_gate_policies(spec)
    symbol_to_sector = _load_symbol_to_sector()

    universe_subset = spec.get("universe", {}).get("subset", "quick_test")
    symbols = _resolve_symbols(args, subset=universe_subset)
    start_date = getattr(args, "start", None) or date(2018, 1, 1)
    end_date = getattr(args, "end", None) or date(2025, 12, 31)

    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    logger.info(
        f"Behavioral validation: {symbols} ({start_date} to {end_date}), "
        f"slope_lookback={slope_lookback}, hist_norm_window={hist_norm_window}"
    )

    cached_data = await prefetch_data(
        symbols=symbols, start_date=start_date, end_date=end_date, timeframe="1d"
    )

    base_generator = MACrossSignalGenerator()
    base_params = {"short_window": 10, "long_window": 50}
    calculator = BehavioralMetricsCalculator()
    symbol_results: list[SymbolResult] = []

    for symbol in symbols:
        data = cached_data.get(symbol)
        if data is None or data.empty:
            logger.warning(f"No data for {symbol}, skipping")
            continue

        metrics, decision_logger, states = _run_gate_for_symbol(
            data,
            symbol,
            base_generator,
            base_params,
            calculator,
            slope_lookback,
            hist_norm_window,
            gate_policies=gate_policies,
            symbol_to_sector=symbol_to_sector,
        )

        warmup_end = _make_gate(
            base_generator,
            slope_lookback,
            hist_norm_window,
            gate_policies=gate_policies,
            symbol_to_sector=symbol_to_sector,
        ).get_warmup_end_date(data)
        post_warmup = decision_logger.get_post_warmup(warmup_end)
        symbol_results.append(SymbolResult(symbol=symbol, metrics=metrics, decisions=post_warmup))

        _write_symbol_report(
            symbol,
            start_date,
            end_date,
            data,
            base_generator,
            base_params,
            metrics,
            decision_logger,
            states,
            slope_lookback,
            hist_norm_window,
            gate_policies=gate_policies,
            symbol_to_sector=symbol_to_sector,
        )
        _print_symbol_summary(symbol, metrics)

    # Generate cross-symbol summary report
    if symbol_results:
        summary_path = OUTPUT_DIR / "index.html"
        generate_summary_report(
            results=symbol_results,
            start_date=start_date,
            end_date=end_date,
            params={"slope_lookback": slope_lookback, "hist_norm_window": hist_norm_window},
            symbol_to_sector=symbol_to_sector,
            gate_policies=gate_policies,
            output_path=summary_path,
        )
        print(f"\nSummary report: {summary_path}")


# ── Full optimization (Optuna) ───────────────────────────────


async def _run_optimization(args: argparse.Namespace) -> None:
    """Run Optuna grid search over slope_lookback × hist_norm_window."""
    import optuna

    from src.domain.strategy.signals.dual_macd_gate import DualMACDGateSignalGenerator
    from src.domain.strategy.signals.ma_cross import MACrossSignalGenerator

    from .analysis.behavioral_metrics import BehavioralMetricsCalculator
    from .analysis.behavioral_models import BehavioralMetrics
    from .analysis.behavioral_report import (
        SymbolResult,
        generate_behavioral_report,
        generate_summary_report,
    )
    from .optimization.behavioral_objective import BehavioralObjective
    from .runner import prefetch_data

    spec = _load_spec_config()
    gate_policies = _load_gate_policies(spec)
    symbol_to_sector = _load_symbol_to_sector()

    universe_subset = spec.get("universe", {}).get("subset", "quick_test")
    symbols = _resolve_symbols(args, subset=universe_subset)
    start_date = getattr(args, "start", None) or date(2018, 1, 1)
    end_date = getattr(args, "end", None) or date(2025, 12, 31)

    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    logger.info(f"Optimization: {symbols} ({start_date} to {end_date})")

    cached_data = await prefetch_data(
        symbols=symbols, start_date=start_date, end_date=end_date, timeframe="1d"
    )

    base_generator = MACrossSignalGenerator()
    base_params = {"short_window": 10, "long_window": 50}
    calculator = BehavioralMetricsCalculator()

    # Aggregate metrics across all symbols for a given param combo
    def run_fn(slope_lookback: int, hist_norm_window: int) -> BehavioralMetrics:
        all_decisions: list = []
        for symbol in symbols:
            data = cached_data.get(symbol)
            if data is None or data.empty:
                continue
            m, dl, _ = _run_gate_for_symbol(
                data,
                symbol,
                base_generator,
                base_params,
                calculator,
                slope_lookback,
                hist_norm_window,
                gate_policies=gate_policies,
                symbol_to_sector=symbol_to_sector,
            )
            warmup_end = DualMACDGateSignalGenerator(
                base_generator,
                slope_lookback,
                hist_norm_window,
                gate_policies=gate_policies,
                symbol_to_sector=symbol_to_sector,
            ).get_warmup_end_date(data)
            all_decisions.extend(dl.get_post_warmup(warmup_end))

        return calculator.calculate(all_decisions)

    # Run Optuna
    objective = BehavioralObjective(run_fn)
    study = optuna.create_study(
        direction="maximize",
        study_name="dual_macd_behavioral",
    )
    study.optimize(objective, n_trials=27, show_progress_bar=True)

    # Print optimization results
    print(f"\n{'='*60}")
    print("Optimization Results")
    print(f"{'='*60}")

    completed = [t for t in study.trials if t.value is not None]
    pruned = [t for t in study.trials if t.value is None]
    print(f"Completed: {len(completed)}, Pruned: {len(pruned)}")

    print(
        f"\n{'slope_lb':>10} {'hist_nw':>10} {'score':>8} {'loss_r':>8} {'allow_r':>8} {'blk_pnl':>8}"
    )
    for trial in study.trials:
        if trial.value is not None:
            ua = trial.user_attrs
            print(
                f"{trial.params.get('slope_lookback', '?'):>10} "
                f"{trial.params.get('hist_norm_window', '?'):>10} "
                f"{trial.value:>8.4f} "
                f"{ua.get('blocked_loss_ratio', 0):>8.2f} "
                f"{ua.get('allowed_count', 0) / max(ua.get('baseline_count', 1), 1):>8.2f} "
                f"{ua.get('blocked_avg_pnl', 0):>+8.2%}"
            )
        else:
            print(
                f"{trial.params.get('slope_lookback', '?'):>10} "
                f"{trial.params.get('hist_norm_window', '?'):>10} "
                f"{'PRUNED':>8}"
            )

    # Determine best params (fall back to defaults if all pruned)
    if completed:
        best_sl = study.best_params["slope_lookback"]
        best_hnw = study.best_params["hist_norm_window"]
        print(f"\nBest score: {study.best_value:.4f}")
        print(f"Best params: slope_lookback={best_sl}, hist_norm_window={best_hnw}")
    else:
        best_sl = 3
        best_hnw = 252
        print(
            "\nWARNING: All trials pruned (constraints too strict). "
            f"Falling back to defaults: slope_lookback={best_sl}, hist_norm_window={best_hnw}"
        )

    print(
        f"\nGenerating reports with params: slope_lookback={best_sl}, hist_norm_window={best_hnw}"
    )

    # Build heatmap data for the report
    heatmap = _build_heatmap(study)
    symbol_results: list[SymbolResult] = []

    for symbol in symbols:
        data = cached_data.get(symbol)
        if data is None or data.empty:
            continue

        metrics, decision_logger, states = _run_gate_for_symbol(
            data,
            symbol,
            base_generator,
            base_params,
            calculator,
            best_sl,
            best_hnw,
            gate_policies=gate_policies,
            symbol_to_sector=symbol_to_sector,
        )

        warmup_end = _make_gate(
            base_generator,
            best_sl,
            best_hnw,
            gate_policies=gate_policies,
            symbol_to_sector=symbol_to_sector,
        ).get_warmup_end_date(data)
        post_warmup = decision_logger.get_post_warmup(warmup_end)
        symbol_results.append(SymbolResult(symbol=symbol, metrics=metrics, decisions=post_warmup))

        _write_symbol_report(
            symbol,
            start_date,
            end_date,
            data,
            base_generator,
            base_params,
            metrics,
            decision_logger,
            states,
            best_sl,
            best_hnw,
            optimization_results={"heatmap": heatmap},
            gate_policies=gate_policies,
            symbol_to_sector=symbol_to_sector,
        )
        _print_symbol_summary(symbol, metrics)

    # Generate cross-symbol summary report
    if symbol_results:
        summary_path = OUTPUT_DIR / "index.html"
        generate_summary_report(
            results=symbol_results,
            start_date=start_date,
            end_date=end_date,
            params={"slope_lookback": best_sl, "hist_norm_window": best_hnw},
            optimization_results={"heatmap": heatmap},
            symbol_to_sector=symbol_to_sector,
            gate_policies=gate_policies,
            output_path=summary_path,
        )
        print(f"\nSummary report: {summary_path}")


def _build_heatmap(study: Any) -> Dict[str, Any]:
    """Build heatmap data from Optuna study for the HTML report."""
    slope_values = [2, 3, 5]
    hist_values = [126, 252, 504]
    z: list[list[float]] = []

    for sl in slope_values:
        row: list[float] = []
        for hnw in hist_values:
            # Find matching trial
            val = 0.0
            for trial in study.trials:
                if (
                    trial.value is not None
                    and trial.params.get("slope_lookback") == sl
                    and trial.params.get("hist_norm_window") == hnw
                ):
                    val = trial.value
                    break
            row.append(val)
        z.append(row)

    return {
        "x": [str(v) for v in hist_values],
        "y": [str(v) for v in slope_values],
        "z": z,
    }


# ── Case studies ─────────────────────────────────────────────


async def _run_case_studies(
    args: argparse.Namespace,
    slope_lookback: int,
    hist_norm_window: int,
    gate_policies: Optional[Dict[str, Any]] = None,
    symbol_to_sector: Optional[Dict[str, str]] = None,
) -> None:
    """Run predefined case studies."""
    from src.domain.strategy.signals.ma_cross import MACrossSignalGenerator

    from .analysis.behavioral_report import generate_behavioral_report
    from .analysis.case_study import PREDEFINED_CASES, CaseStudyRunner
    from .runner import prefetch_data

    base_generator = MACrossSignalGenerator()
    base_params = {"short_window": 10, "long_window": 50}

    runner = CaseStudyRunner(
        base_generator=base_generator,
        slope_lookback=slope_lookback,
        hist_norm_window=hist_norm_window,
        direction="LONG",
        gate_policies=gate_policies,
        symbol_to_sector=symbol_to_sector,
    )

    all_symbols = list({c["symbol"] for c in PREDEFINED_CASES})
    min_start = min(c["start"] for c in PREDEFINED_CASES)
    max_end = max(c["end"] for c in PREDEFINED_CASES)

    cached_data = await prefetch_data(
        symbols=all_symbols, start_date=min_start, end_date=max_end, timeframe="1d"
    )

    def load_data(symbol: str, start: date, end: date) -> pd.DataFrame:
        df = cached_data.get(symbol, pd.DataFrame())
        if df.empty:
            return df
        ts_start = pd.Timestamp(start, tz=df.index.tz)
        ts_end = pd.Timestamp(end, tz=df.index.tz)
        mask = (df.index >= ts_start) & (df.index <= ts_end)
        return df.loc[mask]

    results = runner.run_all(data_loader=load_data, base_params=base_params)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for result in results:
        safe_name = result.name.replace(" ", "_").replace("/", "-")
        output_path = OUTPUT_DIR / f"case_{safe_name}.html"

        generate_behavioral_report(
            symbol=result.config.symbol,
            start_date=result.config.start_date,
            end_date=result.config.end_date,
            close_prices=result.close_prices,
            baseline_entries=result.baseline_entries,
            gated_entries=result.gated_entries,
            decisions=result.decisions.decisions,
            metrics=result.metrics,
            output_path=output_path,
            params={"slope_lookback": slope_lookback, "hist_norm_window": hist_norm_window},
        )

    runner.export_results(results, OUTPUT_DIR / "cases")

    print(f"\n{'='*60}")
    print("Case Study Results")
    print(f"{'='*60}")
    for r in results:
        print(f"\n{r.name}:")
        print(
            f"  Blocked: {r.metrics.blocked_trade_count}, "
            f"Allowed: {r.metrics.allowed_trade_count}, "
            f"Loss ratio: {r.metrics.blocked_trade_loss_ratio:.2%}"
        )
    print(f"\nReports: {OUTPUT_DIR}/")


# ── Shared helpers ───────────────────────────────────────────


def _run_gate_for_symbol(
    data: pd.DataFrame,
    symbol: str,
    base_generator: Any,
    base_params: Dict[str, Any],
    calculator: Any,
    slope_lookback: int,
    hist_norm_window: int,
    gate_policies: Optional[Dict[str, Any]] = None,
    symbol_to_sector: Optional[Dict[str, str]] = None,
) -> tuple:
    """Run gate on a single symbol, return (metrics, decision_logger, macd_states)."""
    from src.domain.signals.indicators.momentum.dual_macd import DualMACDIndicator

    from .analysis.trade_decision_logger import TradeDecisionLogger

    gate = _make_gate(
        base_generator,
        slope_lookback,
        hist_norm_window,
        gate_policies=gate_policies,
        symbol_to_sector=symbol_to_sector,
    )
    params_with_symbol = {**base_params, "symbol": symbol}

    _, _, decision_logger = gate.generate_with_decisions(data, params_with_symbol)

    warmup_end = gate.get_warmup_end_date(data)
    post_warmup = decision_logger.get_post_warmup(warmup_end)

    metrics = calculator.calculate(decisions=post_warmup)

    # Compute MACD states for timeline chart
    macd_indicator = DualMACDIndicator()
    macd_params = {
        "slow_fast": 55,
        "slow_slow": 89,
        "slow_signal": 34,
        "fast_fast": 13,
        "fast_slow": 21,
        "fast_signal": 9,
        "slope_lookback": slope_lookback,
        "hist_norm_window": hist_norm_window,
        "histogram_multiplier": 2.0,
        "eps": 1e-3,
    }
    macd_result = macd_indicator.calculate(data[["close"]], macd_params)
    states = []
    for i in range(len(macd_result)):
        current = macd_result.iloc[i]
        previous = macd_result.iloc[i - 1] if i > 0 else None
        states.append(macd_indicator._get_state(current, previous, macd_params))

    return metrics, decision_logger, states


def _make_gate(
    base_generator: Any,
    slope_lookback: int,
    hist_norm_window: int,
    gate_policies: Optional[Dict[str, Any]] = None,
    symbol_to_sector: Optional[Dict[str, str]] = None,
) -> Any:
    from src.domain.strategy.signals.dual_macd_gate import DualMACDGateSignalGenerator

    return DualMACDGateSignalGenerator(
        base_generator=base_generator,
        slope_lookback=slope_lookback,
        hist_norm_window=hist_norm_window,
        direction="LONG",
        gate_policies=gate_policies,
        symbol_to_sector=symbol_to_sector,
    )


def _write_symbol_report(
    symbol: str,
    start_date: date,
    end_date: date,
    data: pd.DataFrame,
    base_generator: Any,
    base_params: Dict[str, Any],
    metrics: Any,
    decision_logger: Any,
    states: list,
    slope_lookback: int,
    hist_norm_window: int,
    optimization_results: Optional[Dict[str, Any]] = None,
    gate_policies: Optional[Dict[str, Any]] = None,
    symbol_to_sector: Optional[Dict[str, str]] = None,
) -> None:
    """Write HTML report + JSONL for a single symbol."""
    from .analysis.behavioral_report import generate_behavioral_report

    params_with_symbol = {**base_params, "symbol": symbol}
    baseline_entries, _ = base_generator.generate(data, params_with_symbol)

    gate = _make_gate(
        base_generator,
        slope_lookback,
        hist_norm_window,
        gate_policies=gate_policies,
        symbol_to_sector=symbol_to_sector,
    )
    gated_entries, _, _ = gate.generate_with_decisions(data, params_with_symbol)

    warmup_end = gate.get_warmup_end_date(data)
    post_warmup = decision_logger.get_post_warmup(warmup_end)

    output_path = OUTPUT_DIR / f"{symbol}_{start_date}_{end_date}.html"
    generate_behavioral_report(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        close_prices=data["close"],
        baseline_entries=baseline_entries,
        gated_entries=gated_entries,
        decisions=post_warmup,
        metrics=metrics,
        macd_states=states,
        output_path=output_path,
        params={"slope_lookback": slope_lookback, "hist_norm_window": hist_norm_window},
        optimization_results=optimization_results,
    )

    jsonl_path = OUTPUT_DIR / f"{symbol}_{start_date}_{end_date}.jsonl"
    decision_logger.to_jsonl(jsonl_path)


def _print_symbol_summary(symbol: str, metrics: Any) -> None:
    print(f"\n{'='*60}")
    print(f"Behavioral Gate Results: {symbol}")
    print(f"{'='*60}")
    print(f"Baseline trades:         {metrics.baseline_trade_count}")
    print(f"Allowed trades:          {metrics.allowed_trade_count}")
    print(f"Blocked trades:          {metrics.blocked_trade_count}")
    print(f"Blocked loss ratio:      {metrics.blocked_trade_loss_ratio:.2%}")
    print(f"Blocked avg PnL:         {metrics.blocked_trade_avg_pnl:+.2%}")
    print(f"Trade freedom:           {metrics.allowed_trade_ratio:.2%}")
    print(f"Size-down actions:       {metrics.size_down_count}")
    print(f"Bypass actions:          {metrics.bypass_count}")
