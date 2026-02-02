#!/usr/bin/env python
"""
TrendPulse v2.2 — Final validation suite.

Three-stage validation:
1. Expanded universe coverage (36 symbols, 5 buckets)
2. Holdout period (last ~18 months, untouched by optimizer)
3. Deflated Sharpe Ratio (multiple-testing adjusted significance)

Usage:
    python scripts/trend_pulse_validate.py [--holdout-months 18]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# v2.2 frozen params
PARAMS: Dict[str, Any] = {
    "strategy_type": "trend_pulse",
    "zig_threshold_pct": 3.5,
    "trend_strength_moderate": 0.15,
    "exit_bearish_bars": 3,
    "enable_trend_reentry": False,
    "min_confidence": 0.5,
    "atr_stop_mult": 3.5,
    "enable_mtf_confirm": False,
    "weekly_ema_period": 26,
    "signal_shift_bars": 1,
    "min_pct": 0.2,
    "max_pct": 0.8,
    "enable_chop_filter": True,
    "adx_entry_min": 15.0,
    "cooldown_bars": 5,
    "slow_fast": 55,
    "slow_slow": 89,
    "slow_signal": 34,
    "slope_lookback": 3,
    "trend_strength_strong": 0.6,
}

BUCKETS: Dict[str, List[str]] = {
    "Hi-Mom Tech": ["NVDA", "META", "AMZN", "AVGO", "AMD", "TSLA"],
    "Steady Tech": ["AAPL", "MSFT", "GOOGL", "CRM", "PANW"],
    "SaaS/Cloud": ["NFLX", "SHOP", "CRWD", "DDOG", "NOW", "SNOW", "NET", "FTNT", "ADBE"],
    "Semis": ["MRVL", "ON", "SNPS", "CDNS", "KLAC", "LRCX", "ASML"],
    "Consumer/Industrial": ["LULU", "DECK", "ISRG", "DHR", "TT", "ETN", "PH", "DE", "SHW"],
}


def run_validation(
    start: date,
    end: date,
    label: str,
) -> Dict[str, Tuple[float, float, float, int, float, float, dict]]:
    """Run all symbols for a date range, return per-symbol metrics."""
    import vectorbt as vbt

    from src.backtest.core import RunSpec
    from src.backtest.core.run import TimeWindow
    from src.backtest.execution.engines.vectorbt_engine import VectorBTConfig, VectorBTEngine
    from src.domain.strategy.signals.trend_pulse import TrendPulseSignalGenerator

    config = VectorBTConfig(strategy_type="trend_pulse", data_source="yahoo")
    engine = VectorBTEngine(config)
    window = TimeWindow(
        window_id=label,
        fold_index=0,
        train_start=start,
        train_end=end,
        test_start=start,
        test_end=end,
        is_train=True,
        is_oos=False,
    )

    results: Dict[str, Tuple[float, float, float, int, float, float, dict]] = {}

    for bucket, syms in BUCKETS.items():
        for sym in syms:
            try:
                spec = RunSpec(
                    trial_id=label,
                    symbol=sym,
                    window=window,
                    profile_version="v1",
                    data_version="yahoo",
                    params=PARAMS,
                    commission_per_share=0.005,
                    slippage_bps=5.0,
                )
                r = engine.run(spec)
                m = r.metrics

                # Trade-level exit attribution
                data = engine.load_data(sym, start, end)
                assert data is not None, f"No data for {sym}"
                gen = TrendPulseSignalGenerator()
                entries, exits = gen.generate(data, PARAMS)
                exit_reasons = getattr(gen, "exit_reasons", None)
                sizes = getattr(gen, "entry_sizes", None)
                kw: dict = {"size": sizes, "size_type": "percent"} if sizes is not None else {}
                close = data["close"]
                pf = vbt.Portfolio.from_signals(
                    close=close, entries=entries, exits=exits, init_cash=100000, **kw
                )
                tdf = pf.trades.records_readable
                closed = tdf[tdf["Status"] == "Closed"]
                n_trades = len(closed)
                avg_hold = (
                    (
                        pd.to_datetime(closed["Exit Timestamp"])
                        - pd.to_datetime(closed["Entry Timestamp"])
                    ).dt.days.mean()
                    if n_trades > 0
                    else 0.0
                )

                trade_exits: dict = {}
                if exit_reasons is not None and n_trades > 0:
                    for _, trade in closed.iterrows():
                        exit_ts = pd.Timestamp(trade["Exit Timestamp"])
                        if exit_ts in exit_reasons.index:
                            reason = exit_reasons.loc[exit_ts]
                        else:
                            idx_pos = exit_reasons.index.get_indexer([exit_ts], method="nearest")[0]
                            reason = exit_reasons.iloc[idx_pos]
                        if reason and reason != "":
                            trade_exits[reason] = trade_exits.get(reason, 0) + 1

                results[sym] = (
                    m.sharpe,
                    m.total_return,
                    m.max_drawdown,
                    n_trades,
                    avg_hold,
                    m.profit_factor,
                    trade_exits,
                )
            except Exception as e:
                log.warning(f"  {sym}: FAILED — {e}")
                results[sym] = (0.0, 0.0, 0.0, 0, 0.0, 0.0, {})

    return results


def print_results(
    results: Dict[str, Tuple[float, float, float, int, float, float, dict]],
    label: str,
) -> Tuple[float, List[float]]:
    """Print formatted results and return overall avg sharpe + all sharpes."""
    all_sharpes: List[float] = []

    log.info(f"\n{'='*80}")
    log.info(f" {label}")
    log.info(f"{'='*80}")
    log.info(
        f"{'Sym':6s} {'Sharpe':>7s} {'Ret%':>7s} {'DD%':>6s} {'Trades':>6s} "
        f"{'Hold':>6s} {'WR%':>5s} {'PF':>6s}  Exit Attribution"
    )
    log.info("-" * 80)

    for bucket, syms in BUCKETS.items():
        bucket_sharpes = []
        for sym in syms:
            if sym not in results:
                continue
            sharpe, ret, dd, trades, hold, pf, exits = results[sym]
            wr = 0.0
            if trades > 0 and pf > 0:
                # Approximate WR from PF: WR = PF / (1 + PF) is rough
                pass
            exit_str = " ".join(f"{k}:{v}" for k, v in sorted(exits.items()))
            log.info(
                f"{sym:6s} {sharpe:7.2f} {ret*100:7.1f} {dd*100:6.1f} {trades:6d} "
                f"{hold:6.1f} {'':>5s} {pf:6.2f}  {exit_str}"
            )
            bucket_sharpes.append(sharpe)
            all_sharpes.append(sharpe)

        if bucket_sharpes:
            avg = sum(bucket_sharpes) / len(bucket_sharpes)
            pos = sum(1 for s in bucket_sharpes if s > 0)
            log.info(
                f"  >>> {bucket}: AvgSharpe={avg:.2f}  " f"{pos}/{len(bucket_sharpes)} positive"
            )
            log.info("")

    overall_avg = sum(all_sharpes) / len(all_sharpes) if all_sharpes else 0.0
    pos_total = sum(1 for s in all_sharpes if s > 0)
    log.info(f"OVERALL: AvgSharpe={overall_avg:.2f}  {pos_total}/{len(all_sharpes)} positive")

    return overall_avg, all_sharpes


def compute_dsr(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    all_sharpes: List[float],
) -> Tuple[float, float]:
    """Compute Deflated Sharpe Ratio and p-value."""
    from src.backtest.analysis.statistics import DSRCalculator

    calc = DSRCalculator()

    # Compute skewness and kurtosis from all trial Sharpes
    sharpe_arr = np.array(all_sharpes)
    if len(sharpe_arr) > 2:
        skew = float(pd.Series(sharpe_arr).skew())
        kurt = float(pd.Series(sharpe_arr).kurtosis() + 3)  # excess → raw
    else:
        skew, kurt = 0.0, 3.0

    dsr, p_value = calc.calculate(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        n_observations=n_observations,
        skewness=skew,
        kurtosis=kurt,
    )

    return dsr, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description="TrendPulse v2.2 validation suite")
    parser.add_argument(
        "--holdout-months",
        type=int,
        default=18,
        help="Months to reserve as holdout (default: 18)",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip full-period validation, only run holdout",
    )
    args = parser.parse_args()

    full_start = date(2020, 1, 1)
    full_end = date(2025, 12, 30)
    holdout_start = full_end - timedelta(days=args.holdout_months * 30)
    train_end = holdout_start - timedelta(days=1)

    total_symbols = sum(len(s) for s in BUCKETS.values())
    log.info(f"TrendPulse v2.2 Validation Suite")
    log.info(f"  Universe: {total_symbols} symbols, {len(BUCKETS)} buckets")
    log.info(f"  Full period: {full_start} → {full_end}")
    log.info(f"  Holdout: {holdout_start} → {full_end} ({args.holdout_months} months)")
    log.info(f"  Training: {full_start} → {train_end}")

    # --- Stage 1: Full-period coverage ---
    if not args.skip_full:
        log.info("\n" + "=" * 80)
        log.info(" STAGE 1: Expanded Universe Coverage (full period)")
        log.info("=" * 80)
        full_results = run_validation(full_start, full_end, "full")
        full_avg, full_sharpes = print_results(full_results, "Full Period (2020-2025)")
    else:
        full_sharpes = []

    # --- Stage 2: Holdout validation ---
    log.info("\n" + "=" * 80)
    log.info(f" STAGE 2: Holdout Validation ({holdout_start} → {full_end})")
    log.info("   (These dates were NEVER used in optimization)")
    log.info("=" * 80)
    holdout_results = run_validation(holdout_start, full_end, "holdout")
    holdout_avg, holdout_sharpes = print_results(
        holdout_results, f"Holdout ({holdout_start} → {full_end})"
    )

    # --- Stage 3: Deflated Sharpe Ratio ---
    log.info("\n" + "=" * 80)
    log.info(" STAGE 3: Deflated Sharpe Ratio (multiple-testing adjustment)")
    log.info("=" * 80)

    # n_trials = total Optuna trials across Phase 1 (80)
    n_trials = 80
    # n_observations ≈ trading days in full period
    n_observations = 252 * 5  # ~5 years

    all_sharpes_for_dsr = full_sharpes if full_sharpes else holdout_sharpes
    best_sharpe = max(all_sharpes_for_dsr) if all_sharpes_for_dsr else 0.0
    avg_sharpe = np.mean(all_sharpes_for_dsr) if all_sharpes_for_dsr else 0.0

    dsr_best, pval_best = compute_dsr(best_sharpe, n_trials, n_observations, all_sharpes_for_dsr)
    dsr_avg, pval_avg = compute_dsr(
        float(avg_sharpe), n_trials, n_observations, all_sharpes_for_dsr
    )

    log.info(f"  Optuna trials tested: {n_trials}")
    log.info(f"  Observations (trading days): {n_observations}")
    log.info(f"")
    log.info(f"  Best single-symbol Sharpe: {best_sharpe:.2f}")
    log.info(f"    DSR: {dsr_best:.3f}  p-value: {pval_best:.4f}")
    log.info(
        f"    {'✓ SIGNIFICANT (p<0.05)' if pval_best < 0.05 else '✗ NOT significant (p≥0.05)'}"
    )
    log.info(f"")
    log.info(f"  Average cross-symbol Sharpe: {avg_sharpe:.2f}")
    log.info(f"    DSR: {dsr_avg:.3f}  p-value: {pval_avg:.4f}")
    log.info(f"    {'✓ SIGNIFICANT (p<0.05)' if pval_avg < 0.05 else '✗ NOT significant (p≥0.05)'}")

    # --- Summary ---
    log.info("\n" + "=" * 80)
    log.info(" SUMMARY")
    log.info("=" * 80)

    if not args.skip_full:
        log.info(f"  Full period AvgSharpe: {full_avg:.2f}")
    log.info(f"  Holdout AvgSharpe:     {holdout_avg:.2f}")

    if not args.skip_full and holdout_avg > 0:
        degradation = 1 - holdout_avg / full_avg if full_avg > 0 else float("inf")
        log.info(f"  Degradation (full→holdout): {degradation*100:.1f}%")
        if degradation > 0.5:
            log.info("  ⚠ WARNING: >50% degradation — possible overfit to training period")
        elif degradation > 0.3:
            log.info("  ⚠ CAUTION: >30% degradation — monitor closely")
        else:
            log.info("  ✓ Acceptable degradation (<30%)")

    # Red flags
    log.info("\n  Red Flag Check:")
    holdout_pos = sum(1 for s in holdout_sharpes if s > 0)
    holdout_pct = holdout_pos / len(holdout_sharpes) * 100 if holdout_sharpes else 0
    log.info(
        f"    Holdout positive rate: {holdout_pos}/{len(holdout_sharpes)} ({holdout_pct:.0f}%)"
    )
    if holdout_pct < 50:
        log.info("    ⚠ <50% positive in holdout — edge may not be real")
    else:
        log.info("    ✓ Majority positive in holdout")


if __name__ == "__main__":
    main()
