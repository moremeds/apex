#!/usr/bin/env python
"""
TrendPulse v2.2 — Full Universe Backtest with HTML Report.

Runs TrendPulse strategy across every symbol in config/universe.yaml
and generates a self-contained HTML report.

Usage:
    python scripts/trend_pulse_universe.py [--start 2020-01-01] [--end 2025-12-30]
    python scripts/trend_pulse_universe.py --subset quick_test
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

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


def load_universe(universe_path: str, subset: Optional[str] = None) -> Dict[str, List[str]]:
    """Load symbols from universe.yaml, grouped by sector.

    Returns dict of {sector_name: [symbols]}.
    """
    with open(universe_path) as f:
        cfg = yaml.safe_load(f)

    if subset and "subsets" in cfg and subset in cfg["subsets"]:
        symbols = cfg["subsets"][subset]
        return {"subset": symbols}

    grouped: Dict[str, List[str]] = {}

    # Market ETFs
    if "market" in cfg:
        grouped["Market ETFs"] = [m["symbol"] for m in cfg["market"]]

    # Sectors
    if "sectors" in cfg:
        for sector_name, sector_data in cfg["sectors"].items():
            syms = []
            if "etf" in sector_data:
                syms.append(sector_data["etf"])
            if "stocks" in sector_data:
                syms.extend(sector_data["stocks"])
            grouped[sector_name] = syms

    # Speculative
    if "speculative" in cfg:
        spec = cfg["speculative"]
        syms = []
        if "etf" in spec and spec["etf"] != "MEME":
            syms.append(spec["etf"])
        if "stocks" in spec:
            syms.extend(spec["stocks"])
        grouped["speculative"] = syms

    return grouped


def run_backtest(
    symbols: List[str],
    start: date,
    end: date,
    label: str,
) -> Dict[str, Tuple[float, float, float, int, float, float, dict]]:
    """Run TrendPulse on a list of symbols, return per-symbol metrics."""
    from src.backtest.core import RunSpec
    from src.backtest.core.run import TimeWindow
    from src.backtest.execution.engines.vectorbt_engine import (
        VectorBTConfig,
        VectorBTEngine,
    )

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
    total = len(symbols)

    for i, sym in enumerate(symbols, 1):
        try:
            log.info(f"  [{i}/{total}] {sym}...")
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
            results[sym] = (
                m.sharpe,
                m.total_return,
                m.max_drawdown,
                m.total_trades,
                0.0,  # avg_hold placeholder
                m.profit_factor,
                {},  # exit reasons placeholder
            )
        except Exception as e:
            log.warning(f"  [{i}/{total}] {sym}: FAILED — {e}")
            results[sym] = (0.0, 0.0, 0.0, 0, 0.0, 0.0, {})

    return results


def generate_html_report(
    grouped: Dict[str, List[str]],
    results: Dict[str, Tuple[float, float, float, int, float, float, dict]],
    start: date,
    end: date,
    output_path: Path,
) -> None:
    """Generate a self-contained HTML report."""
    # Build sector summary rows
    sector_rows = []
    all_sharpes = []
    all_returns = []

    for sector, syms in grouped.items():
        sector_sharpes = []
        sector_returns = []
        sym_rows = []
        for sym in syms:
            if sym not in results:
                continue
            sharpe, ret, dd, trades, hold, pf, exits = results[sym]
            all_sharpes.append(sharpe)
            all_returns.append(ret)
            sector_sharpes.append(sharpe)
            sector_returns.append(ret)
            color = "#22c55e" if ret >= 0 else "#ef4444"
            sharpe_color = "#22c55e" if sharpe > 0 else "#ef4444"
            sym_rows.append(f"""<tr>
                <td style="padding-left:2em">{sym}</td>
                <td style="color:{sharpe_color}">{sharpe:.2f}</td>
                <td style="color:{color}">{ret*100:.1f}%</td>
                <td>{dd*100:.1f}%</td>
                <td>{trades}</td>
                <td>{pf:.2f}</td>
            </tr>""")
        avg_sh = np.mean(sector_sharpes) if sector_sharpes else 0.0
        avg_ret = np.mean(sector_returns) if sector_returns else 0.0
        pos = sum(1 for s in sector_sharpes if s > 0)
        n = len(sector_sharpes)
        sector_color = "#22c55e" if avg_sh > 0 else "#ef4444"
        sector_rows.append(f"""<tr style="background:#f1f5f9;font-weight:600">
            <td>{sector} ({pos}/{n} positive)</td>
            <td style="color:{sector_color}">{avg_sh:.2f}</td>
            <td>{avg_ret*100:.1f}%</td>
            <td colspan="3"></td>
        </tr>""")
        sector_rows.extend(sym_rows)

    overall_sharpe = np.mean(all_sharpes) if all_sharpes else 0.0
    overall_ret = np.mean(all_returns) if all_returns else 0.0
    pos_total = sum(1 for s in all_sharpes if s > 0)
    total_syms = len(all_sharpes)

    # Build distribution data for chart
    sharpe_bins: Dict[float, int] = {}
    for s in all_sharpes:
        bucket = round(s * 2) / 2  # round to nearest 0.5
        sharpe_bins[bucket] = sharpe_bins.get(bucket, 0) + 1
    sorted_bins = sorted(sharpe_bins.items())
    chart_labels = [str(b) for b, _ in sorted_bins]
    chart_values = [c for _, c in sorted_bins]
    chart_colors = ["#22c55e" if b >= 0 else "#ef4444" for b, _ in sorted_bins]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TrendPulse v2.2 — Universe Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f8fafc; color: #1e293b; padding: 2em; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 1.8em; margin-bottom: 0.3em; }}
.meta {{ color: #64748b; margin-bottom: 2em; }}
.card {{ background: white; border-radius: 12px; padding: 1.5em; margin-bottom: 1.5em;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 1em; margin-bottom: 2em; }}
.kpi {{ background: white; border-radius: 12px; padding: 1.2em; text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.kpi .value {{ font-size: 2em; font-weight: 700; }}
.kpi .label {{ color: #64748b; font-size: 0.85em; margin-top: 0.3em; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
th, td {{ padding: 0.5em 0.8em; text-align: left; border-bottom: 1px solid #e2e8f0; }}
th {{ background: #f1f5f9; font-weight: 600; position: sticky; top: 0; }}
.green {{ color: #22c55e; }} .red {{ color: #ef4444; }}
</style>
</head>
<body>
<div class="container">
    <h1>TrendPulse v2.2 — Universe Backtest</h1>
    <div class="meta">{start} to {end} &bull; {total_syms} symbols &bull;
    v2.2 frozen params</div>

    <div class="kpi-grid">
        <div class="kpi">
            <div class="value" style="color:{'#22c55e' if overall_sharpe > 0 else '#ef4444'}">
                {overall_sharpe:.2f}</div>
            <div class="label">Avg Sharpe</div>
        </div>
        <div class="kpi">
            <div class="value">{overall_ret*100:.1f}%</div>
            <div class="label">Avg Return</div>
        </div>
        <div class="kpi">
            <div class="value">{pos_total}/{total_syms}</div>
            <div class="label">Positive Sharpe</div>
        </div>
        <div class="kpi">
            <div class="value">{pos_total/total_syms*100:.0f}%</div>
            <div class="label">Hit Rate</div>
        </div>
    </div>

    <div class="card">
        <h2 style="margin-bottom:1em">Sharpe Distribution</h2>
        <div id="sharpe-dist" style="height:300px"></div>
    </div>

    <div class="card" style="max-height:80vh;overflow:auto">
        <h2 style="margin-bottom:1em">Per-Symbol Results</h2>
        <table>
            <thead>
                <tr><th>Symbol</th><th>Sharpe</th><th>Return</th>
                    <th>MaxDD</th><th>Trades</th><th>PF</th></tr>
            </thead>
            <tbody>
                {"".join(sector_rows)}
                <tr style="background:#1e293b;color:white;font-weight:700">
                    <td>OVERALL ({pos_total}/{total_syms})</td>
                    <td>{overall_sharpe:.2f}</td>
                    <td>{overall_ret*100:.1f}%</td>
                    <td></td><td></td><td></td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2 style="margin-bottom:0.5em">Parameters (v2.2 frozen)</h2>
        <pre style="font-size:0.85em;color:#475569">{yaml.dump(PARAMS, default_flow_style=False)}</pre>
    </div>
</div>

<script>
Plotly.newPlot('sharpe-dist', [{{
    x: {chart_labels},
    y: {chart_values},
    type: 'bar',
    marker: {{ color: {chart_colors} }}
}}], {{
    xaxis: {{ title: 'Sharpe Ratio (binned 0.5)' }},
    yaxis: {{ title: 'Count' }},
    margin: {{ t: 20, b: 50 }}
}}, {{ responsive: true }});
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    log.info(f"\nReport saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrendPulse v2.2 full universe backtest")
    parser.add_argument(
        "--universe",
        default="config/universe.yaml",
        help="Universe config file (default: config/universe.yaml)",
    )
    parser.add_argument("--subset", default=None, help="Use a named subset")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-30", help="End date")
    parser.add_argument(
        "--output",
        default="out/trend_pulse/universe_report.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    log.info("TrendPulse v2.2 — Universe Backtest")
    log.info(f"  Universe: {args.universe}")
    log.info(f"  Period: {start} → {end}")

    grouped = load_universe(args.universe, args.subset)
    # Deduplicate while preserving sector grouping
    seen: set = set()
    deduped: Dict[str, List[str]] = {}
    for sector, syms in grouped.items():
        unique = [s for s in syms if s not in seen]
        seen.update(unique)
        if unique:
            deduped[sector] = unique

    all_symbols = [s for syms in deduped.values() for s in syms]
    log.info(f"  Symbols: {len(all_symbols)} (deduplicated)")

    if args.subset:
        log.info(f"  Subset: {args.subset}")

    results = run_backtest(all_symbols, start, end, "universe")

    # Print summary to console
    log.info(f"\n{'='*60}")
    sharpes = [results[s][0] for s in all_symbols if s in results]
    pos = sum(1 for s in sharpes if s > 0)
    log.info(
        f"OVERALL: AvgSharpe={np.mean(sharpes):.2f}  "
        f"{pos}/{len(sharpes)} positive ({pos/len(sharpes)*100:.0f}%)"
    )

    generate_html_report(deduped, results, start, end, Path(args.output))
    log.info(f"\nOpen: file://{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
