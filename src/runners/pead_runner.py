"""PEAD Screener CLI runner.

Usage:
    python -m src.runners.pead_runner --update-earnings --universe config/universe.yaml
    python -m src.runners.pead_runner --screen --html-output out/signals/pead.html
    python -m src.runners.pead_runner --full --universe config/universe.yaml --html-output out/pead/pead.html
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.services.earnings_service import EarningsService
from src.services.market_cap_service import MarketCapService, load_universe_symbols
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_pead_config() -> dict[str, Any]:
    """Load PEAD screener config from YAML."""
    path = PROJECT_ROOT / "config" / "pead_screener.yaml"
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _read_current_regime(signals_dir: Path) -> str:
    """Read SPY regime from signal pipeline output."""
    summary_path = signals_dir / "data" / "summary.json"
    if not summary_path.exists():
        logger.warning(
            f"No summary.json at {summary_path}. "
            "Defaulting to R0 — R2 blocking will NOT activate. "
            "Ensure hourly-signals runs before PEAD, or fetch from gh-pages."
        )
        return "R0"

    try:
        data = json.loads(summary_path.read_text())
        for t in data.get("tickers", []):
            if t.get("symbol") == "SPY":
                regime = t.get("regime", "R0")
                logger.info(f"SPY regime: {regime}")
                return str(regime)
    except Exception as e:
        logger.warning(f"Failed to read regime from summary.json: {e}")

    return "R0"


def _write_candidates_json(result: Any, output_dir: Path) -> Path:
    """Write screening results to JSON for email renderer / HTML builder."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "pead_candidates.json"

    candidates_data = []
    for c in result.candidates:
        candidates_data.append(
            {
                "symbol": c.symbol,
                "report_date": c.surprise.report_date.isoformat(),
                "actual_eps": c.surprise.actual_eps,
                "consensus_eps": c.surprise.consensus_eps,
                "sue_score": round(c.surprise.sue_score, 2),
                "earnings_day_gap": round(c.surprise.earnings_day_gap, 4),
                "earnings_day_return": round(c.surprise.earnings_day_return, 4),
                "earnings_day_volume_ratio": round(c.surprise.earnings_day_volume_ratio, 2),
                "revenue_beat": c.surprise.revenue_beat,
                "liquidity_tier": c.surprise.liquidity_tier.value,
                "entry_date": c.entry_date.isoformat(),
                "entry_price": round(c.entry_price, 2),
                "profit_target_pct": c.profit_target_pct,
                "stop_loss_pct": c.stop_loss_pct,
                "trailing_stop_atr": c.trailing_stop_atr,
                "trailing_activation_pct": c.trailing_activation_pct,
                "max_hold_days": c.max_hold_days,
                "position_size_factor": c.position_size_factor,
                "quality_score": round(c.quality_score, 1),
                "quality_label": c.quality_label,
                "regime": c.regime,
                "gap_held": c.gap_held,
                "estimated_slippage_bps": c.estimated_slippage_bps,
            }
        )

    payload = {
        "candidates": candidates_data,
        "screened_count": result.screened_count,
        "passed_filters": result.passed_filters,
        "skipped_count": result.skipped_count,
        "regime": result.regime,
        "generated_at": result.generated_at.isoformat(),
        "errors": result.errors,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote {len(candidates_data)} candidates to {out_path}")
    return out_path


def cmd_update_earnings(symbols: list[str], lookback_days: int = 10) -> None:
    """Update earnings cache from FMP + yfinance."""
    service = EarningsService()
    cache = service.update_earnings(symbols, lookback_days=lookback_days)
    logger.info(
        f"Earnings cache updated: {len(cache.earnings)} symbols, "
        f"{cache.skipped_count} skipped (FMP tier limit)"
    )


def cmd_screen(
    html_output: str | None = None,
    signals_dir: str | None = None,
) -> None:
    """Run PEAD screening from cached data."""
    from src.domain.screeners.pead.screener import PEADScreener

    config = _load_pead_config()
    screener = PEADScreener(config)

    # Read cached earnings
    service = EarningsService()
    earnings = service.get_recent_earnings()
    if not earnings:
        logger.warning("No earnings in cache. Run --update-earnings first.")
        return

    # Read regime
    sig_dir = Path(signals_dir) if signals_dir else PROJECT_ROOT / "out" / "signals"
    regime = _read_current_regime(sig_dir)

    # Read market caps for liquidity tiers
    cap_service = MarketCapService()
    all_caps = cap_service.get_all_cached_caps()

    today = date.today()
    result = screener.generate_pead_candidates(earnings, regime, today, all_caps)

    # Print summary
    print(f"\nPEAD Screen Results — {today.isoformat()}")
    print(
        f"Regime: {regime} | Screened: {result.screened_count} | "
        f"Passed: {result.passed_filters} | Skipped: {result.skipped_count}"
    )
    print("=" * 50)

    for i, c in enumerate(result.candidates, 1):
        print(f"#{i} {c.symbol} [{c.surprise.liquidity_tier.value.upper()}]")
        print(
            f"   SUE:{c.surprise.sue_score:.1f} Gap:{c.surprise.earnings_day_gap:+.1%} "
            f"Vol:{c.surprise.earnings_day_volume_ratio:.1f}x"
        )
        print(
            f"   Quality: {c.quality_score:.0f} {c.quality_label} | "
            f"Rev: {'Y' if c.surprise.revenue_beat else 'N'} | Gap Held: {'Y' if c.gap_held else 'N'}"
        )
        print(
            f"   Target:{c.profit_target_pct:+.0%} Stop:{c.stop_loss_pct:+.0%} "
            f"Trail:{c.trailing_stop_atr:.1f}ATR Size:{c.position_size_factor:.0%}"
        )
        print()

    if not result.candidates:
        print(f"0 PEAD candidates. ({result.screened_count} screened, regime: {regime})")

    # Write JSON
    output_dir = Path(html_output).parent if html_output else PROJECT_ROOT / "out" / "pead"
    _write_candidates_json(result, output_dir)

    # Write HTML if requested
    if html_output:
        from src.infrastructure.reporting.pead.builder import PEADReportBuilder

        builder = PEADReportBuilder()
        html_path = builder.build(result, html_output)
        logger.info(f"PEAD HTML report: {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PEAD Earnings Drift Screener")
    parser.add_argument(
        "--update-earnings", action="store_true", help="Update earnings cache from FMP"
    )
    parser.add_argument("--screen", action="store_true", help="Run PEAD screening from cache")
    parser.add_argument("--full", action="store_true", help="Update + screen in one step")
    parser.add_argument(
        "--universe", type=str, default="config/universe.yaml", help="Universe YAML path"
    )
    parser.add_argument("--lookback-days", type=int, default=10, help="Calendar days to look back")
    parser.add_argument("--html-output", type=str, default=None, help="Output HTML path")
    parser.add_argument("--signals-dir", type=str, default=None, help="Signal pipeline output dir")
    args = parser.parse_args()

    if not (args.update_earnings or args.screen or args.full):
        parser.print_help()
        sys.exit(1)

    if args.update_earnings or args.full:
        universe_path = Path(args.universe)
        if not universe_path.is_absolute():
            universe_path = PROJECT_ROOT / universe_path
        symbols = load_universe_symbols(universe_path)
        logger.info(f"Universe: {len(symbols)} symbols from {universe_path}")
        cmd_update_earnings(symbols, lookback_days=args.lookback_days)

    if args.screen or args.full:
        cmd_screen(html_output=args.html_output, signals_dir=args.signals_dir)


if __name__ == "__main__":
    main()
