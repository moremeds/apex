"""PEAD Screener CLI runner.

Tracker (--track, --update-tracker, --tracker-stats) and attention
(--update-attention) run by default on every invocation. Use --no-*
flags to skip individual steps.

Usage:
    python -m src.runners.pead_runner --update-earnings --universe config/universe.yaml
    python -m src.runners.pead_runner --screen --html-output out/signals/pead.html
    python -m src.runners.pead_runner --full --universe config/universe.yaml --html-output out/pead/pead.html
    python -m src.runners.pead_runner --screen --no-attention --no-track
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.domain.screeners.pead.config import PEADConfig

import yaml

from src.services.earnings_service import EarningsService
from src.services.market_cap_service import MarketCapService, load_universe_symbols
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Fail-closed: unknown regime defaults to R1 (reduced size), not R0 (full trading)
_REGIME_FALLBACK = "R1"

# Module-level reference to last screen result for --track after --screen
_last_screen_result: Any = None


def _load_pead_config() -> dict[str, Any]:
    """Load PEAD screener config from YAML (raw dict for backward compat)."""
    path = PROJECT_ROOT / "config" / "pead_screener.yaml"
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_typed_config() -> "PEADConfig":
    """Load PEAD screener config as typed dataclass."""
    from src.domain.screeners.pead.config import PEADConfig

    path = PROJECT_ROOT / "config" / "pead_screener.yaml"
    return PEADConfig.from_yaml(path)


def _read_current_regime(signals_dir: Path, fallback: str = _REGIME_FALLBACK) -> str:
    """Read SPY regime from signal pipeline output.

    Fail-closed: defaults to fallback (R1) when data is unavailable,
    ensuring conservative position sizing when regime is unknown.
    """
    summary_path = signals_dir / "data" / "summary.json"
    if not summary_path.exists():
        logger.warning(
            f"No summary.json at {summary_path}. "
            f"Defaulting to {fallback} (fail-closed). "
            "Ensure hourly-signals runs before PEAD, or fetch from gh-pages."
        )
        return fallback

    try:
        data = json.loads(summary_path.read_text())
        for t in data.get("tickers", []):
            if t.get("symbol") == "SPY":
                regime = t.get("regime", fallback)
                logger.info(f"SPY regime: {regime}")
                return str(regime)
    except Exception as e:
        logger.warning(f"Failed to read regime from summary.json: {e}")

    return fallback


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
                "multi_quarter_sue": (
                    round(c.surprise.multi_quarter_sue, 2)
                    if c.surprise.multi_quarter_sue is not None
                    else None
                ),
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


# ── Commands ──────────────────────────────────────────────────────────────


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
    regime_fallback: str = _REGIME_FALLBACK,
) -> Any:
    """Run PEAD screening from cached data. Returns screen result."""
    from src.domain.screeners.pead.screener import PEADScreener

    config = _load_typed_config()
    screener = PEADScreener(config)

    # Read cached earnings
    service = EarningsService()
    earnings = service.get_recent_earnings()
    if not earnings:
        logger.warning("No earnings in cache. Run --update-earnings first.")
        return None

    # Read regime
    sig_dir = Path(signals_dir) if signals_dir else PROJECT_ROOT / "out" / "signals"
    regime = _read_current_regime(sig_dir, fallback=regime_fallback)

    # Read market caps for liquidity tiers
    cap_service = MarketCapService()
    all_caps = cap_service.get_all_cached_caps()

    # Read attention data (optional, never blocks screening)
    attention_data: dict[str, str | None] | None = None
    if config.attention_filter.enabled:
        try:
            from src.infrastructure.adapters.earnings.attention_adapter import AttentionAdapter

            adapter = AttentionAdapter()
            attention_data = {}
            for e in earnings:
                sym = e.get("symbol", "")
                rdate = e.get("report_date")
                if isinstance(rdate, str):
                    rdate = date.fromisoformat(rdate)
                if sym and rdate:
                    attention_data[sym] = adapter.get_attention_level(sym, rdate)
        except Exception as e:
            logger.warning(f"Attention data unavailable: {e}")

    today = date.today()
    result = screener.generate_pead_candidates(
        earnings, regime, today, all_caps, attention_data=attention_data
    )

    # Print summary
    print(f"\nPEAD Screen Results — {today.isoformat()}")
    print(
        f"Regime: {regime} | Screened: {result.screened_count} | "
        f"Passed: {result.passed_filters} | Skipped: {result.skipped_count}"
    )
    print("=" * 50)

    for i, c in enumerate(result.candidates, 1):
        mq = f" MQ:{c.surprise.multi_quarter_sue:.1f}" if c.surprise.multi_quarter_sue else ""
        print(f"#{i} {c.symbol} [{c.surprise.liquidity_tier.value.upper()}]")
        print(
            f"   SUE:{c.surprise.sue_score:.1f}{mq} Gap:{c.surprise.earnings_day_gap:+.1%} "
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

    return result


def cmd_update_attention() -> None:
    """Pre-fetch Google Trends attention data for cached earnings.

    Never blocks screening — separate pre-fetch step.
    Requires optional ``pytrends`` dependency.
    """
    try:
        from src.infrastructure.adapters.earnings.attention_adapter import AttentionAdapter
    except ImportError:
        print("Attention filter requires pytrends: pip install pytrends")
        return

    service = EarningsService()
    earnings = service.get_recent_earnings()
    if not earnings:
        logger.info("No earnings in cache. Run --update-earnings first.")
        return

    pairs: list[tuple[date, date]] = []
    for e in earnings:
        sym = e.get("symbol", "")
        rdate = e.get("report_date")
        if isinstance(rdate, str):
            rdate = date.fromisoformat(rdate)
        if sym and rdate:
            pairs.append((sym, rdate))  # type: ignore[arg-type]

    adapter = AttentionAdapter()
    fetched = adapter.update_attention_batch(pairs)  # type: ignore[arg-type]
    print(f"Attention: {fetched} new scores fetched ({len(pairs)} symbols checked)")


def cmd_track(result: Any) -> None:
    """Persist screening candidates to tracker."""
    if result is None or not result.candidates:
        logger.info("No candidates to track")
        return

    from src.services.pead_tracker_service import PEADTrackerService

    tracker = PEADTrackerService()
    added = tracker.add_candidates(result.candidates)
    print(f"Tracker: {added} new candidates added ({len(result.candidates)} screened)")


def cmd_update_tracker() -> None:
    """Resolve open tracker candidates via OHLC first-touch."""
    from src.services.pead_tracker_service import PEADTrackerService

    tracker = PEADTrackerService()
    resolved = tracker.update_outcomes()
    print(f"Tracker: {resolved} candidates resolved")


def cmd_tracker_stats() -> None:
    """Print tracker performance summary."""
    from src.services.pead_tracker_service import PEADTrackerService

    tracker = PEADTrackerService()
    stats = tracker.get_stats()

    print("\nPEAD Tracker Stats")
    print("=" * 40)
    print(f"Total tracked:  {stats.total}")
    print(f"Open:           {stats.open}")
    print(f"Won:            {stats.won}")
    print(f"Lost:           {stats.lost}")
    print(f"Timeout:        {stats.timeout}")

    if stats.win_rate is not None:
        print(f"Win rate:       {stats.win_rate:.1%}")
    if stats.avg_pnl_pct is not None:
        print(f"Avg P&L:        {stats.avg_pnl_pct:+.2%}")
    if stats.avg_hold_days is not None:
        print(f"Avg hold days:  {stats.avg_hold_days:.1f}")

    if stats.by_quality:
        print("\nBy Quality Tier:")
        for label, data in sorted(stats.by_quality.items()):
            wr = data.get("win_rate")
            pnl = data.get("avg_pnl_pct")
            print(
                (
                    f"  {label:10s} n={data['total']:3d}  " f"WR={wr:.1%} "
                    if wr is not None
                    else f"  {label:10s} n={data['total']:3d}  "
                ),
                end="",
            )
            if pnl is not None:
                print(f"P&L={pnl:+.2%}")
            else:
                print()


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
    parser.add_argument(
        "--regime-fallback",
        type=str,
        default=_REGIME_FALLBACK,
        help=f"Regime when summary.json unavailable (default: {_REGIME_FALLBACK})",
    )
    # Attention filter (on by default; --no-attention to skip)
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Skip Google Trends attention pre-fetch",
    )
    # Tracker flags (on by default; --no-track / --no-tracker to skip)
    parser.add_argument(
        "--no-track", action="store_true", help="Skip persisting candidates to tracker"
    )
    parser.add_argument(
        "--no-update-tracker",
        action="store_true",
        help="Skip resolving open tracker candidates",
    )
    parser.add_argument(
        "--no-tracker-stats",
        action="store_true",
        help="Skip printing tracker performance summary",
    )
    args = parser.parse_args()

    has_action = any([args.update_earnings, args.screen, args.full])
    if not has_action:
        parser.print_help()
        sys.exit(1)

    # Update earnings
    if args.update_earnings or args.full:
        universe_path = Path(args.universe)
        if not universe_path.is_absolute():
            universe_path = PROJECT_ROOT / universe_path
        symbols = load_universe_symbols(universe_path)
        logger.info(f"Universe: {len(symbols)} symbols from {universe_path}")
        cmd_update_earnings(symbols, lookback_days=args.lookback_days)

    # Update attention (default on; --no-attention to skip)
    if not args.no_attention:
        cmd_update_attention()

    # Screen
    result = None
    if args.screen or args.full:
        result = cmd_screen(
            html_output=args.html_output,
            signals_dir=args.signals_dir,
            regime_fallback=args.regime_fallback,
        )

    # Track candidates (default on; --no-track to skip)
    if not args.no_track:
        cmd_track(result)

    # Update tracker outcomes (default on; --no-update-tracker to skip)
    if not args.no_update_tracker:
        cmd_update_tracker()

    # Print stats (default on; --no-tracker-stats to skip)
    if not args.no_tracker_stats:
        cmd_tracker_stats()


if __name__ == "__main__":
    main()
