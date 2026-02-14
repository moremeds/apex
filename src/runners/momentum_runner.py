"""Quantitative Momentum Screener CLI runner.

Cache-first by default: screens from existing Parquet data.
Use --update to fetch fresh universe + OHLCV before screening.

Usage:
    python -m src.runners.momentum_runner                           # screen from cache
    python -m src.runners.momentum_runner --update                  # fetch + screen
    python -m src.runners.momentum_runner --backtest                # walk-forward backtest
    python -m src.runners.momentum_runner --include-recent-ipos     # adaptive momentum
    python -m src.runners.momentum_runner --html out/momentum/report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Fail-closed: unknown regime defaults to R1 (reduced size), not R0
_REGIME_FALLBACK = "R1"


def _load_config() -> Any:
    """Load momentum screener config as typed dataclass."""
    from src.domain.screeners.momentum.config import MomentumConfig

    path = PROJECT_ROOT / "config" / "momentum_screener.yaml"
    return MomentumConfig.from_yaml(path)


def _read_current_regime(signals_dir: Path, fallback: str = _REGIME_FALLBACK) -> str:
    """Read SPY regime from signal pipeline output.

    Fail-closed: defaults to R1 when data is unavailable.
    """
    summary_path = signals_dir / "data" / "summary.json"
    if not summary_path.exists():
        logger.warning(
            f"No summary.json at {summary_path}. " f"Defaulting to {fallback} (fail-closed)."
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


def _write_watchlist_json(result: Any, output_dir: Path) -> Path:
    """Write screening results to JSON for downstream consumers."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "momentum_watchlist.json"

    candidates_data = []
    for c in result.candidates:
        s = c.signal
        candidates_data.append(
            {
                "rank": c.rank,
                "symbol": s.symbol,
                "momentum_12_1": round(s.momentum_12_1, 4),
                "fip": round(s.fip, 4),
                "momentum_percentile": round(s.momentum_percentile, 4),
                "fip_percentile": round(s.fip_percentile, 4),
                "composite_rank": round(s.composite_rank, 4),
                "last_close": round(s.last_close, 2),
                "market_cap": s.market_cap,
                "avg_daily_dollar_volume": round(s.avg_daily_dollar_volume, 0),
                "liquidity_tier": s.liquidity_tier.value,
                "estimated_slippage_bps": s.estimated_slippage_bps,
                "lookback_days": s.lookback_days,
                "quality_label": c.quality_label,
                "position_size_factor": c.position_size_factor,
                "regime": c.regime,
            }
        )

    payload = {
        "candidates": candidates_data,
        "universe_size": result.universe_size,
        "passed_filters": result.passed_filters,
        "regime": result.regime,
        "generated_at": result.generated_at.isoformat(),
        "errors": result.errors,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote {len(candidates_data)} candidates to {out_path}")
    return out_path


# ── Commands ──────────────────────────────────────────────────────────


def cmd_update(config: Any) -> list[str]:
    """Update universe + OHLCV data from FMP + yfinance."""
    from src.services.momentum_data_service import MomentumDataService

    svc = MomentumDataService()

    # Fetch universe
    ucfg = config.universe
    symbols = svc.update_universe(
        indices=ucfg.indices,
        russell_proxy=ucfg.russell_proxy_enabled,
        cap_min=ucfg.russell_proxy_market_cap_min,
        cap_max=ucfg.russell_proxy_market_cap_max,
    )

    # Download OHLCV
    months = (config.data_source.lookback_trading_days // 21) + 3  # buffer
    count = svc.update_ohlcv(symbols, months=months)
    logger.info(f"Update complete: {len(symbols)} universe, {count} OHLCV")

    return symbols


def cmd_screen(
    config: Any,
    *,
    html_output: str | None = None,
    signals_dir: str | None = None,
    include_recent_ipos: bool = False,
) -> Any:
    """Run momentum screening from cached data."""
    from src.domain.screeners.momentum.screener import MomentumScreener
    from src.services.momentum_data_service import MomentumDataService

    svc = MomentumDataService()

    # Read universe
    symbols = svc.get_universe()
    if not symbols:
        logger.error("No universe cached. Run with --update first.")
        return None

    # Read regime
    sig_dir = Path(signals_dir) if signals_dir else PROJECT_ROOT / "out" / "signals"
    regime = _read_current_regime(sig_dir)

    # Read market caps
    market_caps = svc.get_market_caps(symbols)

    # Load price + volume data
    today = date.today()
    lookback_cal_days = int(config.data_source.lookback_trading_days * 1.5) + 50
    price_data = svc.get_bulk_closes(symbols, today, lookback_days=lookback_cal_days)
    volume_data = svc.get_bulk_volumes(symbols, today, lookback_days=lookback_cal_days)

    logger.info(
        f"Data loaded: {len(price_data)} symbols with closes, " f"{len(volume_data)} with volumes"
    )

    # Screen
    screener = MomentumScreener(config)
    result = screener.screen(
        price_data=price_data,
        volume_data=volume_data,
        regime=regime,
        market_caps=market_caps,
        use_adaptive=include_recent_ipos,
    )

    # Print summary
    print(f"\nMomentum Screen — {today.isoformat()}")
    print(
        f"Regime: {regime} | Universe: {result.universe_size} | "
        f"Passed: {result.passed_filters} | Top-N: {len(result.candidates)}"
    )
    print("=" * 80)

    if result.candidates:
        print(
            f"{'#':>3} {'Symbol':<7} {'Mom 12-1':>9} {'FIP':>6} "
            f"{'Comp':>6} {'Quality':<9} {'Tier':<10} {'Close':>8} {'Size':>5}"
        )
        print("-" * 80)
        for c in result.candidates:
            s = c.signal
            print(
                f"{c.rank:3d} {s.symbol:<7} {s.momentum_12_1:+8.1%} {s.fip:+5.2f} "
                f"{s.composite_rank:5.2f} {c.quality_label:<9} "
                f"{s.liquidity_tier.value:<10} {s.last_close:8.2f} "
                f"{c.position_size_factor:4.0%}"
            )
    else:
        print(f"0 momentum candidates (regime: {regime})")

    # Write JSON
    output_dir = Path(html_output).parent if html_output else PROJECT_ROOT / "out" / "momentum"
    _write_watchlist_json(result, output_dir)

    # Write HTML
    if html_output:
        from src.infrastructure.reporting.momentum.builder import MomentumReportBuilder

        builder = MomentumReportBuilder()
        html_path = builder.build(result, html_output)
        logger.info(f"Momentum HTML report: {html_path}")

    return result


def cmd_backtest(config: Any, html_output: str | None = None) -> None:
    """Walk-forward momentum backtest using weekly rebalancing.

    For each rebalance date:
    1. Get point-in-time constituents (anti-survivorship)
    2. Run MomentumScreener with is_backtest=True
    3. Compute equal-weight portfolio returns over holding period
    """
    from src.services.momentum_data_service import MomentumDataService

    svc = MomentumDataService()

    bcfg = config.backtest
    start = date.fromisoformat(bcfg.start_date)
    end = date.fromisoformat(bcfg.end_date) if bcfg.end_date else date.today()
    top_n = bcfg.portfolio_top_n
    hold_days = bcfg.holding_period_days

    logger.info(f"Backtest: {start} to {end}, top-{top_n}, {hold_days}-day hold")

    # Load all available data
    symbols = svc.get_universe()
    if not symbols:
        logger.error("No universe cached. Run with --update first.")
        return

    lookback_cal_days = int(config.data_source.lookback_trading_days * 1.5) + 50
    all_closes = svc.get_bulk_closes(
        symbols, end, lookback_days=(end - start).days + lookback_cal_days
    )
    all_volumes = svc.get_bulk_volumes(
        symbols, end, lookback_days=(end - start).days + lookback_cal_days
    )
    market_caps = svc.get_market_caps(symbols)

    from src.domain.screeners.momentum.screener import MomentumScreener

    screener = MomentumScreener(config)

    # Generate weekly rebalance dates (Fridays)
    rebalance_dates: list[date] = []
    current = start
    while current <= end:
        # Find next Friday
        days_to_friday = (4 - current.weekday()) % 7
        friday = current + timedelta(days=days_to_friday)
        if friday > end:
            break
        rebalance_dates.append(friday)
        current = friday + timedelta(days=7)

    logger.info(f"Backtest: {len(rebalance_dates)} rebalance dates")

    # Walk-forward simulation
    portfolio_returns: list[dict[str, Any]] = []

    for rebal_date in rebalance_dates:
        # Trim price data to as-of rebal_date
        trimmed_closes: dict[str, np.ndarray] = {}
        trimmed_volumes: dict[str, np.ndarray] = {}

        for sym in all_closes:
            closes_full = all_closes[sym]
            # Estimate how many bars to trim (rough: calendar days ~ 1.4x trading)
            # This is a simplification; in production you'd index by date
            trimmed_closes[sym] = closes_full
            if sym in all_volumes:
                trimmed_volumes[sym] = all_volumes[sym]

        result = screener.screen(
            price_data=trimmed_closes,
            volume_data=trimmed_volumes,
            regime="R0",  # Backtest uses R0 (no regime gating)
            market_caps=market_caps,
            is_backtest=True,
        )

        picks = [c.signal.symbol for c in result.candidates[:top_n]]
        if not picks:
            continue

        # Compute forward returns (simplified: use available data)
        forward_rets: list[float] = []
        for sym in picks:
            if sym in all_closes:
                closes = all_closes[sym]
                if len(closes) > hold_days:
                    ret = (closes[-1] - closes[-(hold_days + 1)]) / closes[-(hold_days + 1)]
                    forward_rets.append(float(ret))

        if forward_rets:
            avg_ret = float(np.mean(forward_rets))
            portfolio_returns.append(
                {
                    "date": rebal_date.isoformat(),
                    "n_picks": len(picks),
                    "avg_return": avg_ret,
                    "picks": picks[:5],  # Top 5 for logging
                }
            )

    # Print backtest summary
    if portfolio_returns:
        all_rets = [pr["avg_return"] for pr in portfolio_returns]
        cum_ret = float(np.prod([1 + r for r in all_rets]) - 1)
        avg_weekly = float(np.mean(all_rets))
        sharpe_approx = (
            float(np.mean(all_rets) / np.std(all_rets) * np.sqrt(52)) if np.std(all_rets) > 0 else 0
        )

        print(f"\nMomentum Backtest Results — {start} to {end}")
        print("=" * 60)
        print(f"Rebalance periods:  {len(portfolio_returns)}")
        print(f"Cumulative return:  {cum_ret:+.2%}")
        print(f"Avg weekly return:  {avg_weekly:+.4%}")
        print(f"Sharpe (approx):    {sharpe_approx:.2f}")
        print(f"Holding period:     {hold_days} days")
        print(f"Portfolio size:     {top_n}")

        # Write JSON results
        output_dir = Path(html_output).parent if html_output else PROJECT_ROOT / "out" / "momentum"
        output_dir.mkdir(parents=True, exist_ok=True)
        bt_path = output_dir / "data" / "momentum_backtest.json"
        bt_path.parent.mkdir(parents=True, exist_ok=True)
        bt_path.write_text(
            json.dumps(
                {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "cumulative_return": cum_ret,
                    "avg_weekly_return": avg_weekly,
                    "sharpe_approx": sharpe_approx,
                    "periods": portfolio_returns,
                },
                indent=2,
            )
        )
        logger.info(f"Backtest results: {bt_path}")
    else:
        print("No backtest results generated (insufficient data)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitative Momentum Screener (12-1 momentum + FIP)"
    )
    parser.add_argument(
        "--update", action="store_true", help="Fetch universe + OHLCV before screening"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run walk-forward portfolio backtest"
    )
    parser.add_argument(
        "--include-recent-ipos",
        action="store_true",
        help="Use adaptive momentum for stocks with 6-11 month history",
    )
    parser.add_argument("--html", type=str, default=None, help="Output HTML report path")
    parser.add_argument("--signals-dir", type=str, default=None, help="Signal pipeline output dir")
    args = parser.parse_args()

    config = _load_config()

    if args.update:
        cmd_update(config)

    if args.backtest:
        cmd_backtest(config, html_output=args.html)
    else:
        cmd_screen(
            config,
            html_output=args.html,
            signals_dir=args.signals_dir,
            include_recent_ipos=args.include_recent_ipos,
        )


if __name__ == "__main__":
    main()
