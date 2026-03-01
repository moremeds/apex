"""Quantitative Momentum Screener CLI runner.

Always-fresh: refreshes universe + incrementally updates OHLCV before
screening. Incremental update only fetches new bars since each symbol's
last Parquet date, so repeat runs are fast.

Usage:
    python -m src.runners.momentum_runner                           # refresh + screen
    python -m src.runners.momentum_runner --no-refresh              # screen from cache only
    python -m src.runners.momentum_runner --no-earnings             # skip earnings blackout filter
    python -m src.runners.momentum_runner --backtest                # walk-forward + ablation + JSON
    python -m src.runners.momentum_runner --include-recent-ipos     # adaptive momentum
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.regime_display import regime_label

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
                logger.info(f"SPY regime: {regime} ({regime_label(str(regime))})")
                return str(regime)
    except Exception as e:
        logger.warning(f"Failed to read regime from summary.json: {e}")

    return fallback


def _write_watchlist_json(result: Any, output_dir: Path, data_as_of: date | None = None) -> Path:
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

    payload: dict[str, Any] = {
        "candidates": candidates_data,
        "universe_size": result.universe_size,
        "passed_filters": result.passed_filters,
        "regime": result.regime,
        "generated_at": result.generated_at.isoformat(),
        "errors": [f"{k}: {v}" for k, v in result.errors.items()],
    }
    if data_as_of is not None:
        payload["data_as_of"] = data_as_of.isoformat()

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
        fallback_max_symbols=ucfg.fallback_max_symbols,
    )

    # Download OHLCV
    months = (config.data_source.lookback_trading_days // 21) + 3  # buffer
    count = svc.update_ohlcv(symbols, months=months)
    logger.info(f"Update complete: {len(symbols)} universe, {count} OHLCV")

    return symbols


def cmd_screen(
    config: Any,
    *,
    signals_dir: str | None = None,
    include_recent_ipos: bool = False,
    no_earnings: bool = False,
    no_refresh: bool = False,
) -> Any:
    """Run momentum screening with pre-filter → OHLCV → screen pipeline.

    Pipeline order (thesis-aligned):
    [1/6] Load universe
    [2/6] Pre-filter: market cap (from cached JSON, no OHLCV needed)
    [3/6] Pre-filter: earnings blackout (FMP calendar, 1 API call)
    [4/6] Load OHLCV (only for surviving symbols)
    [5/6] Screen (remaining filters + momentum scoring + rank)
    [6/6] Write results

    Args:
        no_earnings: Skip earnings blackout filter.
        no_refresh: Skip universe + OHLCV refresh (use stale cache).
    """
    from src.domain.screeners.momentum.screener import MomentumScreener
    from src.services.momentum_data_service import MomentumDataService

    svc = MomentumDataService()
    today = date.today()

    # [1/6] Load universe
    if no_refresh:
        print("[1/6] Using cached universe...", flush=True)
        symbols = svc.get_universe()
        if not symbols:
            logger.error("No universe cached. Run without --no-refresh first.")
            return None
        print(f"  {len(symbols)} symbols from cache")
    else:
        print("[1/6] Refreshing universe...", flush=True)
        ucfg = config.universe
        symbols = svc.update_universe(
            indices=ucfg.indices,
            russell_proxy=ucfg.russell_proxy_enabled,
            cap_min=ucfg.russell_proxy_market_cap_min,
            cap_max=ucfg.russell_proxy_market_cap_max,
            fallback_max_symbols=ucfg.fallback_max_symbols,
        )

    universe_size_original = len(symbols)
    logger.info(f"[1/6] Universe loaded: {universe_size_original} symbols")

    # [2/6] Pre-filter: market cap (cheap — uses cached JSON, no OHLCV)
    market_caps = svc.get_market_caps(symbols)
    non_zero_caps = sum(1 for v in market_caps.values() if v > 0)
    if non_zero_caps >= len(symbols) * 0.1:  # cache has >= 10% coverage
        pre = len(symbols)
        symbols = [s for s in symbols if market_caps.get(s, 0.0) >= config.filters.min_market_cap]
        logger.info(f"[2/6] Market cap pre-filter: {pre} -> {len(symbols)} symbols")
    else:
        logger.warning(
            f"[2/6] Market cap cache too sparse "
            f"({non_zero_caps}/{len(symbols)}), skipping pre-filter"
        )

    # [3/6] Pre-filter: earnings blackout (live-only, 1 API call)
    earnings_dates = None
    if not no_earnings and config.filters.earnings_blackout_days > 0:
        earnings_dates = svc.get_upcoming_earnings(
            symbols,
            lookahead_days=config.filters.earnings_blackout_days + 2,
        )
        if earnings_dates:
            blackout = {
                s
                for s, d in earnings_dates.items()
                if 0 <= (d - today).days <= config.filters.earnings_blackout_days
            }
            pre = len(symbols)
            symbols = [s for s in symbols if s not in blackout]
            logger.info(
                f"[3/6] Earnings blackout: {pre} -> {len(symbols)} " f"(excluded {len(blackout)})"
            )
        else:
            logger.info("[3/6] Earnings blackout: no upcoming earnings found")
    else:
        logger.info("[3/6] Earnings blackout: skipped")

    # [4/6] Update OHLCV (only for surviving symbols) + load data
    if not no_refresh:
        print("[4/6] Updating OHLCV (incremental)...", flush=True)
        months = (config.data_source.lookback_trading_days // 21) + 3
        svc.update_ohlcv(symbols, months=months)

    lookback_cal_days = int(config.data_source.lookback_trading_days * 1.5) + 50
    price_data = svc.get_bulk_closes(symbols, today, lookback_days=lookback_cal_days)
    volume_data = svc.get_bulk_volumes(symbols, today, lookback_days=lookback_cal_days)
    logger.info(
        f"[4/6] OHLCV loaded: {len(price_data)}/{len(symbols)} symbols with closes, "
        f"{len(volume_data)} with volumes"
    )

    # Check data freshness
    data_as_of = svc.get_data_as_of_date(list(price_data.keys()))
    if data_as_of:
        bdays_behind = int(np.busday_count(data_as_of, today))
        if bdays_behind > 2:
            print(
                f"  WARNING: Data still {bdays_behind} business days behind "
                f"(data: {data_as_of.isoformat()}, today: {today.isoformat()})"
            )

    # Read regime
    sig_dir = Path(signals_dir) if signals_dir else PROJECT_ROOT / "out" / "signals"
    regime = _read_current_regime(sig_dir)

    # [5/6] Screen (remaining filters + momentum scoring + rank)
    print("[5/6] Screening...", end=" ", flush=True)
    screener = MomentumScreener(config)
    result = screener.screen(
        price_data=price_data,
        volume_data=volume_data,
        regime=regime,
        market_caps=market_caps,
        use_adaptive=include_recent_ipos,
        earnings_dates=earnings_dates,
    )
    logger.info(
        f"[5/6] Screening: regime={regime}, passed_filters={result.passed_filters}, "
        f"candidates={len(result.candidates)}"
    )
    print(f"{len(result.candidates)} candidates")

    # Print summary
    data_as_of_str = data_as_of.isoformat() if data_as_of else "unknown"
    print(f"\nMomentum Screen — {today.isoformat()} (data: {data_as_of_str})")
    print(
        f"Regime: {regime} ({regime_label(regime)}) | Universe: {result.universe_size} | "
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
                f"{c.rank:3d} {s.symbol:<7} {s.momentum_12_1:+8.1%} {s.fip:5.2f} "
                f"{s.composite_rank:5.2f} {c.quality_label:<9} "
                f"{s.liquidity_tier.value:<10} {s.last_close:8.2f} "
                f"{c.position_size_factor:4.0%}"
            )
    else:
        print(f"0 momentum candidates (regime: {regime} - {regime_label(regime)})")

    # [6/6] Write results
    output_dir = PROJECT_ROOT / "out" / "momentum"
    _write_watchlist_json(result, output_dir, data_as_of=data_as_of)

    return result


def _generate_rebalance_dates(start: date, end: date) -> list[date]:
    """Generate weekly rebalance dates (Fridays) between start and end."""
    rebalance_dates: list[date] = []
    current = start
    while current <= end:
        days_to_friday = (4 - current.weekday()) % 7
        friday = current + timedelta(days=days_to_friday)
        if friday > end:
            break
        rebalance_dates.append(friday)
        current = friday + timedelta(days=7)
    return rebalance_dates


def _compute_max_drawdown(returns: list[float]) -> float:
    """Max drawdown from a list of period returns. Returns negative value."""
    if not returns:
        return 0.0
    cumulative = np.cumprod([1 + r for r in returns])
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(np.min(drawdown))


def _run_walk_forward(
    screener: Any,
    all_close_series: dict[str, "pd.Series[float]"],
    all_volume_series: dict[str, "pd.Series[float]"],
    market_caps: dict[str, float],
    rebalance_dates: list[date],
    top_n: int,
    hold_days: int,
    log_interval: int = 50,
) -> list[dict[str, Any]]:
    """PIT-correct walk-forward simulation.

    For each rebalance date:
    1. Trim series to as-of rebal_date (only data available at that point)
    2. Approximate PIT market caps via price ratio scaling
    3. Screen with trimmed np.ndarray (screener API unchanged)
    4. Compute forward returns from rebal_date + hold_days

    Market cap approximation: cap(t) ≈ cap(today) × price(t) / price(latest).
    Assumes shares outstanding are roughly stable over the backtest window.

    Args:
        screener: MomentumScreener instance.
        all_close_series: Symbol -> pd.Series (DatetimeIndex) of daily closes.
        all_volume_series: Symbol -> pd.Series (DatetimeIndex) of daily volumes.
        market_caps: Symbol -> market cap in USD (current snapshot).
        rebalance_dates: List of rebalance dates.
        top_n: Number of picks per period.
        hold_days: Trading days to hold positions.
        log_interval: Log progress every N periods.

    Returns:
        List of dicts with date, n_picks, avg_return, picks.
    """
    import pandas as pd

    portfolio_returns: list[dict[str, Any]] = []
    total = len(rebalance_dates)

    # Pre-compute latest close per symbol for PIT market cap scaling
    latest_prices: dict[str, float] = {}
    for sym, series in all_close_series.items():
        if len(series) > 0:
            latest_prices[sym] = float(series.iloc[-1])

    for i, rebal_date in enumerate(rebalance_dates):
        if (i + 1) % log_interval == 0 or i == total - 1:
            logger.info(f"Walk-forward: {i + 1}/{total} periods processed")

        rebal_ts = pd.Timestamp(rebal_date)

        # Trim to PIT: only data available on rebal_date
        trimmed_closes: dict[str, np.ndarray] = {}
        trimmed_volumes: dict[str, np.ndarray] = {}
        pit_market_caps: dict[str, float] = {}
        for sym, series in all_close_series.items():
            pit = series[series.index <= rebal_ts]
            if len(pit) > 0:
                trimmed_closes[sym] = pit.values
                if sym in all_volume_series:
                    vol_pit = all_volume_series[sym][all_volume_series[sym].index <= rebal_ts]
                    trimmed_volumes[sym] = vol_pit.values

                # Approximate PIT market cap: cap(t) ≈ cap(now) × price(t)/price(latest)
                latest_p = latest_prices.get(sym, 0.0)
                current_cap = market_caps.get(sym, 0.0)
                if latest_p > 0 and current_cap > 0:
                    pit_market_caps[sym] = current_cap * (float(pit.iloc[-1]) / latest_p)

        result = screener.screen(
            price_data=trimmed_closes,
            volume_data=trimmed_volumes,
            regime="R0",  # Backtest uses R0 (no regime gating)
            market_caps=pit_market_caps,
            is_backtest=True,
        )

        picks = [c.signal.symbol for c in result.candidates[:top_n]]
        if not picks:
            continue

        # Compute forward returns from rebal_date
        forward_rets: list[float] = []
        for sym in picks:
            if sym not in all_close_series:
                continue
            series = all_close_series[sym]
            # Find last trading day on or before rebal_date.
            # side="right" returns first index > rebal_ts, so -1 gives <=.
            rebal_idx = series.index.searchsorted(rebal_ts, side="right") - 1
            if rebal_idx < 0:
                continue
            if rebal_idx + hold_days < len(series):
                p_entry = series.iloc[rebal_idx]
                p_exit = series.iloc[rebal_idx + hold_days]
                if p_entry > 0:
                    ret = (p_exit - p_entry) / p_entry
                    forward_rets.append(float(ret))

        if forward_rets:
            avg_ret = float(np.mean(forward_rets))
            portfolio_returns.append(
                {
                    "date": rebal_date.isoformat(),
                    "n_picks": len(picks),
                    "avg_return": avg_ret,
                    "picks": picks[:5],
                }
            )

    return portfolio_returns


def _print_backtest_summary(
    portfolio_returns: list[dict[str, Any]],
    start: date,
    end: date,
    top_n: int,
    hold_days: int,
) -> dict[str, float]:
    """Print and return backtest summary metrics."""
    all_rets = [pr["avg_return"] for pr in portfolio_returns]
    cum_ret = float(np.prod([1 + r for r in all_rets]) - 1)
    avg_weekly = float(np.mean(all_rets))
    max_dd = _compute_max_drawdown(all_rets)
    sharpe_approx = (
        float(np.mean(all_rets) / np.std(all_rets) * np.sqrt(52)) if np.std(all_rets) > 0 else 0.0
    )

    print(f"\nMomentum Backtest Results — {start} to {end}")
    print("=" * 60)
    print(f"Rebalance periods:  {len(portfolio_returns)}")
    print(f"Cumulative return:  {cum_ret:+.2%}")
    print(f"Avg weekly return:  {avg_weekly:+.4%}")
    print(f"Sharpe (approx):    {sharpe_approx:.2f}")
    print(f"Max drawdown:       {max_dd:+.2%}")
    print(f"Holding period:     {hold_days} days")
    print(f"Portfolio size:     {top_n}")

    return {
        "cumulative_return": cum_ret,
        "avg_weekly_return": avg_weekly,
        "sharpe_approx": sharpe_approx,
        "max_drawdown": max_dd,
    }


def cmd_backtest(config: Any) -> None:
    """Walk-forward momentum backtest + ablation + JSON output.

    Loads cached data once, then:
    1. Runs full walk-forward simulation (M+FIP+Filters config)
    2. Runs 3 ablation configs (M-only / M+FIP / M+FIP+Filters)
    3. Writes JSON results
    """
    from src.domain.screeners.momentum.config import MomentumFilters, ScoringConfig
    from src.domain.screeners.momentum.screener import MomentumScreener
    from src.services.momentum_data_service import MomentumDataService

    svc = MomentumDataService()

    bcfg = config.backtest
    start = date.fromisoformat(bcfg.start_date)
    end = date.fromisoformat(bcfg.end_date) if bcfg.end_date else date.today()
    top_n = bcfg.portfolio_top_n
    hold_days = bcfg.holding_period_days

    logger.info(f"Backtest: {start} to {end}, top-{top_n}, {hold_days}-day hold")

    # Load all available data once (PIT-correct date-indexed Series)
    symbols = svc.get_universe()
    if not symbols:
        logger.error("No universe cached. Run without --no-refresh first.")
        return

    lookback_cal_days = int(config.data_source.lookback_trading_days * 1.5) + 50
    total_cal_days = (end - start).days + lookback_cal_days
    all_close_series = svc.get_bulk_close_series(symbols, end, lookback_days=total_cal_days)
    all_volume_series = svc.get_bulk_volume_series(symbols, end, lookback_days=total_cal_days)
    market_caps = svc.get_market_caps(symbols)
    rebalance_dates = _generate_rebalance_dates(start, end)

    logger.info(
        f"Loaded {len(all_close_series)} close series, "
        f"{len(all_volume_series)} volume series, "
        f"{len(rebalance_dates)} rebalance dates"
    )

    # ── Walk-forward simulation (full config) ─────────────────────────
    screener = MomentumScreener(config)
    portfolio_returns = _run_walk_forward(
        screener=screener,
        all_close_series=all_close_series,
        all_volume_series=all_volume_series,
        market_caps=market_caps,
        rebalance_dates=rebalance_dates,
        top_n=top_n,
        hold_days=hold_days,
    )

    bt_metrics: dict[str, float] = {}
    if portfolio_returns:
        bt_metrics = _print_backtest_summary(portfolio_returns, start, end, top_n, hold_days)
    else:
        print("No backtest results generated (insufficient data)")

    # ── Ablation: 3 configs ───────────────────────────────────────────
    logger.info("Ablation: running 3 configurations...")

    no_filters = MomentumFilters(
        min_market_cap=0,
        min_avg_daily_dollar_volume=0,
        min_price=0,
        min_daily_turnover_rate=0,
        earnings_blackout_days=0,
    )
    ablation_configs: list[tuple[str, Any]] = [
        (
            "M-only",
            dataclasses.replace(
                config,
                scoring=dataclasses.replace(config.scoring, fip_weight=0.0, momentum_weight=1.0),
                filters=no_filters,
            ),
        ),
        (
            "M+FIP",
            dataclasses.replace(
                config,
                scoring=ScoringConfig(
                    momentum_weight=0.5,
                    fip_weight=0.5,
                    top_n=config.scoring.top_n,
                ),
                filters=no_filters,
            ),
        ),
        (
            "M+FIP+Filters",
            dataclasses.replace(config),
        ),
    ]

    ablation_results: list[tuple[str, dict[str, Any]]] = []

    for idx, (label, abl_config) in enumerate(ablation_configs, 1):
        logger.info(
            f"[{idx}/3] {label}: running walk-forward " f"({len(rebalance_dates)} periods)..."
        )
        abl_screener = MomentumScreener(abl_config)
        abl_returns = _run_walk_forward(
            screener=abl_screener,
            all_close_series=all_close_series,
            all_volume_series=all_volume_series,
            market_caps=market_caps,
            rebalance_dates=rebalance_dates,
            top_n=top_n,
            hold_days=hold_days,
        )

        if abl_returns:
            all_rets = [pr["avg_return"] for pr in abl_returns]
            cum_ret = float(np.prod([1 + r for r in all_rets]) - 1)
            avg_weekly = float(np.mean(all_rets))
            max_dd = _compute_max_drawdown(all_rets)
            sharpe = (
                float(np.mean(all_rets) / np.std(all_rets) * np.sqrt(52))
                if np.std(all_rets) > 0
                else 0.0
            )
            metrics: dict[str, Any] = {
                "cumulative_return": cum_ret,
                "avg_weekly_return": avg_weekly,
                "sharpe_approx": sharpe,
                "max_drawdown": max_dd,
                "n_periods": len(abl_returns),
            }
            logger.info(
                f"[{idx}/3] {label}: cum_ret={cum_ret:+.1%}, "
                f"sharpe={sharpe:.2f}, max_dd={max_dd:+.1%}"
            )
        else:
            metrics = {
                "cumulative_return": 0.0,
                "avg_weekly_return": 0.0,
                "sharpe_approx": 0.0,
                "max_drawdown": 0.0,
                "n_periods": 0,
            }
            logger.info(f"[{idx}/3] {label}: no results")

        ablation_results.append((label, metrics))

    # Print ablation comparison table
    print(f"\nAblation Results — {start} to {end}")
    print("=" * 72)
    print(f"{'Config':<20} {'Cum Return':>12} {'Sharpe':>8} {'Max DD':>10} {'Avg Weekly':>12}")
    print("-" * 72)
    for label, m in ablation_results:
        print(
            f"{label:<20} {m['cumulative_return']:>+11.2%} "
            f"{m['sharpe_approx']:>8.2f} "
            f"{m['max_drawdown']:>+9.2%} "
            f"{m['avg_weekly_return']:>+11.4%}"
        )
    print("=" * 72)

    # ── Write JSON ─────────────────────────────────────────────────────
    data_dir = PROJECT_ROOT / "out" / "momentum" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    bt_data = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "top_n": top_n,
        "hold_days": hold_days,
        **bt_metrics,
        "periods": portfolio_returns,
    }
    abl_data = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "top_n": top_n,
        "hold_days": hold_days,
        "configs": [{"label": label, **m} for label, m in ablation_results],
    }

    (data_dir / "momentum_backtest.json").write_text(json.dumps(bt_data, indent=2))
    (data_dir / "momentum_ablation.json").write_text(json.dumps(abl_data, indent=2))
    logger.info(f"Backtest JSON: {data_dir / 'momentum_backtest.json'}")
    logger.info(f"Ablation JSON: {data_dir / 'momentum_ablation.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitative Momentum Screener (12-1 momentum + FIP)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Legacy alias — universe + OHLCV refresh is now the default behavior",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run walk-forward backtest + ablation comparison",
    )
    parser.add_argument(
        "--include-recent-ipos",
        action="store_true",
        help="Use adaptive momentum for stocks with 6-11 month history",
    )
    parser.add_argument("--signals-dir", type=str, default=None, help="Signal pipeline output dir")
    parser.add_argument("--no-earnings", action="store_true", help="Skip earnings blackout filter")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip universe + OHLCV refresh (screen from stale cache)",
    )
    args = parser.parse_args()

    config = _load_config()

    if args.update:
        cmd_update(config)

    if args.backtest:
        cmd_backtest(config)
    else:
        cmd_screen(
            config,
            signals_dir=args.signals_dir,
            include_recent_ipos=args.include_recent_ipos,
            no_earnings=args.no_earnings,
            # --update already fetched OHLCV; skip redundant refresh
            no_refresh=args.no_refresh or args.update,
        )


if __name__ == "__main__":
    main()
