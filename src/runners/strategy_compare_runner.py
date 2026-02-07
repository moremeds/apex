"""
Strategy Comparison Runner.

Downloads daily OHLCV data and runs multiple strategy signal generators
via VectorBT, then generates a comparison dashboard (strategies.html).

Usage:
    python -m src.runners.strategy_compare_runner \
        --symbols SPY QQQ AAPL NVDA --output out/signals/strategies.html

    # With custom lookback period
    python -m src.runners.strategy_compare_runner \
        --symbols SPY QQQ AAPL --years 2 --output /tmp/signal_test/strategies.html
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal generators to compare (name -> (module_path, class_name, default_params, tier))
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY: Dict[str, Tuple[str, str, Dict[str, Any], str]] = {
    "pulse_dip": (
        "src.domain.strategy.signals.pulse_dip",
        "PulseDipSignalGenerator",
        {
            "ema_trend_period": 99,
            "rsi_period": 14,
            "rsi_entry_threshold": 45.0,
            "atr_stop_mult": 3.0,
            "max_hold_bars": 40,
            "hard_stop_pct": 0.08,
        },
        "TIER 1",
    ),
    "squeeze_play": (
        "src.domain.strategy.signals.squeeze_play",
        "SqueezePlaySignalGenerator",
        {
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_multiplier": 1.5,
            "release_persist_bars": 2,
            "close_outside_bars": 2,
            "atr_stop_mult": 2.5,
            "adx_min": 20.0,
            "hard_stop_pct": 0.08,
        },
        "TIER 1",
    ),
    "trend_pulse": (
        "src.domain.strategy.signals.trend_pulse",
        "TrendPulseSignalGenerator",
        {
            "zig_threshold_pct": 3.5,
            "trend_strength_moderate": 0.2,
            "trend_strength_strong": 0.6,
            "min_confidence": 0.3,
            "atr_stop_mult": 3.0,
            "exit_bearish_bars": 3,
            "enable_trend_reentry": True,
            "enable_chop_filter": True,
            "adx_entry_min": 18.0,
            "cooldown_bars": 3,
            "signal_shift_bars": 1,
        },
        "TIER 1",
    ),
    "buy_and_hold": (
        "src.domain.strategy.signals.buy_and_hold",
        "BuyAndHoldSignalGenerator",
        {},
        "BASELINE",
    ),
}

# Stress windows for survivability testing
STRESS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "covid_crash": ("2020-02-19", "2020-04-30"),
    "bear_2022": ("2022-01-03", "2022-10-13"),
    "ai_meltup_2023": ("2023-01-01", "2023-09-30"),
    "regional_bank_2023": ("2023-03-08", "2023-03-24"),
    "aug_2024_unwind": ("2024-07-10", "2024-08-15"),
}


def _download_data(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.isoformat(), end=end.isoformat(), interval="1d")
    if df.empty:
        logger.warning(f"No data for {symbol}")
        return df
    df.columns = [c.lower() for c in df.columns]
    # Keep only OHLCV
    cols = ["open", "high", "low", "close", "volume"]
    return df[[c for c in cols if c in df.columns]]


def _compute_regime_series(data: pd.DataFrame) -> pd.Series:
    """
    Compute simple SMA-based regime classification.

    R0: Healthy Uptrend  (close > SMA200, SMA50 > SMA200)
    R1: Choppy/Extended  (close > SMA200, SMA50 <= SMA200)
    R2: Risk-Off         (close < SMA200, SMA50 < SMA200)
    R3: Rebound Window   (close < SMA200, SMA50 >= SMA200)
    """
    close = data["close"]
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()

    regime = pd.Series("R1", index=data.index)
    regime[(close > sma200) & (sma50 > sma200)] = "R0"
    regime[(close > sma200) & (sma50 <= sma200)] = "R1"
    regime[(close < sma200) & (sma50 < sma200)] = "R2"
    regime[(close < sma200) & (sma50 >= sma200)] = "R3"
    return regime


def _run_strategy_on_symbol(
    generator: Any,
    data: pd.DataFrame,
    params: Dict[str, Any],
    regime_series: Optional[pd.Series] = None,
    init_cash: float = 100_000.0,
) -> Dict[str, Any]:
    """Run a signal generator on one symbol's data via VectorBT."""
    import vectorbt as vbt

    if len(data) < generator.warmup_bars + 10:
        return {}

    entries, exits = generator.generate(data, params)
    close = data["close"]

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=0.001,
        slippage=0.0005,
        freq="1D",
    )

    # Extract key metrics
    try:
        stats = pf.stats()

        def safe_get(key: str, default: float = 0.0) -> float:
            try:
                val = stats.get(key, default)
                if pd.isna(val):
                    return default
                fval = float(val)
                return default if np.isinf(fval) else fval
            except (TypeError, ValueError):
                return default

        returns_series = pf.returns()
        equity = pf.value()

        # -- Per-regime metrics --
        per_regime_sharpe: Dict[str, float] = {}
        per_regime_return: Dict[str, float] = {}
        if regime_series is not None:
            aligned_regime = regime_series.reindex(returns_series.index, method="ffill")
            for regime_label in ["R0", "R1", "R2", "R3"]:
                mask = aligned_regime == regime_label
                regime_rets = returns_series[mask]
                if len(regime_rets) > 20:
                    mean_r = float(regime_rets.mean())
                    std_r = float(regime_rets.std())
                    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
                    total_r = float((1 + regime_rets).prod() - 1)
                    if not np.isinf(sharpe):
                        per_regime_sharpe[regime_label] = round(sharpe, 3)
                    per_regime_return[regime_label] = round(total_r, 4)

        # -- Monthly returns --
        monthly_returns: Dict[str, float] = {}
        if len(returns_series) > 0:
            monthly = (1 + returns_series).resample("ME").prod() - 1
            for dt, val in monthly.items():
                if not pd.isna(val):
                    monthly_returns[dt.strftime("%Y-%m")] = round(float(val), 4)

        # -- Stress window results --
        stress_results: Dict[str, Dict[str, float]] = {}
        for window_name, (start_str, end_str) in STRESS_WINDOWS.items():
            start_dt = pd.Timestamp(start_str, tz=equity.index.tz)
            end_dt = pd.Timestamp(end_str, tz=equity.index.tz)
            mask = (equity.index >= start_dt) & (equity.index <= end_dt)
            window_equity = equity[mask]
            if len(window_equity) > 3:
                total_ret = float(window_equity.iloc[-1] / window_equity.iloc[0]) - 1
                peak = window_equity.expanding().max()
                dd = float(((window_equity - peak) / peak).min())
                stress_results[window_name] = {
                    "total_return": round(total_ret, 4),
                    "max_drawdown": round(dd, 4),
                }

        # -- Rolling Sharpe (60-day) --
        rolling_sharpe: List[List[float]] = []
        if len(returns_series) > 60:
            roll_mean = returns_series.rolling(60).mean()
            roll_std = returns_series.rolling(60).std()
            roll_sharpe = (roll_mean / roll_std * np.sqrt(252)).dropna()
            for ts, val in zip(roll_sharpe.index, roll_sharpe.values):
                fval = float(val)
                if not np.isnan(fval) and not np.isinf(fval):
                    rolling_sharpe.append([int(ts.timestamp()), fval])

        # Timestamps in SECONDS (template multiplies by 1000 for JS Date)
        return {
            "total_return": safe_get("Total Return [%]", 0) / 100,
            "sharpe": safe_get("Sharpe Ratio", 0),
            "sortino": safe_get("Sortino Ratio", 0),
            "calmar": safe_get("Calmar Ratio", 0),
            "max_drawdown": safe_get("Max Drawdown [%]", 0) / 100,
            "win_rate": safe_get("Win Rate [%]", 0) / 100,
            "profit_factor": safe_get("Profit Factor", 0),
            "total_trades": int(safe_get("Total Trades", 0)),
            "equity_values": equity.values.tolist(),
            "equity_index": [int(ts.timestamp()) for ts in equity.index],
            "returns": returns_series.values.tolist(),
            "per_regime_sharpe": per_regime_sharpe,
            "per_regime_return": per_regime_return,
            "monthly_returns": monthly_returns,
            "stress_results": stress_results,
            "rolling_sharpe": rolling_sharpe,
        }
    except Exception as e:
        logger.warning(f"Metrics extraction failed: {e}")
        return {}


def _aggregate_results(
    per_symbol: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-symbol results into strategy-level metrics."""
    valid = {s: m for s, m in per_symbol.items() if m}
    if not valid:
        return {}

    # Average core metrics across symbols
    sharpes = [m["sharpe"] for m in valid.values()]
    sortinos = [m["sortino"] for m in valid.values()]
    calmars = [m["calmar"] for m in valid.values()]
    returns = [m["total_return"] for m in valid.values()]
    drawdowns = [m["max_drawdown"] for m in valid.values()]
    win_rates = [m["win_rate"] for m in valid.values()]
    pf_factors = [m["profit_factor"] for m in valid.values()]
    trade_counts = [m["total_trades"] for m in valid.values()]

    # Build composite equity curve (equal-weighted average across all symbols)
    eq_curve: List[List[float]] = []
    dd_curve: List[List[float]] = []
    # Collect normalized equity series from each symbol
    norm_series: Dict[str, pd.Series] = {}
    for sym, m in valid.items():
        if m.get("equity_values") and m.get("equity_index"):
            eq_vals = m["equity_values"]
            eq_idx = m["equity_index"]
            if eq_vals and eq_vals[0] > 0:
                base = eq_vals[0]
                idx = pd.Index([pd.Timestamp(t, unit="s") for t in eq_idx])
                norm_series[sym] = pd.Series([(v / base) * 100 for v in eq_vals], index=idx)
    if norm_series:
        # Average across symbols at each timestamp (forward-fill gaps)
        combined = pd.DataFrame(norm_series)
        combined = combined.ffill().bfill()
        avg_equity = combined.mean(axis=1)
        for ts, val in zip(avg_equity.index, avg_equity.values):
            eq_curve.append([int(ts.timestamp()), float(val)])
        # Drawdown from composite equity
        peak = 0.0
        for ts, val in zip(avg_equity.index, avg_equity.values):
            peak = max(peak, float(val))
            dd = (float(val) - peak) / peak if peak > 0 else 0
            dd_curve.append([int(ts.timestamp()), dd])

    per_symbol_sharpe = {s: round(m["sharpe"], 3) for s, m in valid.items()}

    # -- Aggregate per-regime metrics (average across symbols) --
    per_regime_sharpe: Dict[str, float] = {}
    per_regime_return: Dict[str, float] = {}
    for regime_label in ["R0", "R1", "R2", "R3"]:
        regime_sharpes = [
            m["per_regime_sharpe"][regime_label]
            for m in valid.values()
            if regime_label in m.get("per_regime_sharpe", {})
        ]
        regime_returns = [
            m["per_regime_return"][regime_label]
            for m in valid.values()
            if regime_label in m.get("per_regime_return", {})
        ]
        if regime_sharpes:
            per_regime_sharpe[regime_label] = round(float(np.mean(regime_sharpes)), 3)
        if regime_returns:
            per_regime_return[regime_label] = round(float(np.mean(regime_returns)), 4)

    # -- Aggregate stress results (average across symbols) --
    stress_results: Dict[str, Dict[str, float]] = {}
    for window_name in STRESS_WINDOWS:
        window_returns = []
        window_dds = []
        for m in valid.values():
            sr = m.get("stress_results", {}).get(window_name)
            if sr:
                window_returns.append(sr["total_return"])
                window_dds.append(sr["max_drawdown"])
        if window_returns:
            stress_results[window_name] = {
                "total_return": round(float(np.mean(window_returns)), 4),
                "max_drawdown": round(float(np.mean(window_dds)), 4),
            }

    # -- Aggregate monthly returns (average across symbols) --
    monthly_returns: Dict[str, float] = {}
    all_months: set[str] = set()
    for m in valid.values():
        all_months.update(m.get("monthly_returns", {}).keys())
    for month in sorted(all_months):
        vals = [
            m["monthly_returns"][month]
            for m in valid.values()
            if month in m.get("monthly_returns", {})
        ]
        if vals:
            monthly_returns[month] = round(float(np.mean(vals)), 4)

    # -- Rolling Sharpe: average across all symbols --
    rolling_sharpe: List[List[float]] = []
    rs_series: Dict[str, pd.Series] = {}
    for sym, m in valid.items():
        if m.get("rolling_sharpe"):
            rs_data = m["rolling_sharpe"]
            idx = pd.Index([pd.Timestamp(p[0], unit="s") for p in rs_data])
            rs_series[sym] = pd.Series([p[1] for p in rs_data], index=idx)
    if rs_series:
        combined_rs = pd.DataFrame(rs_series).ffill().bfill()
        avg_rs = combined_rs.mean(axis=1)
        for ts, val in zip(avg_rs.index, avg_rs.values):
            fval = float(val)
            if not np.isnan(fval) and not np.isinf(fval):
                rolling_sharpe.append([int(ts.timestamp()), fval])

    return {
        "sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
        "sortino": float(np.mean(sortinos)) if sortinos else 0.0,
        "calmar": float(np.mean(calmars)) if calmars else 0.0,
        "total_return": float(np.mean(returns)) if returns else 0.0,
        "max_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
        "win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
        "profit_factor": float(np.mean(pf_factors)) if pf_factors else 0.0,
        "trade_count": int(np.sum(trade_counts)) if trade_counts else 0,
        "equity_curve": eq_curve,
        "drawdown_curve": dd_curve,
        "per_symbol_sharpe": per_symbol_sharpe,
        "per_regime_sharpe": per_regime_sharpe,
        "per_regime_return": per_regime_return,
        "stress_results": stress_results,
        "monthly_returns": monthly_returns,
        "rolling_sharpe": rolling_sharpe,
    }


def run_comparison(
    symbols: List[str],
    output_path: str,
    years: int = 3,
    strategies: List[str] | None = None,
) -> str:
    """
    Run strategy comparison and generate HTML dashboard.

    Args:
        symbols: List of ticker symbols.
        output_path: Path for strategies.html output.
        years: Lookback period in years.
        strategies: Strategy names to compare (default: all registered).

    Returns:
        Path to generated HTML file.
    """
    from src.infrastructure.reporting.strategy_comparison.builder import (
        StrategyComparisonBuilder,
        StrategyMetrics,
    )

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)

    strategy_names = strategies or list(STRATEGY_REGISTRY.keys())

    print(f"Strategy comparison: {len(strategy_names)} strategies x {len(symbols)} symbols")
    print(f"Period: {start_date} to {end_date} ({years}yr)")

    # Download data for all symbols
    print("Downloading data...")
    all_data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = _download_data(symbol, start_date, end_date)
            if not df.empty:
                all_data[symbol] = df
                print(f"  {symbol}: {len(df)} bars")
            else:
                print(f"  {symbol}: no data (skipped)")
        except Exception as e:
            print(f"  {symbol}: download failed ({e})")

    if not all_data:
        print("ERROR: No data available for any symbol")
        return output_path

    # Compute regime series from SPY (or first available symbol)
    regime_symbol = "SPY" if "SPY" in all_data else list(all_data.keys())[0]
    regime_series = _compute_regime_series(all_data[regime_symbol])
    print(f"\nRegime computed from {regime_symbol}:")
    regime_counts = regime_series.value_counts()
    for r in ["R0", "R1", "R2", "R3"]:
        count = regime_counts.get(r, 0)
        print(f"  {r}: {count} days ({count / len(regime_series) * 100:.0f}%)")

    # Run each strategy
    builder = StrategyComparisonBuilder(
        title="APEX Strategy Comparison Dashboard",
        universe_name=f"{len(all_data)} symbols",
        period=f"{start_date} to {end_date}",
    )
    builder.set_symbols(list(all_data.keys()))

    for strat_name in strategy_names:
        if strat_name not in STRATEGY_REGISTRY:
            print(f"  Unknown strategy: {strat_name} (skipped)")
            continue

        module_path, class_name, default_params, tier = STRATEGY_REGISTRY[strat_name]
        print(f"\nRunning {strat_name}...")

        # Import and instantiate
        from importlib import import_module

        mod = import_module(module_path)
        gen_cls = getattr(mod, class_name)
        generator = gen_cls()

        # Run on each symbol
        per_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, data in all_data.items():
            result = _run_strategy_on_symbol(
                generator, data, default_params, regime_series=regime_series
            )
            per_symbol[symbol] = result
            if result:
                print(
                    f"  {symbol}: Return={result['total_return']:+.1%} "
                    f"Sharpe={result['sharpe']:.2f} "
                    f"Trades={result['total_trades']}"
                )
            else:
                print(f"  {symbol}: insufficient data")

        # Aggregate
        agg = _aggregate_results(per_symbol)
        if not agg:
            print(f"  {strat_name}: no valid results")
            continue

        metrics = StrategyMetrics(
            name=strat_name,
            tier=tier,
            sharpe=agg["sharpe"],
            sortino=agg["sortino"],
            calmar=agg["calmar"],
            total_return=agg["total_return"],
            max_drawdown=agg["max_drawdown"],
            win_rate=agg["win_rate"],
            profit_factor=agg["profit_factor"],
            trade_count=agg["trade_count"],
            equity_curve=agg["equity_curve"],
            drawdown_curve=agg["drawdown_curve"],
            per_symbol_sharpe=agg["per_symbol_sharpe"],
            per_regime_sharpe=agg["per_regime_sharpe"],
            per_regime_return=agg["per_regime_return"],
            stress_results=agg["stress_results"],
            monthly_returns=agg["monthly_returns"],
            rolling_sharpe=agg["rolling_sharpe"],
        )
        builder.add_strategy(strat_name, metrics)
        print(
            f"  => Avg Sharpe={agg['sharpe']:.2f} "
            f"Return={agg['total_return']:+.1%} "
            f"MaxDD={agg['max_drawdown']:.1%}"
        )

    # Build dashboard
    result_path = builder.build(output_path)
    print(f"\nDashboard written to {result_path}")
    return result_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run strategy comparison and generate dashboard",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMZN", "JPM", "XOM", "UNH", "HD"],
        help="Symbols to test (default: 10 diverse stocks)",
    )
    parser.add_argument(
        "--output",
        default="out/signals/strategies.html",
        help="Output path for comparison dashboard",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Lookback period in years (default: 3)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="Strategies to compare (default: all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_comparison(
        symbols=args.symbols,
        output_path=args.output,
        years=args.years,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    main()
