"""
Strategy Comparison Runner.

Downloads daily OHLCV data and runs ALL strategies through BacktestEngine
(event-driven), then generates a comparison dashboard (strategies.html).

All 6 strategies (Tier 1 + Tier 2) use the same engine for apples-to-apples
comparison. VectorBT is NOT used here — it stays for Optuna optimization only.

Usage:
    # With explicit symbols
    python -m src.runners.strategy_compare_runner \
        --symbols SPY QQQ AAPL NVDA --output out/signals/strategies.html

    # With universe config (loads all symbols from YAML)
    python -m src.runners.strategy_compare_runner \
        --universe config/universe.yaml --output out/signals/strategies.html

    # With custom lookback period
    python -m src.runners.strategy_compare_runner \
        --symbols SPY QQQ AAPL --years 2 --output /tmp/signal_test/strategies.html
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from src.domain.signals.indicators.regime.models import MarketRegime, RegimeOutput
from src.domain.strategy.param_loader import get_strategy_metadata, list_strategies
from src.domain.strategy.providers import RegimeProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal generators to compare — loaded from config/strategy/*.yaml
# Each entry: name -> (module_path, class_name, default_params, tier)
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY: Dict[str, Tuple[str, str, Dict[str, Any], str]] = {
    name: get_strategy_metadata(name) for name in list_strategies()
}

# Stress windows for survivability testing
STRESS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "covid_crash": ("2020-02-19", "2020-04-30"),
    "bear_2022": ("2022-01-03", "2022-10-13"),
    "ai_meltup_2023": ("2023-01-01", "2023-09-30"),
    "regional_bank_2023": ("2023-03-08", "2023-03-24"),
    "aug_2024_unwind": ("2024-07-10", "2024-08-15"),
}

# Map regime string labels to MarketRegime enum
_REGIME_MAP: Dict[str, MarketRegime] = {
    "R0": MarketRegime.R0_HEALTHY_UPTREND,
    "R1": MarketRegime.R1_CHOPPY_EXTENDED,
    "R2": MarketRegime.R2_RISK_OFF,
    "R3": MarketRegime.R3_REBOUND_WINDOW,
}


class SeriesRegimeProvider:
    """
    Regime provider backed by a pre-computed pd.Series of regime labels.

    Used by the comparison runner to provide regime context to strategies
    running in BacktestEngine without needing the full regime detector.

    The provider maintains a current timestamp that the compare runner
    advances as bars are processed. get_market_regime() looks up the
    regime label at the current timestamp.
    """

    def __init__(self, regime_series: pd.Series) -> None:
        """
        Args:
            regime_series: pd.Series indexed by datetime with values "R0"-"R3".
        """
        self._regime_series = regime_series
        self._current_regime: Optional[MarketRegime] = None

    def advance_to(self, timestamp: datetime) -> None:
        """Advance the provider to a given timestamp (look up regime label)."""
        # Find the regime at or before this timestamp via forward-fill
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None and self._regime_series.index.tz is not None:
            ts = ts.tz_localize(self._regime_series.index.tz)
        elif ts.tzinfo is not None and self._regime_series.index.tz is None:
            ts = ts.tz_localize(None)

        # Use searchsorted to find the nearest prior index
        idx = self._regime_series.index.searchsorted(ts, side="right") - 1
        if idx >= 0:
            label = str(self._regime_series.iloc[idx])
            self._current_regime = _REGIME_MAP.get(label)
        else:
            self._current_regime = None

    def get_regime(self, symbol: str) -> Optional[RegimeOutput]:
        """Get regime output for a symbol (uses market-level regime)."""
        if self._current_regime is None:
            return None
        return RegimeOutput(final_regime=self._current_regime)

    def get_market_regime(self) -> Optional[MarketRegime]:
        """Get current market-level regime."""
        return self._current_regime


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


def _get_strategy_class(strategy_name: str) -> Optional[Type]:
    """
    Look up a Strategy class from the @register_strategy registry.

    Imports the playbook package first to ensure all strategies are registered.
    """
    # Ensure playbook strategies are registered
    import src.domain.strategy.playbook  # noqa: F401
    from src.domain.strategy.registry import get_strategy_class

    return get_strategy_class(strategy_name)


async def _run_strategy_event_driven(
    strategy_name: str,
    strategy_class: Type,
    data: pd.DataFrame,
    params: Dict[str, Any],
    regime_series: pd.Series,
    init_cash: float = 100_000.0,
) -> Dict[str, Any]:
    """
    Run a strategy on one symbol's data via BacktestEngine (event-driven).

    This replaces the VectorBT path for consistent metrics across all strategies.

    Args:
        strategy_name: Name of the strategy (for logging).
        strategy_class: Strategy class from @register_strategy registry.
        data: OHLCV DataFrame for one symbol.
        params: Strategy parameters from YAML.
        regime_series: Pre-computed regime labels (from SPY).
        init_cash: Initial capital.

    Returns:
        Metrics dict compatible with _aggregate_results(), or {} on failure.
    """
    from src.backtest.data.feeds.memory_feeds import InMemoryDataFeed
    from src.backtest.execution.engines.backtest_engine import BacktestConfig, BacktestEngine

    if len(data) < 260:
        return {}

    symbol = data.attrs.get("symbol", "UNKNOWN")

    # Build InMemoryDataFeed from DataFrame
    feed = InMemoryDataFeed()
    for ts, row in data.iterrows():
        bar_ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if hasattr(bar_ts, "tzinfo") and bar_ts.tzinfo is not None:
            bar_ts = bar_ts.replace(tzinfo=None)
        feed.add_bar(
            symbol=symbol,
            timestamp=bar_ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0)),
        )

    # Create regime provider
    regime_provider = SeriesRegimeProvider(regime_series)

    # Configure BacktestEngine
    start_date = data.index[0].date() if hasattr(data.index[0], "date") else data.index[0]
    end_date = data.index[-1].date() if hasattr(data.index[-1], "date") else data.index[-1]

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=[symbol],
        initial_capital=init_cash,
        strategy_name=strategy_name,
    )

    engine = BacktestEngine(config)

    # Wire regime provider into context
    engine._context.regime_provider = regime_provider

    # Set strategy and data feed
    engine.set_strategy(strategy_class=strategy_class, strategy_name=strategy_name, params=params)
    engine.set_data_feed(feed)

    # Hook: advance regime provider when clock advances
    original_advance = engine._clock.advance_to

    def _advance_with_regime(new_time: datetime) -> int:
        regime_provider.advance_to(new_time)
        return original_advance(new_time)

    engine._clock.advance_to = _advance_with_regime  # type: ignore[assignment]

    # Run backtest
    try:
        result = await engine.run()
    except Exception as e:
        logger.warning(f"BacktestEngine failed for {strategy_name}/{symbol}: {e}")
        return {}

    # Extract metrics into the same dict format as _aggregate_results expects
    return _extract_metrics(result, regime_series, init_cash)


def _extract_metrics(
    result: Any,
    regime_series: pd.Series,
    init_cash: float,
) -> Dict[str, Any]:
    """
    Extract metrics from BacktestResult into the dashboard-compatible dict.

    Maps BacktestResult fields to the same structure that _aggregate_results
    and StrategyComparisonBuilder expect.
    """
    perf = result.performance
    risk = result.risk
    trades = result.trades

    if perf is None or risk is None or trades is None:
        return {}

    total_return = perf.total_return
    sharpe = risk.sharpe_ratio
    max_dd = -(risk.max_drawdown / 100.0) if risk.max_drawdown > 0 else risk.max_drawdown / 100.0
    win_rate = (trades.win_rate / 100.0) if trades.win_rate > 1 else trades.win_rate
    total_trades = trades.total_trades

    # Build equity values/index from equity_curve
    equity_values: List[float] = []
    equity_index: List[int] = []
    for point in result.equity_curve:
        equity_values.append(point["equity"])
        # Parse date string to timestamp
        dt = (
            datetime.fromisoformat(point["date"])
            if isinstance(point["date"], str)
            else point["date"]
        )
        if isinstance(dt, date) and not isinstance(dt, datetime):
            dt = datetime.combine(dt, datetime.min.time())
        equity_index.append(int(dt.timestamp()))

    # Build returns series from equity
    returns_list: List[float] = []
    for i in range(1, len(equity_values)):
        if equity_values[i - 1] > 0:
            returns_list.append(equity_values[i] / equity_values[i - 1] - 1)
        else:
            returns_list.append(0.0)

    # Per-regime metrics
    per_regime_sharpe: Dict[str, float] = {}
    per_regime_return: Dict[str, float] = {}
    if len(returns_list) > 0 and len(equity_index) > 1:
        ret_idx = pd.Index([pd.Timestamp(t, unit="s") for t in equity_index[1:]])
        returns_series = pd.Series(returns_list, index=ret_idx)
        aligned_regime = regime_series.reindex(returns_series.index, method="ffill")
        for regime_label in ["R0", "R1", "R2", "R3"]:
            mask = aligned_regime == regime_label
            regime_rets = returns_series[mask]
            if len(regime_rets) > 20:
                mean_r = float(regime_rets.mean())
                std_r = float(regime_rets.std())
                s = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
                total_r = float((1 + regime_rets).prod() - 1)
                if not np.isinf(s):
                    per_regime_sharpe[regime_label] = round(s, 3)
                per_regime_return[regime_label] = round(total_r, 4)

    # Monthly returns
    monthly_returns: Dict[str, float] = {}
    if len(returns_list) > 0 and len(equity_index) > 1:
        ret_idx = pd.Index([pd.Timestamp(t, unit="s") for t in equity_index[1:]])
        returns_series = pd.Series(returns_list, index=ret_idx)
        monthly = (1 + returns_series).resample("ME").prod() - 1
        for dt, val in monthly.items():
            if not pd.isna(val):
                monthly_returns[dt.strftime("%Y-%m")] = round(float(val), 4)

    # Stress window results
    stress_results: Dict[str, Dict[str, float]] = {}
    if equity_values and equity_index:
        eq_idx_pd = pd.Index([pd.Timestamp(t, unit="s") for t in equity_index])
        eq_series = pd.Series(equity_values, index=eq_idx_pd)
        for window_name, (start_str, end_str) in STRESS_WINDOWS.items():
            start_dt = pd.Timestamp(start_str)
            end_dt = pd.Timestamp(end_str)
            if eq_idx_pd.tz is not None:
                start_dt = start_dt.tz_localize(eq_idx_pd.tz)
                end_dt = end_dt.tz_localize(eq_idx_pd.tz)
            mask = (eq_series.index >= start_dt) & (eq_series.index <= end_dt)
            window_equity = eq_series[mask]
            if len(window_equity) > 3:
                total_ret = float(window_equity.iloc[-1] / window_equity.iloc[0]) - 1
                peak = window_equity.expanding().max()
                dd = float(((window_equity - peak) / peak).min())
                stress_results[window_name] = {
                    "total_return": round(total_ret, 4),
                    "max_drawdown": round(dd, 4),
                }

    # Rolling Sharpe (60-day)
    rolling_sharpe: List[List[float]] = []
    if len(returns_list) > 60:
        ret_idx = pd.Index([pd.Timestamp(t, unit="s") for t in equity_index[1:]])
        returns_series = pd.Series(returns_list, index=ret_idx)
        roll_mean = returns_series.rolling(60).mean()
        roll_std = returns_series.rolling(60).std()
        roll_sharpe = (roll_mean / roll_std * np.sqrt(252)).dropna()
        for ts, val in zip(roll_sharpe.index, roll_sharpe.values):
            fval = float(val)
            if not np.isnan(fval) and not np.isinf(fval):
                rolling_sharpe.append([int(ts.timestamp()), fval])

    # Sortino (from daily returns)
    sortino = 0.0
    if len(returns_list) > 20:
        ret_arr = np.array(returns_list)
        mean_r = float(ret_arr.mean())
        downside = ret_arr[ret_arr < 0]
        if len(downside) > 0:
            downside_std = float(np.std(downside))
            if downside_std > 0:
                sortino = mean_r / downside_std * np.sqrt(252)

    # Calmar
    calmar = 0.0
    if max_dd != 0:
        calmar = abs(total_return / max_dd)

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": trades.profit_factor,
        "total_trades": total_trades,
        "equity_values": equity_values,
        "equity_index": equity_index,
        "returns": returns_list,
        "per_regime_sharpe": per_regime_sharpe,
        "per_regime_return": per_regime_return,
        "monthly_returns": monthly_returns,
        "stress_results": stress_results,
        "rolling_sharpe": rolling_sharpe,
    }


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

    # Slim per-symbol metrics for Per-Stock tab (no equity curves)
    per_symbol_metrics: Dict[str, Dict[str, float]] = {
        s: {
            "sharpe": round(m["sharpe"], 3),
            "total_return": round(m["total_return"], 4),
            "max_drawdown": round(m["max_drawdown"], 4),
            "win_rate": round(m["win_rate"], 3),
            "total_trades": m["total_trades"],
            "sortino": round(m["sortino"], 3),
            "calmar": round(m["calmar"], 3),
            "profit_factor": round(m["profit_factor"], 3),
        }
        for s, m in valid.items()
    }

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
        "per_symbol_metrics": per_symbol_metrics,
        "per_regime_sharpe": per_regime_sharpe,
        "per_regime_return": per_regime_return,
        "stress_results": stress_results,
        "monthly_returns": monthly_returns,
        "rolling_sharpe": rolling_sharpe,
    }


def _load_sector_map(universe_path: str) -> Dict[str, List[str]]:
    """Parse sector mapping from universe YAML.

    Returns {sector_name: [etf, stock1, stock2, ...]}.
    """
    from pathlib import Path

    import yaml

    data = yaml.safe_load(Path(universe_path).read_text())
    sectors = data.get("sectors", {})
    result: Dict[str, List[str]] = {}
    for sector_name, sector_data in sectors.items():
        symbols: List[str] = []
        etf = sector_data.get("etf")
        if etf:
            symbols.append(etf)
        symbols.extend(sector_data.get("stocks", []))
        result[sector_name] = symbols
    return result


def run_comparison(
    symbols: List[str],
    output_path: str,
    years: int = 3,
    strategies: List[str] | None = None,
    universe_path: str | None = None,
) -> str:
    """
    Run strategy comparison and generate HTML dashboard.

    All strategies run through BacktestEngine (event-driven) for consistent
    metrics. No VectorBT in the comparison path.

    Args:
        symbols: List of ticker symbols.
        output_path: Path for strategies.html output.
        years: Lookback period in years.
        strategies: Strategy names to compare (default: all registered).
        universe_path: Path to universe YAML (for sector mapping).

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
    print("Engine: BacktestEngine (event-driven, all strategies)")

    # Download data for all symbols
    print("Downloading data...")
    all_data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = _download_data(symbol, start_date, end_date)
            if not df.empty:
                # Store symbol name in DataFrame attrs for downstream use
                df.attrs["symbol"] = symbol
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

    # Load sector map for sector-level aggregation
    if universe_path:
        sector_map = _load_sector_map(universe_path)
    else:
        sector_map = {"all": list(all_data.keys())}
    builder.set_sector_map(sector_map)

    for strat_name in strategy_names:
        if strat_name not in STRATEGY_REGISTRY:
            print(f"  Unknown strategy: {strat_name} (skipped)")
            continue

        _module_path, _class_name, default_params, tier = STRATEGY_REGISTRY[strat_name]
        print(f"\nRunning {strat_name} (event-driven)...")

        # Look up Strategy class from @register_strategy registry
        strategy_class = _get_strategy_class(strat_name)
        if strategy_class is None:
            print(f"  {strat_name}: not found in strategy registry (skipped)")
            continue

        # Run on each symbol
        per_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, data in all_data.items():
            result = asyncio.run(
                _run_strategy_event_driven(
                    strategy_name=strat_name,
                    strategy_class=strategy_class,
                    data=data,
                    params=default_params,
                    regime_series=regime_series,
                )
            )
            per_symbol[symbol] = result
            if result:
                print(
                    f"  {symbol}: Return={result['total_return']:+.1%} "
                    f"Sharpe={result['sharpe']:.2f} "
                    f"Trades={result['total_trades']}"
                )
            else:
                print(f"  {symbol}: insufficient data or backtest failed")

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
            per_symbol_metrics=agg["per_symbol_metrics"],
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
        default=None,
        help="Symbols to test (default: 10 diverse stocks if --universe not given)",
    )
    parser.add_argument(
        "--universe",
        help="Path to universe YAML config (mutually exclusive with --symbols)",
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

    if args.symbols and args.universe:
        parser.error("--symbols and --universe are mutually exclusive")

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve symbols: --universe > --symbols > default fallback
    if args.universe:
        from pathlib import Path

        from src.services.market_cap_service import load_universe_symbols

        symbols = load_universe_symbols(Path(args.universe))
        print(f"Loaded {len(symbols)} symbols from {args.universe}")
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMZN", "JPM", "XOM", "UNH", "HD"]

    run_comparison(
        symbols=symbols,
        output_path=args.output,
        years=args.years,
        strategies=args.strategies,
        universe_path=args.universe,
    )


if __name__ == "__main__":
    main()
