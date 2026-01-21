"""
Helper functions for M2 Validation Runner.

Provides data loading utilities for validation.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from ..domain.signals.validation.schemas import LabelerThreshold

logger = logging.getLogger(__name__)

# Bars per trading day by timeframe
BARS_PER_DAY: Dict[str, float] = {
    "1d": 1.0,
    "4h": 1.625,
    "2h": 3.25,
    "1h": 6.5,
    "30m": 13.0,
    "15m": 26.0,
    "5m": 78.0,
    "1m": 390.0,
}


def get_bars_per_day(tf: str) -> float:
    """Get bars per trading day for a timeframe."""
    return BARS_PER_DAY.get(tf, 1.0)


def load_universe_from_yaml(path: str) -> Dict[str, List[str]]:
    """
    Load universe from YAML file.

    Args:
        path: Path to universe YAML

    Returns:
        Dict with 'training_universe' and 'holdout_universe' lists
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        logger.warning(f"Universe file not found: {path}")
        return {"training_universe": [], "holdout_universe": []}

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    return {
        "training_universe": data.get("training_universe", []),
        "holdout_universe": data.get("holdout_universe", []),
    }


def load_labeler_thresholds_from_yaml(path: str) -> Dict[str, LabelerThreshold]:
    """
    Load labeler thresholds from YAML file.

    Args:
        path: Path to thresholds YAML

    Returns:
        Dict mapping timeframe -> LabelerThreshold
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        logger.warning(f"Thresholds file not found: {path}, using defaults")
        return _default_labeler_thresholds()

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    thresholds = {}
    for tf, config in data.get("timeframes", {}).items():
        thresholds[tf] = LabelerThreshold(
            version=data.get("version", "v1.0"),
            trending_forward_return_min=config.get("trending_forward_return_min", 0.10),
            trending_sharpe_min=config.get("trending_sharpe_min", 1.0),
            choppy_volatility_min=config.get("choppy_volatility_min", 0.25),
            choppy_drawdown_max=config.get("choppy_drawdown_max", -0.10),
        )

    return thresholds


def _default_labeler_thresholds() -> Dict[str, LabelerThreshold]:
    """Return default labeler thresholds."""
    return {
        "1d": LabelerThreshold(
            version="v1.0",
            trending_forward_return_min=0.10,
            trending_sharpe_min=1.0,
            choppy_volatility_min=0.25,
            choppy_drawdown_max=-0.10,
        ),
        "4h": LabelerThreshold(
            version="v1.0",
            trending_forward_return_min=0.06,
            trending_sharpe_min=0.8,
            choppy_volatility_min=0.30,
            choppy_drawdown_max=-0.08,
        ),
        "2h": LabelerThreshold(
            version="v1.0",
            trending_forward_return_min=0.04,
            trending_sharpe_min=0.6,
            choppy_volatility_min=0.35,
            choppy_drawdown_max=-0.06,
        ),
        "1h": LabelerThreshold(
            version="v1.0",
            trending_forward_return_min=0.03,
            trending_sharpe_min=0.5,
            choppy_volatility_min=0.40,
            choppy_drawdown_max=-0.05,
        ),
    }


def load_bars_yahoo(
    symbols: List[str],
    timeframe: str = "1d",
    days: int = 500,
    end_date: Optional[date] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load historical bars from Yahoo Finance.

    Args:
        symbols: List of symbols to load
        timeframe: Timeframe (1d, 4h, 1h)
        days: Number of calendar days of history
        end_date: End date (default: today)

    Returns:
        Dict mapping symbol -> OHLCV DataFrame
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return {}

    if end_date is None:
        end_date = date.today()

    start_date = end_date - timedelta(days=days)

    # Map timeframe to yfinance interval
    interval_map = {
        "1d": "1d",
        "4h": "1h",  # Aggregate 1h -> 4h
        "2h": "1h",  # Aggregate 1h -> 2h
        "1h": "1h",
    }
    interval = interval_map.get(timeframe, "1d")

    bars_by_symbol: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                interval=interval,
            )

            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            for col in ["dividends", "stock splits", "adj close"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Aggregate if needed
            if timeframe == "4h" and interval == "1h":
                df = _aggregate_bars(df, "4h")
            elif timeframe == "2h" and interval == "1h":
                df = _aggregate_bars(df, "2h")

            bars_by_symbol[symbol] = df
            logger.debug(f"Loaded {len(df)} {timeframe} bars for {symbol}")

        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")

    return bars_by_symbol


def _aggregate_bars(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Aggregate bars to target timeframe."""
    resample_map = {"4h": "4h", "2h": "2h"}
    rule = resample_map.get(target_tf, "4h")

    return (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


def generate_synthetic_bars(
    symbols: List[str],
    days: int = 500,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data for CI testing.

    Creates deterministic price series with both trending and choppy characteristics
    to exercise the validation logic without external API dependencies.

    Args:
        symbols: List of symbol names to generate
        days: Number of trading days to generate
        seed: Random seed for reproducibility

    Returns:
        Dict mapping symbol -> OHLCV DataFrame
    """
    import numpy as np

    np.random.seed(seed)

    bars_by_symbol: Dict[str, pd.DataFrame] = {}
    end_date = date.today()
    dates = pd.bdate_range(end=end_date, periods=days)

    for i, symbol in enumerate(symbols):
        # Use different seed per symbol for variety
        np.random.seed(seed + i)

        # Generate price series with mixed trending/choppy regimes
        n = len(dates)
        returns = np.zeros(n)

        # Alternate between trending and choppy regimes
        regime_length = n // 4
        for j in range(4):
            start_idx = j * regime_length
            end_idx = min((j + 1) * regime_length, n)

            if j % 2 == 0:
                # Trending regime: strong directional drift with low variance
                # Drift of 0.5% daily with 1% volatility gives clear trends
                drift = 0.005 if i % 2 == 0 else -0.005  # Uptrend/downtrend
                regime_returns = np.random.normal(drift, 0.01, end_idx - start_idx)
            else:
                # Choppy regime: zero mean with high variance (mean-reverting behavior)
                regime_returns = np.random.normal(0.0, 0.02, end_idx - start_idx)
                # Add mean-reversion to make it clearly non-trending
                for k in range(1, len(regime_returns)):
                    regime_returns[k] -= 0.3 * regime_returns[k - 1]

            returns[start_idx:end_idx] = regime_returns

        # Convert returns to prices
        base_price = 100.0 + i * 50  # Different base for each symbol
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        volatility = np.abs(returns) + 0.005
        high = prices * (1 + volatility * 0.5)
        low = prices * (1 - volatility * 0.5)
        open_prices = np.roll(prices, 1)
        open_prices[0] = base_price
        volume = np.random.randint(1_000_000, 10_000_000, n)

        df = pd.DataFrame(
            {
                "open": open_prices,
                "high": high,
                "low": low,
                "close": prices,
                "volume": volume,
            },
            index=dates,
        )
        df.index.name = "Date"

        bars_by_symbol[symbol] = df
        logger.debug(f"Generated {len(df)} synthetic bars for {symbol}")

    return bars_by_symbol
