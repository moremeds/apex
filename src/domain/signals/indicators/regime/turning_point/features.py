"""
Feature Extraction for Turning Point Detection (Phase 4).

Features are extracted from regime components:
- Trend: MA slopes, price vs MAs
- Volatility: ATR percentiles, vol expansion/contraction
- Chop: Choppiness percentiles, trending vs ranging
- Extension: ATR units from mean

All features use PAST data only to avoid leakage with forward-looking labels.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TurningPointFeatures:
    """
    Feature set for turning point model.

    All features are computed from historical data only.
    Features are designed to capture conditions that precede
    market turning points.
    """

    # Trend features
    price_vs_ma20: float = 0.0  # (close - MA20) / ATR
    price_vs_ma50: float = 0.0  # (close - MA50) / ATR
    price_vs_ma200: float = 0.0  # (close - MA200) / ATR
    ma20_slope: float = 0.0  # MA20 slope (normalized)
    ma50_slope: float = 0.0  # MA50 slope (normalized)
    ma20_vs_ma50: float = 0.0  # MA alignment

    # Volatility features
    atr_pct_63: float = 50.0  # ATR percentile (63-day)
    atr_pct_252: float = 50.0  # ATR percentile (252-day)
    atr_expansion_rate: float = 0.0  # Rate of change in ATR
    vol_regime: int = 0  # -1=low, 0=normal, 1=high

    # Chop/Range features
    chop_pct_252: float = 50.0  # Choppiness percentile
    adx_value: float = 0.0  # ADX if available
    range_position: float = 0.5  # Position in recent range [0, 1]

    # Extension features
    ext_atr_units: float = 0.0  # Distance from mean in ATR units
    ext_zscore: float = 0.0  # Z-score of price
    rsi_14: float = 50.0  # RSI for momentum

    # Rate of change features
    roc_5: float = 0.0  # 5-bar rate of change
    roc_10: float = 0.0  # 10-bar rate of change
    roc_20: float = 0.0  # 20-bar rate of change

    # Delta features (change from previous bar)
    delta_atr_pct: float = 0.0  # Change in ATR percentile
    delta_chop_pct: float = 0.0  # Change in chop percentile
    delta_ext: float = 0.0  # Change in extension

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array(
            [
                self.price_vs_ma20,
                self.price_vs_ma50,
                self.price_vs_ma200,
                self.ma20_slope,
                self.ma50_slope,
                self.ma20_vs_ma50,
                self.atr_pct_63,
                self.atr_pct_252,
                self.atr_expansion_rate,
                self.vol_regime,
                self.chop_pct_252,
                self.adx_value,
                self.range_position,
                self.ext_atr_units,
                self.ext_zscore,
                self.rsi_14,
                self.roc_5,
                self.roc_10,
                self.roc_20,
                self.delta_atr_pct,
                self.delta_chop_pct,
                self.delta_ext,
            ]
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "price_vs_ma20": self.price_vs_ma20,
            "price_vs_ma50": self.price_vs_ma50,
            "price_vs_ma200": self.price_vs_ma200,
            "ma20_slope": self.ma20_slope,
            "ma50_slope": self.ma50_slope,
            "ma20_vs_ma50": self.ma20_vs_ma50,
            "atr_pct_63": self.atr_pct_63,
            "atr_pct_252": self.atr_pct_252,
            "atr_expansion_rate": self.atr_expansion_rate,
            "vol_regime": float(self.vol_regime),
            "chop_pct_252": self.chop_pct_252,
            "adx_value": self.adx_value,
            "range_position": self.range_position,
            "ext_atr_units": self.ext_atr_units,
            "ext_zscore": self.ext_zscore,
            "rsi_14": self.rsi_14,
            "roc_5": self.roc_5,
            "roc_10": self.roc_10,
            "roc_20": self.roc_20,
            "delta_atr_pct": self.delta_atr_pct,
            "delta_chop_pct": self.delta_chop_pct,
            "delta_ext": self.delta_ext,
        }

    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names."""
        return [
            "price_vs_ma20",
            "price_vs_ma50",
            "price_vs_ma200",
            "ma20_slope",
            "ma50_slope",
            "ma20_vs_ma50",
            "atr_pct_63",
            "atr_pct_252",
            "atr_expansion_rate",
            "vol_regime",
            "chop_pct_252",
            "adx_value",
            "range_position",
            "ext_atr_units",
            "ext_zscore",
            "rsi_14",
            "roc_5",
            "roc_10",
            "roc_20",
            "delta_atr_pct",
            "delta_chop_pct",
            "delta_ext",
        ]


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling percentile rank of current value."""

    def pct_rank(x):
        if len(x) < 2:
            return 50.0
        rank = (x.iloc[-1] > x.iloc[:-1]).sum()
        return 100.0 * rank / (len(x) - 1)

    return series.rolling(window=window).apply(pct_rank, raw=False).fillna(50.0)


def compute_choppiness(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Choppiness Index."""
    atr = (
        pd.concat(
            [
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1)),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(window=period)
        .mean()
    )

    high_low_range = high.rolling(window=period).max() - low.rolling(window=period).min()

    # Choppiness Index formula
    chop = 100 * np.log10(atr * period / high_low_range.replace(0, np.nan)) / np.log10(period)
    return chop.fillna(50.0)


def extract_features(
    df: pd.DataFrame,
    prev_features: Optional[TurningPointFeatures] = None,
) -> pd.DataFrame:
    """
    Extract features from OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        prev_features: Previous bar's features for delta calculation

    Returns:
        DataFrame with one row per bar, feature columns
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Moving averages
    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()
    ma200 = close.rolling(window=200).mean()

    # ATR
    tr = pd.concat(
        [
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1)),
        ],
        axis=1,
    ).max(axis=1)
    atr20 = tr.rolling(window=20).mean()

    # Slopes
    ma20_slope = (ma20 - ma20.shift(5)) / (atr20 * 5).replace(0, np.nan)
    ma50_slope = (ma50 - ma50.shift(10)) / (atr20 * 10).replace(0, np.nan)

    # Price vs MAs (normalized by ATR)
    price_vs_ma20 = (close - ma20) / atr20.replace(0, np.nan)
    price_vs_ma50 = (close - ma50) / atr20.replace(0, np.nan)
    price_vs_ma200 = (close - ma200) / atr20.replace(0, np.nan)

    # MA alignment
    ma20_vs_ma50 = (ma20 - ma50) / atr20.replace(0, np.nan)

    # ATR percentiles
    atr_pct_63 = compute_percentile_rank(atr20, 63)
    atr_pct_252 = compute_percentile_rank(atr20, 252)

    # ATR expansion rate
    atr_expansion_rate = (atr20 - atr20.shift(5)) / atr20.shift(5).replace(0, np.nan)

    # Vol regime
    vol_regime = pd.Series(0, index=df.index)
    vol_regime = vol_regime.where(atr_pct_63 <= 80, 1)  # High vol
    vol_regime = vol_regime.where(atr_pct_63 >= 20, -1)  # Low vol

    # Choppiness
    chop = compute_choppiness(high, low, close)
    chop_pct_252 = compute_percentile_rank(chop, 252)

    # Range position
    high_20 = high.rolling(window=20).max()
    low_20 = low.rolling(window=20).min()
    range_position = (close - low_20) / (high_20 - low_20).replace(0, np.nan)

    # Extension
    mean_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    ext_atr_units = (close - mean_20) / atr20.replace(0, np.nan)
    ext_zscore = (close - mean_20) / std_20.replace(0, np.nan)

    # RSI
    rsi_14 = compute_rsi(close, 14)

    # Rate of change
    roc_5 = (close - close.shift(5)) / close.shift(5).replace(0, np.nan) * 100
    roc_10 = (close - close.shift(10)) / close.shift(10).replace(0, np.nan) * 100
    roc_20 = (close - close.shift(20)) / close.shift(20).replace(0, np.nan) * 100

    # Deltas
    delta_atr_pct = atr_pct_63.diff()
    delta_chop_pct = chop_pct_252.diff()
    delta_ext = ext_atr_units.diff()

    # Build feature DataFrame
    features = pd.DataFrame(
        {
            "price_vs_ma20": price_vs_ma20,
            "price_vs_ma50": price_vs_ma50,
            "price_vs_ma200": price_vs_ma200,
            "ma20_slope": ma20_slope,
            "ma50_slope": ma50_slope,
            "ma20_vs_ma50": ma20_vs_ma50,
            "atr_pct_63": atr_pct_63,
            "atr_pct_252": atr_pct_252,
            "atr_expansion_rate": atr_expansion_rate,
            "vol_regime": vol_regime,
            "chop_pct_252": chop_pct_252,
            "adx_value": pd.Series(0.0, index=df.index),  # Placeholder
            "range_position": range_position,
            "ext_atr_units": ext_atr_units,
            "ext_zscore": ext_zscore,
            "rsi_14": rsi_14,
            "roc_5": roc_5,
            "roc_10": roc_10,
            "roc_20": roc_20,
            "delta_atr_pct": delta_atr_pct,
            "delta_chop_pct": delta_chop_pct,
            "delta_ext": delta_ext,
        },
        index=df.index,
    )

    # Fill NaN with defaults
    features = features.fillna(0.0)

    return features


def extract_single_bar_features(
    close: float,
    high: float,
    low: float,
    ma20: float,
    ma50: float,
    ma200: float,
    atr20: float,
    chop: float,
    rsi: float,
    prev_close: float,
    prev_atr_pct: float = 50.0,
    prev_chop_pct: float = 50.0,
    prev_ext: float = 0.0,
    high_20: float = 0.0,
    low_20: float = 0.0,
    atr_pct_63: float = 50.0,
    atr_pct_252: float = 50.0,
    chop_pct_252: float = 50.0,
) -> TurningPointFeatures:
    """
    Extract features for a single bar from pre-computed indicators.

    Useful for real-time inference where indicators are already available.
    """
    # Normalize by ATR
    atr = atr20 if atr20 > 0 else 1e-6

    price_vs_ma20 = (close - ma20) / atr
    price_vs_ma50 = (close - ma50) / atr
    price_vs_ma200 = (close - ma200) / atr
    ma20_vs_ma50 = (ma20 - ma50) / atr

    # Extension
    mean_20 = ma20  # Approximate
    ext_atr_units = (close - mean_20) / atr

    # Range position
    if high_20 > low_20:
        range_position = (close - low_20) / (high_20 - low_20)
    else:
        range_position = 0.5

    # Vol regime
    if atr_pct_63 >= 80:
        vol_regime = 1
    elif atr_pct_63 <= 20:
        vol_regime = -1
    else:
        vol_regime = 0

    # ROC
    roc = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

    return TurningPointFeatures(
        price_vs_ma20=price_vs_ma20,
        price_vs_ma50=price_vs_ma50,
        price_vs_ma200=price_vs_ma200,
        ma20_slope=0.0,  # Would need history
        ma50_slope=0.0,  # Would need history
        ma20_vs_ma50=ma20_vs_ma50,
        atr_pct_63=atr_pct_63,
        atr_pct_252=atr_pct_252,
        atr_expansion_rate=0.0,  # Would need history
        vol_regime=vol_regime,
        chop_pct_252=chop_pct_252,
        adx_value=0.0,
        range_position=range_position,
        ext_atr_units=ext_atr_units,
        ext_zscore=0.0,  # Would need std
        rsi_14=rsi,
        roc_5=roc,
        roc_10=0.0,
        roc_20=0.0,
        delta_atr_pct=atr_pct_63 - prev_atr_pct,
        delta_chop_pct=chop_pct_252 - prev_chop_pct,
        delta_ext=ext_atr_units - prev_ext,
    )
