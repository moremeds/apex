"""
Validation Service - Connects RegimeDetector with Labeler for validation.

Orchestrates the validation pipeline:
1. Load historical bars for symbols
2. Run RegimeDetector to get predictions
3. Run RegimeLabeler to get ground truth labels
4. Compute R0 rates and other metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..indicators.regime.models import MarketRegime
from ..indicators.regime.regime_detector import RegimeDetectorIndicator
from .confirmation import (
    ConfirmationResult,
    compare_strategies,
)
from .earliness import EarlinessResult, compute_earliness
from .labeler_contract import RegimeLabel, RegimeLabeler, RegimeLabelerConfig
from .statistics import StatisticalResult, SymbolMetrics, compute_symbol_level_stats

logger = logging.getLogger(__name__)


@dataclass
class SymbolValidationResult:
    """Validation result for a single symbol."""

    symbol: str
    timeframe: str
    n_bars: int
    n_trending_bars: int
    n_choppy_bars: int
    r0_rate_trending: float  # % of trending bars where detector predicted R0
    r0_rate_choppy: float  # % of choppy bars where detector predicted R0
    predictions: List[bool]  # True = R0 predicted
    labels: List[RegimeLabel]


@dataclass
class ValidationServiceConfig:
    """Configuration for validation service."""

    timeframes: List[str] = field(default_factory=lambda: ["1d"])
    horizon_days: int = 20
    min_bars: int = 252  # Minimum bars for warmup
    detector_params: Dict = field(default_factory=dict)


class ValidationService:
    """
    Service that runs validation against real regime detector.

    Connects:
    - RegimeDetector: Makes predictions (R0, R1, R2, R3)
    - RegimeLabeler: Generates ground truth labels (TRENDING, CHOPPY, NEUTRAL)
    """

    def __init__(self, config: ValidationServiceConfig):
        self.config = config
        self._detector = RegimeDetectorIndicator()
        self._labelers: Dict[str, RegimeLabeler] = {}

        # Initialize labelers for each timeframe
        for tf in config.timeframes:
            labeler_config = RegimeLabelerConfig.load_v1(tf, config.horizon_days)
            self._labelers[tf] = RegimeLabeler(labeler_config)

    def validate_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str,
        end_date: Optional[date] = None,
    ) -> SymbolValidationResult:
        """
        Validate detector performance on a single symbol.

        Args:
            symbol: Symbol to validate
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            timeframe: Timeframe (1d, 4h, etc.)
            end_date: End date for labels (prevents lookahead)

        Returns:
            SymbolValidationResult with R0 rates
        """
        if len(df) < self.config.min_bars:
            logger.warning(f"{symbol}: Only {len(df)} bars, need {self.config.min_bars}")
            return SymbolValidationResult(
                symbol=symbol,
                timeframe=timeframe,
                n_bars=len(df),
                n_trending_bars=0,
                n_choppy_bars=0,
                r0_rate_trending=0.0,
                r0_rate_choppy=0.0,
                predictions=[],
                labels=[],
            )

        # Get detector predictions
        params = {**self._detector._default_params, **self.config.detector_params}
        result_df = self._detector._calculate(df, params)

        # Get ground truth labels
        if end_date is None:
            end_date = df.index[-1].date() if hasattr(df.index[-1], "date") else date.today()

        labeler = self._labelers.get(timeframe)
        if labeler is None:
            raise ValueError(f"No labeler configured for timeframe: {timeframe}")

        labeled_periods = labeler.label_period(df, end_date)

        # Align predictions with labels
        predictions: List[bool] = []
        labels: List[RegimeLabel] = []

        for i, lp in enumerate(labeled_periods):
            if i >= len(result_df):
                break

            # Prediction: R0 = healthy uptrend
            # Note: Detector returns string 'R0', 'R1', etc. or enum MarketRegime
            regime = result_df.iloc[i].get("regime", "R1")
            is_r0 = (
                regime == MarketRegime.R0_HEALTHY_UPTREND
                or regime == "R0"
                or (hasattr(regime, "name") and regime.name == "R0_HEALTHY_UPTREND")
            )

            predictions.append(is_r0)
            labels.append(lp.label)

        # Compute R0 rates by label type
        trending_indices = [i for i, l in enumerate(labels) if l == RegimeLabel.TRENDING]
        choppy_indices = [i for i, l in enumerate(labels) if l == RegimeLabel.CHOPPY]

        r0_rate_trending = 0.0
        if trending_indices:
            r0_count = sum(1 for i in trending_indices if predictions[i])
            r0_rate_trending = r0_count / len(trending_indices)

        r0_rate_choppy = 0.0
        if choppy_indices:
            r0_count = sum(1 for i in choppy_indices if predictions[i])
            r0_rate_choppy = r0_count / len(choppy_indices)

        return SymbolValidationResult(
            symbol=symbol,
            timeframe=timeframe,
            n_bars=len(predictions),
            n_trending_bars=len(trending_indices),
            n_choppy_bars=len(choppy_indices),
            r0_rate_trending=r0_rate_trending,
            r0_rate_choppy=r0_rate_choppy,
            predictions=predictions,
            labels=labels,
        )

    def validate_universe(
        self,
        symbols: List[str],
        bars_by_symbol: Dict[str, pd.DataFrame],
        timeframe: str,
    ) -> Tuple[List[SymbolValidationResult], StatisticalResult]:
        """
        Validate detector across a universe of symbols.

        Args:
            symbols: List of symbols to validate
            bars_by_symbol: Dict mapping symbol -> OHLCV DataFrame
            timeframe: Timeframe to validate

        Returns:
            Tuple of (list of per-symbol results, aggregated statistical result)
        """
        results: List[SymbolValidationResult] = []

        for symbol in symbols:
            df = bars_by_symbol.get(symbol)
            if df is None or df.empty:
                logger.warning(f"No data for {symbol}, skipping")
                continue

            result = self.validate_symbol(symbol, df, timeframe)
            results.append(result)

        # Aggregate to statistical result
        symbol_metrics = self._aggregate_to_symbol_metrics(results)
        statistical_result = compute_symbol_level_stats(symbol_metrics)

        return results, statistical_result

    def _aggregate_to_symbol_metrics(
        self, results: List[SymbolValidationResult]
    ) -> List[SymbolMetrics]:
        """Convert validation results to SymbolMetrics for statistics."""
        metrics = []

        for r in results:
            if r.n_trending_bars > 0:
                r0_bars_trending = int(r.r0_rate_trending * r.n_trending_bars)
                metrics.append(
                    SymbolMetrics(
                        symbol=r.symbol,
                        label_type="TRENDING",
                        r0_rate=r.r0_rate_trending,
                        total_bars=r.n_trending_bars,
                        r0_bars=r0_bars_trending,
                    )
                )

            if r.n_choppy_bars > 0:
                r0_bars_choppy = int(r.r0_rate_choppy * r.n_choppy_bars)
                metrics.append(
                    SymbolMetrics(
                        symbol=r.symbol,
                        label_type="CHOPPY",
                        r0_rate=r.r0_rate_choppy,
                        total_bars=r.n_choppy_bars,
                        r0_bars=r0_bars_choppy,
                    )
                )

        return metrics

    def compute_fast_validation(
        self,
        symbols: List[str],
        bars_by_symbol: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Tuple[float, float, bool]:
        """
        Compute fast validation metrics (for PR gate).

        Args:
            symbols: Symbols to validate
            bars_by_symbol: Bar data by symbol
            timeframe: Timeframe

        Returns:
            Tuple of (trending_r0_rate, choppy_r0_rate, causality_passed)
        """
        results, _ = self.validate_universe(symbols, bars_by_symbol, timeframe)

        # Aggregate R0 rates
        trending_rates = [r.r0_rate_trending for r in results if r.n_trending_bars > 0]
        choppy_rates = [r.r0_rate_choppy for r in results if r.n_choppy_bars > 0]

        trending_r0_rate = sum(trending_rates) / len(trending_rates) if trending_rates else 0.0
        choppy_r0_rate = sum(choppy_rates) / len(choppy_rates) if choppy_rates else 0.0

        # Causality check: trending R0 rate should be significantly higher than choppy
        causality_passed = trending_r0_rate > choppy_r0_rate + 0.10

        return trending_r0_rate, choppy_r0_rate, causality_passed

    def compute_earliness(
        self,
        bars_1d: Dict[str, pd.DataFrame],
        bars_4h: Dict[str, pd.DataFrame],
        symbols: List[str],
    ) -> EarlinessResult:
        """
        Compute earliness of 4h vs 1d detection.

        Args:
            bars_1d: 1d bars by symbol
            bars_4h: 4h bars by symbol
            symbols: Symbols to analyze

        Returns:
            EarlinessResult comparing 4h vs 1d
        """
        # Get signal dates for each symbol
        signals_1d: Dict[date, bool] = {}
        signals_4h: Dict[date, bool] = {}

        for symbol in symbols:
            df_1d = bars_1d.get(symbol)
            df_4h = bars_4h.get(symbol)

            if df_1d is None or df_4h is None:
                continue

            # Get 1d signals
            result_1d = self.validate_symbol(symbol, df_1d, "1d")
            for i, is_r0 in enumerate(result_1d.predictions):
                if i < len(df_1d):
                    dt = df_1d.index[i]
                    d = dt.date() if hasattr(dt, "date") else dt
                    signals_1d[d] = is_r0

            # Get 4h signals (aggregate to daily)
            result_4h = self.validate_symbol(symbol, df_4h, "4h")
            for i, is_r0 in enumerate(result_4h.predictions):
                if i < len(df_4h):
                    dt = df_4h.index[i]
                    d = dt.date() if hasattr(dt, "date") else dt
                    if d not in signals_4h or is_r0:
                        signals_4h[d] = is_r0

        return compute_earliness(signals_1d, signals_4h)

    def compute_confirmation(
        self,
        bars_1d: Dict[str, pd.DataFrame],
        bars_4h: Dict[str, pd.DataFrame],
        symbols: List[str],
    ) -> ConfirmationResult:
        """
        Compute confirmation analysis (1d only vs 1d AND 4h).

        Args:
            bars_1d: 1d bars by symbol
            bars_4h: 4h bars by symbol
            symbols: Symbols to analyze

        Returns:
            ConfirmationResult comparing strategies
        """
        s1_predictions: List[bool] = []  # 1d only
        s2_predictions: List[bool] = []  # 1d AND 4h
        actuals: List[bool] = []  # Ground truth (TRENDING = True)

        for symbol in symbols:
            df_1d = bars_1d.get(symbol)
            df_4h = bars_4h.get(symbol)

            if df_1d is None:
                continue

            result_1d = self.validate_symbol(symbol, df_1d, "1d")

            # Get 4h predictions (aggregate to daily if available)
            preds_4h_by_date: Dict[date, bool] = {}
            if df_4h is not None:
                result_4h = self.validate_symbol(symbol, df_4h, "4h")
                for i, is_r0 in enumerate(result_4h.predictions):
                    if i < len(df_4h):
                        dt = df_4h.index[i]
                        d = dt.date() if hasattr(dt, "date") else dt
                        if d not in preds_4h_by_date or is_r0:
                            preds_4h_by_date[d] = is_r0

            # Align predictions
            for i, (pred_1d, label) in enumerate(zip(result_1d.predictions, result_1d.labels)):
                if i >= len(df_1d):
                    break

                dt = df_1d.index[i]
                d = dt.date() if hasattr(dt, "date") else dt

                s1_predictions.append(pred_1d)

                # S2: 1d AND 4h both signal R0
                pred_4h = preds_4h_by_date.get(d, False)
                s2_predictions.append(pred_1d and pred_4h)

                # Actual: TRENDING = positive
                actuals.append(label == RegimeLabel.TRENDING)

        return compare_strategies(s1_predictions, s2_predictions, actuals)


def load_bars_from_yahoo(
    symbols: List[str],
    timeframe: str,
    start_date: date,
    end_date: date,
) -> Dict[str, pd.DataFrame]:
    """
    Load historical bars from Yahoo Finance.

    Args:
        symbols: List of symbols
        timeframe: Timeframe (1d, 4h, 1h, etc.)
        start_date: Start date
        end_date: End date

    Returns:
        Dict mapping symbol -> DataFrame with OHLCV
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return {}

    # Map timeframe to yfinance interval
    interval_map = {
        "1d": "1d",
        "4h": "1h",  # Yahoo doesn't support 4h, use 1h and aggregate
        "1h": "1h",
        "5m": "5m",
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
            if "adj close" in df.columns:
                df = df.drop(columns=["adj close"])

            # Aggregate to 4h if needed
            if timeframe == "4h" and interval == "1h":
                df = _aggregate_to_4h(df)

            bars_by_symbol[symbol] = df
            logger.info(f"Loaded {len(df)} bars for {symbol}")

        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")

    return bars_by_symbol


def _aggregate_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h bars to 4h."""
    return (
        df.resample("4h")
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
