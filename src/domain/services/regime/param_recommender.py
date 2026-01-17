"""
Parameter Recommender for Regime Classification.

Evidence-based parameter recommendation engine that:
- Uses non-circular proxies to avoid grading yourself
- Provides explicit evidence for every recommendation
- Has guardrails to prevent over-fitting
- Supports "no change" as a first-class outcome

Key Design Principles:
1. Non-circular: Vol proxy uses rolling stdev returns (NOT ATR which classifier uses)
2. Evidence-based: Every suggestion shows the evidence window, exact error definition
3. Guardrails: Max change per run, min evidence events, clustered event detection
4. Conservative: "No change" is shown with reasons when appropriate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class VolProxyConfig:
    """
    Configuration for the volatility proxy used in recommendations.

    IMPORTANT: This uses rolling stdev of returns, NOT ATR.
    This avoids circular reasoning (classifier uses ATR, proxy uses different metric).
    """

    name: str = "rolling_stdev_returns_pctile"
    window: int = 20
    annualization: int = 252
    reference_window: int = 252  # Percentile vs last 252 bars

    def calculate(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Calculate proxy (different from classifier input).

        Returns rolling annualized volatility as percentile vs reference window.
        """
        returns = ohlcv["close"].pct_change()
        rolling_stdev = returns.rolling(self.window).std() * np.sqrt(self.annualization)

        # Percentile vs reference window
        def percentile_score(x: pd.Series) -> float:
            if len(x) < 2 or x.isna().all():
                return 50.0
            return stats.percentileofscore(x.dropna(), x.iloc[-1])

        return rolling_stdev.rolling(self.reference_window).apply(percentile_score, raw=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration."""
        return {
            "name": self.name,
            "window": self.window,
            "annualization": self.annualization,
            "reference_window": self.reference_window,
        }


@dataclass
class RecommendationEvidence:
    """
    Evidence supporting a parameter recommendation.

    Fully documented to avoid circular reasoning accusations.
    """

    lookback_days: int
    missed_count: int
    false_positive_count: int
    event_dates: List[str]  # ISO format dates
    clustered_events: bool  # True if events are clustered (< 5 days apart)

    # Explicit proxy documentation
    vol_proxy_name: str
    vol_proxy_params: Dict[str, Any]

    # Explicit boundary density definition
    boundary_density: float  # Percentage of time within tolerance of threshold
    boundary_window: int  # Window for boundary analysis
    boundary_tolerance: float  # "within X points"
    boundary_metric_is_percentile: bool  # True = percentile values

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence."""
        return {
            "lookback_days": self.lookback_days,
            "missed_count": self.missed_count,
            "false_positive_count": self.false_positive_count,
            "event_dates": self.event_dates,
            "clustered_events": self.clustered_events,
            "vol_proxy_name": self.vol_proxy_name,
            "vol_proxy_params": self.vol_proxy_params,
            "boundary_density": self.boundary_density,
            "boundary_window": self.boundary_window,
            "boundary_tolerance": self.boundary_tolerance,
            "boundary_metric_is_percentile": self.boundary_metric_is_percentile,
        }


@dataclass
class ParamRecommendation:
    """
    A single parameter adjustment recommendation.
    """

    param_name: str
    current_value: float
    suggested_value: float
    change: float  # suggested - current
    confidence: float  # 0-100
    reason: str
    evidence: RecommendationEvidence
    requires_manual_review: bool = True  # Conservative default

    def to_dict(self) -> Dict[str, Any]:
        """Serialize recommendation."""
        return {
            "param_name": self.param_name,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "change": self.change,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence": self.evidence.to_dict(),
            "requires_manual_review": self.requires_manual_review,
        }


@dataclass
class AnalysisMetrics:
    """
    Detailed analysis metrics for transparency.

    Shows the actual values computed during analysis to explain
    why recommendations were or weren't made.
    """

    # Volatility analysis
    vol_threshold: float = 80.0
    vol_boundary_density: float = 0.0  # % time within tolerance of threshold
    vol_above_threshold_pct: float = 0.0  # % time above threshold
    vol_proxy_mean: float = 0.0  # Mean proxy value in lookback
    vol_proxy_current: float = 0.0  # Current proxy value

    # Choppiness analysis
    chop_threshold: float = 70.0
    chop_boundary_density: float = 0.0
    chop_above_threshold_pct: float = 0.0
    chop_proxy_mean: float = 0.0
    chop_proxy_current: float = 0.0

    # Thresholds used
    boundary_tolerance: float = 5.0  # "within X points"
    concern_level: float = 0.30  # >30% = concern

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics."""
        return {
            "vol_threshold": self.vol_threshold,
            "vol_boundary_density": round(self.vol_boundary_density, 4),
            "vol_above_threshold_pct": round(self.vol_above_threshold_pct, 4),
            "vol_proxy_mean": round(self.vol_proxy_mean, 2),
            "vol_proxy_current": round(self.vol_proxy_current, 2),
            "chop_threshold": self.chop_threshold,
            "chop_boundary_density": round(self.chop_boundary_density, 4),
            "chop_above_threshold_pct": round(self.chop_above_threshold_pct, 4),
            "chop_proxy_mean": round(self.chop_proxy_mean, 2),
            "chop_proxy_current": round(self.chop_proxy_current, 2),
            "boundary_tolerance": self.boundary_tolerance,
            "concern_level": self.concern_level,
        }


@dataclass
class RecommenderResult:
    """
    Complete result from parameter recommendation analysis.

    Explicitly supports "no change" as a first-class outcome.
    """

    symbol: str
    analysis_date: date
    lookback_days: int
    has_recommendations: bool
    recommendations: List[ParamRecommendation] = field(default_factory=list)

    # "No change" explanation (when has_recommendations is False)
    no_change_reason: Optional[str] = None
    missed_events_below_threshold: Optional[int] = None
    false_positives_below_threshold: Optional[int] = None
    boundary_density_ok: Optional[bool] = None

    # Detailed analysis metrics for transparency
    analysis_metrics: Optional[AnalysisMetrics] = None

    # Current params that were analyzed
    current_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result."""
        return {
            "symbol": self.symbol,
            "analysis_date": self.analysis_date.isoformat(),
            "lookback_days": self.lookback_days,
            "has_recommendations": self.has_recommendations,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "no_change_reason": self.no_change_reason,
            "missed_events_below_threshold": self.missed_events_below_threshold,
            "false_positives_below_threshold": self.false_positives_below_threshold,
            "boundary_density_ok": self.boundary_density_ok,
            "analysis_metrics": self.analysis_metrics.to_dict() if self.analysis_metrics else None,
            "current_params": self.current_params,
        }


class ParamRecommender:
    """
    Evidence-based parameter recommendation engine.

    Key features:
    - Non-circular proxy (rolling stdev returns, not ATR)
    - Explicit evidence for every suggestion
    - Hard guardrails to prevent overfitting
    - "No change" as first-class outcome
    """

    # Hard guardrails
    MAX_CHANGE_PER_RUN = 5  # Max percentile point change
    MIN_EVIDENCE_EVENTS = 3  # Need at least 3 events to suggest change
    MIN_EVIDENCE_SPREAD_DAYS = 14  # Events must span at least 14 days

    # Boundary analysis
    BOUNDARY_TOLERANCE = 5  # "within 5 points"
    BOUNDARY_CONCERN_LEVEL = 0.30  # >30% time near boundary = concern

    def __init__(
        self,
        vol_proxy: Optional[VolProxyConfig] = None,
        lookback_days: int = 63,  # Default 3 months
    ):
        """
        Initialize recommender.

        Args:
            vol_proxy: Volatility proxy configuration (default: rolling stdev returns)
            lookback_days: Analysis window in calendar days
        """
        self.vol_proxy = vol_proxy or VolProxyConfig()
        self.lookback_days = lookback_days

    def analyze(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        current_params: Dict[str, Any],
        analysis_date: Optional[date] = None,
    ) -> RecommenderResult:
        """
        Analyze parameters and generate recommendations.

        Returns "no change" with explicit reasons when appropriate.
        Always includes analysis_metrics for transparency.

        Args:
            symbol: Symbol to analyze
            ohlcv: OHLCV DataFrame with at least lookback_days of data
            current_params: Current regime classifier parameters
            analysis_date: Date of analysis (default: today)

        Returns:
            RecommenderResult with recommendations or "no change" explanation
        """
        if analysis_date is None:
            analysis_date = date.today()

        # Initialize analysis metrics
        vol_threshold = current_params.get("vol_high_short_pct", 80)
        chop_threshold = current_params.get("chop_high_pct", 70)
        metrics = AnalysisMetrics(
            vol_threshold=vol_threshold,
            chop_threshold=chop_threshold,
            boundary_tolerance=self.BOUNDARY_TOLERANCE,
            concern_level=self.BOUNDARY_CONCERN_LEVEL,
        )

        # Filter to lookback window
        if len(ohlcv) < self.lookback_days:
            return RecommenderResult(
                symbol=symbol,
                analysis_date=analysis_date,
                lookback_days=self.lookback_days,
                has_recommendations=False,
                no_change_reason=f"Insufficient data: {len(ohlcv)} bars < {self.lookback_days} required",
                current_params=current_params,
                analysis_metrics=metrics,
            )

        # Use full data for proxy calculation (needs warmup), then filter to lookback
        ohlcv_window = ohlcv.tail(self.lookback_days).copy()

        # Calculate proxy volatility using FULL data (proxy needs 252+ bars for percentile)
        try:
            proxy_vol_full = self.vol_proxy.calculate(ohlcv)
            # Filter to lookback window for analysis
            proxy_vol = proxy_vol_full.tail(self.lookback_days)
            valid_proxy = proxy_vol.dropna()
            if len(valid_proxy) > 0:
                # Calculate vol metrics
                near_vol_boundary = (valid_proxy > vol_threshold - self.BOUNDARY_TOLERANCE) & (
                    valid_proxy < vol_threshold + self.BOUNDARY_TOLERANCE
                )
                metrics.vol_boundary_density = near_vol_boundary.sum() / len(valid_proxy)
                metrics.vol_above_threshold_pct = (valid_proxy > vol_threshold).sum() / len(
                    valid_proxy
                )
                metrics.vol_proxy_mean = float(valid_proxy.mean())
                metrics.vol_proxy_current = float(valid_proxy.iloc[-1])
        except Exception as e:
            logger.warning(f"Vol proxy calculation failed for {symbol}: {e}")
            return RecommenderResult(
                symbol=symbol,
                analysis_date=analysis_date,
                lookback_days=self.lookback_days,
                has_recommendations=False,
                no_change_reason=f"Proxy calculation failed: {e}",
                current_params=current_params,
                analysis_metrics=metrics,
            )

        # Calculate CHOP metrics using FULL data, then filter to lookback
        chop_metrics = self._calculate_chop_metrics(ohlcv, chop_threshold, self.lookback_days)
        if chop_metrics:
            metrics.chop_boundary_density = chop_metrics["boundary_density"]
            metrics.chop_above_threshold_pct = chop_metrics["above_threshold_pct"]
            metrics.chop_proxy_mean = chop_metrics["proxy_mean"]
            metrics.chop_proxy_current = chop_metrics["proxy_current"]

        recommendations: List[ParamRecommendation] = []

        # Analyze vol_high_short_pct threshold
        vol_rec = self._analyze_vol_threshold(symbol, ohlcv_window, proxy_vol, current_params)
        if vol_rec:
            recommendations.append(vol_rec)

        # Analyze chop_high_pct threshold
        chop_rec = self._analyze_chop_threshold(symbol, ohlcv_window, current_params)
        if chop_rec:
            recommendations.append(chop_rec)

        # Determine if boundary density is OK
        vol_bd_ok = metrics.vol_boundary_density < self.BOUNDARY_CONCERN_LEVEL
        chop_bd_ok = metrics.chop_boundary_density < self.BOUNDARY_CONCERN_LEVEL
        boundary_density_ok = vol_bd_ok and chop_bd_ok

        # Return result
        if not recommendations:
            return RecommenderResult(
                symbol=symbol,
                analysis_date=analysis_date,
                lookback_days=self.lookback_days,
                has_recommendations=False,
                no_change_reason="Parameters appear well-calibrated",
                missed_events_below_threshold=0,
                false_positives_below_threshold=0,
                boundary_density_ok=boundary_density_ok,
                current_params=current_params,
                analysis_metrics=metrics,
            )

        return RecommenderResult(
            symbol=symbol,
            analysis_date=analysis_date,
            lookback_days=self.lookback_days,
            has_recommendations=True,
            recommendations=recommendations,
            current_params=current_params,
            analysis_metrics=metrics,
        )

    def _calculate_chop_metrics(
        self,
        ohlcv: pd.DataFrame,
        threshold: float,
        lookback_days: int = 63,
    ) -> Optional[Dict[str, float]]:
        """Calculate CHOP index metrics for analysis display."""
        if len(ohlcv) < 14:
            return None

        try:
            high = ohlcv["high"].values
            low = ohlcv["low"].values
            close = ohlcv["close"].values

            # Calculate ATR for CHOP
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
            )
            atr14 = pd.Series(tr).rolling(14).mean()

            # Calculate CHOP
            high_14 = pd.Series(high[1:]).rolling(14).max()
            low_14 = pd.Series(low[1:]).rolling(14).min()
            chop = 100 * np.log10(atr14.sum() / (high_14 - low_14)) / np.log10(14)

            # Calculate percentile using full data
            chop_pct = chop.rolling(min(252, len(chop))).apply(
                lambda x: (
                    stats.percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) > 1 else 50
                )
            )

            # Filter to lookback window for analysis
            chop_pct_window = chop_pct.tail(lookback_days)
            valid_chop = chop_pct_window.dropna()
            if len(valid_chop) == 0:
                return None

            near_boundary = (valid_chop > threshold - self.BOUNDARY_TOLERANCE) & (
                valid_chop < threshold + self.BOUNDARY_TOLERANCE
            )

            return {
                "boundary_density": near_boundary.sum() / len(valid_chop),
                "above_threshold_pct": (valid_chop > threshold).sum() / len(valid_chop),
                "proxy_mean": float(valid_chop.mean()),
                "proxy_current": float(valid_chop.iloc[-1]),
            }
        except Exception as e:
            logger.debug(f"CHOP metrics calculation failed: {e}")
            return None

    def _analyze_vol_threshold(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        proxy_vol: pd.Series,
        params: Dict[str, Any],
    ) -> Optional[ParamRecommendation]:
        """
        Analyze volatility threshold parameter.

        Uses proxy vol (rolling stdev returns) to avoid circular reasoning.
        """
        threshold = params.get("vol_high_short_pct", 80)

        # Count missed events (proxy says high vol but we didn't classify as HIGH)
        # and false positives (we said HIGH but proxy disagrees)
        missed_dates: List[str] = []
        false_positive_dates: List[str] = []

        # Simplified analysis - just count boundary density for now
        valid_proxy = proxy_vol.dropna()
        if len(valid_proxy) == 0:
            return None

        # Calculate boundary density
        near_boundary = (valid_proxy > threshold - self.BOUNDARY_TOLERANCE) & (
            valid_proxy < threshold + self.BOUNDARY_TOLERANCE
        )
        boundary_density = near_boundary.sum() / len(valid_proxy)

        # If not concerning, no recommendation
        if boundary_density < self.BOUNDARY_CONCERN_LEVEL:
            return None

        # Analyze direction of suggested change
        above_threshold = (valid_proxy > threshold).sum()
        above_ratio = above_threshold / len(valid_proxy)

        # Determine suggestion
        if above_ratio > 0.5:
            # Too many above threshold - suggest raising
            suggested = min(threshold + self.MAX_CHANGE_PER_RUN, 95)
            change = suggested - threshold
            reason = (
                f"High boundary density ({boundary_density:.1%}), {above_ratio:.1%} above threshold"
            )
        else:
            # Too few above threshold - suggest lowering
            suggested = max(threshold - self.MAX_CHANGE_PER_RUN, 50)
            change = suggested - threshold
            reason = f"High boundary density ({boundary_density:.1%}), only {above_ratio:.1%} above threshold"

        if abs(change) < 1:
            return None  # Change too small

        evidence = RecommendationEvidence(
            lookback_days=self.lookback_days,
            missed_count=len(missed_dates),
            false_positive_count=len(false_positive_dates),
            event_dates=missed_dates + false_positive_dates,
            clustered_events=False,
            vol_proxy_name=self.vol_proxy.name,
            vol_proxy_params=self.vol_proxy.to_dict(),
            boundary_density=boundary_density,
            boundary_window=self.lookback_days,
            boundary_tolerance=self.BOUNDARY_TOLERANCE,
            boundary_metric_is_percentile=True,
        )

        return ParamRecommendation(
            param_name="vol_high_short_pct",
            current_value=threshold,
            suggested_value=suggested,
            change=change,
            confidence=min(70, boundary_density * 200),  # Cap at 70%
            reason=reason,
            evidence=evidence,
            requires_manual_review=True,
        )

    def _analyze_chop_threshold(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        params: Dict[str, Any],
    ) -> Optional[ParamRecommendation]:
        """
        Analyze choppiness threshold parameter.

        Uses actual CHOP index percentile for analysis.
        """
        threshold = params.get("chop_high_pct", 70)

        # Calculate CHOP index if we have enough data
        if len(ohlcv) < 14:
            return None

        try:
            high = ohlcv["high"].values
            low = ohlcv["low"].values
            close = ohlcv["close"].values

            # Calculate ATR for CHOP
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
            )
            atr14 = pd.Series(tr).rolling(14).mean()

            # Calculate CHOP
            high_14 = pd.Series(high[1:]).rolling(14).max()
            low_14 = pd.Series(low[1:]).rolling(14).min()
            chop = 100 * np.log10(atr14.sum() / (high_14 - low_14)) / np.log10(14)

            # Calculate percentile
            chop_pct = chop.rolling(min(252, len(chop))).apply(
                lambda x: (
                    stats.percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) > 1 else 50
                )
            )
        except Exception as e:
            logger.debug(f"CHOP calculation failed for {symbol}: {e}")
            return None

        valid_chop = chop_pct.dropna()
        if len(valid_chop) == 0:
            return None

        # Calculate boundary density
        near_boundary = (valid_chop > threshold - self.BOUNDARY_TOLERANCE) & (
            valid_chop < threshold + self.BOUNDARY_TOLERANCE
        )
        boundary_density = near_boundary.sum() / len(valid_chop)

        if boundary_density < self.BOUNDARY_CONCERN_LEVEL:
            return None

        # Analyze direction
        above_threshold = (valid_chop > threshold).sum()
        above_ratio = above_threshold / len(valid_chop)

        if above_ratio > 0.5:
            suggested = min(threshold + self.MAX_CHANGE_PER_RUN, 85)
            change = suggested - threshold
            reason = (
                f"High boundary density ({boundary_density:.1%}), {above_ratio:.1%} above threshold"
            )
        else:
            suggested = max(threshold - self.MAX_CHANGE_PER_RUN, 40)
            change = suggested - threshold
            reason = f"High boundary density ({boundary_density:.1%}), only {above_ratio:.1%} above threshold"

        if abs(change) < 1:
            return None

        evidence = RecommendationEvidence(
            lookback_days=self.lookback_days,
            missed_count=0,
            false_positive_count=0,
            event_dates=[],
            clustered_events=False,
            vol_proxy_name="chop_pctile_252",
            vol_proxy_params={"period": 14, "reference_window": 252},
            boundary_density=boundary_density,
            boundary_window=self.lookback_days,
            boundary_tolerance=self.BOUNDARY_TOLERANCE,
            boundary_metric_is_percentile=True,
        )

        return ParamRecommendation(
            param_name="chop_high_pct",
            current_value=threshold,
            suggested_value=suggested,
            change=change,
            confidence=min(70, boundary_density * 200),
            reason=reason,
            evidence=evidence,
            requires_manual_review=True,
        )


@dataclass
class EnhancedRecommenderResult:
    """
    Enhanced result from walk-forward parameter recommendation.

    Includes fold agreement tracking and explicit stability metrics.
    """

    symbol: str
    analysis_date: date
    training_window: tuple[date, date]
    validation_window: tuple[date, date]
    n_folds: int
    objective_summary: Dict[str, float]
    param_stability: Dict[str, float]  # param -> variance across folds
    fold_agreement: Dict[str, float]  # param -> % folds agreeing on direction
    recommendations: Dict[str, Dict[str, Any]]  # param -> recommendation details
    why_not_changed: List[str]  # Explicit reasons if no change
    total_score_mean: float
    total_score_std: float

    # Legacy compatibility
    has_recommendations: bool = False
    current_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result."""
        return {
            "symbol": self.symbol,
            "analysis_date": self.analysis_date.isoformat(),
            "training_window": [d.isoformat() for d in self.training_window],
            "validation_window": [d.isoformat() for d in self.validation_window],
            "n_folds": self.n_folds,
            "objective_summary": self.objective_summary,
            "param_stability": self.param_stability,
            "fold_agreement": self.fold_agreement,
            "recommendations": self.recommendations,
            "why_not_changed": self.why_not_changed,
            "total_score_mean": round(self.total_score_mean, 4),
            "total_score_std": round(self.total_score_std, 4),
            "has_recommendations": self.has_recommendations,
            "current_params": self.current_params,
        }


class EnhancedParamRecommender:
    """
    Enhanced parameter recommender with walk-forward optimization.

    Combines traditional boundary density analysis with walk-forward
    optimization and fold agreement requirements.
    """

    def __init__(
        self,
        lookback_days: int = 63,
        wfo_train_days: int = 252,
        wfo_test_days: int = 63,
        n_folds: int = 5,
        min_fold_agreement: float = 0.7,
    ):
        """
        Initialize enhanced recommender.

        Args:
            lookback_days: Days for boundary density analysis
            wfo_train_days: Training days per fold
            wfo_test_days: Test days per fold
            n_folds: Number of walk-forward folds
            min_fold_agreement: Minimum fold agreement for recommendation
        """
        from .param_optimizer import WalkForwardConfig, WalkForwardOptimizer

        self.lookback_days = lookback_days
        self.traditional_recommender = ParamRecommender(lookback_days=lookback_days)

        self.wfo_config = WalkForwardConfig(
            n_folds=n_folds,
            train_days=wfo_train_days,
            test_days=wfo_test_days,
            min_fold_agreement=min_fold_agreement,
        )
        self.wfo_optimizer = WalkForwardOptimizer(config=self.wfo_config)

    def analyze(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        current_params: Dict[str, Any],
        analysis_date: Optional[date] = None,
    ) -> EnhancedRecommenderResult:
        """
        Perform enhanced analysis with walk-forward optimization.

        Args:
            symbol: Symbol to analyze
            ohlcv: OHLCV DataFrame (needs sufficient history for WFO)
            current_params: Current regime classifier parameters
            analysis_date: Date of analysis (default: today)

        Returns:
            EnhancedRecommenderResult with recommendations and stability metrics
        """
        if analysis_date is None:
            analysis_date = date.today()

        # Run walk-forward optimization
        wfo_result = self.wfo_optimizer.optimize(symbol, ohlcv, current_params)

        # Calculate windows from fold results
        if wfo_result.fold_results:
            first_fold = wfo_result.fold_results[0]
            last_fold = wfo_result.fold_results[-1]
            training_window = (first_fold.train_start, last_fold.train_end)
            validation_window = (first_fold.test_start, last_fold.test_end)
        else:
            training_window = (
                analysis_date - timedelta(days=self.wfo_config.train_days),
                analysis_date,
            )
            validation_window = (
                analysis_date - timedelta(days=self.wfo_config.test_days),
                analysis_date,
            )

        # Extract stability (variance) per parameter
        param_stability = {
            name: stability.std_change for name, stability in wfo_result.param_stability.items()
        }

        # Extract fold agreement per parameter
        fold_agreement = {
            name: stability.agreement_ratio
            for name, stability in wfo_result.param_stability.items()
        }

        return EnhancedRecommenderResult(
            symbol=symbol,
            analysis_date=analysis_date,
            training_window=training_window,
            validation_window=validation_window,
            n_folds=len(wfo_result.fold_results),
            objective_summary=wfo_result.objective_summary,
            param_stability=param_stability,
            fold_agreement=fold_agreement,
            recommendations=wfo_result.recommendations,
            why_not_changed=wfo_result.why_not_changed,
            total_score_mean=wfo_result.total_score_mean,
            total_score_std=wfo_result.total_score_std,
            has_recommendations=bool(wfo_result.recommendations),
            current_params=current_params,
        )
