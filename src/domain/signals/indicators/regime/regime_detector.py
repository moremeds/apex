"""
Regime Detector Indicator.

Implements a 3-level hierarchical regime detection system for market classification:
- R0 (Healthy Uptrend): TrendUp + NormalVol + Trending
- R1 (Choppy/Extended): TrendUp but Choppy OR Overbought
- R2 (Risk-Off): TrendDown OR (HighVol + below MA50) OR IV_HIGH
- R3 (Rebound Window): HighVol + Oversold + NOT TrendDown + structural confirm

Key Features:
- Priority-based decision tree prevents parallel regime triggers
- Proper hysteresis with pending_regime/pending_count pattern
- Component-based architecture for testability and optimization
- Full explainability with RuleTrace for every decision
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

from ...data.quality_validator import validate_close_for_regime

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
from ...models import SignalCategory
from ..base import IndicatorBase
from .components import (
    calculate_chop_state,
    calculate_ext_state,
    calculate_trend_state,
    calculate_vol_state,
)
from .models import (
    ENTRY_HYSTERESIS,
    EXIT_HYSTERESIS,
    MARKET_BENCHMARKS,
    ChopState,
    ComponentStates,
    ComponentValues,
    DataQuality,
    DataWindow,
    DerivedMetrics,
    ExtState,
    InputsUsed,
    IVState,
    MarketRegime,
    RegimeOutput,
    RegimeState,
    RegimeTransitionState,
    TrendState,
    VolState,
)
from .rule_trace import (
    RuleTrace,
    ThresholdInfo,
    create_categorical_rule_trace,
    create_threshold_rule_trace,
)

# Phase 4: Turning point imports (lazy loaded to avoid circular imports)
# TurningPointModel, TurningPointFeatures, TurningPointOutput are imported in method


logger = get_logger(__name__)


class RegimeDetectorIndicator(IndicatorBase):
    """
    Regime Detector indicator for market condition classification.

    Classifies market into one of four regimes:
    - R0: Healthy Uptrend - Full trading allowed
    - R1: Choppy/Extended - Reduced frequency, wider spreads
    - R2: Risk-Off - No new positions
    - R3: Rebound Window - Small defined-risk positions only

    Default Parameters:
        ma50_period: 50
        ma200_period: 200
        ma20_period: 20
        slope_lookback: 20
        atr_period: 20
        atr_pct_short_window: 63 (3 months)
        atr_pct_long_window: 252 (1 year)
        vol_high_short_pct: 80
        vol_high_long_pct: 85
        vol_low_pct: 20
        chop_period: 14
        chop_pct_window: 252
        ma20_cross_lookback: 10
        chop_high_pct: 70
        chop_low_pct: 30
        chop_cross_high: 4
        chop_cross_low: 1
        ext_overbought: 2.0
        ext_oversold: -2.0
        ext_slightly_high: 1.5
        ext_slightly_low: -1.5

    State Output:
        regime: "R0", "R1", "R2", "R3"
        regime_name: Human-readable name
        confidence: 0-100 regime confidence
        component_states: Dict of component classifications
        components: Dict of raw numeric values
        transition: Dict with regime_changed, previous_regime, bars_in_regime
    """

    name = "regime_detector"
    category = SignalCategory.REGIME
    required_fields = ["high", "low", "close"]
    warmup_periods = 252  # Need 1 year for percentile calculations

    _default_params = {
        # MAs
        "ma50_period": 50,
        "ma200_period": 200,
        "ma20_period": 20,
        "slope_lookback": 20,
        # Volatility (dual-window)
        "atr_period": 20,
        "atr_pct_short_window": 63,
        "atr_pct_long_window": 252,
        "vol_high_short_pct": 80,
        "vol_high_long_pct": 85,
        "vol_low_pct": 20,
        # IV (market level only)
        "iv_pct_window": 63,
        "iv_high_pct": 75,
        "iv_elevated_pct": 50,
        "iv_low_pct": 25,
        # Choppiness
        "chop_period": 14,
        "chop_pct_window": 252,
        "ma20_cross_lookback": 10,
        "chop_high_pct": 70,
        "chop_low_pct": 30,
        "chop_cross_high": 4,
        "chop_cross_low": 1,
        # Extension
        "ext_overbought": 2.0,
        "ext_oversold": -2.0,
        "ext_slightly_high": 1.5,
        "ext_slightly_low": -1.5,
    }

    def __init__(self) -> None:
        """Initialize regime detector with tracking state."""
        super().__init__()
        # Track regime state per symbol for hysteresis (thread-safe)
        self._regime_states: Dict[str, RegimeState] = {}
        self._state_lock = threading.Lock()

        # Phase 4: Turning point models (loaded on demand)
        self._turning_point_models: Dict[str, Any] = {}
        self._tp_model_load_attempted: Dict[str, bool] = {}

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate regime indicators for all bars.

        Returns DataFrame with columns for regime classification and all
        component values for rule evaluation and reporting.
        """
        n = len(data)
        if n == 0:
            return pd.DataFrame(
                {
                    "regime": pd.Series(dtype=str),
                    "regime_confidence": pd.Series(dtype=int),
                    "trend_state": pd.Series(dtype=str),
                    "vol_state": pd.Series(dtype=str),
                    "chop_state": pd.Series(dtype=str),
                    "ext_state": pd.Series(dtype=str),
                },
                index=data.index,
            )

        # Extract OHLC arrays
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        # Calculate all components with error handling
        try:
            trend_states, trend_details = calculate_trend_state(close, params)
        except Exception as e:
            logger.error(f"Trend state calculation failed: {e}", exc_info=True)
            trend_states = np.array([TrendState.NEUTRAL] * n)
            trend_details = {
                "ma50": np.full(n, np.nan),
                "ma200": np.full(n, np.nan),
                "ma50_slope": np.full(n, np.nan),
            }

        try:
            vol_states, vol_details = calculate_vol_state(high, low, close, params)
        except Exception as e:
            logger.error(f"Vol state calculation failed: {e}", exc_info=True)
            vol_states = np.array([VolState.NORMAL] * n)
            vol_details = {
                "atr": np.full(n, np.nan),
                "atr_pct": np.full(n, np.nan),
                "atr_pct_63": np.full(n, 50.0),
                "atr_pct_252": np.full(n, 50.0),
            }

        try:
            chop_states, chop_details = calculate_chop_state(high, low, close, params)
        except Exception as e:
            logger.error(f"Chop state calculation failed: {e}", exc_info=True)
            chop_states = np.array([ChopState.NEUTRAL] * n)
            chop_details = {
                "chop": np.full(n, 50.0),
                "chop_pct_252": np.full(n, 50.0),
                "ma20_crosses": np.zeros(n),
            }

        try:
            ext_states, ext_details = calculate_ext_state(high, low, close, params)
        except Exception as e:
            logger.error(f"Ext state calculation failed: {e}", exc_info=True)
            ext_states = np.array([ExtState.NEUTRAL] * n)
            ext_details = {"ext": np.zeros(n), "ma20": np.full(n, np.nan)}

        # Get MA20 from ext_details (already calculated)
        ma20 = ext_details["ma20"]

        # Calculate 5-bar high for R3 structural confirmation
        last_5_bar_high = np.full(n, np.nan)
        for i in range(4, n):
            last_5_bar_high[i] = np.max(high[i - 4 : i + 1])

        # Classify regime for each bar
        regimes = []
        confidences = []

        for i in range(n):
            # Build state dict for classification
            state = {
                "close": close[i],
                "ma20": ma20[i] if not np.isnan(ma20[i]) else close[i],
                "ma50": trend_details["ma50"][i],
                "ma200": trend_details["ma200"][i],
                "ma50_slope": trend_details["ma50_slope"][i],
                "last_5_bar_high": (
                    last_5_bar_high[i] if not np.isnan(last_5_bar_high[i]) else close[i]
                ),
                "trend_state": trend_states[i],
                "vol_state": vol_states[i],
                "chop_state": chop_states[i],
                "ext_state": ext_states[i],
                "iv_state": IVState.NA,  # IV calculated separately at service level
                "is_market_level": False,
            }

            # Compute regime (without hysteresis for batch calculation)
            regime, _ = self._evaluate_decision_tree(state)
            confidence = self._compute_confidence(state, regime)

            regimes.append(regime.value)
            confidences.append(confidence)

        # Build result DataFrame
        result = pd.DataFrame(
            {
                "regime": regimes,
                "regime_confidence": confidences,
                # Component states
                "trend_state": [ts.value for ts in trend_states],
                "vol_state": [vs.value for vs in vol_states],
                "chop_state": [cs.value for cs in chop_states],
                "ext_state": [es.value for es in ext_states],
                # Component values
                "ma20": ma20,
                "ma50": trend_details["ma50"],
                "ma200": trend_details["ma200"],
                "ma50_slope": trend_details["ma50_slope"],
                "atr20": vol_details["atr"],
                "atr_pct": vol_details["atr_pct"],
                "atr_pct_63": vol_details["atr_pct_63"],
                "atr_pct_252": vol_details["atr_pct_252"],
                "chop": chop_details["chop"],
                "chop_pct_252": chop_details["chop_pct_252"],
                "ma20_crosses": chop_details["ma20_crosses"],
                "ext": ext_details["ext"],
                "last_5_bar_high": last_5_bar_high,
            },
            index=data.index,
        )

        return result

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract regime state for rule evaluation.

        Applies hysteresis for stable regime transitions.
        """
        regime_str = current.get("regime", "R1")
        confidence = current.get("regime_confidence", 50)

        # Parse component states
        trend_state_str = current.get("trend_state", "neutral")
        vol_state_str = current.get("vol_state", "vol_normal")
        chop_state_str = current.get("chop_state", "neutral")
        ext_state_str = current.get("ext_state", "neutral")

        # Convert strings to enums with logging
        try:
            regime = MarketRegime(regime_str)
        except ValueError:
            logger.warning(f"Invalid regime string '{regime_str}', falling back to R1")
            regime = MarketRegime.R1_CHOPPY_EXTENDED

        # Detect regime change
        regime_changed = False
        previous_regime = None
        if previous is not None:
            prev_regime_str = previous.get("regime", "R1")
            if prev_regime_str != regime_str:
                regime_changed = True
                try:
                    previous_regime = MarketRegime(prev_regime_str)
                except ValueError:
                    logger.warning(f"Invalid previous regime string '{prev_regime_str}'")
                    previous_regime = None

        # Build state dict
        return {
            "regime": regime.value,
            "regime_name": regime.display_name,
            "confidence": int(confidence) if not pd.isna(confidence) else 50,
            # Component states
            "trend_state": trend_state_str,
            "vol_state": vol_state_str,
            "chop_state": chop_state_str,
            "ext_state": ext_state_str,
            "iv_state": "na",  # IV handled at service level
            # Component values
            "components": {
                "close": self._safe_float(current.get("close")),
                "ma20": self._safe_float(current.get("ma20")),
                "ma50": self._safe_float(current.get("ma50")),
                "ma200": self._safe_float(current.get("ma200")),
                "ma50_slope": self._safe_float(current.get("ma50_slope")),
                "atr20": self._safe_float(current.get("atr20")),
                "atr_pct": self._safe_float(current.get("atr_pct")),
                "atr_pct_63": self._safe_float(current.get("atr_pct_63")),
                "atr_pct_252": self._safe_float(current.get("atr_pct_252")),
                "chop": self._safe_float(current.get("chop")),
                "chop_pct_252": self._safe_float(current.get("chop_pct_252")),
                "ma20_crosses": int(current.get("ma20_crosses", 0)),
                "ext": self._safe_float(current.get("ext")),
            },
            # Transition
            "regime_changed": regime_changed,
            "previous_regime": previous_regime.value if previous_regime else None,
        }

    def _evaluate_decision_tree(
        self, state: Dict[str, Any]
    ) -> Tuple[MarketRegime, List[RuleTrace]]:
        """
        Evaluate the priority-based decision tree with full rule tracing.

        This is a PURE function - it only computes the decision regime based on
        current state without any hysteresis. Hysteresis is applied separately.

        Priority Order (highest to lowest):
        1. R2 (Risk-Off) - Veto power, always checked first
        2. R3 (Rebound)  - Only if NOT in active downtrend + structural confirm
        3. R1 (Choppy)   - Only if NOT in strong trend acceleration
        4. R0 (Healthy)  - Default when conditions are favorable

        Returns:
            Tuple of (decision_regime, rules_fired) where rules_fired contains
            traces for all rules evaluated with pass/fail status.
        """
        rules_fired: List[RuleTrace] = []

        # Price and MA values
        close = state["close"]
        ma20 = state["ma20"]
        ma50 = state["ma50"]
        ma200 = state["ma200"]
        ma50_slope = state["ma50_slope"]
        last_5_bar_high = state["last_5_bar_high"]

        # Additional numeric metrics for evidence
        atr_pct_63 = state.get("atr_pct_63", 50.0)
        atr_pct_252 = state.get("atr_pct_252", 50.0)
        chop_pct = state.get("chop_pct_252", 50.0)
        ext_value = state.get("ext", 0.0)

        # Component states - normalize to enums (state can be string or enum)
        trend_state_raw = state["trend_state"]
        vol_state_raw = state["vol_state"]
        chop_state_raw = state["chop_state"]
        ext_state_raw = state["ext_state"]
        iv_state_raw = state.get("iv_state", IVState.NA)
        is_market_level = state.get("is_market_level", False)

        # Convert strings to enums for consistent comparison
        def _to_enum(val: Any, enum_class: type, default: Any) -> Any:
            """Convert string value to enum, or return as-is if already enum."""
            if isinstance(val, enum_class):
                return val
            if isinstance(val, str):
                try:
                    return enum_class(val)
                except ValueError:
                    return default
            return default

        trend_state = _to_enum(trend_state_raw, TrendState, TrendState.NEUTRAL)
        vol_state = _to_enum(vol_state_raw, VolState, VolState.NORMAL)
        chop_state = _to_enum(chop_state_raw, ChopState, ChopState.NEUTRAL)
        ext_state = _to_enum(ext_state_raw, ExtState, ExtState.NEUTRAL)
        iv_state = _to_enum(iv_state_raw, IVState, IVState.NA)

        # Handle NaN values - fallback to R1
        if np.isnan(ma50) or np.isnan(ma200):
            rules_fired.append(
                RuleTrace(
                    rule_id="fallback_nan",
                    description="Fallback: MA values are NaN",
                    passed=True,
                    evidence={"ma50": ma50, "ma200": ma200},
                    regime_target="R1",
                    category="fallback",
                    priority=0,
                )
            )
            return MarketRegime.R1_CHOPPY_EXTENDED, rules_fired

        # ========================================================================
        # R2 CHECK (Highest Priority - Veto)
        # ========================================================================
        trend_state_str = trend_state.value if hasattr(trend_state, "value") else str(trend_state)
        vol_state_str = vol_state.value if hasattr(vol_state, "value") else str(vol_state)
        iv_state_str = iv_state.value if hasattr(iv_state, "value") else str(iv_state)

        # R2 Rule 1: Trend is DOWN (categorical)
        r2_trend_down = trend_state == TrendState.DOWN
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r2_trend_down",
                description="R2: Trend state is DOWN",
                passed=r2_trend_down,
                evidence={
                    "trend_state": trend_state_str,
                    "close": close,
                    "ma50": ma50,
                    "ma200": ma200,
                    "ma50_slope": ma50_slope if not np.isnan(ma50_slope) else 0.0,
                },
                regime_target="R2",
                category="trend",
                priority=1,
            )
        )

        # R2 Rule 2: High vol + below MA50 (compound: categorical + threshold)
        # Split into two sub-rules for better explainability
        r2_vol_high = vol_state == VolState.HIGH
        r2_below_ma50 = close < ma50
        r2_vol_breakdown = r2_vol_high and r2_below_ma50

        # Sub-rule 2a: Vol state is HIGH (categorical)
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r2_vol_high",
                description="R2: Volatility is HIGH",
                passed=r2_vol_high,
                evidence={
                    "vol_state": vol_state_str,
                    "atr_pct_63": atr_pct_63,
                    "atr_pct_252": atr_pct_252,
                },
                regime_target="R2",
                category="vol",
                priority=1,
            )
        )

        # Sub-rule 2b: Close below MA50 (threshold) - uses eval_condition
        rules_fired.append(
            create_threshold_rule_trace(
                rule_id="r2_below_ma50",
                description="R2: Close < MA50",
                metric_name="close",
                current_value=close,
                threshold=ma50,
                operator="<",
                unit="$",
                evidence={"close": close, "ma50": ma50},
                regime_target="R2",
                category="vol",
                priority=1,
            )
        )

        # R2 Rule 3: IV HIGH at market level (categorical)
        r2_iv_high = iv_state == IVState.HIGH and is_market_level
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r2_iv_high",
                description="R2: IV is HIGH (market level)",
                passed=r2_iv_high,
                evidence={
                    "iv_state": iv_state_str,
                    "is_market_level": is_market_level,
                },
                regime_target="R2",
                category="iv",
                priority=1,
            )
        )

        # R2 aggregate check
        if r2_trend_down or r2_vol_breakdown or r2_iv_high:
            return MarketRegime.R2_RISK_OFF, rules_fired

        # ========================================================================
        # R3 CHECK (Rebound Window)
        # ========================================================================
        # R3 requires ALL conditions:
        structural_confirm = close > ma20 or close > last_5_bar_high

        r3_vol_high = vol_state == VolState.HIGH
        r3_oversold = ext_state == ExtState.OVERSOLD
        r3_not_downtrend = trend_state != TrendState.DOWN
        r3_above_ma200 = close > ma200
        r3_slope_ok = np.isnan(ma50_slope) or ma50_slope > -0.02
        r3_structural = structural_confirm

        ext_state_str = ext_state.value if hasattr(ext_state, "value") else str(ext_state)
        trend_state_str = trend_state.value if hasattr(trend_state, "value") else str(trend_state)

        # Create rule traces for each R3 condition using helper functions
        # R3 vol_high: categorical check with threshold context
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r3_vol_high",
                description="R3: Volatility is HIGH",
                passed=r3_vol_high,
                evidence={
                    "vol_state": vol_state_str,
                    "atr_pct_63": atr_pct_63,
                    "atr_pct_252": atr_pct_252,
                },
                regime_target="R3",
                category="vol",
                priority=2,
            )
        )

        # R3 oversold: threshold-based (ext_value <= -2.0 ATR)
        rules_fired.append(
            create_threshold_rule_trace(
                rule_id="r3_oversold",
                description="R3: Extension is OVERSOLD",
                metric_name="ext_atr_units",
                current_value=ext_value,
                threshold=-2.0,
                operator="<=",
                unit=" ATR",
                evidence={
                    "ext_state": ext_state_str,
                    "ext_value": ext_value,
                },
                regime_target="R3",
                category="ext",
                priority=2,
            )
        )

        # R3 not_downtrend: categorical check
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r3_not_downtrend",
                description="R3: Trend is NOT DOWN",
                passed=r3_not_downtrend,
                evidence={
                    "trend_state": trend_state_str,
                    "close": close,
                    "ma50": ma50,
                    "ma200": ma200,
                },
                regime_target="R3",
                category="trend",
                priority=2,
            )
        )

        # R3 above_ma200: threshold-based (close > ma200)
        rules_fired.append(
            create_threshold_rule_trace(
                rule_id="r3_above_ma200",
                description="R3: Close > MA200",
                metric_name="close",
                current_value=close,
                threshold=ma200,
                operator=">",
                unit="$",
                evidence={"close": close, "ma200": ma200},
                regime_target="R3",
                category="trend",
                priority=2,
            )
        )

        # R3 slope_ok: threshold-based with NaN handling
        # When slope is NaN, we treat it as passing (no penalty for missing data)
        slope_for_eval = ma50_slope if not np.isnan(ma50_slope) else 0.0
        rules_fired.append(
            create_threshold_rule_trace(
                rule_id="r3_slope_ok",
                description="R3: MA50 slope > -2%",
                metric_name="ma50_slope",
                current_value=slope_for_eval,
                threshold=-0.02,
                operator=">",
                unit="%",
                evidence={"ma50_slope": ma50_slope, "is_nan": np.isnan(ma50_slope)},
                regime_target="R3",
                category="trend",
                priority=2,
            )
        )

        # R3 structural: categorical compound check
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r3_structural",
                description="R3: Structural confirm (close > MA20 or 5-bar high)",
                passed=r3_structural,
                evidence={
                    "close": close,
                    "ma20": ma20,
                    "last_5_bar_high": last_5_bar_high,
                    "close_above_ma20": close > ma20,
                    "close_above_5bar_high": close > last_5_bar_high,
                },
                regime_target="R3",
                category="ext",
                priority=2,
            )
        )

        # R3 aggregate check (ALL must pass)
        r3_all_pass = (
            r3_vol_high
            and r3_oversold
            and r3_not_downtrend
            and r3_above_ma200
            and r3_slope_ok
            and r3_structural
        )
        if r3_all_pass:
            return MarketRegime.R3_REBOUND_WINDOW, rules_fired

        # ========================================================================
        # R1 CHECK (Choppy/Extended)
        # ========================================================================
        chop_state_str = chop_state.value if hasattr(chop_state, "value") else str(chop_state)

        # Check for strong trend acceleration (exception to R1)
        is_strong_trend_acceleration = (
            trend_state == TrendState.UP
            and not np.isnan(ma50_slope)
            and ma50_slope > 0.03
            and chop_state == ChopState.TRENDING
        )

        # R1 acceleration exception: compound categorical check
        accel_slope = ma50_slope if not np.isnan(ma50_slope) else 0.0
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r1_trend_acceleration",
                description="R1 Exception: Strong trend acceleration",
                passed=is_strong_trend_acceleration,
                evidence={
                    "trend_state": trend_state_str,
                    "ma50_slope": accel_slope,
                    "chop_state": chop_state_str,
                    "chop_pct": chop_pct,
                    "trend_up": trend_state == TrendState.UP,
                    "slope_above_3pct": accel_slope > 0.03,
                    "is_trending": chop_state == ChopState.TRENDING,
                },
                regime_target="R0",  # This exception favors R0
                category="trend",
                priority=3,
            )
        )

        # R1 Rule 1: Trend UP + Choppy (compound categorical)
        r1_trend_choppy = trend_state == TrendState.UP and chop_state == ChopState.CHOPPY
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r1_trend_choppy",
                description="R1: Trend UP + Choppy market",
                passed=r1_trend_choppy,
                evidence={
                    "trend_state": trend_state_str,
                    "chop_state": chop_state_str,
                    "chop_pct": chop_pct,
                    "close": close,
                    "ma50": ma50,
                    "trend_up": trend_state == TrendState.UP,
                    "is_choppy": chop_state == ChopState.CHOPPY,
                },
                regime_target="R1",
                category="chop",
                priority=3,
            )
        )

        # R1 Rule 2: Trend UP + Overbought (unless strong acceleration)
        r1_trend_overbought = (
            trend_state == TrendState.UP
            and ext_state == ExtState.OVERBOUGHT
            and not is_strong_trend_acceleration
        )
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r1_trend_overbought",
                description="R1: Trend UP + Overbought (not accelerating)",
                passed=r1_trend_overbought,
                evidence={
                    "trend_state": trend_state_str,
                    "ext_state": ext_state_str,
                    "ext_value": ext_value,
                    "is_strong_trend_acceleration": is_strong_trend_acceleration,
                    "trend_up": trend_state == TrendState.UP,
                    "is_overbought": ext_state == ExtState.OVERBOUGHT,
                },
                regime_target="R1",
                category="ext",
                priority=3,
            )
        )

        # R1 aggregate check
        if r1_trend_choppy or r1_trend_overbought:
            return MarketRegime.R1_CHOPPY_EXTENDED, rules_fired

        # ========================================================================
        # R0 CHECK (Healthy Uptrend)
        # ========================================================================
        r0_trend_up = trend_state == TrendState.UP
        r0_vol_not_high = vol_state != VolState.HIGH
        r0_not_choppy = chop_state != ChopState.CHOPPY

        # R0 trend_up: categorical state check
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="r0_trend_up",
                description="R0: Trend is UP",
                passed=r0_trend_up,
                evidence={
                    "trend_state": trend_state_str,
                    "close": close,
                    "ma50": ma50,
                    "ma200": ma200,
                    "ma50_slope": ma50_slope if not np.isnan(ma50_slope) else 0.0,
                    "is_trend_up": trend_state == TrendState.UP,
                },
                regime_target="R0",
                category="trend",
                priority=4,
            )
        )

        # R0 vol_not_high: threshold-based (atr_pct_63 < 80)
        rules_fired.append(
            create_threshold_rule_trace(
                rule_id="r0_vol_not_high",
                description="R0: Volatility is NOT HIGH",
                metric_name="atr_pct_63",
                current_value=atr_pct_63,
                threshold=80.0,
                operator="<",
                unit="%",
                evidence={
                    "vol_state": vol_state_str,
                    "atr_pct_63": atr_pct_63,
                    "atr_pct_252": atr_pct_252,
                },
                regime_target="R0",
                category="vol",
                priority=4,
            )
        )

        # R0 not_choppy: threshold-based (chop_pct < 70)
        rules_fired.append(
            create_threshold_rule_trace(
                rule_id="r0_not_choppy",
                description="R0: Market is NOT Choppy",
                metric_name="chop_pct",
                current_value=chop_pct,
                threshold=70.0,
                operator="<",
                unit="%",
                evidence={
                    "chop_state": chop_state_str,
                    "chop_pct": chop_pct,
                },
                regime_target="R0",
                category="chop",
                priority=4,
            )
        )

        # R0 aggregate check (ALL must pass)
        if r0_trend_up and r0_vol_not_high and r0_not_choppy:
            return MarketRegime.R0_HEALTHY_UPTREND, rules_fired

        # ========================================================================
        # FALLBACK to R1
        # ========================================================================
        rules_fired.append(
            create_categorical_rule_trace(
                rule_id="fallback_r1",
                description="Fallback: No regime conditions fully met",
                passed=True,
                evidence={
                    "trend_state": trend_state_str,
                    "vol_state": vol_state_str,
                    "chop_state": chop_state_str,
                    "ext_state": ext_state_str,
                    "close": close,
                    "ma50": ma50,
                    "ma200": ma200,
                },
                regime_target="R1",
                category="fallback",
                priority=5,
            )
        )
        return MarketRegime.R1_CHOPPY_EXTENDED, rules_fired

    def _compute_confidence(self, state: Dict[str, Any], regime: MarketRegime) -> int:
        """
        Compute confidence score (0-100) for the regime classification.

        Confidence is based on how clearly the conditions are met.
        """
        confidence = 50  # Base confidence

        trend_state = state["trend_state"]
        vol_state = state["vol_state"]
        chop_state = state["chop_state"]
        ext_state = state["ext_state"]

        if regime == MarketRegime.R0_HEALTHY_UPTREND:
            # R0: Higher confidence if all conditions strongly met
            if trend_state == TrendState.UP:
                confidence += 20
            if vol_state == VolState.LOW:
                confidence += 15
            elif vol_state == VolState.NORMAL:
                confidence += 10
            if chop_state == ChopState.TRENDING:
                confidence += 15

        elif regime == MarketRegime.R1_CHOPPY_EXTENDED:
            # R1: Moderate confidence
            if chop_state == ChopState.CHOPPY:
                confidence += 15
            if ext_state in (ExtState.OVERBOUGHT, ExtState.SLIGHTLY_HIGH):
                confidence += 10

        elif regime == MarketRegime.R2_RISK_OFF:
            # R2: High confidence when conditions are clear
            if trend_state == TrendState.DOWN:
                confidence += 25
            if vol_state == VolState.HIGH:
                confidence += 20

        elif regime == MarketRegime.R3_REBOUND_WINDOW:
            # R3: Confidence based on oversold depth and vol spike
            if ext_state == ExtState.OVERSOLD:
                confidence += 20
            if vol_state == VolState.HIGH:
                confidence += 15

        return min(100, max(0, confidence))

    # Pre-computed valid enum value sets for performance
    _VALID_TREND_STATES = {e.value for e in TrendState}
    _VALID_VOL_STATES = {e.value for e in VolState}
    _VALID_CHOP_STATES = {e.value for e in ChopState}
    _VALID_EXT_STATES = {e.value for e in ExtState}
    _VALID_IV_STATES = {e.value for e in IVState}

    @staticmethod
    def _parse_enum_value(value: Any, enum_class: Any, valid_set: set, default: Any) -> Any:
        """
        Parse a value into an enum, handling both string and enum inputs.

        Args:
            value: Input value (string or enum)
            enum_class: Target enum class (e.g., TrendState)
            valid_set: Pre-computed set of valid string values
            default: Default enum value if parsing fails

        Returns:
            Enum value
        """
        if isinstance(value, str):
            return enum_class(value) if value in valid_set else default
        return value if value is not None else default

    def update_with_hysteresis(
        self,
        symbol: str,
        state: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> RegimeOutput:
        """
        Update regime with proper pending/count hysteresis.

        Use this method for real-time regime updates to get stable transitions.
        Thread-safe: uses internal lock for regime state access.

        Args:
            symbol: Symbol being analyzed
            state: Current state dict from get_state() or flat state dict
            timestamp: Optional timestamp for the update

        Returns:
            RegimeOutput with stable regime classification and full explainability
        """
        # Flatten state if components are nested (from get_state())
        flat_state = self._flatten_state(state)

        # Evaluate decision tree (outside lock - pure computation)
        decision_regime, rules_fired_decision = self._evaluate_decision_tree(flat_state)

        # Compute confidence for decision regime
        confidence = self._compute_confidence(flat_state, decision_regime)

        # Thread-safe state access
        with self._state_lock:
            # Get or create regime state for this symbol
            if symbol not in self._regime_states:
                self._regime_states[symbol] = RegimeState()

            regime_state = self._regime_states[symbol]

            # Save old regime BEFORE applying hysteresis (fixes previous_regime bug)
            old_regime = regime_state.current_regime

            # Apply hysteresis (creates copy internally, returns traces)
            updated_state, rules_fired_hysteresis, transition_reason = self._apply_hysteresis(
                regime_state, decision_regime, timestamp
            )
            self._regime_states[symbol] = updated_state

        # Build output (outside lock - uses immutable updated_state)
        regime_changed = updated_state.current_regime != old_regime
        final_regime = updated_state.current_regime

        # Convert string states to enums for ComponentStates (using helper)
        trend_state_val = self._parse_enum_value(
            flat_state.get("trend_state"),
            TrendState,
            self._VALID_TREND_STATES,
            TrendState.NEUTRAL,
        )
        vol_state_val = self._parse_enum_value(
            flat_state.get("vol_state"),
            VolState,
            self._VALID_VOL_STATES,
            VolState.NORMAL,
        )
        chop_state_val = self._parse_enum_value(
            flat_state.get("chop_state"),
            ChopState,
            self._VALID_CHOP_STATES,
            ChopState.NEUTRAL,
        )
        ext_state_val = self._parse_enum_value(
            flat_state.get("ext_state"),
            ExtState,
            self._VALID_EXT_STATES,
            ExtState.NEUTRAL,
        )
        iv_state_val = self._parse_enum_value(
            flat_state.get("iv_state"), IVState, self._VALID_IV_STATES, IVState.NA
        )

        # PR-A: Validate close price before building ComponentValues
        raw_close = flat_state.get("close", 0.0)
        is_valid_close, close_error = validate_close_for_regime(
            raw_close, symbol, "update_with_hysteresis"
        )
        if not is_valid_close:
            logger.error(close_error)
            # Log available state keys for debugging
            logger.debug(
                f"[{symbol}] flat_state keys: {list(flat_state.keys())}, "
                f"close value type: {type(raw_close)}"
            )

        # Use validated close (or 0.0 if invalid - will be flagged in quality)
        validated_close = raw_close if is_valid_close else 0.0

        # Build component values with validated close
        component_values = ComponentValues(
            close=validated_close,
            ma20=flat_state.get("ma20", 0.0),
            ma50=flat_state.get("ma50", 0.0),
            ma200=flat_state.get("ma200", 0.0),
            ma50_slope=flat_state.get("ma50_slope", 0.0),
            atr20=flat_state.get("atr20", 0.0),
            atr_pct=flat_state.get("atr_pct", 0.0),
            atr_pct_63=flat_state.get("atr_pct_63", 50.0),
            atr_pct_252=flat_state.get("atr_pct_252", 50.0),
            chop=flat_state.get("chop", 50.0),
            chop_pct_252=flat_state.get("chop_pct_252", 50.0),
            ma20_crosses=flat_state.get("ma20_crosses", 0),
            ext=flat_state.get("ext", 0.0),
            last_5_bar_high=flat_state.get("last_5_bar_high", 0.0),
        )

        # Build derived metrics with explicit naming
        derived_metrics = DerivedMetrics.from_component_values(component_values)

        # Build inputs used with validated close
        inputs_used = InputsUsed(
            close=validated_close,
            high=flat_state.get("high", 0.0),
            low=flat_state.get("low", 0.0),
            volume=flat_state.get("volume", 0.0),
        )

        # Build data quality (PR-A: track close validity)
        is_market = symbol.upper() in MARKET_BENCHMARKS
        component_issues: Dict[str, str] = {}
        if not is_market:
            component_issues["iv"] = "not available for non-market symbols"
        if not is_valid_close:
            component_issues["close"] = close_error

        quality = DataQuality(
            warmup_ok=True,  # Assume warmup OK at this point
            warmup_bars_needed=self.warmup_periods,
            warmup_bars_available=self.warmup_periods,  # Conservative
            component_validity={
                "trend": True,
                "vol": True,
                "chop": True,
                "ext": True,
                "iv": is_market,
                "close": is_valid_close,  # PR-A: track close validity
            },
            component_issues=component_issues,
        )

        # Build transition state
        transition = RegimeTransitionState(
            pending_regime=updated_state.pending_regime,
            pending_count=updated_state.pending_count,
            entry_threshold=(
                ENTRY_HYSTERESIS.get(updated_state.pending_regime, 0)
                if updated_state.pending_regime
                else 0
            ),
            exit_threshold=EXIT_HYSTERESIS.get(final_regime, 0),
            bars_in_current=updated_state.bars_in_current,
            last_transition_ts=updated_state.last_regime_change,
            transition_reason=transition_reason,
        )

        # Phase 4: Get turning point prediction
        turning_point_output = self._get_turning_point_prediction(symbol, flat_state)

        # Use timestamp or current time for output
        effective_ts = timestamp or datetime.now(timezone.utc)

        return RegimeOutput(
            # Schema & Identity
            schema_version="regime_output@1.0",
            symbol=symbol,
            asof_ts=effective_ts,
            bar_interval="1d",
            data_window=DataWindow(
                start_ts=effective_ts,
                end_ts=effective_ts,
                bars=1,
            ),
            # Regime Classification (Separated)
            decision_regime=decision_regime,
            final_regime=final_regime,
            regime_name=final_regime.display_name,
            confidence=confidence,
            # Component States & Values
            component_states=ComponentStates(
                trend_state=trend_state_val,
                vol_state=vol_state_val,
                chop_state=chop_state_val,
                ext_state=ext_state_val,
                iv_state=iv_state_val,
            ),
            component_values=component_values,
            # Explainability
            inputs_used=inputs_used,
            derived_metrics=derived_metrics,
            rules_fired_decision=rules_fired_decision,
            rules_fired_hysteresis=rules_fired_hysteresis,
            quality=quality,
            # Transition State
            transition=transition,
            regime_changed=regime_changed,
            previous_regime=old_regime if regime_changed else None,
            # Phase 4: Turning Point Detection
            turning_point=turning_point_output,
        )

    def _apply_hysteresis(
        self,
        regime_state: RegimeState,
        candidate: MarketRegime,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[RegimeState, List[RuleTrace], Optional[str]]:
        """
        Apply hysteresis to regime transition with full tracing.

        Creates a copy of the input state to avoid mutation side effects.

        1. Compute candidate_regime from priority tree
        2. If candidate != current AND candidate != pending: reset pending
        3. If candidate == pending: increment pending_count
        4. If pending_count >= entry_threshold: switch regime
        5. Exit hysteresis: can't leave current until bars_in_current >= exit_threshold

        Returns:
            Tuple of (updated_state, rules_fired_hysteresis, transition_reason)
        """
        # Create a copy to avoid mutating the input
        state = replace(regime_state)
        rules_fired: List[RuleTrace] = []
        transition_reason: Optional[str] = None

        # === STEP 1: Exit hysteresis check ===
        if candidate != state.current_regime:
            exit_threshold = EXIT_HYSTERESIS[state.current_regime]

            # Add exit hysteresis rule trace
            exit_blocked = state.bars_in_current < exit_threshold
            rules_fired.append(
                RuleTrace(
                    rule_id="hysteresis_exit_check",
                    description=f"Exit check: bars_in_current >= exit_threshold",
                    passed=not exit_blocked,
                    evidence={
                        "bars_in_current": state.bars_in_current,
                        "exit_threshold": exit_threshold,
                        "current_regime": state.current_regime.value,
                    },
                    regime_target=state.current_regime.value,
                    category="hysteresis",
                    priority=0,
                    threshold_info=ThresholdInfo(
                        metric_name="bars_in_current",
                        current_value=state.bars_in_current,
                        threshold=exit_threshold,
                        operator=">=",
                        gap=exit_threshold - state.bars_in_current,
                        unit=" bars",
                    ),
                )
            )

            if exit_blocked:
                # Stay in current regime - blocked by exit hysteresis
                state.bars_in_current += 1
                transition_reason = (
                    f"Exit blocked: need {exit_threshold} bars in "
                    f"{state.current_regime.value}, have {state.bars_in_current - 1}"
                )
                return state, rules_fired, transition_reason

        # === STEP 2: Entry hysteresis (pending/count) ===
        if candidate != state.current_regime:
            if candidate != state.pending_regime:
                # New candidate - reset pending
                state.pending_regime = candidate
                state.pending_count = 1

                rules_fired.append(
                    RuleTrace(
                        rule_id="hysteresis_new_candidate",
                        description=f"New candidate regime detected",
                        passed=True,
                        evidence={
                            "candidate": candidate.value,
                            "previous_pending": (
                                regime_state.pending_regime.value
                                if regime_state.pending_regime
                                else None
                            ),
                        },
                        regime_target=candidate.value,
                        category="hysteresis",
                        priority=0,
                    )
                )
            else:
                # Same candidate - increment count
                state.pending_count += 1

                rules_fired.append(
                    RuleTrace(
                        rule_id="hysteresis_accumulate",
                        description=f"Accumulating confirmation for {candidate.value}",
                        passed=True,
                        evidence={
                            "candidate": candidate.value,
                            "pending_count": state.pending_count,
                        },
                        regime_target=candidate.value,
                        category="hysteresis",
                        priority=0,
                    )
                )

            # Check if candidate confirmed
            entry_threshold = ENTRY_HYSTERESIS[candidate]
            entry_confirmed = state.pending_count >= entry_threshold

            rules_fired.append(
                RuleTrace(
                    rule_id="hysteresis_entry_check",
                    description=f"Entry check: pending_count >= entry_threshold",
                    passed=entry_confirmed,
                    evidence={
                        "pending_count": state.pending_count,
                        "entry_threshold": entry_threshold,
                        "candidate": candidate.value,
                    },
                    regime_target=candidate.value,
                    category="hysteresis",
                    priority=0,
                    threshold_info=ThresholdInfo(
                        metric_name="pending_count",
                        current_value=state.pending_count,
                        threshold=entry_threshold,
                        operator=">=",
                        gap=entry_threshold - state.pending_count,
                        unit=" bars",
                    ),
                )
            )

            if entry_confirmed:
                # SWITCH to new regime
                old_regime = state.current_regime
                state.current_regime = candidate
                state.pending_regime = None
                state.pending_count = 0
                state.bars_in_current = 1
                state.last_regime_change = timestamp
                transition_reason = (
                    f"Confirmed: {old_regime.value} -> {candidate.value} "
                    f"(after {entry_threshold} bars confirmation)"
                )
            else:
                # Stay in current, waiting for confirmation
                state.bars_in_current += 1
                transition_reason = (
                    f"Pending: {candidate.value} " f"({state.pending_count}/{entry_threshold} bars)"
                )
        else:
            # Candidate == current, clear pending
            if state.pending_regime:
                rules_fired.append(
                    RuleTrace(
                        rule_id="hysteresis_pending_cleared",
                        description="Pending regime cleared (candidate = current)",
                        passed=True,
                        evidence={
                            "candidate": candidate.value,
                            "cleared_pending": state.pending_regime.value,
                        },
                        regime_target=candidate.value,
                        category="hysteresis",
                        priority=0,
                    )
                )

            state.pending_regime = None
            state.pending_count = 0
            state.bars_in_current += 1
            transition_reason = f"Stable: {candidate.value} (no pending transition)"

        return state, rules_fired, transition_reason

    def _flatten_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten state dict by merging nested 'components' into top level.

        The get_state() method returns component values nested under 'components' key,
        but _evaluate_decision_tree() expects them at the top level.

        Args:
            state: State dict from get_state() or already flat

        Returns:
            Flat state dict with all component values at top level
        """
        flat = dict(state)

        # If components are nested, merge them to top level
        if "components" in flat:
            components = flat.pop("components")
            if components:  # Guard against None
                for key, value in components.items():
                    if key not in flat:  # Don't overwrite existing top-level keys
                        flat[key] = value

        # Ensure last_5_bar_high has a default
        if "last_5_bar_high" not in flat:
            flat["last_5_bar_high"] = flat.get("close", 0.0)

        return flat

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            logger.debug(f"_safe_float conversion failed: {value!r} -> {default}, error: {e}")
            return default

    def _train_turning_point_model(self, symbol: str, days: int = 750) -> Optional[Any]:
        """
        Train a turning point model for a symbol on-demand.

        Args:
            symbol: Symbol to train model for
            days: Days of historical data to use

        Returns:
            Trained TurningPointModel or None if training fails
        """
        from .turning_point import TurningPointLabeler, TurningPointModel
        from .turning_point.features import extract_features

        logger.info(f"Auto-training turning point model for {symbol}...")

        try:
            # Fetch historical data
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days + 50}d", interval="1d")
            df.columns = df.columns.str.lower()

            if len(df) < 300:
                logger.warning(f"Insufficient data for {symbol}: only {len(df)} bars")
                return None

            # Generate labels
            labeler = TurningPointLabeler(
                atr_period=14,
                zigzag_threshold=2.0,
                risk_horizon=10,
                risk_threshold=1.5,
            )
            y_top, y_bottom, _ = labeler.generate_combined_labels(df)

            # Extract features
            features_df = extract_features(df)

            # Align data
            valid_mask = ~features_df.isna().any(axis=1)
            valid_idx = features_df.index[valid_mask][:-10]  # Exclude last horizon bars

            X = features_df.loc[valid_idx].values
            y_top_arr = y_top.loc[valid_idx].values
            y_bottom_arr = y_bottom.loc[valid_idx].values

            if len(X) < 200:
                logger.warning(f"Insufficient training samples for {symbol}: {len(X)}")
                return None

            # Train model
            model = TurningPointModel(model_type="logistic", confidence_threshold=0.7)
            model.train(
                X=X,
                y_top=y_top_arr,
                y_bottom=y_bottom_arr,
                cv_splits=5,
                label_horizon=10,
                embargo=2,
            )

            # Save model
            model_dir = PROJECT_ROOT / "models/turning_point"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{symbol.lower()}_logistic.pkl"
            model.save(model_path)

            logger.info(f"Auto-trained and saved turning point model for {symbol} to {model_path}")
            return model

        except Exception as e:
            logger.warning(f"Auto-training failed for {symbol}: {e}")
            return None

    def _get_turning_point_prediction(
        self,
        symbol: str,
        flat_state: Dict[str, Any],
        auto_train: bool = True,
    ) -> Optional[Any]:
        """
        Get turning point prediction for the current bar.

        Loads model on demand from models/turning_point/{symbol.lower()}_logistic.pkl.
        If model not found and auto_train=True, trains a new model automatically.

        Args:
            symbol: Symbol to predict for (used to find model file)
            flat_state: Flattened state dict with component values
            auto_train: Whether to auto-train if model not found (default: True)

        Returns:
            TurningPointOutput or None if model unavailable
        """
        # Lazy import to avoid circular dependency
        from .turning_point.features import TurningPointFeatures
        from .turning_point.model import TurningPointModel

        # Try to load model if not attempted yet
        symbol_key = symbol.upper()
        if symbol_key not in self._tp_model_load_attempted:
            self._tp_model_load_attempted[symbol_key] = True
            # Check both new format (symbol/active.pkl) and legacy format (symbol_logistic.pkl)
            new_model_path = PROJECT_ROOT / "models/turning_point" / symbol.lower() / "active.pkl"
            legacy_model_path = PROJECT_ROOT / "models/turning_point" / f"{symbol.lower()}_logistic.pkl"
            model_path = new_model_path if new_model_path.exists() else legacy_model_path
            if model_path.exists():
                try:
                    self._turning_point_models[symbol_key] = TurningPointModel.load(model_path)
                    logger.info(f"Loaded turning point model for {symbol} from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load turning point model for {symbol}: {e}")
            elif auto_train:
                # Auto-train model if not found
                model = self._train_turning_point_model(symbol)
                if model:
                    self._turning_point_models[symbol_key] = model
            else:
                logger.debug(f"No turning point model found at {model_path}")

        # Get model if available
        model = self._turning_point_models.get(symbol_key)
        if model is None:
            return None

        # Build features from flat_state
        # Note: Some features need to be computed from raw state, using defaults where unavailable
        try:
            # Map vol_state to vol_regime integer
            vol_state_str = flat_state.get("vol_state", "vol_normal")
            if vol_state_str == "vol_high":
                vol_regime = 1
            elif vol_state_str == "vol_low":
                vol_regime = -1
            else:
                vol_regime = 0

            # Compute features from available state
            # Note: Some features like rsi_14, roc_*, adx_value are not in standard state
            # so we use defaults for now. A full implementation would need these computed.
            close = flat_state.get("close", 0.0)
            ma20 = flat_state.get("ma20", close)
            ma50 = flat_state.get("ma50", close)
            ma200 = flat_state.get("ma200", close)
            atr = flat_state.get("atr20", 1.0) or 1.0  # Avoid division by zero

            features = TurningPointFeatures(
                # Trend features (normalized by ATR where applicable)
                price_vs_ma20=(close - ma20) / atr if atr > 0 else 0.0,
                price_vs_ma50=(close - ma50) / atr if atr > 0 else 0.0,
                price_vs_ma200=(close - ma200) / atr if atr > 0 else 0.0,
                ma20_slope=flat_state.get("ma50_slope", 0.0),  # Using ma50_slope as proxy
                ma50_slope=flat_state.get("ma50_slope", 0.0),
                ma20_vs_ma50=(ma20 - ma50) / atr if atr > 0 else 0.0,
                # Volatility features
                atr_pct_63=flat_state.get("atr_pct_63", 50.0),
                atr_pct_252=flat_state.get("atr_pct_252", 50.0),
                atr_expansion_rate=0.0,  # Would need historical ATR
                vol_regime=vol_regime,
                # Chop/Range features
                chop_pct_252=flat_state.get("chop_pct_252", 50.0),
                adx_value=25.0,  # Default - would need ADX indicator
                range_position=0.5,  # Default - would need range calculation
                # Extension features
                ext_atr_units=flat_state.get("ext", 0.0),
                ext_zscore=flat_state.get("ext", 0.0),  # Using ext as proxy
                rsi_14=50.0,  # Default - would need RSI indicator
                # Rate of change features (defaults)
                roc_5=0.0,
                roc_10=0.0,
                roc_20=0.0,
                # Delta features (defaults - would need previous bar)
                delta_atr_pct=0.0,
                delta_chop_pct=0.0,
                delta_ext=0.0,
            )

            return model.predict(features)

        except Exception as e:
            logger.warning(f"Turning point prediction failed for {symbol}: {e}")
            return None

    def reset_state(self, symbol: Optional[str] = None) -> None:
        """
        Reset regime state for a symbol or all symbols.

        Thread-safe: uses internal lock for state access.

        Args:
            symbol: Symbol to reset, or None to reset all
        """
        with self._state_lock:
            if symbol is None:
                self._regime_states.clear()
            elif symbol in self._regime_states:
                del self._regime_states[symbol]
