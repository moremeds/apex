"""
Decision tree evaluator for regime classification.

Priority-based evaluation with full rule tracing:
1. R2 (Risk-Off) - Veto power, always checked first
2. R3 (Rebound)  - Only if NOT in active downtrend + structural confirm
3. R1 (Choppy)   - Only if NOT in strong trend acceleration
4. R0 (Healthy)  - Default when conditions are favorable
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..models import (
    ChopState,
    ExtState,
    IVState,
    MarketRegime,
    TrendState,
    VolState,
)
from ..rule_trace import (
    RuleTrace,
    create_categorical_rule_trace,
    create_threshold_rule_trace,
)


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


def evaluate_decision_tree(state: Dict[str, Any]) -> Tuple[MarketRegime, List[RuleTrace]]:
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

    # Component states - normalize to enums
    trend_state = _to_enum(state["trend_state"], TrendState, TrendState.NEUTRAL)
    vol_state = _to_enum(state["vol_state"], VolState, VolState.NORMAL)
    chop_state = _to_enum(state["chop_state"], ChopState, ChopState.NEUTRAL)
    ext_state = _to_enum(state["ext_state"], ExtState, ExtState.NEUTRAL)
    iv_state = _to_enum(state.get("iv_state", IVState.NA), IVState, IVState.NA)
    is_market_level = state.get("is_market_level", False)

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

    # Get string values for evidence
    trend_state_str = trend_state.value if hasattr(trend_state, "value") else str(trend_state)
    vol_state_str = vol_state.value if hasattr(vol_state, "value") else str(vol_state)
    iv_state_str = iv_state.value if hasattr(iv_state, "value") else str(iv_state)
    chop_state_str = chop_state.value if hasattr(chop_state, "value") else str(chop_state)
    ext_state_str = ext_state.value if hasattr(ext_state, "value") else str(ext_state)

    # ========================================================================
    # R2 CHECK (Highest Priority - Veto)
    # ========================================================================
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

    # R2 Rule 2: High vol + below MA50
    r2_vol_high = vol_state == VolState.HIGH
    r2_below_ma50 = close < ma50
    r2_vol_breakdown = r2_vol_high and r2_below_ma50

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

    # R2 Rule 3: IV HIGH at market level
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

    if r2_trend_down or r2_vol_breakdown or r2_iv_high:
        return MarketRegime.R2_RISK_OFF, rules_fired

    # ========================================================================
    # R3 CHECK (Rebound Window)
    # ========================================================================
    structural_confirm = close > ma20 or close > last_5_bar_high

    r3_vol_high = vol_state == VolState.HIGH
    r3_oversold = ext_state == ExtState.OVERSOLD
    r3_not_downtrend = trend_state != TrendState.DOWN
    r3_above_ma200 = close > ma200
    r3_slope_ok = np.isnan(ma50_slope) or ma50_slope > -0.02
    r3_structural = structural_confirm

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
    is_strong_trend_acceleration = (
        trend_state == TrendState.UP
        and not np.isnan(ma50_slope)
        and ma50_slope > 0.03
        and chop_state == ChopState.TRENDING
    )

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
            regime_target="R0",
            category="trend",
            priority=3,
        )
    )

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

    if r1_trend_choppy or r1_trend_overbought:
        return MarketRegime.R1_CHOPPY_EXTENDED, rules_fired

    # ========================================================================
    # R0 CHECK (Healthy Uptrend)
    # ========================================================================
    r0_trend_up = trend_state == TrendState.UP
    r0_vol_not_high = vol_state != VolState.HIGH
    r0_not_choppy = chop_state != ChopState.CHOPPY

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


def compute_confidence(state: Dict[str, Any], regime: MarketRegime) -> int:
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
        if trend_state == TrendState.UP:
            confidence += 20
        if vol_state == VolState.LOW:
            confidence += 15
        elif vol_state == VolState.NORMAL:
            confidence += 10
        if chop_state == ChopState.TRENDING:
            confidence += 15

    elif regime == MarketRegime.R1_CHOPPY_EXTENDED:
        if chop_state == ChopState.CHOPPY:
            confidence += 15
        if ext_state in (ExtState.OVERBOUGHT, ExtState.SLIGHTLY_HIGH):
            confidence += 10

    elif regime == MarketRegime.R2_RISK_OFF:
        if trend_state == TrendState.DOWN:
            confidence += 25
        if vol_state == VolState.HIGH:
            confidence += 20

    elif regime == MarketRegime.R3_REBOUND_WINDOW:
        if ext_state == ExtState.OVERSOLD:
            confidence += 20
        if vol_state == VolState.HIGH:
            confidence += 15

    return min(100, max(0, confidence))
