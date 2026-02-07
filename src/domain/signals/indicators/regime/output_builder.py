"""
Output builder for RegimeOutput construction.

Separates the output building logic from the main detector class.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.domain.signals.data.quality_validator import validate_close_for_regime
from src.utils.logging_setup import get_logger

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
from .rule_trace import RuleTrace

logger = get_logger(__name__)

# Pre-computed valid enum value sets for performance
_VALID_TREND_STATES = {e.value for e in TrendState}
_VALID_VOL_STATES = {e.value for e in VolState}
_VALID_CHOP_STATES = {e.value for e in ChopState}
_VALID_EXT_STATES = {e.value for e in ExtState}
_VALID_IV_STATES = {e.value for e in IVState}


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    import numpy as np

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_regime_output(
    symbol: str,
    flat_state: Dict[str, Any],
    decision_regime: MarketRegime,
    final_regime: MarketRegime,
    old_regime: MarketRegime,
    confidence: int,
    updated_state: RegimeState,
    rules_fired_decision: List[RuleTrace],
    rules_fired_hysteresis: List[RuleTrace],
    transition_reason: Optional[str],
    turning_point_output: Optional[Any],
    timestamp: Optional[datetime] = None,
    timeframe: str = "1d",
    warmup_periods: int = 252,
) -> RegimeOutput:
    """
    Build a complete RegimeOutput from component data.

    Args:
        symbol: Symbol being analyzed
        flat_state: Flattened state dict
        decision_regime: Regime from decision tree
        final_regime: Final regime after hysteresis
        old_regime: Previous regime (before this update)
        confidence: Confidence score
        updated_state: Updated RegimeState after hysteresis
        rules_fired_decision: Rule traces from decision tree
        rules_fired_hysteresis: Rule traces from hysteresis
        transition_reason: Human-readable transition reason
        turning_point_output: Turning point prediction (or None)
        timestamp: Optional timestamp
        timeframe: Bar interval
        warmup_periods: Required warmup periods

    Returns:
        Fully constructed RegimeOutput
    """
    regime_changed = final_regime != old_regime

    # Convert string states to enums for ComponentStates
    trend_state_val = _parse_enum_value(
        flat_state.get("trend_state"),
        TrendState,
        _VALID_TREND_STATES,
        TrendState.NEUTRAL,
    )
    vol_state_val = _parse_enum_value(
        flat_state.get("vol_state"),
        VolState,
        _VALID_VOL_STATES,
        VolState.NORMAL,
    )
    chop_state_val = _parse_enum_value(
        flat_state.get("chop_state"),
        ChopState,
        _VALID_CHOP_STATES,
        ChopState.NEUTRAL,
    )
    ext_state_val = _parse_enum_value(
        flat_state.get("ext_state"),
        ExtState,
        _VALID_EXT_STATES,
        ExtState.NEUTRAL,
    )
    iv_state_val = _parse_enum_value(
        flat_state.get("iv_state"), IVState, _VALID_IV_STATES, IVState.NA
    )

    # Validate close price
    raw_close = flat_state.get("close", 0.0)
    is_valid_close, close_error = validate_close_for_regime(
        raw_close, symbol, "build_regime_output"
    )
    if not is_valid_close:
        logger.error(close_error)
        logger.debug(
            f"[{symbol}] flat_state keys: {list(flat_state.keys())}, "
            f"close value type: {type(raw_close)}"
        )

    validated_close = raw_close if is_valid_close else 0.0

    # Build component values
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

    derived_metrics = DerivedMetrics.from_component_values(component_values)

    inputs_used = InputsUsed(
        close=validated_close,
        high=flat_state.get("high", 0.0),
        low=flat_state.get("low", 0.0),
        volume=flat_state.get("volume", 0.0),
    )

    # Build data quality
    is_market = symbol.upper() in MARKET_BENCHMARKS
    component_issues: Dict[str, str] = {}
    if not is_market:
        component_issues["iv"] = "not available for non-market symbols"
    if not is_valid_close:
        component_issues["close"] = close_error

    quality = DataQuality(
        warmup_ok=True,
        warmup_bars_needed=warmup_periods,
        warmup_bars_available=warmup_periods,
        component_validity={
            "trend": True,
            "vol": True,
            "chop": True,
            "ext": True,
            "iv": is_market,
            "close": is_valid_close,
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

    effective_ts = timestamp or datetime.now(timezone.utc)

    return RegimeOutput(
        # Schema & Identity
        schema_version="regime_output@1.0",
        symbol=symbol,
        asof_ts=effective_ts,
        bar_interval=timeframe,
        data_window=DataWindow(
            start_ts=effective_ts,
            end_ts=effective_ts,
            bars=1,
        ),
        # Regime Classification
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
        # Turning Point Detection
        turning_point=turning_point_output,
        # Composite Scoring
        composite_score=flat_state.get("composite_score"),
        composite_factors=(
            {
                "trend": flat_state.get("composite_trend"),
                "trend_short": flat_state.get("composite_trend_short"),
                "momentum": flat_state.get("composite_momentum"),
                "volatility": flat_state.get("composite_volatility"),
                "breadth": flat_state.get("composite_breadth"),
            }
            if flat_state.get("composite_score")
            else None
        ),
    )
