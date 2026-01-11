"""
Pre-built rules for volume indicators.

Includes rules for:
- OBV: Trend divergence alerts
- VWAP: Price vs VWAP crosses
- Force Index: Momentum confirmation
- Volume Profile: Key level alerts
"""

from ..models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    SignalRule,
)

VOLUME_RULES = [
    # =========================================================================
    # OBV (On-Balance Volume) Rules
    # =========================================================================
    SignalRule(
        name="obv_trend_up",
        indicator="obv",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.BUY,
        strength=55,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "trend",
            "from": ["down", "bearish", "neutral"],
            "to": ["up", "bullish"],
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} OBV trend turned bullish (accumulation)",
    ),
    SignalRule(
        name="obv_trend_down",
        indicator="obv",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.SELL,
        strength=55,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "trend",
            "from": ["up", "bullish", "neutral"],
            "to": ["down", "bearish"],
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} OBV trend turned bearish (distribution)",
    ),
    SignalRule(
        name="obv_divergence_bullish",
        indicator="obv",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.BUY,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "divergence",
            "from": ["none", "neutral"],
            "to": ["bullish"],
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} OBV bullish divergence detected",
    ),
    SignalRule(
        name="obv_divergence_bearish",
        indicator="obv",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.SELL,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "divergence",
            "from": ["none", "neutral"],
            "to": ["bearish"],
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} OBV bearish divergence detected",
    ),
    # =========================================================================
    # VWAP Rules
    # =========================================================================
    SignalRule(
        name="vwap_cross_above",
        indicator="vwap",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.BUY,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "position",
            "from": ["below"],
            "to": ["above"],
        },
        timeframes=("1h", "4h"),
        cooldown_seconds=3600,
        message_template="{symbol} price crossed above VWAP (bullish)",
    ),
    SignalRule(
        name="vwap_cross_below",
        indicator="vwap",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.SELL,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "position",
            "from": ["above"],
            "to": ["below"],
        },
        timeframes=("1h", "4h"),
        cooldown_seconds=3600,
        message_template="{symbol} price crossed below VWAP (bearish)",
    ),
    # =========================================================================
    # Force Index Rules
    # =========================================================================
    SignalRule(
        name="force_bullish_cross",
        indicator="force",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.BUY,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_UP,
        condition_config={
            "field": "value",
            "threshold": 0,
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Force Index turned positive (buying pressure)",
    ),
    SignalRule(
        name="force_bearish_cross",
        indicator="force",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.SELL,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
        condition_config={
            "field": "value",
            "threshold": 0,
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Force Index turned negative (selling pressure)",
    ),
    # =========================================================================
    # Volume Spike Rules
    # =========================================================================
    SignalRule(
        name="volume_spike",
        indicator="volume",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.ALERT,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "spike",
            "from": [False, 0, "false", "normal"],
            "to": [True, 1, "true", "spike", "high"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} volume spike detected - significant activity",
    ),
    SignalRule(
        name="volume_dry_up",
        indicator="volume",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.ALERT,
        strength=50,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "relative",
            "from": ["normal", "average", "high"],
            "to": ["low", "dry"],
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} volume dried up - potential consolidation",
    ),
    # =========================================================================
    # CMF (Chaikin Money Flow) Rules
    # =========================================================================
    SignalRule(
        name="cmf_bullish",
        indicator="cmf",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.BUY,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_UP,
        condition_config={
            "field": "value",
            "threshold": 0,
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} CMF turned positive (money flow bullish)",
    ),
    SignalRule(
        name="cmf_bearish",
        indicator="cmf",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.SELL,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
        condition_config={
            "field": "value",
            "threshold": 0,
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} CMF turned negative (money flow bearish)",
    ),
]
