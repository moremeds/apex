"""
Pre-built rules for trend indicators.

Includes rules for:
- SuperTrend: Direction changes
- EMA: Golden/death cross
- ADX: Trend strength changes
- Aroon: Trend direction signals
"""

from ..models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    SignalRule,
)

TREND_RULES = [
    # =========================================================================
    # SuperTrend Rules
    # =========================================================================
    SignalRule(
        name="supertrend_bullish",
        indicator="supertrend",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=75,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "direction",
            "from": ["bearish", "down", "-1"],
            "to": ["bullish", "up", "1"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} SuperTrend turned bullish",
    ),
    SignalRule(
        name="supertrend_bearish",
        indicator="supertrend",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=75,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "direction",
            "from": ["bullish", "up", "1"],
            "to": ["bearish", "down", "-1"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} SuperTrend turned bearish",
    ),
    # =========================================================================
    # EMA Cross Rules
    # =========================================================================
    SignalRule(
        name="ema_golden_cross",
        indicator="ema",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "ema_fast",
            "line_b": "ema_slow",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,  # 4 hours
        message_template="{symbol} EMA golden cross (fast crossed above slow)",
    ),
    SignalRule(
        name="ema_death_cross",
        indicator="ema",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "ema_fast",
            "line_b": "ema_slow",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,  # 4 hours
        message_template="{symbol} EMA death cross (fast crossed below slow)",
    ),
    # =========================================================================
    # SMA Cross Rules
    # =========================================================================
    SignalRule(
        name="sma_golden_cross",
        indicator="sma",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=65,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "sma_fast",
            "line_b": "sma_slow",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} SMA golden cross",
    ),
    SignalRule(
        name="sma_death_cross",
        indicator="sma",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=65,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "sma_fast",
            "line_b": "sma_slow",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} SMA death cross",
    ),
    # =========================================================================
    # ADX Rules
    # =========================================================================
    SignalRule(
        name="adx_trend_strong",
        indicator="adx",
        category=SignalCategory.TREND,
        direction=SignalDirection.ALERT,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_UP,
        condition_config={
            "field": "adx",
            "threshold": 25,
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} ADX above 25 - strong trend developing",
    ),
    SignalRule(
        name="adx_trend_very_strong",
        indicator="adx",
        category=SignalCategory.TREND,
        direction=SignalDirection.ALERT,
        strength=75,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.THRESHOLD_CROSS_UP,
        condition_config={
            "field": "adx",
            "threshold": 40,
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} ADX above 40 - very strong trend",
    ),
    SignalRule(
        name="adx_di_bullish_cross",
        indicator="adx",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "di_plus",
            "line_b": "di_minus",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} ADX +DI crossed above -DI (bullish trend)",
    ),
    SignalRule(
        name="adx_di_bearish_cross",
        indicator="adx",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "di_plus",
            "line_b": "di_minus",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} ADX +DI crossed below -DI (bearish trend)",
    ),
    # =========================================================================
    # Aroon Rules
    # =========================================================================
    SignalRule(
        name="aroon_bullish_cross",
        indicator="aroon",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "aroon_up",
            "line_b": "aroon_down",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Aroon Up crossed above Aroon Down",
    ),
    SignalRule(
        name="aroon_bearish_cross",
        indicator="aroon",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "aroon_up",
            "line_b": "aroon_down",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Aroon Up crossed below Aroon Down",
    ),
    # =========================================================================
    # PSAR Rules
    # =========================================================================
    SignalRule(
        name="psar_bullish_flip",
        indicator="psar",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "direction",
            "from": ["bearish", "down"],
            "to": ["bullish", "up"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} PSAR flipped bullish",
    ),
    SignalRule(
        name="psar_bearish_flip",
        indicator="psar",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "direction",
            "from": ["bullish", "up"],
            "to": ["bearish", "down"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} PSAR flipped bearish",
    ),
    # =========================================================================
    # Ichimoku Cloud Rules
    # =========================================================================
    SignalRule(
        name="ichimoku_tk_bullish_cross",
        indicator="ichimoku",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "tenkan",
            "line_b": "kijun",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Ichimoku TK bullish cross (Tenkan above Kijun)",
    ),
    SignalRule(
        name="ichimoku_tk_bearish_cross",
        indicator="ichimoku",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "tenkan",
            "line_b": "kijun",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Ichimoku TK bearish cross (Tenkan below Kijun)",
    ),
    SignalRule(
        name="ichimoku_cloud_bullish_breakout",
        indicator="ichimoku",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=75,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "price_vs_cloud",
            "from": ["inside", "below"],
            "to": ["above"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} Ichimoku price broke above cloud (strong bullish)",
    ),
    SignalRule(
        name="ichimoku_cloud_bearish_breakout",
        indicator="ichimoku",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=75,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "price_vs_cloud",
            "from": ["inside", "above"],
            "to": ["below"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} Ichimoku price broke below cloud (strong bearish)",
    ),
    SignalRule(
        name="ichimoku_kumo_bullish_twist",
        indicator="ichimoku",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "cloud_color",
            "from": ["red", "neutral"],
            "to": ["green"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} Ichimoku kumo twist to bullish (green cloud)",
    ),
    SignalRule(
        name="ichimoku_kumo_bearish_twist",
        indicator="ichimoku",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "cloud_color",
            "from": ["green", "neutral"],
            "to": ["red"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} Ichimoku kumo twist to bearish (red cloud)",
    ),
    # =========================================================================
    # Vortex Indicator Rules
    # =========================================================================
    SignalRule(
        name="vortex_bullish_cross",
        indicator="vortex",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "vi_plus",
            "line_b": "vi_minus",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Vortex bullish cross (VI+ above VI-)",
    ),
    SignalRule(
        name="vortex_bearish_cross",
        indicator="vortex",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "vi_plus",
            "line_b": "vi_minus",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Vortex bearish cross (VI+ below VI-)",
    ),
    # =========================================================================
    # Zero-Lag EMA Rules
    # =========================================================================
    SignalRule(
        name="zerolag_slope_bullish",
        indicator="zerolag",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "slope",
            "from": ["falling", "flat"],
            "to": ["rising"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} Zero-Lag EMA slope turned bullish",
    ),
    SignalRule(
        name="zerolag_slope_bearish",
        indicator="zerolag",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "slope",
            "from": ["rising", "flat"],
            "to": ["falling"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} Zero-Lag EMA slope turned bearish",
    ),
    # =========================================================================
    # Aroon Rules
    # =========================================================================
    SignalRule(
        name="aroon_osc_bullish_cross",
        indicator="aroon",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=55,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_UP,
        condition_config={
            "field": "oscillator",
            "threshold": 0,
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Aroon Oscillator crossed above zero (bullish trend)",
    ),
    SignalRule(
        name="aroon_osc_bearish_cross",
        indicator="aroon",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=55,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
        condition_config={
            "field": "oscillator",
            "threshold": 0,
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Aroon Oscillator crossed below zero (bearish trend)",
    ),
    # =========================================================================
    # TRIX Rules
    # =========================================================================
    SignalRule(
        name="trix_bullish_cross",
        indicator="trix",
        category=SignalCategory.TREND,
        direction=SignalDirection.BUY,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "trix",
            "line_b": "signal",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} TRIX bullish crossover (TRIX above signal)",
    ),
    SignalRule(
        name="trix_bearish_cross",
        indicator="trix",
        category=SignalCategory.TREND,
        direction=SignalDirection.SELL,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "trix",
            "line_b": "signal",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} TRIX bearish crossover (TRIX below signal)",
    ),
]
