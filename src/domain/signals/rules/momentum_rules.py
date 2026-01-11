"""
Pre-built rules for momentum indicators.

Includes rules for:
- RSI: Oversold/overbought zone exits
- MACD: Bullish/bearish crossovers, histogram reversals
- KDJ: Stochastic crossovers
- MFI: Money flow zone exits
"""

from ..models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    SignalRule,
)

MOMENTUM_RULES = [
    # =========================================================================
    # RSI Rules
    # =========================================================================
    SignalRule(
        name="rsi_oversold_exit",
        indicator="rsi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.BUY,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["oversold"],
            "to": ["neutral"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} RSI exiting oversold zone",
    ),
    SignalRule(
        name="rsi_overbought_exit",
        indicator="rsi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.SELL,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["overbought"],
            "to": ["neutral"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} RSI exiting overbought zone",
    ),
    SignalRule(
        name="rsi_50_cross_up",
        indicator="rsi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.BUY,
        strength=50,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_UP,
        condition_config={
            "field": "value",
            "threshold": 50,
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} RSI crossed above 50 (bullish momentum)",
    ),
    SignalRule(
        name="rsi_50_cross_down",
        indicator="rsi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.SELL,
        strength=50,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
        condition_config={
            "field": "value",
            "threshold": 50,
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} RSI crossed below 50 (bearish momentum)",
    ),
    # =========================================================================
    # MACD Rules
    # =========================================================================
    SignalRule(
        name="macd_bullish_cross",
        indicator="macd",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.BUY,
        strength=65,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "macd",
            "line_b": "signal",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} MACD bullish crossover",
    ),
    SignalRule(
        name="macd_bearish_cross",
        indicator="macd",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.SELL,
        strength=65,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "macd",
            "line_b": "signal",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} MACD bearish crossover",
    ),
    SignalRule(
        name="macd_histogram_bullish",
        indicator="macd",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.BUY,
        strength=55,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_UP,
        condition_config={
            "field": "histogram",
            "threshold": 0,
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} MACD histogram turned positive",
    ),
    SignalRule(
        name="macd_histogram_bearish",
        indicator="macd",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.SELL,
        strength=55,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
        condition_config={
            "field": "histogram",
            "threshold": 0,
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} MACD histogram turned negative",
    ),
    # =========================================================================
    # KDJ (Stochastic) Rules
    # =========================================================================
    SignalRule(
        name="kdj_bullish_cross",
        indicator="kdj",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.BUY,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "k",
            "line_b": "d",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} KDJ bullish crossover (K crossed above D)",
    ),
    SignalRule(
        name="kdj_bearish_cross",
        indicator="kdj",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.SELL,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "k",
            "line_b": "d",
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} KDJ bearish crossover (K crossed below D)",
    ),
    SignalRule(
        name="kdj_oversold_exit",
        indicator="kdj",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.BUY,
        strength=65,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["oversold"],
            "to": ["neutral"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} KDJ exiting oversold zone",
    ),
    # =========================================================================
    # MFI (Money Flow Index) Rules
    # =========================================================================
    SignalRule(
        name="mfi_oversold_exit",
        indicator="mfi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.BUY,
        strength=65,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["oversold"],
            "to": ["neutral"],
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} MFI exiting oversold zone (buying pressure)",
    ),
    SignalRule(
        name="mfi_overbought_exit",
        indicator="mfi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.SELL,
        strength=65,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["overbought"],
            "to": ["neutral"],
        },
        timeframes=("4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} MFI exiting overbought zone (selling pressure)",
    ),
]
