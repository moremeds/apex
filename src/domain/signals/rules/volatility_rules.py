"""
Pre-built rules for volatility indicators.

Includes rules for:
- Bollinger Bands: Band touches, squeeze/expansion
- ATR: Volatility expansion
- Keltner Channels: Breakouts
- Squeeze Momentum: Squeeze release signals
"""

from ..models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    SignalRule,
)

VOLATILITY_RULES = [
    # =========================================================================
    # Bollinger Bands Rules
    # =========================================================================
    SignalRule(
        name="bollinger_lower_touch",
        indicator="bollinger",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.BUY,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["middle", "neutral"],
            "to": ["below_lower", "lower"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} price touched lower Bollinger Band",
    ),
    SignalRule(
        name="bollinger_upper_touch",
        indicator="bollinger",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.SELL,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["middle", "neutral"],
            "to": ["above_upper", "upper"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=3600,
        message_template="{symbol} price touched upper Bollinger Band",
    ),
    SignalRule(
        name="bollinger_squeeze",
        indicator="bollinger",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "squeeze",
            "from": [False, "false", "no", 0],
            "to": [True, "true", "yes", 1],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} Bollinger Bands squeeze detected - breakout pending",
    ),
    SignalRule(
        name="bollinger_expansion",
        indicator="bollinger",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "squeeze",
            "from": [True, "true", "yes", 1],
            "to": [False, "false", "no", 0],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} Bollinger Bands expanding - breakout in progress",
    ),
    # =========================================================================
    # ATR Rules
    # =========================================================================
    SignalRule(
        name="atr_expansion",
        indicator="atr",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "trend",
            "from": ["contracting", "stable", "neutral"],
            "to": ["expanding", "increasing"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} ATR expanding - increased volatility",
    ),
    SignalRule(
        name="atr_contraction",
        indicator="atr",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=55,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "trend",
            "from": ["expanding", "increasing", "neutral"],
            "to": ["contracting", "decreasing"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} ATR contracting - volatility decreasing",
    ),
    # =========================================================================
    # Keltner Channel Rules
    # =========================================================================
    SignalRule(
        name="keltner_upper_breakout",
        indicator="keltner",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.BUY,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["middle", "neutral"],
            "to": ["above_upper", "upper"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Keltner Channel upper breakout (bullish momentum)",
    ),
    SignalRule(
        name="keltner_lower_breakout",
        indicator="keltner",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.SELL,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["middle", "neutral"],
            "to": ["below_lower", "lower"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Keltner Channel lower breakout (bearish momentum)",
    ),
    # =========================================================================
    # Squeeze Momentum Rules
    # =========================================================================
    SignalRule(
        name="squeeze_fire_bullish",
        indicator="squeeze",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.BUY,
        strength=75,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "signal",
            "from": ["squeeze", "neutral", "off"],
            "to": ["bullish", "long", "buy"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Squeeze fired bullish - momentum breakout",
    ),
    SignalRule(
        name="squeeze_fire_bearish",
        indicator="squeeze",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.SELL,
        strength=75,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "signal",
            "from": ["squeeze", "neutral", "off"],
            "to": ["bearish", "short", "sell"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Squeeze fired bearish - momentum breakdown",
    ),
    SignalRule(
        name="squeeze_on",
        indicator="squeeze",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "squeeze_on",
            "from": [False, 0, "false"],
            "to": [True, 1, "true"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} Squeeze activated - consolidation phase",
    ),
    # =========================================================================
    # Donchian Channel Rules
    # =========================================================================
    SignalRule(
        name="donchian_upper_breakout",
        indicator="donchian",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.BUY,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "breakout",
            "from": ["none", "neutral"],
            "to": ["upper", "high"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Donchian upper channel breakout",
    ),
    SignalRule(
        name="donchian_lower_breakout",
        indicator="donchian",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.SELL,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "breakout",
            "from": ["none", "neutral"],
            "to": ["lower", "low"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Donchian lower channel breakout",
    ),
    # =========================================================================
    # Chaikin Volatility Rules
    # =========================================================================
    SignalRule(
        name="chaikin_vol_expanding",
        indicator="chaikin_vol",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "direction",
            "from": ["contracting", "neutral"],
            "to": ["expanding"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Chaikin Volatility expanding - range increasing",
    ),
    SignalRule(
        name="chaikin_vol_contracting",
        indicator="chaikin_vol",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=55,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "direction",
            "from": ["expanding", "neutral"],
            "to": ["contracting"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Chaikin Volatility contracting - range decreasing",
    ),
    # =========================================================================
    # Historical Volatility Rules
    # =========================================================================
    SignalRule(
        name="hvol_high_regime",
        indicator="hvol",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "regime",
            "from": ["normal_vol", "low_vol"],
            "to": ["high_vol"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} Historical Volatility entered high regime (>30%)",
    ),
    SignalRule(
        name="hvol_low_regime",
        indicator="hvol",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=60,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "regime",
            "from": ["normal_vol", "high_vol"],
            "to": ["low_vol"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=14400,
        message_template="{symbol} Historical Volatility entered low regime (<15%) - breakout potential",
    ),
    # =========================================================================
    # Standard Deviation Rules
    # =========================================================================
    SignalRule(
        name="stddev_spike",
        indicator="stddev",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=65,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "volatility",
            "from": ["normal", "low"],
            "to": ["high"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Standard Deviation spike - volatility increasing",
    ),
    SignalRule(
        name="stddev_compression",
        indicator="stddev",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=55,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "volatility",
            "from": ["normal", "high"],
            "to": ["low"],
        },
        timeframes=("1h", "4h", "1d"),
        cooldown_seconds=7200,
        message_template="{symbol} Standard Deviation compression - potential breakout setup",
    ),
]
