"""
Pre-built rules for short-timeframe (1m, 5m, 15m) signals.

These rules target intraday trading with lower strength values and shorter
cooldowns compared to swing/position trading rules. All signals are initially
ALERT direction (informational) and can be upgraded to BUY/SELL after validation.

Design Considerations:
- Shorter cooldowns (60-300s) match bar frequency
- Lower strength (30-50) reflects higher noise in short timeframes
- ALERT direction prevents premature trading decisions
- LOW/MEDIUM priority reduces signal feed noise

Includes rules for:
- RSI: Zone exits and extreme conditions
- MACD: Crossovers and histogram reversals
- KDJ: Stochastic crossovers
- Williams %R: Zone exits
- ROC: Momentum direction changes
- SuperTrend: Short-term trend flips
- ATR: Volatility changes
- OBV/CVD: Volume pressure changes
"""

from ..models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    SignalRule,
)

SHORT_TIMEFRAME_RULES = [
    # =========================================================================
    # RSI Short-Timeframe Rules
    # =========================================================================
    SignalRule(
        name="rsi_st_oversold_exit",
        indicator="rsi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=40,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["oversold"],
            "to": ["neutral"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} RSI exiting oversold (short-term)",
    ),
    SignalRule(
        name="rsi_st_overbought_exit",
        indicator="rsi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=40,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["overbought"],
            "to": ["neutral"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} RSI exiting overbought (short-term)",
    ),
    SignalRule(
        name="rsi_st_extreme_oversold",
        indicator="rsi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=50,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
        condition_config={
            "field": "value",
            "threshold": 20,
            "detect_initial": True,  # Detect extreme conditions on first evaluation
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=180,
        message_template="{symbol} RSI extremely oversold (<20)",
    ),
    SignalRule(
        name="rsi_st_extreme_overbought",
        indicator="rsi",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=50,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.THRESHOLD_CROSS_UP,
        condition_config={
            "field": "value",
            "threshold": 80,
            "detect_initial": True,  # Detect extreme conditions on first evaluation
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=180,
        message_template="{symbol} RSI extremely overbought (>80)",
    ),
    # =========================================================================
    # MACD Short-Timeframe Rules
    # =========================================================================
    SignalRule(
        name="macd_st_bullish_cross",
        indicator="macd",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=45,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "macd",
            "line_b": "signal",
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=180,
        message_template="{symbol} MACD bullish crossover (short-term)",
    ),
    SignalRule(
        name="macd_st_bearish_cross",
        indicator="macd",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=45,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "macd",
            "line_b": "signal",
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=180,
        message_template="{symbol} MACD bearish crossover (short-term)",
    ),
    SignalRule(
        name="macd_st_histogram_bullish",
        indicator="macd",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.THRESHOLD_CROSS_UP,
        condition_config={
            "field": "histogram",
            "threshold": 0,
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} MACD histogram turned positive (short-term)",
    ),
    SignalRule(
        name="macd_st_histogram_bearish",
        indicator="macd",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
        condition_config={
            "field": "histogram",
            "threshold": 0,
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} MACD histogram turned negative (short-term)",
    ),
    # =========================================================================
    # KDJ (Stochastic) Short-Timeframe Rules
    # =========================================================================
    SignalRule(
        name="kdj_st_bullish_cross",
        indicator="kdj",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=40,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.CROSS_UP,
        condition_config={
            "line_a": "k",
            "line_b": "d",
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} KDJ bullish crossover (short-term)",
    ),
    SignalRule(
        name="kdj_st_bearish_cross",
        indicator="kdj",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=40,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.CROSS_DOWN,
        condition_config={
            "line_a": "k",
            "line_b": "d",
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} KDJ bearish crossover (short-term)",
    ),
    SignalRule(
        name="kdj_st_oversold_exit",
        indicator="kdj",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=45,
        priority=SignalPriority.MEDIUM,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["oversold"],
            "to": ["neutral"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=180,
        message_template="{symbol} KDJ exiting oversold (short-term)",
    ),
    # =========================================================================
    # Williams %R Short-Timeframe Rules
    # =========================================================================
    SignalRule(
        name="williams_st_oversold_exit",
        indicator="williams_r",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["oversold"],
            "to": ["neutral"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} Williams %R exiting oversold (short-term)",
    ),
    SignalRule(
        name="williams_st_overbought_exit",
        indicator="williams_r",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "zone",
            "from": ["overbought"],
            "to": ["neutral"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} Williams %R exiting overbought (short-term)",
    ),
    # =========================================================================
    # ROC Short-Timeframe Rules
    # =========================================================================
    SignalRule(
        name="roc_st_bullish_cross",
        indicator="roc",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "direction",
            "from": ["bearish", "neutral"],
            "to": ["bullish"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} ROC turned positive (short-term momentum)",
    ),
    SignalRule(
        name="roc_st_bearish_cross",
        indicator="roc",
        category=SignalCategory.MOMENTUM,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "direction",
            "from": ["bullish", "neutral"],
            "to": ["bearish"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} ROC turned negative (short-term momentum)",
    ),
    # =========================================================================
    # SuperTrend Short-Timeframe Rules
    # =========================================================================
    SignalRule(
        name="supertrend_st_bullish_flip",
        indicator="supertrend",
        category=SignalCategory.TREND,
        direction=SignalDirection.ALERT,
        strength=45,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "trend",
            "from": ["bearish"],
            "to": ["bullish"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=180,
        message_template="{symbol} SuperTrend flipped bullish (short-term)",
    ),
    SignalRule(
        name="supertrend_st_bearish_flip",
        indicator="supertrend",
        category=SignalCategory.TREND,
        direction=SignalDirection.ALERT,
        strength=45,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "trend",
            "from": ["bullish"],
            "to": ["bearish"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=180,
        message_template="{symbol} SuperTrend flipped bearish (short-term)",
    ),
    # =========================================================================
    # ATR Short-Timeframe Rules
    # =========================================================================
    SignalRule(
        name="atr_st_expansion",
        indicator="atr",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=40,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "volatility",
            "from": ["normal", "low"],
            "to": ["high"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=180,
        message_template="{symbol} ATR expanding - volatility spike (short-term)",
    ),
    SignalRule(
        name="atr_st_contraction",
        indicator="atr",
        category=SignalCategory.VOLATILITY,
        direction=SignalDirection.ALERT,
        strength=30,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "volatility",
            "from": ["normal", "high"],
            "to": ["low"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=300,
        message_template="{symbol} ATR contracting - low volatility (short-term)",
    ),
    # =========================================================================
    # OBV Short-Timeframe Rules
    # =========================================================================
    SignalRule(
        name="obv_st_accumulation",
        indicator="obv",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "trend",
            "from": ["distribution", "neutral"],
            "to": ["accumulation"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} OBV shows accumulation (short-term)",
    ),
    SignalRule(
        name="obv_st_distribution",
        indicator="obv",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "trend",
            "from": ["accumulation", "neutral"],
            "to": ["distribution"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} OBV shows distribution (short-term)",
    ),
    # =========================================================================
    # CVD Short-Timeframe Rules
    # =========================================================================
    SignalRule(
        name="cvd_st_buying_pressure",
        indicator="cvd",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "pressure",
            "from": ["selling", "neutral"],
            "to": ["buying"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} CVD buying pressure detected (short-term)",
    ),
    SignalRule(
        name="cvd_st_selling_pressure",
        indicator="cvd",
        category=SignalCategory.VOLUME,
        direction=SignalDirection.ALERT,
        strength=35,
        priority=SignalPriority.LOW,
        condition_type=ConditionType.STATE_CHANGE,
        condition_config={
            "field": "pressure",
            "from": ["buying", "neutral"],
            "to": ["selling"],
        },
        timeframes=("1m", "5m", "15m"),
        cooldown_seconds=120,
        message_template="{symbol} CVD selling pressure detected (short-term)",
    ),
]
