"""
Regime Detection Models and Enums.

Defines the core types for the 3-level hierarchical regime detection system:
- MarketRegime: R0 (Healthy), R1 (Choppy), R2 (Risk-Off), R3 (Rebound)
- Component states: TrendState, VolState, ChopState, ExtState, IVState
- RegimeState: Tracks regime with hysteresis for stable transitions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MarketRegime(Enum):
    """
    Market regime classification with trading implications.

    Priority Order (highest to lowest for classification):
    1. R2 (Risk-Off) - Veto power, always checked first
    2. R3 (Rebound)  - Only if NOT in active downtrend + structural confirm
    3. R1 (Choppy)   - Only if NOT in strong trend acceleration
    4. R0 (Healthy)  - Default when conditions are favorable
    """

    R0_HEALTHY_UPTREND = "R0"
    R1_CHOPPY_EXTENDED = "R1"
    R2_RISK_OFF = "R2"
    R3_REBOUND_WINDOW = "R3"

    @property
    def display_name(self) -> str:
        """Human-readable name for display."""
        names = {
            "R0": "Healthy Uptrend",
            "R1": "Choppy/Extended",
            "R2": "Risk-Off",
            "R3": "Rebound Window",
        }
        return names.get(self.value, self.value)

    @property
    def severity(self) -> int:
        """Severity level for comparison (higher = more restrictive)."""
        severity_map = {
            "R0": 0,
            "R1": 1,
            "R3": 2,  # R3 is between R1 and R2 in severity
            "R2": 3,
        }
        return severity_map.get(self.value, 0)


class TrendState(Enum):
    """
    Trend state based on MA relationships and slope.

    Classification:
    - UP: close > MA200 AND MA50_slope > 0 AND close > MA50
    - DOWN: close < MA200 AND MA50_slope < 0
    - NEUTRAL: otherwise
    """

    UP = "trend_up"
    DOWN = "trend_down"
    NEUTRAL = "neutral"


class VolState(Enum):
    """
    Volatility state based on dual-window ATR percentiles.

    Classification (realized volatility):
    - HIGH: ATR_pct_63 > 80 OR ATR_pct_252 > 85
    - LOW: ATR_pct_63 < 20 AND ATR_pct_252 < 25
    - NORMAL: otherwise
    """

    HIGH = "vol_high"
    NORMAL = "vol_normal"
    LOW = "vol_low"


class IVState(Enum):
    """
    Implied volatility state based on VIX/VXN percentile.

    Only applicable at MARKET level (QQQ/SPY).
    - HIGH: VIX_pct_63 > 75 (risk warning for short put)
    - ELEVATED: VIX_pct_63 in [50, 75] (caution)
    - NORMAL: VIX_pct_63 in [25, 50] (average volatility)
    - LOW: VIX_pct_63 < 25 (quiet market, favorable for premium selling)
    - NA: IV data unavailable
    """

    HIGH = "iv_high"
    ELEVATED = "iv_elevated"
    NORMAL = "iv_normal"
    LOW = "iv_low"
    NA = "na"


class ChopState(Enum):
    """
    Choppiness state based on CHOP index percentile and MA20 crosses.

    Classification:
    - CHOPPY: CHOP_pct_252 > 70 OR MA20_crosses >= 4 (in last 10 bars)
    - TRENDING: CHOP_pct_252 < 30 AND MA20_crosses <= 1
    - NEUTRAL: otherwise

    Note: 61.8/38.2 levels are for display only, not classification.
    """

    CHOPPY = "choppy"
    TRENDING = "trending"
    NEUTRAL = "neutral"


class ExtState(Enum):
    """
    Extension state (distance from mean).

    Classification (ext = (close - MA20) / ATR20):
    - OVERBOUGHT: ext > 2.0
    - OVERSOLD: ext < -2.0
    - SLIGHTLY_HIGH: ext > 1.5
    - SLIGHTLY_LOW: ext < -1.5
    - NEUTRAL: otherwise
    """

    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    SLIGHTLY_HIGH = "slightly_high"
    SLIGHTLY_LOW = "slightly_low"
    NEUTRAL = "neutral"


# Hysteresis thresholds for regime transitions
ENTRY_HYSTERESIS = {
    MarketRegime.R2_RISK_OFF: 0,  # Immediate - no delay for risk-off
    MarketRegime.R3_REBOUND_WINDOW: 2,  # Require 2 bars to confirm bottom
    MarketRegime.R1_CHOPPY_EXTENDED: 3,  # Require 3 bars to avoid noise
    MarketRegime.R0_HEALTHY_UPTREND: 2,  # Require 2 bars to confirm recovery
}

EXIT_HYSTERESIS = {
    MarketRegime.R2_RISK_OFF: 5,  # Stay cautious for 5 bars after R2
    MarketRegime.R3_REBOUND_WINDOW: 3,  # Don't exit R3 too quickly
    MarketRegime.R1_CHOPPY_EXTENDED: 2,
    MarketRegime.R0_HEALTHY_UPTREND: 1,  # Can exit R0 quickly if conditions worsen
}


@dataclass
class RegimeState:
    """
    Tracks regime with proper hysteresis for stable transitions.

    The pending_regime/pending_count pattern ensures:
    1. R2 triggers immediately (entry_hysteresis = 0)
    2. Other regimes require confirmation bars
    3. Exit from current regime respects exit_hysteresis
    """

    current_regime: MarketRegime = MarketRegime.R1_CHOPPY_EXTENDED
    pending_regime: Optional[MarketRegime] = None
    pending_count: int = 0
    bars_in_current: int = 0
    last_regime_change: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "current_regime": self.current_regime.value,
            "pending_regime": self.pending_regime.value if self.pending_regime else None,
            "pending_count": self.pending_count,
            "bars_in_current": self.bars_in_current,
            "last_regime_change": (
                self.last_regime_change.isoformat() if self.last_regime_change else None
            ),
        }


@dataclass
class ComponentStates:
    """
    Aggregated component states for regime classification.

    Contains all the individual component classifications that
    feed into the regime decision tree.
    """

    trend_state: TrendState = TrendState.NEUTRAL
    vol_state: VolState = VolState.NORMAL
    chop_state: ChopState = ChopState.NEUTRAL
    ext_state: ExtState = ExtState.NEUTRAL
    iv_state: IVState = IVState.NA

    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary."""
        return {
            "trend_state": self.trend_state.value,
            "vol_state": self.vol_state.value,
            "chop_state": self.chop_state.value,
            "ext_state": self.ext_state.value,
            "iv_state": self.iv_state.value,
        }


@dataclass
class ComponentValues:
    """
    Raw numeric values from component calculations.

    Used for debugging, reporting, and threshold tuning.
    """

    # Price and MAs
    close: float = 0.0
    ma20: float = 0.0
    ma50: float = 0.0
    ma200: float = 0.0
    ma50_slope: float = 0.0

    # Volatility
    atr20: float = 0.0
    atr_pct: float = 0.0  # ATR as % of close
    atr_pct_63: float = 50.0  # ATR percentile (3-month)
    atr_pct_252: float = 50.0  # ATR percentile (1-year)

    # Implied volatility (market level only)
    iv_value: Optional[float] = None  # VIX/VXN raw value
    iv_pct_63: Optional[float] = None  # IV percentile

    # Choppiness
    chop: float = 50.0
    chop_pct_252: float = 50.0
    ma20_crosses: int = 0

    # Extension
    ext: float = 0.0

    # Structural (for R3 confirmation)
    last_5_bar_high: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "close": self.close,
            "ma20": self.ma20,
            "ma50": self.ma50,
            "ma200": self.ma200,
            "ma50_slope": self.ma50_slope,
            "atr20": self.atr20,
            "atr_pct": self.atr_pct,
            "atr_pct_63": self.atr_pct_63,
            "atr_pct_252": self.atr_pct_252,
            "iv_value": self.iv_value,
            "iv_pct_63": self.iv_pct_63,
            "chop": self.chop,
            "chop_pct_252": self.chop_pct_252,
            "ma20_crosses": self.ma20_crosses,
            "ext": self.ext,
            "last_5_bar_high": self.last_5_bar_high,
        }


@dataclass
class RegimeOutput:
    """
    Complete regime detection output for a single bar.

    Combines regime classification, confidence, component states,
    and transition information.
    """

    # Core classification
    regime: MarketRegime = MarketRegime.R1_CHOPPY_EXTENDED
    regime_name: str = "Choppy/Extended"
    confidence: int = 50  # 0-100

    # Component states
    component_states: ComponentStates = field(default_factory=ComponentStates)
    component_values: ComponentValues = field(default_factory=ComponentValues)

    # Transition tracking
    regime_changed: bool = False
    previous_regime: Optional[MarketRegime] = None
    bars_in_regime: int = 0

    # Metadata
    timestamp: Optional[datetime] = None
    symbol: str = ""
    is_market_level: bool = False  # True for QQQ/SPY

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "regime": self.regime.value,
            "regime_name": self.regime_name,
            "confidence": self.confidence,
            "component_states": self.component_states.to_dict(),
            "components": self.component_values.to_dict(),
            "transition": {
                "regime_changed": self.regime_changed,
                "previous_regime": self.previous_regime.value if self.previous_regime else None,
                "bars_in_regime": self.bars_in_regime,
            },
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "is_market_level": self.is_market_level,
        }
