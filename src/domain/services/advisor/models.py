"""Domain models for the Trading Advisor feature."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VRPResult:
    """Volatility Risk Premium calculation result."""

    iv30: float  # Implied vol (VIX close)
    rv30: float  # 30-day realized vol (annualized)
    vrp: float  # iv30 - rv30
    vrp_zscore: float  # Z-scored over 252-day trailing window
    iv_percentile: float  # VIX percentile rank (0-100)


@dataclass(frozen=True)
class LegTemplate:
    """Template for constructing an option leg."""

    side: str  # "buy" | "sell"
    option_type: str  # "put" | "call"
    delta_target: int  # Target delta (absolute, e.g. 25)
    dte_target: int  # Target DTE


@dataclass(frozen=True)
class PremiumStrategyDef:
    """Definition of a premium-selling strategy."""

    name: str  # "short_put", "iron_condor", etc.
    display_name: str  # "Short Put", "Iron Condor"
    direction: str  # "bullish" | "bearish" | "neutral"
    risk_profile: str  # "undefined" | "defined"
    regime_fit: frozenset[str]  # {"R0", "R1"} -- suitable regimes
    leg_templates: tuple[LegTemplate, ...]


@dataclass(frozen=True)
class LegSpec:
    """Concrete option leg with estimated strike."""

    side: str
    option_type: str
    target_delta: int
    target_dte: int
    estimated_strike: float


@dataclass(frozen=True)
class PremiumAdvice:
    """Premium strategy recommendation for a symbol."""

    symbol: str
    action: str  # "SELL" | "HOLD" | "BLOCKED"
    strategy: str | None  # Strategy name or None if blocked
    display_name: str | None
    confidence: float  # 0-100
    legs: list[LegSpec]
    vrp_zscore: float
    iv_percentile: float
    term_structure_ratio: float
    regime: str
    earnings_warning: str | None
    reasoning: list[str]


@dataclass(frozen=True)
class EquityAdvice:
    """Equity buy/sell recommendation for a symbol."""

    symbol: str
    sector: str
    action: str  # "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL"
    confidence: float  # 0-100
    regime: str
    signal_summary: dict[str, int]  # {"bullish": N, "bearish": N, "neutral": N}
    top_signals: list[dict]  # Top 3 signals by strength
    trend_pulse: dict | None  # TrendPulse-specific state
    key_levels: dict[str, float]  # {"support": ..., "resistance": ...}
    reasoning: list[str]


@dataclass(frozen=True)
class MarketContext:
    """Market-level context shared by all advisor responses."""

    regime: str
    regime_name: str
    regime_confidence: float
    vix: float
    vix_percentile: float
    vrp_zscore: float
    term_structure_ratio: float
    term_structure_state: str  # "contango" | "flat" | "inverted"
    timestamp: str
