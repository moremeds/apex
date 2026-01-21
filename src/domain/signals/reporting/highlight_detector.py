"""
Highlight Detector - Detects notable patterns and conditions for summary reports.

M3 PR-03 Deliverable: Identifies divergences, regime changes, and other highlights
for the summary.json highlights section.

Constraints:
- Returns max 20 highlights to fit within HIGHLIGHTS_BUDGET_KB
- Evidence is pruned to fit budget
- Priority-ranked by importance
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    from ..indicators.regime import RegimeOutput


class HighlightType(Enum):
    """Types of highlights that can be detected."""

    REGIME_CHANGE = "regime_change"
    DIVERGENCE_RSI_PRICE = "divergence_rsi_price"
    DIVERGENCE_MACD_PRICE = "divergence_macd_price"
    DIVERGENCE_VOLUME_PRICE = "divergence_volume_price"
    EXTREME_VOLATILITY = "extreme_volatility"
    TREND_REVERSAL = "trend_reversal"
    BREAKOUT = "breakout"
    SUPPORT_RESISTANCE = "support_resistance"
    CONFLUENCE = "confluence"
    TURNING_POINT = "turning_point"


class HighlightPriority(Enum):
    """Priority levels for highlights."""

    CRITICAL = 1  # Immediate attention
    HIGH = 2  # Important
    MEDIUM = 3  # Notable
    LOW = 4  # Informational


@dataclass
class Highlight:
    """A detected highlight/notable condition."""

    type: HighlightType
    priority: HighlightPriority
    symbol: str
    timeframe: str
    title: str
    description: str
    timestamp: datetime
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_evidence: bool = True) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {
            "type": self.type.value,
            "priority": self.priority.value,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
        }
        if include_evidence and self.evidence:
            result["evidence"] = self.evidence
        return result


class HighlightDetector:
    """
    Detects notable patterns and conditions for summary reports.

    Analyzes indicator data and regime outputs to identify:
    - Divergences (RSI vs price, MACD vs price, volume vs price)
    - Regime changes (TRENDING -> CHOPPY transitions)
    - Extreme volatility conditions
    - Trend reversals
    - Breakouts
    - Multi-indicator confluence
    """

    MAX_HIGHLIGHTS = 20  # Maximum highlights to return

    def __init__(
        self,
        divergence_lookback: int = 20,
        volatility_percentile_threshold: float = 0.90,
    ) -> None:
        """
        Initialize highlight detector.

        Args:
            divergence_lookback: Bars to look back for divergence detection
            volatility_percentile_threshold: Percentile for extreme volatility
        """
        self.divergence_lookback = divergence_lookback
        self.volatility_percentile_threshold = volatility_percentile_threshold

    def detect(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        regime_outputs: Optional[Dict[str, "RegimeOutput"]] = None,
        previous_regimes: Optional[Dict[str, str]] = None,
    ) -> List[Highlight]:
        """
        Detect highlights from data and regime outputs.

        Args:
            data: Dict mapping (symbol, timeframe) to DataFrame
            regime_outputs: Optional dict mapping symbol to RegimeOutput
            previous_regimes: Optional dict of previous regime states

        Returns:
            List of Highlight objects, sorted by priority, max MAX_HIGHLIGHTS
        """
        highlights: List[Highlight] = []

        # Detect regime changes
        if regime_outputs and previous_regimes:
            regime_highlights = self._detect_regime_changes(
                regime_outputs, previous_regimes
            )
            highlights.extend(regime_highlights)

        # Detect turning points from regime outputs
        if regime_outputs:
            tp_highlights = self._detect_turning_points(regime_outputs)
            highlights.extend(tp_highlights)

        # Detect divergences and other patterns from data
        for (symbol, timeframe), df in data.items():
            if len(df) < self.divergence_lookback:
                continue

            # RSI divergence
            rsi_div = self._detect_rsi_divergence(df, symbol, timeframe)
            if rsi_div:
                highlights.append(rsi_div)

            # MACD divergence
            macd_div = self._detect_macd_divergence(df, symbol, timeframe)
            if macd_div:
                highlights.append(macd_div)

            # Volume divergence
            vol_div = self._detect_volume_divergence(df, symbol, timeframe)
            if vol_div:
                highlights.append(vol_div)

            # Extreme volatility
            vol_extreme = self._detect_extreme_volatility(df, symbol, timeframe)
            if vol_extreme:
                highlights.append(vol_extreme)

        # Sort by priority (lower = higher priority)
        highlights.sort(key=lambda h: h.priority.value)

        # Truncate to max
        return highlights[: self.MAX_HIGHLIGHTS]

    def _detect_regime_changes(
        self,
        regime_outputs: Dict[str, "RegimeOutput"],
        previous_regimes: Dict[str, str],
    ) -> List[Highlight]:
        """Detect regime changes from previous state."""
        highlights = []

        for symbol, regime_output in regime_outputs.items():
            current = regime_output.final_regime.value
            previous = previous_regimes.get(symbol)

            if previous and current != previous:
                # Determine priority based on transition type
                if current == "R2":
                    priority = HighlightPriority.CRITICAL
                    title = f"{symbol}: Risk-Off Triggered"
                elif current == "R0" and previous == "R2":
                    priority = HighlightPriority.HIGH
                    title = f"{symbol}: Recovery to Healthy Uptrend"
                elif current == "R3":
                    priority = HighlightPriority.HIGH
                    title = f"{symbol}: Rebound Window Active"
                else:
                    priority = HighlightPriority.MEDIUM
                    title = f"{symbol}: Regime Change"

                highlights.append(
                    Highlight(
                        type=HighlightType.REGIME_CHANGE,
                        priority=priority,
                        symbol=symbol,
                        timeframe="1d",
                        title=title,
                        description=f"Regime changed from {previous} to {current}",
                        timestamp=datetime.now(),
                        evidence={
                            "previous_regime": previous,
                            "current_regime": current,
                            "confidence": regime_output.confidence,
                        },
                    )
                )

        return highlights

    def _detect_turning_points(
        self,
        regime_outputs: Dict[str, "RegimeOutput"],
    ) -> List[Highlight]:
        """Detect turning point predictions from regime outputs."""
        highlights = []

        for symbol, regime_output in regime_outputs.items():
            if not regime_output.turning_point:
                continue

            tp = regime_output.turning_point
            if tp.turn_confidence >= 0.7:  # High probability turning point
                priority = HighlightPriority.HIGH
                turn_state = tp.turn_state.value
                title = f"{symbol}: Turning Point Detected ({turn_state})"
                description = f"High confidence ({tp.turn_confidence:.0%}) {turn_state} detected"

                highlights.append(
                    Highlight(
                        type=HighlightType.TURNING_POINT,
                        priority=priority,
                        symbol=symbol,
                        timeframe="1d",
                        title=title,
                        description=description,
                        timestamp=datetime.now(),
                        evidence={
                            "turn_state": turn_state,
                            "confidence": tp.turn_confidence,
                        },
                    )
                )

        return highlights

    def _detect_rsi_divergence(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Optional[Highlight]:
        """Detect RSI vs price divergence."""
        rsi_col = None
        for col in df.columns:
            if "rsi" in col.lower():
                rsi_col = col
                break

        if rsi_col is None or rsi_col not in df.columns:
            return None

        recent = df.tail(self.divergence_lookback)
        if len(recent) < 10:
            return None

        price = recent["close"].values
        rsi = recent[rsi_col].values

        # Skip if too many NaN
        if pd.isna(rsi).sum() > len(rsi) * 0.3:
            return None

        # Bearish divergence: price making higher highs, RSI making lower highs
        price_trend = price[-1] > price[0]
        rsi_trend = rsi[-1] > rsi[0] if not pd.isna(rsi[-1]) and not pd.isna(rsi[0]) else None

        if rsi_trend is None:
            return None

        if price_trend and not rsi_trend:
            return Highlight(
                type=HighlightType.DIVERGENCE_RSI_PRICE,
                priority=HighlightPriority.MEDIUM,
                symbol=symbol,
                timeframe=timeframe,
                title=f"{symbol}: Bearish RSI Divergence",
                description="Price making higher highs while RSI makes lower highs",
                timestamp=datetime.now(),
                evidence={
                    "price_start": float(price[0]),
                    "price_end": float(price[-1]),
                    "rsi_start": float(rsi[0]) if not pd.isna(rsi[0]) else None,
                    "rsi_end": float(rsi[-1]) if not pd.isna(rsi[-1]) else None,
                },
            )
        elif not price_trend and rsi_trend:
            return Highlight(
                type=HighlightType.DIVERGENCE_RSI_PRICE,
                priority=HighlightPriority.MEDIUM,
                symbol=symbol,
                timeframe=timeframe,
                title=f"{symbol}: Bullish RSI Divergence",
                description="Price making lower lows while RSI makes higher lows",
                timestamp=datetime.now(),
                evidence={
                    "price_start": float(price[0]),
                    "price_end": float(price[-1]),
                    "rsi_start": float(rsi[0]) if not pd.isna(rsi[0]) else None,
                    "rsi_end": float(rsi[-1]) if not pd.isna(rsi[-1]) else None,
                },
            )

        return None

    def _detect_macd_divergence(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Optional[Highlight]:
        """Detect MACD vs price divergence."""
        macd_col = None
        for col in df.columns:
            if "macd" in col.lower() and "signal" not in col.lower() and "histogram" not in col.lower():
                macd_col = col
                break

        if macd_col is None:
            return None

        recent = df.tail(self.divergence_lookback)
        if len(recent) < 10:
            return None

        price = recent["close"].values
        macd = recent[macd_col].values

        if pd.isna(macd).sum() > len(macd) * 0.3:
            return None

        price_trend = price[-1] > price[0]
        macd_trend = macd[-1] > macd[0] if not pd.isna(macd[-1]) and not pd.isna(macd[0]) else None

        if macd_trend is None:
            return None

        if price_trend and not macd_trend:
            return Highlight(
                type=HighlightType.DIVERGENCE_MACD_PRICE,
                priority=HighlightPriority.MEDIUM,
                symbol=symbol,
                timeframe=timeframe,
                title=f"{symbol}: Bearish MACD Divergence",
                description="Price trending up while MACD trending down",
                timestamp=datetime.now(),
                evidence={
                    "price_change_pct": float((price[-1] - price[0]) / price[0] * 100),
                },
            )
        elif not price_trend and macd_trend:
            return Highlight(
                type=HighlightType.DIVERGENCE_MACD_PRICE,
                priority=HighlightPriority.MEDIUM,
                symbol=symbol,
                timeframe=timeframe,
                title=f"{symbol}: Bullish MACD Divergence",
                description="Price trending down while MACD trending up",
                timestamp=datetime.now(),
                evidence={
                    "price_change_pct": float((price[-1] - price[0]) / price[0] * 100),
                },
            )

        return None

    def _detect_volume_divergence(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Optional[Highlight]:
        """Detect volume vs price divergence."""
        if "volume" not in df.columns:
            return None

        recent = df.tail(self.divergence_lookback)
        if len(recent) < 10:
            return None

        price = recent["close"].values
        volume = recent["volume"].values

        # Calculate trends
        price_up = price[-1] > price[0]
        avg_vol_first_half = volume[: len(volume) // 2].mean()
        avg_vol_second_half = volume[len(volume) // 2 :].mean()
        volume_up = avg_vol_second_half > avg_vol_first_half

        # Divergence: price up but volume declining
        if price_up and not volume_up and avg_vol_first_half > 0:
            vol_change = (avg_vol_second_half - avg_vol_first_half) / avg_vol_first_half * 100
            if abs(vol_change) > 20:  # Significant volume change
                return Highlight(
                    type=HighlightType.DIVERGENCE_VOLUME_PRICE,
                    priority=HighlightPriority.LOW,
                    symbol=symbol,
                    timeframe=timeframe,
                    title=f"{symbol}: Volume Declining on Price Rise",
                    description="Price rising but volume support weakening",
                    timestamp=datetime.now(),
                    evidence={
                        "volume_change_pct": float(vol_change),
                    },
                )

        return None

    def _detect_extreme_volatility(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Optional[Highlight]:
        """Detect extreme volatility conditions."""
        if len(df) < 50:
            return None

        # Calculate recent volatility
        returns = df["close"].pct_change().dropna()
        if len(returns) < 20:
            return None

        recent_vol = returns.tail(10).std()
        historical_vol = returns.std()

        # Check if recent volatility is extreme
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 0

        if vol_ratio > 2.0:  # More than 2x historical
            return Highlight(
                type=HighlightType.EXTREME_VOLATILITY,
                priority=HighlightPriority.HIGH,
                symbol=symbol,
                timeframe=timeframe,
                title=f"{symbol}: Extreme Volatility",
                description=f"Recent volatility {vol_ratio:.1f}x historical average",
                timestamp=datetime.now(),
                evidence={
                    "recent_volatility": float(recent_vol * 100),
                    "historical_volatility": float(historical_vol * 100),
                    "ratio": float(vol_ratio),
                },
            )

        return None

    def prune_evidence(
        self,
        highlights: List[Highlight],
        max_evidence_keys: int = 3,
    ) -> List[Highlight]:
        """
        Prune evidence to fit within budget.

        Args:
            highlights: List of highlights
            max_evidence_keys: Maximum keys to keep in evidence

        Returns:
            Highlights with pruned evidence
        """
        for h in highlights:
            if len(h.evidence) > max_evidence_keys:
                # Keep only most important keys
                important_keys = list(h.evidence.keys())[:max_evidence_keys]
                h.evidence = {k: h.evidence[k] for k in important_keys}

        return highlights
