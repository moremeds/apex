"""
Cross-Indicator Analysis.

Analyzes agreement/disagreement across multiple indicators to determine
confluence and identify conflicting signals.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils.logging_setup import get_logger

from ..models import ConfluenceScore

logger = get_logger(__name__)


# Type for indicator direction checking function
DirectionChecker = Callable[[Dict[str, Any]], bool]


class CrossIndicatorAnalyzer:
    """
    Analyzes agreement/disagreement across multiple indicators.

    Determines overall market sentiment by scoring how many indicators
    are bullish vs bearish, and identifies conflicting indicator pairs.

    Example:
        analyzer = CrossIndicatorAnalyzer()
        score = analyzer.analyze(
            symbol='AAPL',
            timeframe='1h',
            indicator_states={
                'rsi': {'value': 35, 'zone': 'oversold'},
                'macd': {'histogram': -0.5, 'cross': 'bearish'},
                'supertrend': {'direction': 'bullish'},
            }
        )
        print(f"Alignment: {score.alignment_score}, Bullish: {score.bullish_count}")
    """

    def __init__(self) -> None:
        """Initialize with default signal detection rules."""
        # Bullish signal detectors per indicator
        self._bullish_signals: Dict[str, DirectionChecker] = {
            "rsi": lambda s: s.get("zone") == "oversold",
            "macd": lambda s: s.get("histogram", 0) > 0 or s.get("cross") == "bullish",
            "supertrend": lambda s: s.get("direction") in ["bullish", "up", "1"],
            "bollinger": lambda s: s.get("zone") in ["below_lower", "lower"],
            "adx": lambda s: s.get("di_plus", 0) > s.get("di_minus", 0),
            "kdj": lambda s: s.get("zone") == "oversold" or s.get("cross") == "bullish",
            "ema": lambda s: s.get("cross") == "bullish" or s.get("trend") == "up",
            "sma": lambda s: s.get("cross") == "bullish" or s.get("trend") == "up",
            "psar": lambda s: s.get("direction") in ["bullish", "up"],
            "aroon": lambda s: s.get("aroon_up", 0) > s.get("aroon_down", 0),
            "obv": lambda s: s.get("trend") in ["up", "bullish"],
            "vwap": lambda s: s.get("position") == "above",
            "mfi": lambda s: s.get("zone") == "oversold",
            "squeeze": lambda s: s.get("signal") in ["bullish", "long", "buy"],
        }

        # Bearish signal detectors per indicator
        self._bearish_signals: Dict[str, DirectionChecker] = {
            "rsi": lambda s: s.get("zone") == "overbought",
            "macd": lambda s: s.get("histogram", 0) < 0 or s.get("cross") == "bearish",
            "supertrend": lambda s: s.get("direction") in ["bearish", "down", "-1"],
            "bollinger": lambda s: s.get("zone") in ["above_upper", "upper"],
            "adx": lambda s: s.get("di_minus", 0) > s.get("di_plus", 0),
            "kdj": lambda s: s.get("zone") == "overbought" or s.get("cross") == "bearish",
            "ema": lambda s: s.get("cross") == "bearish" or s.get("trend") == "down",
            "sma": lambda s: s.get("cross") == "bearish" or s.get("trend") == "down",
            "psar": lambda s: s.get("direction") in ["bearish", "down"],
            "aroon": lambda s: s.get("aroon_down", 0) > s.get("aroon_up", 0),
            "obv": lambda s: s.get("trend") in ["down", "bearish"],
            "vwap": lambda s: s.get("position") == "below",
            "mfi": lambda s: s.get("zone") == "overbought",
            "squeeze": lambda s: s.get("signal") in ["bearish", "short", "sell"],
        }

    def register_indicator(
        self,
        indicator_name: str,
        bullish_check: DirectionChecker,
        bearish_check: DirectionChecker,
    ) -> None:
        """
        Register custom signal detection for an indicator.

        Args:
            indicator_name: Name of the indicator
            bullish_check: Function that returns True if state is bullish
            bearish_check: Function that returns True if state is bearish
        """
        self._bullish_signals[indicator_name] = bullish_check
        self._bearish_signals[indicator_name] = bearish_check

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        indicator_states: Dict[str, Dict[str, Any]],
    ) -> ConfluenceScore:
        """
        Analyze indicator confluence.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            indicator_states: Dict mapping indicator names to their state dicts

        Returns:
            ConfluenceScore with alignment analysis
        """
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        diverging_pairs: List[Tuple[str, str, str]] = []

        indicator_directions: Dict[str, str] = {}

        # Determine direction for each indicator
        for indicator, state in indicator_states.items():
            direction = self._get_direction(indicator, state)
            indicator_directions[indicator] = direction

            if direction == "bullish":
                bullish_count += 1
            elif direction == "bearish":
                bearish_count += 1
            else:
                neutral_count += 1

        # Find diverging pairs (indicators with opposite directions)
        indicators = list(indicator_directions.keys())
        for i in range(len(indicators)):
            for j in range(i + 1, len(indicators)):
                ind1, ind2 = indicators[i], indicators[j]
                dir1, dir2 = indicator_directions[ind1], indicator_directions[ind2]

                if dir1 != "neutral" and dir2 != "neutral" and dir1 != dir2:
                    reason = f"{ind1} is {dir1}, {ind2} is {dir2}"
                    diverging_pairs.append((ind1, ind2, reason))

        # Calculate alignment score (-100 to +100)
        total = bullish_count + bearish_count + neutral_count
        if total > 0:
            alignment_score = int(((bullish_count - bearish_count) / total) * 100)
        else:
            alignment_score = 0

        # Determine strongest signal direction
        strongest_signal: Optional[str] = None
        if bullish_count > bearish_count and bullish_count > neutral_count:
            strongest_signal = "bullish"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            strongest_signal = "bearish"

        score = ConfluenceScore(
            symbol=symbol,
            timeframe=timeframe,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            alignment_score=alignment_score,
            diverging_pairs=diverging_pairs,
            strongest_signal=strongest_signal,
        )

        # Debug log for confluence analysis
        logger.debug(
            "Cross-indicator analysis completed",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": neutral_count,
                "alignment_score": alignment_score,
                "diverging_pairs": [(p[0], p[1]) for p in diverging_pairs],
                "strongest_signal": strongest_signal,
                "indicators_analyzed": list(indicator_states.keys()),
            },
        )

        # Info log for strong confluence (actionable signal)
        if abs(alignment_score) >= 60:
            logger.info(
                "Strong confluence detected",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "alignment_score": alignment_score,
                    "direction": strongest_signal,
                    "bullish": bullish_count,
                    "bearish": bearish_count,
                },
            )

        # Info log for significant divergences
        if len(diverging_pairs) >= 3:
            logger.info(
                "Multiple indicator divergences detected",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "diverging_count": len(diverging_pairs),
                    "pairs": [(p[0], p[1]) for p in diverging_pairs[:5]],
                },
            )

        return score

    def _get_direction(self, indicator: str, state: Dict[str, Any]) -> str:
        """
        Determine indicator direction from its state.

        Args:
            indicator: Indicator name
            state: Indicator state dictionary

        Returns:
            "bullish", "bearish", or "neutral"
        """
        # Try bullish check first
        bullish_check = self._bullish_signals.get(indicator)
        if bullish_check is not None:
            try:
                if bullish_check(state):
                    return "bullish"
            except Exception as e:
                logger.warning(
                    "Bullish direction check failed",
                    extra={"indicator": indicator, "error": str(e)},
                )

        # Try bearish check
        bearish_check = self._bearish_signals.get(indicator)
        if bearish_check is not None:
            try:
                if bearish_check(state):
                    return "bearish"
            except Exception as e:
                logger.warning(
                    "Bearish direction check failed",
                    extra={"indicator": indicator, "error": str(e)},
                )

        # Default to neutral
        return "neutral"

    def get_registered_indicators(self) -> List[str]:
        """Return list of indicators with registered signal detection."""
        return list(set(self._bullish_signals.keys()) | set(self._bearish_signals.keys()))
