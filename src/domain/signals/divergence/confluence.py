"""
Multi-Timeframe Divergence and Confluence Analysis.

Provides tools for:
- Detecting timeframe alignment/divergence
- Scoring confluence across multiple timeframes
- Identifying higher-timeframe confirmation patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logging_setup import get_logger

from ..models import ConfluenceScore
from .cross_divergence import CrossIndicatorAnalyzer

logger = get_logger(__name__)


@dataclass
class MTFAlignment:
    """
    Multi-timeframe alignment result.

    Represents the degree of agreement between signals across different
    timeframes for a given symbol.
    """

    symbol: str
    timeframes: List[str]
    tf_scores: Dict[str, ConfluenceScore]
    alignment_strength: str  # "strong", "moderate", "weak", "unknown"
    dominant_direction: Optional[str]  # "bullish", "bearish", or None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframes": self.timeframes,
            "tf_scores": {tf: score.to_dict() for tf, score in self.tf_scores.items()},
            "alignment_strength": self.alignment_strength,
            "dominant_direction": self.dominant_direction,
            "timestamp": self.timestamp.isoformat(),
        }


class MTFDivergenceAnalyzer:
    """
    Detects timeframe alignment/divergence.

    Analyzes indicator confluence across multiple timeframes to determine
    if higher timeframes confirm or conflict with lower timeframe signals.

    Example:
        analyzer = MTFDivergenceAnalyzer()
        alignment = analyzer.analyze(
            symbol='AAPL',
            timeframes=['1h', '4h', '1d'],
            states_by_tf={
                '1h': {'rsi': {...}, 'macd': {...}},
                '4h': {'rsi': {...}, 'macd': {...}},
                '1d': {'rsi': {...}, 'macd': {...}},
            }
        )
        print(f"Alignment: {alignment.alignment_strength}")
    """

    # Standard timeframe ordering (lower to higher)
    TIMEFRAME_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    def __init__(self, cross_analyzer: Optional[CrossIndicatorAnalyzer] = None) -> None:
        """
        Initialize the analyzer.

        Args:
            cross_analyzer: CrossIndicatorAnalyzer instance (creates default if None)
        """
        self._cross_analyzer = cross_analyzer or CrossIndicatorAnalyzer()

    def analyze(
        self,
        symbol: str,
        timeframes: List[str],
        states_by_tf: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> MTFAlignment:
        """
        Analyze signal alignment across timeframes.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to analyze (e.g., ["1h", "4h", "1d"])
            states_by_tf: Dict mapping timeframe -> indicator_name -> state

        Returns:
            MTFAlignment result

        Strength levels:
        - Strong: All timeframes agree on direction
        - Moderate: Higher timeframes (2+) agree
        - Weak: Timeframes conflict
        - Unknown: Insufficient data
        """
        # Sort timeframes by order
        sorted_tfs = self._sort_timeframes(timeframes)

        # Calculate confluence score for each timeframe
        tf_scores: Dict[str, ConfluenceScore] = {}
        for tf in sorted_tfs:
            if tf in states_by_tf:
                score = self._cross_analyzer.analyze(symbol, tf, states_by_tf[tf])
                tf_scores[tf] = score

        if not tf_scores:
            return MTFAlignment(
                symbol=symbol,
                timeframes=timeframes,
                tf_scores={},
                alignment_strength="unknown",
                dominant_direction=None,
            )

        # Extract directions with their alignment scores
        directions: List[tuple[str, str, int]] = []  # (tf, direction, score)
        for tf, score in tf_scores.items():
            if score.strongest_signal:
                directions.append((tf, score.strongest_signal, score.alignment_score))

        # Determine alignment strength
        alignment_strength = self._determine_alignment_strength(directions, sorted_tfs)

        # Determine dominant direction (prefer higher timeframes)
        dominant_direction = self._determine_dominant_direction(directions, sorted_tfs)

        alignment = MTFAlignment(
            symbol=symbol,
            timeframes=sorted_tfs,
            tf_scores=tf_scores,
            alignment_strength=alignment_strength,
            dominant_direction=dominant_direction,
        )

        # Debug log for MTF analysis
        logger.debug(
            "MTF alignment analysis completed",
            extra={
                "symbol": symbol,
                "timeframes": sorted_tfs,
                "strength": alignment_strength,
                "direction": dominant_direction,
                "tf_directions": {tf: score.strongest_signal for tf, score in tf_scores.items()},
            },
        )

        # Info log for strong or weak alignment (actionable)
        if alignment_strength == "strong":
            logger.info(
                "Strong MTF alignment detected",
                extra={
                    "symbol": symbol,
                    "strength": alignment_strength,
                    "direction": dominant_direction,
                    "timeframes": sorted_tfs,
                },
            )
        elif alignment_strength == "weak":
            logger.info(
                "Weak MTF alignment - timeframes in conflict",
                extra={
                    "symbol": symbol,
                    "strength": alignment_strength,
                    "timeframes": sorted_tfs,
                    "tf_directions": {
                        tf: score.strongest_signal for tf, score in tf_scores.items()
                    },
                },
            )

        return alignment

    def _determine_alignment_strength(
        self,
        directions: List[tuple[str, str, int]],
        sorted_tfs: List[str],
    ) -> str:
        """
        Determine overall alignment strength.

        Returns:
            "strong", "moderate", "weak", or "unknown"
        """
        if len(directions) < 2:
            return "unknown"

        # Extract just the direction strings
        dir_values = [d[1] for d in directions]

        # Check if all agree
        if all(d == dir_values[0] for d in dir_values):
            return "strong"

        # Check if higher timeframes agree (last 2 in sorted order)
        higher_tf_directions = []
        for tf, direction, _ in directions:
            if tf in sorted_tfs[-2:]:
                higher_tf_directions.append(direction)

        if len(higher_tf_directions) >= 2 and len(set(higher_tf_directions)) == 1:
            return "moderate"

        return "weak"

    def _determine_dominant_direction(
        self,
        directions: List[tuple[str, str, int]],
        sorted_tfs: List[str],
    ) -> Optional[str]:
        """
        Determine the dominant direction, prioritizing higher timeframes.

        Uses a weighted scoring system where higher timeframes have more weight.
        """
        if not directions:
            return None

        # Weight by timeframe position (higher = more weight)
        bullish_weight = 0
        bearish_weight = 0

        for tf, direction, alignment_score in directions:
            try:
                tf_index = sorted_tfs.index(tf)
            except ValueError:
                tf_index = 0

            weight = tf_index + 1  # 1-based weight

            if direction == "bullish":
                bullish_weight += weight * abs(alignment_score)
            elif direction == "bearish":
                bearish_weight += weight * abs(alignment_score)

        if bullish_weight > bearish_weight:
            return "bullish"
        elif bearish_weight > bullish_weight:
            return "bearish"

        return None

    def _sort_timeframes(self, timeframes: List[str]) -> List[str]:
        """Sort timeframes from lowest to highest."""

        def get_order(tf: str) -> int:
            try:
                return self.TIMEFRAME_ORDER.index(tf)
            except ValueError:
                return len(self.TIMEFRAME_ORDER)

        return sorted(timeframes, key=get_order)

    def get_higher_tf_confirmation(
        self,
        symbol: str,
        current_tf: str,
        direction: str,
        states_by_tf: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> bool:
        """
        Check if higher timeframes confirm a given direction.

        Args:
            symbol: Trading symbol
            current_tf: Current timeframe
            direction: Direction to confirm ("bullish" or "bearish")
            states_by_tf: Indicator states by timeframe

        Returns:
            True if at least one higher timeframe confirms the direction
        """
        try:
            current_idx = self.TIMEFRAME_ORDER.index(current_tf)
        except ValueError:
            return False

        higher_tfs = self.TIMEFRAME_ORDER[current_idx + 1 :]

        for tf in higher_tfs:
            if tf not in states_by_tf:
                continue

            score = self._cross_analyzer.analyze(symbol, tf, states_by_tf[tf])
            if score.strongest_signal == direction:
                return True

        return False
