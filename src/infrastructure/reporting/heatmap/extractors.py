"""
Data Extractors - Utility functions for extracting data from ticker summaries.

Extracts regime, alignment score, and other metrics from ticker data structures.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def extract_regime(ticker: Dict[str, Any]) -> Optional[str]:
    """
    Extract regime string from ticker data.

    Handles multiple possible locations for regime data:
    - Direct "regime" field
    - Nested "regime_output.regime" field

    Args:
        ticker: Ticker summary dictionary

    Returns:
        Regime string (e.g., "R0", "R1", "R2", "R3") or None
    """
    # Try direct regime field
    regime = ticker.get("regime")
    if regime:
        return str(regime)

    # Try nested regime_output
    regime_output = ticker.get("regime_output", {})
    if isinstance(regime_output, dict):
        nested_regime = regime_output.get("regime")
        if nested_regime:
            return str(nested_regime)

    return None


def extract_alignment_score(ticker: Dict[str, Any]) -> Optional[float]:
    """
    Extract alignment score from ticker data.

    Handles multiple possible locations:
    - Direct "alignment_score" field
    - Nested "confluence.alignment_score" field

    Args:
        ticker: Ticker summary dictionary

    Returns:
        Alignment score as float or None
    """
    # Try direct alignment_score
    score = ticker.get("alignment_score")
    if score is not None:
        return float(score)

    # Try nested confluence
    confluence = ticker.get("confluence", {})
    if isinstance(confluence, dict):
        score = confluence.get("alignment_score")
        if score is not None:
            return float(score)

    return None


def extract_composite_score(ticker: Dict[str, Any]) -> Optional[float]:
    """
    Extract average composite score from ticker data.

    The composite_score_avg is the average across all timeframes (0-100).

    Args:
        ticker: Ticker summary dictionary

    Returns:
        Composite score as float (0-100) or None
    """
    score = ticker.get("composite_score_avg")
    if score is not None:
        return float(score)
    return None
