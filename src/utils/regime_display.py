"""Shared regime label utility — single source of truth via MarketRegime.display_name."""

from __future__ import annotations


def regime_label(code: str) -> str:
    """Convert regime code string to human-readable display name.

    Args:
        code: Regime code, e.g. "R0", "R1", "R2", "R3".

    Returns:
        Display name like "Healthy Uptrend", or "Unknown" for invalid codes.
    """
    from src.domain.signals.indicators.regime.models import MarketRegime

    try:
        return MarketRegime(code).display_name
    except ValueError:
        return "Unknown"
