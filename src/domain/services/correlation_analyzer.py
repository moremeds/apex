"""
Correlation Analyzer - Sector concentration and beta-weighted delta risk.

Detects:
- Sector concentration > 60% of total delta
- Correlated positions (same sector exposure)
- Beta-weighted delta calculations (vs SPY/SPX)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...models.risk_signal import (
    RiskSignal,
    SignalLevel,
    SignalSeverity,
    SuggestedAction,
)
from ...models.risk_snapshot import RiskSnapshot
from ...utils.logging_setup import get_logger
from .risk.threshold import Threshold, ThresholdDirection

logger = get_logger(__name__)


class CorrelationAnalyzer:
    """
    Analyzes portfolio correlation and sector concentration risk.

    Features:
    - Sector concentration detection (> 60% in single sector)
    - Beta-weighted delta calculation
    - Correlated risk identification
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize correlation analyzer.

        Args:
            config: Configuration dictionary with correlation_risk section
        """
        self.config = config
        correlation_config = config.get("risk_signals", {}).get("correlation_risk", {})

        # Configuration
        self.enabled = correlation_config.get("enabled", True)
        self.max_sector_concentration_pct = correlation_config.get(
            "max_sector_concentration_pct", 0.60
        )
        self.beta_reference = correlation_config.get("beta_reference", "SPY")

        # Threshold for sector concentration (warning at 80% of limit, critical at limit)
        self.concentration_threshold = Threshold(
            warning=self.max_sector_concentration_pct * 0.8,
            critical=self.max_sector_concentration_pct,
            direction=ThresholdDirection.ABOVE,
        )

        # Sector mappings
        self.sector_map = self._build_sector_map(correlation_config.get("sectors", {}))

        logger.info(
            f"CorrelationAnalyzer initialized: "
            f"enabled={self.enabled}, "
            f"max_concentration={self.max_sector_concentration_pct:.0%}, "
            f"sectors={len(self.sector_map)} symbols mapped"
        )

    def _build_sector_map(self, sectors_config: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Build symbol -> sector mapping.

        Args:
            sectors_config: Dict of sector_name -> [symbols]

        Returns:
            Dict of symbol -> sector_name
        """
        sector_map = {}
        for sector_name, symbols in sectors_config.items():
            for symbol in symbols:
                sector_map[symbol] = sector_name
        return sector_map

    def check(self, snapshot: RiskSnapshot) -> List[RiskSignal]:
        """
        Check portfolio for correlation and concentration risks.

        Args:
            snapshot: Risk snapshot with position risks

        Returns:
            List of risk signals
        """
        if not self.enabled:
            return []

        signals: List[RiskSignal] = []

        # Check sector concentration
        concentration_signal = self._check_sector_concentration(snapshot)
        if concentration_signal:
            signals.append(concentration_signal)

        return signals

    def _check_sector_concentration(self, snapshot: RiskSnapshot) -> Optional[RiskSignal]:
        """
        Check if portfolio has excessive concentration in a single sector.

        Returns signal if any sector exceeds max_sector_concentration_pct of total delta.
        Uses Threshold helper for standardized severity checking.
        """
        # Calculate delta by sector
        sector_deltas: Dict[str, float] = defaultdict(float)
        total_delta = 0.0

        for pos_risk in snapshot.position_risks:
            symbol = pos_risk.symbol
            delta = pos_risk.delta or 0.0

            # Map to sector
            sector = self.sector_map.get(symbol, "Other")
            sector_deltas[sector] += delta
            total_delta += abs(delta)

        if total_delta == 0:
            return None

        # Check each sector concentration using Threshold helper
        for sector, sector_delta in sector_deltas.items():
            concentration = abs(sector_delta) / total_delta

            severity_str = self.concentration_threshold.check(concentration)
            if severity_str:
                breach_pct = self.concentration_threshold.breach_pct(concentration) * 100
                severity = (
                    SignalSeverity.CRITICAL
                    if severity_str == "CRITICAL"
                    else SignalSeverity.WARNING
                )

                # Get symbols in this sector
                sector_symbols = [
                    pos_risk.symbol
                    for pos_risk in snapshot.position_risks
                    if self.sector_map.get(pos_risk.symbol, "Other") == sector
                ]

                return RiskSignal(
                    signal_id=f"PORTFOLIO:Sector_Concentration:{sector}",
                    timestamp=datetime.now(),
                    level=SignalLevel.PORTFOLIO,
                    severity=severity,
                    trigger_rule="Sector_Concentration",
                    current_value=concentration * 100,
                    threshold=self.max_sector_concentration_pct * 100,
                    breach_pct=breach_pct,
                    suggested_action=SuggestedAction.HEDGE,
                    action_details=(
                        f"{sector} sector represents {concentration*100:.1f}% of total delta "
                        f"(limit: {self.max_sector_concentration_pct*100:.0f}%). "
                        f"Concentrated exposure to {len(sector_symbols)} symbols. "
                        f"Consider diversifying or adding sector hedge."
                    ),
                    layer=2,
                    metadata={
                        "sector": sector,
                        "sector_delta": sector_delta,
                        "total_delta": total_delta,
                        "concentration_pct": concentration * 100,
                        "symbols": sector_symbols[:10],  # Limit to first 10
                        "symbol_count": len(sector_symbols),
                    },
                )

        return None

    def calculate_beta_weighted_delta(
        self,
        snapshot: RiskSnapshot,
        beta_map: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate beta-weighted delta vs reference (SPY).

        Args:
            snapshot: Risk snapshot
            beta_map: Optional symbol -> beta mapping. If None, assumes beta=1.0

        Returns:
            Beta-weighted delta
        """
        if beta_map is None:
            beta_map = {}

        beta_weighted_delta = 0.0

        for pos_risk in snapshot.position_risks:
            symbol = pos_risk.symbol
            delta = pos_risk.delta or 0.0

            # Get beta (default 1.0 if not provided)
            beta = beta_map.get(symbol, 1.0)

            beta_weighted_delta += delta * beta

        return beta_weighted_delta

    def get_sector_breakdown(self, snapshot: RiskSnapshot) -> Dict[str, Dict[str, float]]:
        """
        Get sector exposure breakdown.

        Args:
            snapshot: Risk snapshot

        Returns:
            Dict of sector -> {delta, notional, symbol_count}
        """
        sector_breakdown: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"delta": 0.0, "notional": 0.0, "symbol_count": 0}
        )

        for pos_risk in snapshot.position_risks:
            symbol = pos_risk.symbol
            sector = self.sector_map.get(symbol, "Other")

            sector_breakdown[sector]["delta"] += pos_risk.delta or 0.0
            sector_breakdown[sector]["notional"] += pos_risk.notional or 0.0
            sector_breakdown[sector]["symbol_count"] += 1

        return dict(sector_breakdown)

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"CorrelationAnalyzer(enabled={self.enabled}, "
            f"max_concentration={self.max_sector_concentration_pct:.0%}, "
            f"sectors={len(set(self.sector_map.values()))})"
        )
