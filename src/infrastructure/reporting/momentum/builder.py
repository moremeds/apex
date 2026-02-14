"""Momentum report builder — generates standalone HTML dashboard.

Serializes MomentumScreenResult to JSON and passes to templates for rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.domain.screeners.momentum.models import MomentumScreenResult
from src.utils.logging_setup import get_logger

from .templates import render_momentum_html

logger = get_logger(__name__)


class MomentumReportBuilder:
    """Builds the momentum screener HTML report from screening results."""

    def build(self, result: MomentumScreenResult, output_path: str | Path) -> Path:
        """Build and write the momentum HTML report.

        Args:
            result: Screening result with ranked candidates.
            output_path: Path to write the HTML file.

        Returns:
            Path to the written HTML file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = self._serialize(result)
        html = render_momentum_html(data)
        output.write_text(html, encoding="utf-8")

        logger.info(f"Momentum report written: {output} ({len(result.candidates)} candidates)")
        return output

    @staticmethod
    def _serialize(result: MomentumScreenResult) -> dict[str, Any]:
        """Convert MomentumScreenResult to template-friendly dict."""
        candidates = []
        for c in result.candidates:
            s = c.signal
            candidates.append(
                {
                    "rank": c.rank,
                    "symbol": s.symbol,
                    "momentum_12_1": round(s.momentum_12_1, 4),
                    "fip": round(s.fip, 4),
                    "momentum_percentile": round(s.momentum_percentile, 4),
                    "fip_percentile": round(s.fip_percentile, 4),
                    "composite_rank": round(s.composite_rank, 4),
                    "last_close": round(s.last_close, 2),
                    "market_cap": s.market_cap,
                    "avg_daily_dollar_volume": round(s.avg_daily_dollar_volume, 0),
                    "liquidity_tier": s.liquidity_tier.value,
                    "estimated_slippage_bps": s.estimated_slippage_bps,
                    "lookback_days": s.lookback_days,
                    "quality_label": c.quality_label,
                    "position_size_factor": c.position_size_factor,
                    "regime": c.regime,
                }
            )

        return {
            "candidates": candidates,
            "universe_size": result.universe_size,
            "passed_filters": result.passed_filters,
            "regime": result.regime,
            "generated_at": result.generated_at.isoformat(),
            "errors": result.errors,
        }
