"""PEAD report builder — generates standalone HTML dashboard.

Serializes PEADScreenResult to JSON and passes to templates for rendering.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from src.domain.screeners.pead.models import PEADScreenResult
from src.utils.logging_setup import get_logger

from .templates import render_pead_html

logger = get_logger(__name__)


class PEADReportBuilder:
    """Builds the PEAD HTML report from screening results."""

    def build(self, result: PEADScreenResult, output_path: str | Path) -> Path:
        """Build and write the PEAD HTML report.

        Args:
            result: Screening result with scored candidates.
            output_path: Path to write the HTML file.

        Returns:
            Path to the written HTML file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = self._serialize(result)
        html = render_pead_html(data)
        output.write_text(html, encoding="utf-8")

        logger.info(f"PEAD report written: {output} ({len(result.candidates)} candidates)")
        return output

    @staticmethod
    def _serialize(result: PEADScreenResult) -> dict[str, Any]:
        """Convert PEADScreenResult to template-friendly dict."""
        candidates = []
        for c in result.candidates:
            candidates.append(
                {
                    "symbol": c.symbol,
                    "report_date": c.surprise.report_date.isoformat(),
                    "actual_eps": c.surprise.actual_eps,
                    "consensus_eps": c.surprise.consensus_eps,
                    "surprise_pct": round(c.surprise.surprise_pct, 2),
                    "sue_score": round(c.surprise.sue_score, 2),
                    "earnings_day_gap": round(c.surprise.earnings_day_gap, 4),
                    "earnings_day_return": round(c.surprise.earnings_day_return, 4),
                    "earnings_day_volume_ratio": round(c.surprise.earnings_day_volume_ratio, 2),
                    "revenue_beat": c.surprise.revenue_beat,
                    "liquidity_tier": c.surprise.liquidity_tier.value,
                    "forward_pe": c.surprise.forward_pe,
                    "entry_date": c.entry_date.isoformat(),
                    "entry_price": round(c.entry_price, 2),
                    "profit_target_pct": c.profit_target_pct,
                    "stop_loss_pct": c.stop_loss_pct,
                    "trailing_stop_atr": c.trailing_stop_atr,
                    "trailing_activation_pct": c.trailing_activation_pct,
                    "max_hold_days": c.max_hold_days,
                    "position_size_factor": c.position_size_factor,
                    "quality_score": round(c.quality_score, 1),
                    "quality_label": c.quality_label,
                    "regime": c.regime,
                    "gap_held": c.gap_held,
                    "estimated_slippage_bps": c.estimated_slippage_bps,
                }
            )

        return {
            "candidates": candidates,
            "screened_count": result.screened_count,
            "passed_filters": result.passed_filters,
            "skipped_count": result.skipped_count,
            "regime": result.regime,
            "generated_at": result.generated_at.isoformat(),
            "errors": result.errors,
        }
