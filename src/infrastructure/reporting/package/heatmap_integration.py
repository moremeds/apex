"""
Heatmap Integration - Wrapper for heatmap generation.

Integrates heatmap generation into the package builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_heatmap(
    summary: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """
    Build heatmap landing page from summary data.

    PR-C: Creates an interactive treemap visualization for quick market overview.

    Args:
        summary: Summary.json data structure
        output_dir: Package output directory

    Returns:
        Path to heatmap.html or None if generation failed
    """
    try:
        from src.services.market_cap_service import MarketCapService

        from ..heatmap.builder import HeatmapBuilder

        # Load market cap service
        cap_service = MarketCapService()

        # Build heatmap model
        builder = HeatmapBuilder(market_cap_service=cap_service)

        # Build manifest for report URL mapping
        # Link to report.html with symbol parameter (heatmap is now index.html)
        manifest = {
            "symbol_reports": {
                ticker["symbol"]: f"report.html?symbol={ticker['symbol']}"
                for ticker in summary.get("tickers", [])
                if ticker.get("symbol")
            }
        }

        model = builder.build_heatmap_model(summary, manifest)

        # Render and save as index.html (heatmap is the landing page)
        heatmap_path = builder.save_heatmap(model, output_dir, "index.html")

        logger.info(
            f"Heatmap generated: {model.symbol_count} symbols, "
            f"{model.cap_missing_count} missing caps"
        )

        return heatmap_path

    except ImportError as e:
        logger.warning(f"Heatmap generation skipped: {e}")
        return None
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        return None
