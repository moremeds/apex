"""
Package Builder - Orchestrator for signal report package generation.

Directory-based signal report package with lazy loading (PR-02).
Generates a complete package with heatmap landing page, data files, and assets.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logging_setup import get_logger

from .constants import PACKAGE_FORMAT_VERSION
from .file_writers import (
    write_data_files,
    write_indicators_file,
    write_manifest_file,
    write_regime_html_files,
    write_snapshot_file,
    write_summary_file,
)
from .heatmap_integration import build_heatmap
from .html_assets import (
    build_css,
    build_index_html,
    get_theme_colors,
    timeframe_seconds,
)
from .javascript import build_javascript
from .score_history import ScoreHistoryManager
from .summary_builder import PackageManifest, SummaryBuilder

if TYPE_CHECKING:
    from src.domain.signals.indicators.base import Indicator
    from src.domain.signals.indicators.regime import RegimeOutput
    from src.domain.signals.models import SignalRule

logger = get_logger(__name__)


class PackageBuilder:
    """
    Build a signal package with lazy loading and full feature parity.

    The package structure:
        output_dir/
            index.html           # Heatmap landing page (PR-C)
            report.html          # Signal report shell (lazy loads data)
            manifest.json        # Package metadata
            assets/
                styles.css       # Combined CSS
                app.js           # JavaScript application
                heatmap-theme.css # Heatmap-specific CSS
            data/
                summary.json     # Symbol summaries and metadata
                indicators.json  # Indicator descriptions
                regime/          # Pre-rendered regime HTML per symbol
                AAPL_1d.json     # Per-symbol data files
                ...
            snapshots/
                payload_snapshot.json  # For diffing (optional)
    """

    def __init__(
        self,
        theme: str = "dark",
        enforce_budget: bool = False,
        with_heatmap: bool = True,  # Kept for API compatibility, heatmap always generated
    ) -> None:
        """
        Initialize package builder.

        Args:
            theme: Color theme ("dark" or "light")
            enforce_budget: If True, raise SizeBudgetExceeded for over-budget sections
            with_heatmap: Ignored - heatmap landing page is always generated
        """
        self.theme = theme
        self._colors = get_theme_colors(theme)
        self._summary_builder = SummaryBuilder(enforce_budget=enforce_budget)

    def build(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        indicators: List["Indicator"],
        rules: List["SignalRule"],
        output_dir: Path,
        regime_outputs: Optional[Dict[str, "RegimeOutput"]] = None,
        validation_url: Optional[str] = None,
        score_history_path: Optional[Path] = None,
    ) -> PackageManifest:
        """
        Build the complete signal package.

        Args:
            data: Dict mapping (symbol, timeframe) to DataFrame
            indicators: List of computed indicators
            rules: List of signal rules
            output_dir: Output directory for the package
            regime_outputs: Optional dict mapping symbol to RegimeOutput
            validation_url: Optional URL to validation results page

        Returns:
            PackageManifest with package metadata
        """
        logger.info(f"Building signal package in {output_dir}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract unique symbols and timeframes
        symbols = sorted(set(sym for sym, tf in data.keys()))
        timeframes = sorted(
            set(tf for sym, tf in data.keys()),
            key=lambda x: timeframe_seconds(x),
        )
        regime_outputs = regime_outputs or {}

        # Create directory structure
        data_dir = output_dir / "data"
        assets_dir = output_dir / "assets"
        regime_dir = data_dir / "regime"
        snapshots_dir = output_dir / "snapshots"

        for d in [data_dir, assets_dir, regime_dir, snapshots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Build and write summary.json
        summary = self._summary_builder.build_summary(data, symbols, timeframes, regime_outputs)
        summary_size_kb = write_summary_file(summary, output_dir)

        # Update score history (append current scores, save alongside package)
        history_file = output_dir / "data" / "score_history.json"
        score_mgr = ScoreHistoryManager()
        if score_history_path and score_history_path.exists():
            score_mgr.load(score_history_path)
        elif history_file.exists():
            score_mgr.load(history_file)
        score_mgr.append_from_summary(summary)
        score_mgr.save(history_file)

        # Extract sparklines once for reuse
        sparklines = score_mgr.get_all_sparklines()

        # Build heatmap landing page (index.html) with sparklines
        build_heatmap(summary, output_dir, sparklines)

        # Write per-symbol data files
        data_files = write_data_files(data, indicators, rules, data_dir)

        # Write indicators.json
        write_indicators_file(indicators, rules, data_dir)

        # Write pre-rendered regime HTML files with sparklines
        write_regime_html_files(
            regime_outputs,
            regime_dir,
            self.theme,
            all_symbols=symbols,
            score_sparklines=sparklines,
        )

        # Write snapshot for diffing
        write_snapshot_file(data, regime_outputs, symbols, timeframes, output_dir)

        # Build and write assets
        css = build_css(self.theme)
        (assets_dir / "styles.css").write_text(css, encoding="utf-8")

        js = build_javascript(symbols, timeframes, self._colors)
        (assets_dir / "app.js").write_text(js, encoding="utf-8")

        # Build and write report.html (symbol analysis page)
        report_html = build_index_html(
            symbols, timeframes, self._colors, regime_outputs, validation_url
        )
        (output_dir / "report.html").write_text(report_html, encoding="utf-8")

        # Build manifest
        manifest = PackageManifest(
            version=PACKAGE_FORMAT_VERSION,
            created_at=datetime.now().isoformat(),
            symbols=tuple(symbols),
            timeframes=tuple(timeframes),
            total_data_files=len(data_files),
            summary_size_kb=round(summary_size_kb, 2),
            theme=self.theme,
        )

        # Write manifest.json
        write_manifest_file(manifest, output_dir)

        logger.info(
            f"Package built: {len(symbols)} symbols, {len(timeframes)} timeframes, "
            f"{len(data_files)} data files, {summary_size_kb:.1f}KB summary"
        )

        return manifest
