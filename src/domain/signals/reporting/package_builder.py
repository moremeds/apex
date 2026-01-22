"""
Package Builder - Directory-based signal report package with lazy loading.

PR-02 Deliverable: Produces a directory structure instead of monolithic HTML.

Package Structure:
    signal_package/
    ├── index.html              # Shell (~50 KB)
    ├── assets/
    │   ├── styles.css          # Extracted CSS
    │   └── app.js              # JavaScript with lazy loading
    ├── data/
    │   ├── summary.json        # ≤200 KB (all symbols condensed)
    │   ├── AAPL_1d.json        # Full data for lazy load
    │   ├── SPY_1d.json
    │   └── ...
    └── snapshots/
        └── payload_snapshot.json  # For diff (NO HTML PARSING)

Usage:
    builder = PackageBuilder(theme="dark")
    package_path = builder.build(
        data=data,
        indicators=indicators,
        rules=rules,
        output_dir=Path("results/signals/package"),
        regime_outputs=regime_outputs,
    )
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logging_setup import get_logger

from ..data.quality_validator import get_last_valid_close, validate_close_for_regime

# Import regime HTML generators for 1:1 feature parity
from .regime import (
    generate_components_4block_html,
    generate_decision_tree_html,
    generate_hysteresis_html,
    generate_methodology_html,
    generate_optimization_html,
    generate_quality_html,
    generate_recommendations_html,
    generate_regime_one_liner_html,
    generate_regime_styles,
    generate_report_header_html,
    generate_turning_point_html,
)
from .snapshot_builder import SnapshotBuilder

if TYPE_CHECKING:
    from ..indicators.base import Indicator
    from ..indicators.regime import RegimeOutput
    from ..models import SignalRule

logger = get_logger(__name__)

# Indicator grouping for chart layout (same as SignalReportGenerator)
OVERLAY_INDICATORS = {
    "bollinger",
    "supertrend",
    "sma",
    "ema",
    "vwap",
    "keltner",
    "donchian",
    "ichimoku",
}
BOUNDED_OSCILLATORS = {"rsi", "stochastic", "kdj", "williams_r", "mfi", "cci", "adx"}
UNBOUNDED_OSCILLATORS = {"macd", "momentum", "roc", "cmf", "pvo", "force_index"}

# Version for package format
PACKAGE_FORMAT_VERSION = "1.0"

# Size budget constants (KB) - M3 PR-03
SUMMARY_BUDGET_KB = 200
MARKET_BUDGET_KB = 8
SECTORS_BUDGET_KB = 20
TICKERS_BUDGET_KB = 100
HIGHLIGHTS_BUDGET_KB = 40
CONFLUENCE_BUDGET_KB = 30
DATA_QUALITY_BUDGET_KB = 2


@dataclass(frozen=True)
class PackageManifest:
    """Manifest for a signal package."""

    version: str
    created_at: str
    symbols: Tuple[str, ...]
    timeframes: Tuple[str, ...]
    total_data_files: int
    summary_size_kb: float
    theme: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "total_data_files": self.total_data_files,
            "summary_size_kb": self.summary_size_kb,
            "theme": self.theme,
        }


class PackageBuilder:
    """
    Builds a directory-based signal report package.

    Advantages over monolithic HTML:
    - Lazy loading: Only load data for selected symbol/timeframe
    - Smaller initial load: index.html ~50KB vs 50MB
    - Machine-readable: JSON files can be diffed/processed
    - Cacheable: Browser can cache assets and data files
    """

    # Theme color schemes
    THEMES = {
        "dark": {
            "bg": "#0f172a",
            "card_bg": "#1e293b",
            "border": "#334155",
            "text": "#f8fafc",
            "text_muted": "#94a3b8",
            "primary": "#3b82f6",
            "success": "#22c55e",
            "warning": "#eab308",
            "danger": "#ef4444",
        },
        "light": {
            "bg": "#ffffff",
            "card_bg": "#f8fafc",
            "border": "#e2e8f0",
            "text": "#1e293b",
            "text_muted": "#64748b",
            "primary": "#2563eb",
            "success": "#16a34a",
            "warning": "#ca8a04",
            "danger": "#dc2626",
        },
    }

    # Class-level budget constants for M3 PR-03 enforcement
    MAX_SUMMARY_KB = SUMMARY_BUDGET_KB
    TICKERS_BUDGET_KB = TICKERS_BUDGET_KB
    MARKET_BUDGET_KB = MARKET_BUDGET_KB
    CONFLUENCE_BUDGET_KB = CONFLUENCE_BUDGET_KB

    def __init__(
        self,
        theme: str = "dark",
        enforce_budget: bool = False,
        with_heatmap: bool = False,
    ) -> None:
        """
        Initialize package builder.

        Args:
            theme: Color theme ("dark" or "light")
            enforce_budget: If True, raise SizeBudgetExceeded on overflow
            with_heatmap: If True, generate heatmap landing page (PR-C)
        """
        self.theme = theme
        self.enforce_budget = enforce_budget
        self.with_heatmap = with_heatmap
        self._colors = self.THEMES.get(theme, self.THEMES["dark"])
        self._snapshot_builder = SnapshotBuilder()

    def _check_budget(self, section: str, data: Dict, budget_kb: int) -> None:
        """
        Check if section data exceeds budget.

        Args:
            section: Section name
            data: Section data dictionary
            budget_kb: Budget limit in KB

        Raises:
            SizeBudgetExceeded: If enforce_budget is True and budget exceeded
        """
        from .exceptions import SizeBudgetExceeded

        size_bytes = len(json.dumps(data, default=str))
        size_kb = size_bytes / 1024

        if size_kb > budget_kb:
            if self.enforce_budget:
                top_contributors = self._find_top_contributors(data)
                raise SizeBudgetExceeded(
                    section=section,
                    actual_kb=size_kb,
                    budget_kb=budget_kb,
                    top_contributors=top_contributors,
                )
            else:
                logger.warning(
                    f"Section '{section}' exceeds budget: " f"{size_kb:.1f}KB > {budget_kb}KB"
                )

    def _find_top_contributors(self, data: Dict) -> List:
        """
        Find top contributors to section size.

        Args:
            data: Section data dictionary

        Returns:
            List of BudgetContributor objects
        """
        from .exceptions import BudgetContributor

        total_size = len(json.dumps(data, default=str))
        contributors = []

        for key, value in data.items():
            item_size = len(json.dumps(value, default=str))
            pct = (item_size / total_size) * 100 if total_size > 0 else 0
            contributors.append(
                BudgetContributor(key=key, size_bytes=item_size, pct_of_section=pct)
            )

        # Sort by size descending
        contributors.sort(key=lambda x: x.size_bytes, reverse=True)
        return contributors[:10]

    def build(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        indicators: List["Indicator"],
        rules: List["SignalRule"],
        output_dir: Path,
        regime_outputs: Optional[Dict[str, "RegimeOutput"]] = None,
        validation_url: Optional[str] = None,
    ) -> Path:
        """
        Build the signal package.

        Args:
            data: Dict mapping (symbol, timeframe) to DataFrame with OHLCV + indicators
            indicators: List of computed indicators
            rules: List of signal rules
            output_dir: Directory to create package in
            regime_outputs: Optional dict mapping symbol to RegimeOutput
            validation_url: Optional URL to validation results page

        Returns:
            Path to the created package directory
        """
        output_dir = Path(output_dir)

        # Clean and create directory structure
        if output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True)
        (output_dir / "assets").mkdir()
        (output_dir / "data").mkdir()
        (output_dir / "data" / "regime").mkdir()
        (output_dir / "snapshots").mkdir()

        symbols = sorted(set(sym for sym, tf in data.keys()))
        timeframes = sorted(
            set(tf for sym, tf in data.keys()),
            key=lambda x: self._timeframe_seconds(x),
        )

        # 1. Write per-symbol data files (for lazy loading)
        data_files = self._write_data_files(data, indicators, rules, output_dir / "data")

        # 1b. Write indicators.json (for Indicators section)
        self._write_indicators_file(indicators, rules, output_dir / "data")

        # 2. Write summary.json
        summary = self._build_summary(
            data=data,
            symbols=symbols,
            timeframes=timeframes,
            regime_outputs=regime_outputs or {},
        )
        summary_path = output_dir / "data" / "summary.json"
        summary_json = json.dumps(summary, indent=2, default=str)
        summary_path.write_text(summary_json, encoding="utf-8")
        summary_size_kb = len(summary_json.encode("utf-8")) / 1024

        # 3. Write regime HTML files (for 1:1 feature parity)
        regime_files = self._write_regime_html_files(
            regime_outputs or {},
            output_dir / "data" / "regime",
        )

        # 4. Write CSS (including regime styles)
        css_content = self._build_css()
        (output_dir / "assets" / "styles.css").write_text(css_content, encoding="utf-8")

        # 5. Write JavaScript
        js_content = self._build_javascript(symbols, timeframes)
        (output_dir / "assets" / "app.js").write_text(js_content, encoding="utf-8")

        # 6. Write index.html (shell)
        html_content = self._build_index_html(symbols, timeframes, regime_outputs, validation_url)
        (output_dir / "index.html").write_text(html_content, encoding="utf-8")

        # 7. Write payload snapshot for diff
        snapshot = self._snapshot_builder.build(
            data=data,
            regime_outputs=regime_outputs or {},
            symbols=symbols,
            timeframes=timeframes,
        )
        snapshot_path = output_dir / "snapshots" / "payload_snapshot.json"
        snapshot_path.write_text(
            json.dumps(snapshot, indent=2, default=str),
            encoding="utf-8",
        )

        # 8. Write manifest
        manifest = PackageManifest(
            version=PACKAGE_FORMAT_VERSION,
            created_at=datetime.now().isoformat(),
            symbols=tuple(symbols),
            timeframes=tuple(timeframes),
            total_data_files=len(data_files),
            summary_size_kb=round(summary_size_kb, 2),
            theme=self.theme,
        )
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )

        # 9. Create .nojekyll file for GitHub Pages (prevents Jekyll processing)
        nojekyll_path = output_dir / ".nojekyll"
        nojekyll_path.write_text("", encoding="utf-8")

        # 10. Generate heatmap landing page if enabled (PR-C)
        heatmap_path = None
        if self.with_heatmap:
            heatmap_path = self._build_heatmap(summary, output_dir)

        logger.info(
            f"Package built: {output_dir} "
            f"({len(symbols)} symbols, {len(data_files)} data files, "
            f"{len(regime_files)} regime files, summary={summary_size_kb:.1f}KB"
            f"{', heatmap=yes' if heatmap_path else ''}"
            ")"
        )

        return output_dir

    def _build_heatmap(
        self,
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

            from .heatmap_builder import HeatmapBuilder

            # Load market cap service
            cap_service = MarketCapService()

            # Build heatmap model
            builder = HeatmapBuilder(market_cap_service=cap_service)

            # Build manifest for report URL mapping
            manifest = {
                "symbol_reports": {
                    ticker["symbol"]: f"data/regime/{ticker['symbol']}.html"
                    for ticker in summary.get("tickers", [])
                    if ticker.get("symbol")
                }
            }

            model = builder.build_heatmap_model(summary, manifest)

            # Render and save
            heatmap_path = builder.save_heatmap(model, output_dir, "heatmap.html")

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

    def _write_indicators_file(
        self,
        indicators: List["Indicator"],
        rules: List["SignalRule"],
        data_dir: Path,
    ) -> None:
        """Write indicators.json with indicator and rule information."""
        # Group indicators by category
        categories: Dict[str, List[Dict[str, Any]]] = {}

        # Build rule lookup by indicator
        rule_lookup: Dict[str, List[Dict[str, Any]]] = {}
        for rule in rules:
            ind_name = rule.indicator.lower()
            if ind_name not in rule_lookup:
                rule_lookup[ind_name] = []
            # Extract direction value from enum if needed
            direction_str: str = (
                rule.direction.value if hasattr(rule.direction, "value") else str(rule.direction)
            )
            rule_lookup[ind_name].append(
                {
                    "id": rule.name,  # SignalRule uses 'name' not 'id'
                    "direction": direction_str,
                    "description": getattr(rule, "message_template", None)
                    or f"{direction_str.upper()} when {rule.indicator} {rule.condition_type.value}",
                }
            )

        for indicator in indicators:
            category = indicator.category.value if hasattr(indicator, "category") else "other"
            if category not in categories:
                categories[category] = []

            ind_name = indicator.name.lower()
            params = {}
            if hasattr(indicator, "get_params"):
                params = indicator.get_params()

            categories[category].append(
                {
                    "name": indicator.name.upper(),
                    "description": getattr(indicator, "description", f"{indicator.name} indicator"),
                    "params": params,
                    "rules": rule_lookup.get(ind_name, []),
                }
            )

        # Sort categories
        category_order = ["momentum", "trend", "volatility", "volume", "pattern", "other"]
        sorted_categories = []
        for cat in category_order:
            if cat in categories:
                sorted_categories.append(
                    {
                        "name": cat.title() + " Indicators",
                        "indicators": categories[cat],
                    }
                )

        indicators_data = {
            "categories": sorted_categories,
            "total_indicators": len(indicators),
            "total_rules": len(rules),
        }

        file_path = data_dir / "indicators.json"
        file_path.write_text(
            json.dumps(indicators_data, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_data_files(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        indicators: List["Indicator"],
        rules: List["SignalRule"],
        data_dir: Path,
    ) -> List[str]:
        """Write individual JSON data files for each symbol/timeframe."""
        from .signal_report_generator import detect_historical_signals

        files_written = []

        for (symbol, timeframe), df in data.items():
            key = f"{symbol}_{timeframe}"

            # Convert DataFrame to JSON-serializable format
            chart_data = self._df_to_chart_data(df)

            # Detect signals for this symbol/timeframe
            signals = detect_historical_signals(df, rules, symbol, timeframe)

            file_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "generated_at": datetime.now().isoformat(),
                "bar_count": len(df),
                "chart_data": chart_data,
                "signals": signals,
            }

            file_path = data_dir / f"{key}.json"
            file_path.write_text(
                json.dumps(file_data, indent=2, default=str),
                encoding="utf-8",
            )
            files_written.append(key)

        return files_written

    def _df_to_chart_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert DataFrame to chart-ready JSON format.

        Structures indicator data by category for multi-subplot rendering:
        - overlays: Bollinger Bands, SuperTrend, etc (same Y-axis as price)
        - rsi: RSI indicator (0-100 scale)
        - macd: MACD, Signal, Histogram (unbounded scale)
        - oscillators: Other bounded oscillators
        - volume_ind: Volume indicators
        """
        # Convert index to ISO strings
        timestamps = [ts.isoformat() if hasattr(ts, "isoformat") else str(ts) for ts in df.index]

        # OHLCV data
        chart_data: Dict[str, Any] = {
            "timestamps": timestamps,
            "open": df["open"].tolist() if "open" in df else [],
            "high": df["high"].tolist() if "high" in df else [],
            "low": df["low"].tolist() if "low" in df else [],
            "close": df["close"].tolist() if "close" in df else [],
            "volume": df["volume"].tolist() if "volume" in df else [],
            # Structured indicator data for chart subplots
            "overlays": {},
            "rsi": {},
            "macd": {},
            "oscillators": {},
            "volume_ind": {},
        }

        # Categorize indicator columns (same logic as SignalReportGenerator)
        ohlcv_cols = {"open", "high", "low", "close", "volume", "timestamp"}
        oscillator_names = BOUNDED_OSCILLATORS | UNBOUNDED_OSCILLATORS

        for col in df.columns:
            if col.lower() in ohlcv_cols:
                continue

            # Convert to list, handling NaN
            values = df[col].tolist()
            values = [
                None if pd.isna(v) else float(v) if isinstance(v, (int, float)) else str(v)
                for v in values
            ]

            # Parse indicator name from prefixed column (e.g., "macd_histogram" → "macd")
            parts = col.split("_")
            ind_name = parts[0].lower() if parts else col.lower()

            # Route to appropriate subplot bucket
            if ind_name in OVERLAY_INDICATORS:
                chart_data["overlays"][col] = values
            elif ind_name == "rsi":
                chart_data["rsi"][col] = values
            elif ind_name == "macd":
                chart_data["macd"][col] = values
            elif ind_name in oscillator_names:
                chart_data["oscillators"][col] = values
            else:
                # Default to oscillators bucket
                chart_data["oscillators"][col] = values

        return chart_data

    def _build_summary(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        symbols: List[str],
        timeframes: List[str],
        regime_outputs: Dict[str, "RegimeOutput"],
    ) -> Dict[str, Any]:
        """
        Build summary.json content.

        Target: ≤200KB total with ~1.5KB per symbol.
        Includes confluence data for each symbol/timeframe combination.

        PR-B: Added run_data_quality section for aggregate quality metrics.
        """
        from .signal_report_generator import calculate_confluence

        from ..schemas import DataQualityReport

        summary: Dict[str, Any] = {
            "version": PACKAGE_FORMAT_VERSION,
            "generated_at": datetime.now().isoformat(),
            "symbols": symbols,
            "timeframes": timeframes,
            "symbol_count": len(symbols),
            "timeframe_count": len(timeframes),
        }

        # PR-B: Track aggregate data quality
        data_quality_report = DataQualityReport()

        # Per-symbol summaries (condensed)
        ticker_summaries = []
        for symbol in symbols:
            regime = regime_outputs.get(symbol)
            # Find DataFrame for this symbol
            df = None
            for tf in ["1d", "1h", "5m"]:
                if (symbol, tf) in data:
                    df = data[(symbol, tf)]
                    break

            ticker_summary = self._build_ticker_summary(symbol, data, regime)

            # PR-B: Add per-ticker data_quality and aggregate
            ticker_quality = self._extract_ticker_data_quality(symbol, df, regime)
            ticker_summary["data_quality"] = ticker_quality

            # Update aggregate report
            if not ticker_quality.get("regime_trustworthy", True):
                data_quality_report.invalid_symbol_count += 1
                if len(data_quality_report.worst_symbols) < 10:
                    data_quality_report.worst_symbols.append(symbol)

            ticker_summaries.append(ticker_summary)

        summary["tickers"] = ticker_summaries

        # PR-B: Add run-level data quality to summary
        summary["run_data_quality"] = data_quality_report.to_dict()

        # Check tickers budget
        self._check_budget("tickers", {"tickers": ticker_summaries}, TICKERS_BUDGET_KB)

        # Market overview (for benchmarks like SPY, QQQ)
        market_overview = self._build_market_overview(regime_outputs)
        summary["market"] = market_overview

        # Check market budget
        self._check_budget("market", market_overview, MARKET_BUDGET_KB)

        # Calculate confluence scores for each symbol/timeframe
        confluence_scores = calculate_confluence(data)
        confluence_data: Dict[str, Dict[str, Any]] = {}
        for key, score in confluence_scores.items():
            confluence_data[key] = {
                "symbol": score.symbol,
                "timeframe": score.timeframe,
                "bullish_count": score.bullish_count,
                "bearish_count": score.bearish_count,
                "neutral_count": score.neutral_count,
                "alignment_score": score.alignment_score,
                "diverging_pairs": [
                    {"ind1": p[0], "ind2": p[1], "reason": p[2]} for p in score.diverging_pairs
                ],
                "strongest_signal": score.strongest_signal,
            }
        summary["confluence"] = confluence_data

        # Check confluence budget
        self._check_budget("confluence", confluence_data, CONFLUENCE_BUDGET_KB)

        # Check data quality budget
        self._check_budget("data_quality", summary["run_data_quality"], DATA_QUALITY_BUDGET_KB)

        return summary

    def _build_ticker_summary(
        self,
        symbol: str,
        data: Dict[Tuple[str, str], pd.DataFrame],
        regime: Optional["RegimeOutput"],
    ) -> Dict[str, Any]:
        """
        Build full ticker summary with complete regime data for 1:1 feature parity.

        Uses RegimeOutput.to_dict() to include all fields from the original report:
        - Component states and values
        - Decision vs final regime
        - Transition state (hysteresis)
        - Derived metrics with percentiles
        - Rules fired
        - Data quality
        - Turning point prediction
        """
        # Find the best timeframe data for this symbol (prefer 1d)
        df = None
        for tf in ["1d", "1h", "5m"]:
            if (symbol, tf) in data:
                df = data[(symbol, tf)]
                break

        summary: Dict[str, Any] = {"symbol": symbol}

        # Add full regime info if available (1:1 feature parity)
        if regime:
            # Use full to_dict() for complete data
            regime_dict = regime.to_dict(precision=4)

            # Extract key fields to top level for easy access
            summary["regime"] = regime_dict["final_regime"]
            summary["regime_name"] = regime_dict["regime_name"]
            summary["confidence"] = regime_dict["confidence"]
            summary["regime_changed"] = regime_dict["regime_changed"]

            # Include decision vs final regime (important for hysteresis display)
            summary["decision_regime"] = regime_dict["decision_regime"]

            # Component states and values
            summary["component_states"] = regime_dict["component_states"]
            summary["component_values"] = regime_dict["component_values"]

            # PR-A: Validate close from regime output
            regime_close = regime_dict.get("component_values", {}).get("close", 0.0)
            is_valid, error_msg = validate_close_for_regime(
                regime_close, symbol, "ticker_summary"
            )
            if not is_valid:
                logger.warning(error_msg)
                # Try to get valid close from DataFrame as fallback
                if df is not None and not df.empty:
                    fallback_close, fallback_ts = get_last_valid_close(df, symbol)
                    if fallback_close > 0:
                        logger.info(
                            f"[{symbol}] Using fallback close from DataFrame: {fallback_close}"
                        )
                        summary["component_values"]["close"] = round(fallback_close, 2)

            # Derived metrics with percentiles
            summary["derived_metrics"] = regime_dict["derived_metrics"]

            # Transition state (for hysteresis display)
            summary["transition"] = regime_dict["transition"]

            # Data quality (for validation display)
            summary["quality"] = regime_dict["quality"]

            # Rules fired for decision tree display
            summary["rules_fired_decision"] = regime_dict["rules_fired_decision"]

            # Turning point if available
            if regime_dict.get("turning_point"):
                summary["turning_point"] = regime_dict["turning_point"]

            # Data window info
            summary["data_window"] = regime_dict["data_window"]
            summary["asof_ts"] = regime_dict["asof_ts"]

        elif df is not None and not df.empty:
            # PR-A: Use get_last_valid_close for fallback
            # This ensures we don't get 0.0 or -1.0 sentinel values
            valid_close, close_ts = get_last_valid_close(df, symbol)
            summary["component_values"] = {
                "close": round(valid_close, 2) if valid_close > 0 else 0.0,
            }
            if valid_close <= 0:
                logger.warning(f"[{symbol}] No valid close found in DataFrame for summary")

        return summary

    def _build_market_overview(
        self,
        regime_outputs: Dict[str, "RegimeOutput"],
    ) -> Dict[str, Any]:
        """Build market overview section (benchmarks like SPY, QQQ)."""
        benchmarks = ["SPY", "QQQ", "IWM", "DIA"]
        overview: Dict[str, Any] = {"benchmarks": {}}

        for symbol in benchmarks:
            if symbol in regime_outputs:
                regime = regime_outputs[symbol]
                overview["benchmarks"][symbol] = {
                    "regime": regime.final_regime.value,
                    "confidence": regime.confidence,
                }

        return overview

    def _extract_ticker_data_quality(
        self,
        symbol: str,
        df: Optional[pd.DataFrame],
        regime: Optional["RegimeOutput"],
    ) -> Dict[str, Any]:
        """
        Extract per-ticker data quality information for PR-B.

        Returns:
            Dict with data quality metrics for this ticker:
            - usable_bars: Number of usable bars after cleaning
            - dropped_bars: Number of bars dropped
            - sentinel_count: Number of sentinel values detected
            - gaps_detected: Number of gaps in data
            - reasons: List of quality issues found
            - regime_trustworthy: Whether regime classification can be trusted
        """
        quality: Dict[str, Any] = {
            "usable_bars": 0,
            "dropped_bars": 0,
            "sentinel_count": 0,
            "gaps_detected": 0,
            "reasons": [],
            "regime_trustworthy": True,
        }

        # Get quality from DataFrame
        if df is not None and not df.empty:
            quality["usable_bars"] = len(df)

            # Check for sentinel values in close column
            if "close" in df.columns:
                sentinel_count = (df["close"] == -1.0).sum()
                quality["sentinel_count"] = int(sentinel_count)
                if sentinel_count > 0:
                    quality["reasons"].append("SENTINEL_VALUES")
                    quality["regime_trustworthy"] = False

            # Check for NaN values
            if "close" in df.columns:
                nan_count = df["close"].isna().sum()
                if nan_count > 0:
                    quality["reasons"].append("NAN_VALUES")

        # Get quality from regime output if available
        if regime and hasattr(regime, "quality"):
            regime_quality = regime.quality
            if hasattr(regime_quality, "component_validity"):
                validity = regime_quality.component_validity
                # Check if close is marked as invalid
                if isinstance(validity, dict) and not validity.get("close", True):
                    quality["regime_trustworthy"] = False
                    if "INVALID_CLOSE" not in quality["reasons"]:
                        quality["reasons"].append("INVALID_CLOSE")

            if hasattr(regime_quality, "component_issues"):
                issues = regime_quality.component_issues
                if isinstance(issues, dict) and issues.get("close"):
                    quality["regime_trustworthy"] = False

        return quality

    def _write_regime_html_files(
        self,
        regime_outputs: Dict[str, "RegimeOutput"],
        regime_dir: Path,
    ) -> List[str]:
        """
        Write pre-rendered regime HTML files for each symbol.

        This provides 1:1 feature parity with SignalReportGenerator by using
        the same HTML generation functions from the regime package.

        Each file contains:
        - Report header
        - One-liner summary
        - Methodology section
        - Decision tree
        - Components 4-block
        - Quality section
        - Hysteresis section
        - Turning point section
        - Optimization section (placeholder if no provenance)
        - Recommendations section (placeholder if no results)
        """
        files_written = []

        for symbol, regime_output in regime_outputs.items():
            try:
                html_sections = []

                # Generate all regime sections using the regime package functions
                # Note: Some functions don't take theme arg, they use CSS variables
                html_sections.append(generate_report_header_html(regime_output, theme=self.theme))
                html_sections.append(generate_regime_one_liner_html(regime_output))
                html_sections.append(generate_methodology_html(theme=self.theme))
                html_sections.append(generate_decision_tree_html(regime_output, theme=self.theme))
                html_sections.append(
                    generate_components_4block_html(regime_output, theme=self.theme)
                )
                html_sections.append(generate_quality_html(regime_output, theme=self.theme))
                html_sections.append(generate_hysteresis_html(regime_output, theme=self.theme))
                html_sections.append(generate_turning_point_html(regime_output, theme=self.theme))

                # Optimization sections (placeholders - actual data comes from param services)
                html_sections.append(generate_optimization_html(provenance=None, theme=self.theme))
                html_sections.append(generate_recommendations_html(result=None, theme=self.theme))

                # Combine into full regime HTML
                regime_html = f"""
<!-- Regime Analysis for {symbol} - Generated by PackageBuilder -->
<div class="regime-report-container" data-symbol="{symbol}">
    {''.join(html_sections)}
</div>
"""
                # Write to file
                file_path = regime_dir / f"{symbol}.html"
                file_path.write_text(regime_html, encoding="utf-8")
                files_written.append(symbol)

                logger.debug(f"Wrote regime HTML: {file_path}")
            except Exception as e:
                # Log warning but continue - don't fail entire build for regime HTML
                logger.warning(f"Failed to generate regime HTML for {symbol}: {e}")

        return files_written

    def _build_css(self) -> str:
        """Build CSS content for styles.css including regime styles."""
        c = self._colors

        # Get regime styles (function doesn't take theme arg - uses CSS vars)
        regime_css = generate_regime_styles()

        return f"""
/* Signal Package Styles - Theme: {self.theme} */
/* Generated by PackageBuilder - PR-02 Feature Parity */

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: {c['bg']};
    color: {c['text']};
    line-height: 1.6;
}}

.container {{
    max-width: 1600px;
    margin: 0 auto;
    padding: 20px;
}}

.header {{
    text-align: center;
    padding: 24px;
    margin-bottom: 24px;
    background: linear-gradient(135deg, #1e40af 0%, {c['primary']} 100%);
    border-radius: 12px;
    color: white;
}}

.header-top {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 24px;
    margin-bottom: 8px;
}}

.header h1 {{
    font-size: 28px;
    font-weight: 600;
    margin: 0;
}}

.validation-link {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: rgba(255,255,255,0.15);
    color: white;
    text-decoration: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    transition: background 0.2s;
}}

.validation-link:hover {{
    background: rgba(255,255,255,0.25);
}}

.header .meta {{
    display: flex;
    justify-content: center;
    gap: 24px;
    font-size: 14px;
    opacity: 0.9;
}}

.controls {{
    display: flex;
    gap: 24px;
    align-items: end;
    margin-bottom: 24px;
    padding: 16px;
    background: {c['card_bg']};
    border-radius: 12px;
    border: 1px solid {c['border']};
}}

.control-group {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}

.control-group label {{
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    color: {c['text_muted']};
}}

.control-group select {{
    padding: 10px 16px;
    font-size: 14px;
    border: 1px solid {c['border']};
    border-radius: 8px;
    background: {c['bg']};
    color: {c['text']};
    cursor: pointer;
    min-width: 150px;
}}

.timeframe-buttons {{
    display: flex;
    gap: 4px;
}}

.tf-btn {{
    padding: 10px 16px;
    font-size: 14px;
    font-weight: 500;
    border: 1px solid {c['border']};
    border-radius: 8px;
    background: {c['bg']};
    color: {c['text']};
    cursor: pointer;
    transition: all 0.2s;
}}

.tf-btn:hover {{
    border-color: {c['primary']};
}}

.tf-btn.active {{
    background: {c['primary']};
    border-color: {c['primary']};
    color: white;
}}

.chart-container {{
    background: {c['card_bg']};
    border-radius: 12px;
    border: 1px solid {c['border']};
    padding: 16px;
    margin-bottom: 24px;
}}

#main-chart {{
    min-height: 800px;
}}

#main-chart.loading {{
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 800px;
    color: {c['text_muted']};
}}

.loading-spinner {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.loading-spinner::after {{
    content: '';
    width: 24px;
    height: 24px;
    border: 3px solid {c['border']};
    border-top-color: {c['primary']};
    border-radius: 50%;
    animation: spin 1s linear infinite;
}}

.loading-error {{
    color: {c['danger']};
    text-align: center;
    padding: 40px;
}}

@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}

/* Section Headers */
.section {{
    margin-bottom: 24px;
}}

.section-header {{
    cursor: pointer;
    padding: 16px;
    background: {c['card_bg']};
    border: 1px solid {c['border']};
    border-radius: 8px 8px 0 0;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 16px;
    font-weight: 600;
    user-select: none;
}}

.section-header:hover {{
    background: {c['bg']};
}}

.toggle-icon {{
    font-size: 12px;
    transition: transform 0.2s ease;
}}

.section-content {{
    padding: 16px;
    background: {c['card_bg']};
    border: 1px solid {c['border']};
    border-top: none;
    border-radius: 0 0 8px 8px;
}}

.section-content.collapsed {{
    display: none;
}}

/* Confluence Panel */
.confluence-panel {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}}

.confluence-score {{
    display: flex;
    flex-direction: column;
    gap: 16px;
}}

.alignment-meter {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}

.alignment-bar {{
    height: 24px;
    background: linear-gradient(to right, {c['danger']} 0%, {c['text_muted']} 50%, {c['success']} 100%);
    border-radius: 12px;
    position: relative;
    overflow: visible;
}}

.alignment-indicator {{
    position: absolute;
    top: 50%;
    transform: translateX(-50%) translateY(-50%);
    width: 4px;
    height: 32px;
    background: white;
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(0,0,0,0.5);
}}

.alignment-value {{
    font-size: 28px;
    font-weight: 700;
    text-align: center;
}}

.alignment-value.bullish {{ color: {c['success']}; }}
.alignment-value.bearish {{ color: {c['danger']}; }}
.alignment-value.neutral {{ color: {c['text_muted']}; }}

.signal-counts {{
    display: flex;
    justify-content: center;
    gap: 24px;
}}

.count-item {{
    text-align: center;
}}

.count-value {{
    font-size: 24px;
    font-weight: 600;
}}

.count-value.bullish {{ color: {c['success']}; }}
.count-value.bearish {{ color: {c['danger']}; }}
.count-value.neutral {{ color: {c['text_muted']}; }}

.count-label {{
    font-size: 12px;
    color: {c['text_muted']};
    text-transform: uppercase;
}}

.strongest-signal {{
    text-align: center;
    padding: 12px;
    background: {c['bg']};
    border-radius: 8px;
}}

.strongest-signal .label {{
    font-size: 12px;
    color: {c['text_muted']};
    text-transform: uppercase;
}}

.strongest-signal .value {{
    font-size: 18px;
    font-weight: 600;
}}

.strongest-signal .value.bullish {{ color: {c['success']}; }}
.strongest-signal .value.bearish {{ color: {c['danger']}; }}
.strongest-signal .value.neutral {{ color: {c['text_muted']}; }}

.divergence-section h4 {{
    margin-bottom: 12px;
    color: {c['text_muted']};
    font-size: 12px;
    text-transform: uppercase;
}}

.divergence-item {{
    padding: 12px;
    background: {c['bg']};
    border-radius: 8px;
    margin-bottom: 8px;
}}

.divergence-item .indicators {{
    font-weight: 600;
    margin-bottom: 4px;
}}

.divergence-item .reason {{
    color: {c['text_muted']};
    font-size: 12px;
}}

.no-divergences, .no-confluence, .no-regime, .no-signals {{
    text-align: center;
    color: {c['text_muted']};
    padding: 24px;
    font-style: italic;
}}

/* Regime Dashboard */
.regime-dashboard {{
    display: grid;
    gap: 16px;
}}

.regime-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    background: {c['bg']};
    border-radius: 8px;
}}

.regime-badge {{
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 700;
}}

.regime-name {{
    font-size: 16px;
    font-weight: 600;
    flex: 1;
}}

.regime-confidence {{
    color: {c['text_muted']};
    font-size: 14px;
}}

.regime-components, .regime-metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
    padding: 16px;
    background: {c['bg']};
    border-radius: 8px;
}}

.component-row, .metric-row {{
    display: flex;
    justify-content: space-between;
    padding: 8px;
}}

.component-label, .metric-label {{
    color: {c['text_muted']};
    font-size: 13px;
}}

.component-value, .metric-value {{
    font-weight: 500;
    font-size: 13px;
}}

/* Signal History Table */
.signal-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

.signal-table th {{
    text-align: left;
    padding: 12px 8px;
    border-bottom: 2px solid {c['border']};
    color: {c['text_muted']};
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
}}

.signal-table td {{
    padding: 10px 8px;
    border-bottom: 1px solid {c['border']};
}}

.signal-table tr:hover {{
    background: {c['bg']};
}}

.signal-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}}

.signal-badge.buy {{
    background: rgba(34, 197, 94, 0.2);
    color: {c['success']};
}}

.signal-badge.sell {{
    background: rgba(239, 68, 68, 0.2);
    color: {c['danger']};
}}

.signal-badge.alert {{
    background: rgba(59, 130, 246, 0.2);
    color: {c['primary']};
}}

/* Indicators Section */
.category-group {{
    margin-bottom: 24px;
}}

.category-title {{
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid {c['border']};
    color: {c['text']};
}}

.indicator-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
}}

.indicator-card {{
    background: {c['bg']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 12px;
}}

.indicator-card h3 {{
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
    color: {c['primary']};
}}

.indicator-card .description {{
    font-size: 12px;
    color: {c['text_muted']};
    margin-bottom: 8px;
}}

.indicator-card .rules {{
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid {c['border']};
}}

.indicator-card .rules h4 {{
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 6px;
    color: {c['text_muted']};
    text-transform: uppercase;
}}

.rule-item {{
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-bottom: 6px;
    padding: 4px 6px;
    background: {c['card_bg']};
    border-radius: 4px;
}}

.rule-name {{
    font-size: 11px;
    font-weight: 600;
}}

.rule-name.direction-buy {{
    color: {c['success']};
}}

.rule-name.direction-sell {{
    color: {c['danger']};
}}

.rule-name.direction-alert {{
    color: {c['primary']};
}}

.rule-desc {{
    font-size: 10px;
    color: {c['text_muted']};
}}

.no-indicators {{
    text-align: center;
    color: {c['text_muted']};
    padding: 24px;
    font-style: italic;
}}

/* Responsive */
@media (max-width: 768px) {{
    .controls {{
        flex-direction: column;
        align-items: stretch;
    }}
    .timeframe-buttons {{
        flex-wrap: wrap;
    }}
    .confluence-panel {{
        grid-template-columns: 1fr;
    }}
    .regime-header {{
        flex-direction: column;
        text-align: center;
    }}
}}

/* Regime Report Styles (1:1 Feature Parity) */
{regime_css}
"""

    def _build_javascript(self, symbols: List[str], timeframes: List[str]) -> str:
        """Build JavaScript content for app.js with lazy loading and full chart rendering."""
        c = self._colors
        colors_json = json.dumps(c)
        return f"""
// Signal Package JavaScript
// Generated by PackageBuilder - supports lazy loading with full chart rendering

const CONFIG = {{
    symbols: {json.dumps(symbols)},
    timeframes: {json.dumps(timeframes)},
    dataCache: {{}},
    summary: null,
    currentSymbol: '{symbols[0] if symbols else ''}',
    currentTimeframe: '{timeframes[0] if timeframes else '1d'}'
}};

const colors = {colors_json};

function getDataKey() {{
    return `${{CONFIG.currentSymbol}}_${{CONFIG.currentTimeframe}}`;
}}

// Load summary on page load
async function loadSummary() {{
    try {{
        const response = await fetch('data/summary.json');
        CONFIG.summary = await response.json();
        console.log('Summary loaded:', CONFIG.summary.symbol_count, 'symbols');
        await updateChart();
    }} catch (error) {{
        console.error('Failed to load summary:', error);
    }}
}}

// Lazy load data for a symbol/timeframe
async function loadData(symbol, timeframe) {{
    const key = `${{symbol}}_${{timeframe}}`;

    // Check cache first
    if (CONFIG.dataCache[key]) {{
        return CONFIG.dataCache[key];
    }}

    // Show loading state
    const chartEl = document.getElementById('main-chart');
    chartEl.classList.add('loading');
    chartEl.innerHTML = '<div class="loading-spinner">Loading data...</div>';

    try {{
        const response = await fetch(`data/${{key}}.json`);
        if (!response.ok) {{
            throw new Error(`Data not found for ${{key}}`);
        }}
        const data = await response.json();
        CONFIG.dataCache[key] = data;
        console.log(`Loaded ${{key}}: ${{data.bar_count}} bars`);
        return data;
    }} catch (error) {{
        console.error(`Failed to load ${{key}}:`, error);
        chartEl.innerHTML = '<div class="loading-error">Failed to load data for ' + key + '</div>';
        return null;
    }}
}}

// Update chart with current selection
async function updateChart() {{
    const symbol = document.getElementById('symbol-select').value;
    const timeframe = CONFIG.currentTimeframe;

    CONFIG.currentSymbol = symbol;

    const data = await loadData(symbol, timeframe);
    if (!data) return;

    renderMainChart(data);
    updateSignalHistoryTable();
    updateConfluencePanel();
    updateRegimeSection();
}}

// Render full multi-subplot chart using Plotly (matches SignalReportGenerator)
function renderMainChart(data) {{
    const chartEl = document.getElementById('main-chart');
    // Clear loading state
    chartEl.classList.remove('loading');
    chartEl.innerHTML = '';

    const chartData = data.chart_data;
    const traces = [];
    const hasData = (values) => values && values.length > 0 && !values.every(v => v === null);

    // Issue 4: Debug logging for indicator data
    console.log('=== Chart Data Debug ===');
    console.log('Timestamps count:', chartData.timestamps?.length || 0);
    console.log('OHLCV counts:', {{
        open: chartData.open?.length || 0,
        high: chartData.high?.length || 0,
        low: chartData.low?.length || 0,
        close: chartData.close?.length || 0,
        volume: chartData.volume?.length || 0
    }});
    console.log('Overlay indicators:', Object.keys(chartData.overlays || {{}}));
    console.log('RSI data keys:', Object.keys(chartData.rsi || {{}}));
    console.log('MACD data keys:', Object.keys(chartData.macd || {{}}));
    console.log('RSI values (first 5):', chartData.rsi?.rsi_rsi?.slice(0, 5));
    console.log('MACD values (first 5):', chartData.macd?.macd_macd?.slice(0, 5));
    console.log('Bollinger values (first 5):', chartData.overlays?.bollinger_bb_upper?.slice(0, 5));

    // For intraday charts, use index-based x-axis to avoid gaps
    const isIntraday = ['1m', '5m', '15m', '30m', '1h', '4h'].includes(data.timeframe);
    const xValues = isIntraday
        ? chartData.timestamps.map((_, i) => i)
        : chartData.timestamps;

    // Row 1: Price candlesticks
    traces.push({{
        type: 'candlestick',
        x: xValues,
        open: chartData.open,
        high: chartData.high,
        low: chartData.low,
        close: chartData.close,
        name: 'Price',
        increasing: {{ line: {{ color: colors.success }}, fillcolor: colors.success }},
        decreasing: {{ line: {{ color: colors.danger }}, fillcolor: colors.danger }},
        xaxis: 'x',
        yaxis: 'y',
    }});

    // Overlay indicators: Bollinger Bands, SuperTrend
    const overlayConfig = {{
        'bollinger_bb_upper': {{ color: '#3b82f6', dash: 'dot' }},
        'bollinger_bb_middle': {{ color: '#6366f1', dash: 'solid' }},
        'bollinger_bb_lower': {{ color: '#3b82f6', dash: 'dot' }},
        'supertrend_supertrend': {{ color: '#f59e0b', dash: 'solid' }},
    }};
    for (const [name, config] of Object.entries(overlayConfig)) {{
        const values = chartData.overlays[name];
        if (!hasData(values)) continue;
        traces.push({{
            type: 'scatter',
            mode: 'lines',
            x: xValues,
            y: values,
            name: name.replace('bollinger_bb_', 'BB ').replace('supertrend_', 'ST '),
            line: {{ color: config.color, width: 1, dash: config.dash }},
            xaxis: 'x',
            yaxis: 'y',
        }});
    }}

    // Row 2: RSI with threshold lines
    const rsiValues = chartData.rsi['rsi_rsi'];
    if (hasData(rsiValues)) {{
        traces.push({{
            type: 'scatter',
            mode: 'lines',
            x: xValues,
            y: rsiValues,
            name: 'RSI',
            line: {{ color: '#8b5cf6', width: 1.5 }},
            xaxis: 'x',
            yaxis: 'y2',
        }});
        const boundsX = [xValues[0], xValues[xValues.length - 1]];
        const rsiLevels = [
            {{ value: 70, name: 'Overbought', color: colors.danger }},
            {{ value: 30, name: 'Oversold', color: colors.success }},
        ];
        for (const level of rsiLevels) {{
            traces.push({{
                type: 'scatter',
                mode: 'lines',
                x: boundsX,
                y: [level.value, level.value],
                name: level.name,
                line: {{ color: level.color, width: 1, dash: 'dash' }},
                xaxis: 'x',
                yaxis: 'y2',
                showlegend: false,
            }});
        }}
    }}

    // Row 3: MACD subplot
    const macdHist = chartData.macd['macd_histogram'];
    if (hasData(macdHist)) {{
        const barColors = macdHist.map(v => v >= 0 ? colors.success : colors.danger);
        traces.push({{
            type: 'bar',
            x: xValues,
            y: macdHist,
            name: 'MACD Hist',
            marker: {{ color: barColors }},
            xaxis: 'x',
            yaxis: 'y3',
        }});
    }}
    const macdLines = [
        {{ key: 'macd_macd', name: 'MACD', color: '#3b82f6' }},
        {{ key: 'macd_signal', name: 'Signal', color: '#f59e0b' }},
    ];
    for (const {{ key, name, color }} of macdLines) {{
        const values = chartData.macd[key];
        if (!hasData(values)) continue;
        traces.push({{
            type: 'scatter',
            mode: 'lines',
            x: xValues,
            y: values,
            name,
            line: {{ color, width: 1.5 }},
            xaxis: 'x',
            yaxis: 'y3',
        }});
    }}

    // Row 4: Volume bars
    if (chartData.volume && chartData.volume.length > 0) {{
        const volColors = chartData.close.map((c, i) => {{
            if (i === 0) return colors.text_muted;
            return c >= chartData.close[i-1] ? colors.success : colors.danger;
        }});
        traces.push({{
            type: 'bar',
            x: xValues,
            y: chartData.volume,
            name: 'Volume',
            marker: {{ color: volColors, opacity: 0.5 }},
            xaxis: 'x',
            yaxis: 'y4',
        }});
    }}

    // Add signal markers on price chart
    const key = getDataKey();
    const signals = data.signals || [];
    const buySignals = signals.filter(s => s.direction === 'buy');
    const sellSignals = signals.filter(s => s.direction === 'sell');

    if (buySignals.length > 0) {{
        const buyData = buySignals.map(s => {{
            const idx = chartData.timestamps.findIndex(t => t === s.timestamp);
            if (idx < 0) return null;
            return {{
                x: isIntraday ? idx : s.timestamp,
                y: chartData.low[idx] * 0.995,
                rule: s.rule
            }};
        }}).filter(d => d !== null);

        if (buyData.length > 0) {{
            traces.push({{
                type: 'scatter',
                mode: 'markers',
                x: buyData.map(d => d.x),
                y: buyData.map(d => d.y),
                name: 'Buy Signal',
                marker: {{
                    symbol: 'triangle-up',
                    size: 12,
                    color: colors.success,
                    line: {{ color: 'white', width: 1 }}
                }},
                hovertemplate: '%{{text}}<extra></extra>',
                text: buyData.map(d => d.rule),
                xaxis: 'x',
                yaxis: 'y',
            }});
        }}
    }}

    if (sellSignals.length > 0) {{
        const sellData = sellSignals.map(s => {{
            const idx = chartData.timestamps.findIndex(t => t === s.timestamp);
            if (idx < 0) return null;
            return {{
                x: isIntraday ? idx : s.timestamp,
                y: chartData.high[idx] * 1.005,
                rule: s.rule
            }};
        }}).filter(d => d !== null);

        if (sellData.length > 0) {{
            traces.push({{
                type: 'scatter',
                mode: 'markers',
                x: sellData.map(d => d.x),
                y: sellData.map(d => d.y),
                name: 'Sell Signal',
                marker: {{
                    symbol: 'triangle-down',
                    size: 12,
                    color: colors.danger,
                    line: {{ color: 'white', width: 1 }}
                }},
                hovertemplate: '%{{text}}<extra></extra>',
                text: sellData.map(d => d.rule),
                xaxis: 'x',
                yaxis: 'y',
            }});
        }}
    }}

    // Layout with 4 subplots
    const isDaily = ['1d', '1w', '1D', '1W'].includes(data.timeframe);
    let rangebreaks = [];
    if (isDaily) {{
        rangebreaks = [{{ bounds: ['sat', 'mon'] }}];
    }}

    // Custom tick labels for intraday
    let tickvals = null;
    let ticktext = null;
    if (isIntraday && chartData.timestamps.length > 0) {{
        const n = chartData.timestamps.length;
        const step = Math.max(1, Math.floor(n / 15));
        tickvals = [];
        ticktext = [];
        for (let i = 0; i < n; i += step) {{
            tickvals.push(i);
            const ts = new Date(chartData.timestamps[i]);
            const month = String(ts.getUTCMonth() + 1).padStart(2, '0');
            const day = String(ts.getUTCDate()).padStart(2, '0');
            const hour = String(ts.getUTCHours()).padStart(2, '0');
            const min = String(ts.getUTCMinutes()).padStart(2, '0');
            ticktext.push(`${{month}}/${{day}} ${{hour}}:${{min}}`);
        }}
    }}

    const layout = {{
        title: {{
            text: `${{data.symbol}} - ${{data.timeframe}} (${{data.bar_count}} bars)`,
            font: {{ color: colors.text, size: 18 }},
        }},
        showlegend: true,
        legend: {{
            orientation: 'h',
            y: -0.08,
            font: {{ color: colors.text, size: 10 }},
        }},
        paper_bgcolor: colors.card_bg,
        plot_bgcolor: colors.card_bg,
        font: {{ color: colors.text }},
        margin: {{ t: 50, r: 50, b: 80, l: 50 }},
        hovermode: 'x unified',
        bargap: 0.1,

        xaxis: {{
            title: {{ text: 'Time (UTC)', standoff: 10, font: {{ size: 11, color: colors.text_muted }} }},
            gridcolor: colors.border,
            showgrid: true,
            rangeslider: {{ visible: false }},
            tickangle: -45,
            domain: [0, 1],
            rangebreaks: rangebreaks,
            ...(isIntraday && tickvals ? {{ tickvals: tickvals, ticktext: ticktext }} : {{ nticks: 15 }}),
        }},

        // Y1: Price (48% with gap) - CRITICAL: autorange: true
        yaxis: {{
            title: 'Price',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.52, 1.00],
            autorange: true,
        }},

        // Y2: RSI (12% with gap)
        yaxis2: {{
            title: 'RSI',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.36, 0.48],
            range: [0, 100],
            dtick: 25,
        }},

        // Y3: MACD (12% with gap)
        yaxis3: {{
            title: 'MACD',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.20, 0.32],
        }},

        // Y4: Volume (16%)
        yaxis4: {{
            title: 'Vol',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.00, 0.16],
        }},
    }};

    const config = {{
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    }};

    Plotly.newPlot('main-chart', traces, layout, config);
}}

// Update signal history table
function updateSignalHistoryTable() {{
    const key = getDataKey();
    const cachedData = CONFIG.dataCache[key];
    const container = document.getElementById('signals-content');
    if (!container) return;

    const signals = cachedData ? cachedData.signals || [] : [];

    if (signals.length === 0) {{
        container.innerHTML = '<div class="no-signals">No signals detected for this symbol/timeframe</div>';
        return;
    }}

    let html = `
        <table class="signal-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Signal</th>
                    <th>Direction</th>
                    <th>Indicator</th>
                    <th>Message</th>
                </tr>
            </thead>
            <tbody>
    `;

    // Show most recent first
    const sortedSignals = [...signals].reverse();
    for (const sig of sortedSignals) {{
        const time = new Date(sig.timestamp).toLocaleString();
        const direction = sig.direction || 'alert';
        html += `
            <tr>
                <td>${{time}}</td>
                <td>${{sig.rule}}</td>
                <td><span class="signal-badge ${{direction}}">${{direction}}</span></td>
                <td>${{sig.indicator}}</td>
                <td>${{sig.message || '-'}}</td>
            </tr>
        `;
    }}

    html += '</tbody></table>';
    container.innerHTML = html;
}}

// Update confluence panel
function updateConfluencePanel() {{
    const key = getDataKey();
    const container = document.getElementById('confluence-content');
    if (!container || !CONFIG.summary) return;

    const confluence = CONFIG.summary.confluence ? CONFIG.summary.confluence[key] : null;

    if (!confluence) {{
        container.innerHTML = '<div class="no-confluence">No confluence data available for this symbol/timeframe</div>';
        return;
    }}

    const alignmentPct = (confluence.alignment_score + 100) / 2;
    const alignmentClass = confluence.alignment_score > 20 ? 'bullish' : confluence.alignment_score < -20 ? 'bearish' : 'neutral';
    const strongestClass = confluence.strongest_signal === 'bullish' ? 'bullish' : confluence.strongest_signal === 'bearish' ? 'bearish' : 'neutral';

    let divergenceHtml = '';
    if (confluence.diverging_pairs && confluence.diverging_pairs.length > 0) {{
        divergenceHtml = confluence.diverging_pairs.slice(0, 5).map(p => `
            <div class="divergence-item">
                <div class="indicators">${{p.ind1}} - ${{p.ind2}}</div>
                <div class="reason">${{p.reason}}</div>
            </div>
        `).join('');
    }} else {{
        divergenceHtml = '<div class="no-divergences">No divergences detected - indicators are aligned</div>';
    }}

    container.innerHTML = `
        <div class="confluence-panel">
            <div class="confluence-score">
                <div class="alignment-meter">
                    <div class="alignment-bar">
                        <div class="alignment-indicator" style="left: ${{alignmentPct}}%"></div>
                    </div>
                </div>
                <div class="alignment-value ${{alignmentClass}}">${{confluence.alignment_score > 0 ? '+' : ''}}${{Math.round(confluence.alignment_score)}}</div>
                <div class="signal-counts">
                    <div class="count-item">
                        <div class="count-value bullish">&#9650; ${{confluence.bullish_count}}</div>
                        <div class="count-label">Bullish</div>
                    </div>
                    <div class="count-item">
                        <div class="count-value neutral">&#9679; ${{confluence.neutral_count}}</div>
                        <div class="count-label">Neutral</div>
                    </div>
                    <div class="count-item">
                        <div class="count-value bearish">&#9660; ${{confluence.bearish_count}}</div>
                        <div class="count-label">Bearish</div>
                    </div>
                </div>
                <div class="strongest-signal">
                    <div class="label">Strongest Signal</div>
                    <div class="value ${{strongestClass}}">${{confluence.strongest_signal ? confluence.strongest_signal.toUpperCase() : 'NONE'}}</div>
                </div>
            </div>
            <div class="divergence-section">
                <h4>Diverging Indicators</h4>
                ${{divergenceHtml}}
            </div>
        </div>
    `;
}}

// Regime HTML cache
const regimeHtmlCache = {{}};

// Update regime section - Load pre-rendered HTML for 1:1 feature parity
async function updateRegimeSection() {{
    const container = document.getElementById('regime-content');
    if (!container) return;

    const symbol = CONFIG.currentSymbol;

    // Check cache first
    if (regimeHtmlCache[symbol]) {{
        container.innerHTML = regimeHtmlCache[symbol];
        return;
    }}

    // Show loading state
    container.innerHTML = '<div class="no-regime">Loading regime analysis...</div>';

    try {{
        const response = await fetch(`data/regime/${{symbol}}.html`);
        if (!response.ok) {{
            // Fall back to summary data if no pre-rendered HTML
            fallbackRegimeSection(container, symbol);
            return;
        }}
        const html = await response.text();
        regimeHtmlCache[symbol] = html;
        container.innerHTML = html;
        console.log(`Loaded regime HTML for ${{symbol}}`);
    }} catch (error) {{
        console.error(`Failed to load regime HTML for ${{symbol}}:`, error);
        fallbackRegimeSection(container, symbol);
    }}
}}

// Fallback: render basic regime info from summary data
function fallbackRegimeSection(container, symbol) {{
    if (!CONFIG.summary) {{
        container.innerHTML = '<div class="no-regime">No regime data available for ' + symbol + '</div>';
        return;
    }}

    const tickers = CONFIG.summary.tickers || [];
    const ticker = tickers.find(t => t.symbol === symbol);

    if (!ticker || !ticker.regime) {{
        container.innerHTML = '<div class="no-regime">No regime data available for ' + symbol + '</div>';
        return;
    }}

    const regimeColors = {{
        'R0': colors.success,
        'R1': colors.warning,
        'R2': colors.danger,
        'R3': colors.primary,
    }};
    const regimeNames = {{
        'R0': 'Healthy Uptrend',
        'R1': 'Choppy/Extended',
        'R2': 'Risk-Off',
        'R3': 'Rebound Window',
    }};
    const regimeColor = regimeColors[ticker.regime] || colors.text_muted;

    const components = ticker.component_states || {{}};
    const cv = ticker.component_values || {{}};
    const transition = ticker.transition || {{}};
    const quality = ticker.quality || {{}};

    const stateColors = {{
        'trend_up': colors.success, 'trend_down': colors.danger, 'neutral': colors.text_muted,
        'vol_high': colors.danger, 'vol_normal': colors.text_muted, 'vol_low': colors.success,
        'choppy': colors.warning, 'trending': colors.success,
        'overbought': colors.danger, 'oversold': colors.success, 'slightly_high': colors.warning, 'slightly_low': colors.primary,
    }};

    const formatState = (state) => {{
        const color = stateColors[state] || colors.text_muted;
        return `<span style="color: ${{color}}; font-weight: 500;">${{state ? state.toUpperCase().replace('_', ' ') : 'N/A'}}</span>`;
    }};

    container.innerHTML = `
        <div class="regime-dashboard">
            <div class="regime-header">
                <div class="regime-badge" style="background: ${{regimeColor}}20; color: ${{regimeColor}}; border: 2px solid ${{regimeColor}}; padding: 12px 24px; border-radius: 12px; font-size: 24px; font-weight: 700;">
                    ${{ticker.regime}}
                </div>
                <div style="flex: 1;">
                    <div class="regime-name" style="font-size: 18px; font-weight: 600;">${{ticker.regime_name || regimeNames[ticker.regime] || 'Unknown'}}</div>
                    <div class="regime-confidence" style="color: ${{colors.text_muted}};">Confidence: ${{ticker.confidence || 0}}%</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
                <div class="regime-components" style="background: ${{colors.bg}}; border-radius: 8px; padding: 16px;">
                    <h4 style="margin: 0 0 12px 0; color: ${{colors.text_muted}}; font-size: 12px; text-transform: uppercase;">Component States</h4>
                    <div class="component-row" style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span class="component-label">Trend:</span>
                        ${{formatState(components.trend_state)}}
                    </div>
                    <div class="component-row" style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span class="component-label">Volatility:</span>
                        ${{formatState(components.vol_state)}}
                    </div>
                    <div class="component-row" style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span class="component-label">Choppiness:</span>
                        ${{formatState(components.chop_state)}}
                    </div>
                    <div class="component-row" style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span class="component-label">Extension:</span>
                        ${{formatState(components.ext_state)}}
                    </div>
                </div>
                <div class="regime-metrics" style="background: ${{colors.bg}}; border-radius: 8px; padding: 16px;">
                    <h4 style="margin: 0 0 12px 0; color: ${{colors.text_muted}}; font-size: 12px; text-transform: uppercase;">Key Metrics</h4>
                    <div class="metric-row" style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span class="metric-label">Close:</span>
                        <span class="metric-value" style="font-weight: 500;">${{cv.close ? '$' + cv.close.toFixed(2) : 'N/A'}}</span>
                    </div>
                    <div class="metric-row" style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span class="metric-label">MA50:</span>
                        <span class="metric-value">${{cv.ma50 ? '$' + cv.ma50.toFixed(2) : 'N/A'}}</span>
                    </div>
                </div>
            </div>
            <p style="margin-top: 16px; color: ${{colors.text_muted}}; font-size: 12px;">
                <em>Note: Full regime analysis not available. Showing summary data.</em>
            </p>
        </div>
    `;
}}

// Set timeframe
function setTimeframe(tf) {{
    CONFIG.currentTimeframe = tf;

    // Update button states
    document.querySelectorAll('.tf-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.tf === tf);
    }});

    updateChart();
}}

// Toggle section visibility - handles both string ID and DOM element (for regime report sections)
function toggleSection(arg) {{
    if (typeof arg === 'string') {{
        // Original behavior: arg is content ID
        const content = document.getElementById(arg);
        if (!content) return;
        const header = content.previousElementSibling;
        const icon = header ? header.querySelector('.toggle-icon') : null;

        content.classList.toggle('collapsed');
        if (icon) {{
            icon.style.transform = content.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
        }}
    }} else {{
        // New behavior: arg is the header element (from onclick="toggleSection(this)")
        const header = arg;
        const section = header.parentElement;
        if (!section || !section.classList.contains('report-section')) return;

        section.classList.toggle('collapsed');
        const indicator = header.querySelector('.collapse-indicator');
        if (indicator) {{
            indicator.textContent = section.classList.contains('collapsed') ? '▶' : '▼';
        }}
    }}
}}

// Load and render indicators section
async function loadIndicatorsSection() {{
    const container = document.getElementById('indicators-content');
    if (!container) return;

    try {{
        const response = await fetch('data/indicators.json');
        if (!response.ok) {{
            container.innerHTML = '<div class="no-indicators">Indicators data not available</div>';
            return;
        }}
        const data = await response.json();
        renderIndicatorsSection(container, data);
    }} catch (error) {{
        console.error('Failed to load indicators:', error);
        container.innerHTML = '<div class="no-indicators">Failed to load indicators</div>';
    }}
}}

function renderIndicatorsSection(container, data) {{
    if (!data.categories || data.categories.length === 0) {{
        container.innerHTML = '<div class="no-indicators">No indicators configured</div>';
        return;
    }}

    let html = '';
    for (const category of data.categories) {{
        html += `
            <div class="category-group">
                <div class="category-title">${{category.name}}</div>
                <div class="indicator-cards">
        `;

        for (const ind of category.indicators) {{
            const paramsStr = Object.entries(ind.params || {{}})
                .map(([k, v]) => `${{k}}=${{v}}`)
                .join(', ');

            let rulesHtml = '';
            if (ind.rules && ind.rules.length > 0) {{
                rulesHtml = '<div class="rules"><h4>Rules</h4>';
                for (const rule of ind.rules) {{
                    rulesHtml += `
                        <div class="rule-item">
                            <span class="rule-name direction-${{rule.direction}}">${{rule.id}}</span>
                            <div class="rule-desc">${{rule.description}}</div>
                        </div>
                    `;
                }}
                rulesHtml += '</div>';
            }}

            html += `
                <div class="indicator-card">
                    <h3>${{ind.name}}</h3>
                    <div class="description">${{ind.description}}${{paramsStr ? '. Params: ' + paramsStr : ''}}</div>
                    ${{rulesHtml}}
                </div>
            `;
        }}

        html += '</div></div>';
    }}

    container.innerHTML = html;
}}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {{
    loadSummary();
    loadIndicatorsSection();

    // Set initial timeframe button
    const initialTf = CONFIG.currentTimeframe;
    document.querySelectorAll('.tf-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.tf === initialTf);
    }});
}});
"""

    def _build_index_html(
        self,
        symbols: List[str],
        timeframes: List[str],
        regime_outputs: Optional[Dict[str, "RegimeOutput"]],
        validation_url: Optional[str] = None,
    ) -> str:
        """Build the index.html shell with full feature parity sections."""
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        symbol_options = "\n".join(
            f'                    <option value="{s}">{s}</option>' for s in symbols
        )

        timeframe_buttons = "\n".join(
            f'                    <button class="tf-btn" data-tf="{tf}" onclick="setTimeframe(\'{tf}\')">{tf}</button>'
            for tf in timeframes
        )

        # Validation link if URL provided
        validation_link = ""
        if validation_url:
            validation_link = (
                f'<a href="{validation_url}" class="validation-link">📊 Validation Results</a>'
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="header-top">
                <h1>Signal Analysis Report</h1>
                {validation_link}
            </div>
            <div class="meta">
                <span><strong>Symbols:</strong> {len(symbols)}</span>
                <span><strong>Timeframes:</strong> {', '.join(timeframes)}</span>
                <span><strong>Generated:</strong> {generated_at}</span>
            </div>
        </header>

        <div class="controls">
            <div class="control-group">
                <label>Symbol</label>
                <select id="symbol-select" onchange="updateChart()">
{symbol_options}
                </select>
            </div>
            <div class="control-group">
                <label>Timeframe</label>
                <div class="timeframe-buttons">
{timeframe_buttons}
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div id="main-chart" class="loading">
                <div class="loading-spinner">Loading chart data...</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('confluence-content')">
                <span class="toggle-icon">&#9660;</span> Confluence Analysis
            </h2>
            <div id="confluence-content" class="section-content">
                <div class="no-confluence">Loading confluence data...</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('regime-content')">
                <span class="toggle-icon">&#9660;</span> Regime Analysis
            </h2>
            <div id="regime-content" class="section-content">
                <div class="no-regime">Loading regime data...</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('signals-content')">
                <span class="toggle-icon">&#9660;</span> Signal History
            </h2>
            <div id="signals-content" class="section-content">
                <div class="no-signals">Loading signal history...</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('indicators-content')">
                <span class="toggle-icon">&#9660;</span> Indicators
            </h2>
            <div id="indicators-content" class="section-content collapsed">
                <div class="no-indicators">Loading indicators...</div>
            </div>
        </div>
    </div>

    <script src="assets/app.js"></script>
</body>
</html>"""

    @staticmethod
    def _timeframe_seconds(tf: str) -> int:
        """Convert timeframe string to seconds for sorting."""
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
            "1w": 604800,
        }
        return mapping.get(tf, 0)
