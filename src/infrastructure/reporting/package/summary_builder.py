"""
Summary Builder - PackageManifest and budget enforcement.

Handles summary.json creation with M3 PR-03 size budget enforcement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from src.domain.signals.data.quality_validator import (
    get_last_valid_close,
    validate_close_for_regime,
)
from src.utils.logging_setup import get_logger

from .constants import (
    CONFLUENCE_BUDGET_KB,
    DATA_QUALITY_BUDGET_KB,
    MARKET_BUDGET_KB,
    PACKAGE_FORMAT_VERSION,
    SUMMARY_BUDGET_KB,
    TICKERS_BUDGET_KB,
)

if TYPE_CHECKING:
    from src.domain.signals.indicators.regime import RegimeOutput

logger = get_logger(__name__)


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


class SummaryBuilder:
    """Builds summary.json with budget enforcement."""

    # Class-level budget constants for M3 PR-03 enforcement
    MAX_SUMMARY_KB = SUMMARY_BUDGET_KB
    TICKERS_BUDGET_KB = TICKERS_BUDGET_KB
    MARKET_BUDGET_KB = MARKET_BUDGET_KB
    CONFLUENCE_BUDGET_KB = CONFLUENCE_BUDGET_KB

    def __init__(self, enforce_budget: bool = False) -> None:
        self.enforce_budget = enforce_budget

    def check_budget(self, section: str, data: Dict, budget_kb: int) -> None:
        """
        Check if section data exceeds budget.

        Args:
            section: Section name
            data: Section data dictionary
            budget_kb: Budget limit in KB

        Raises:
            SizeBudgetExceeded: If enforce_budget is True and budget exceeded
        """
        from ..exceptions import SizeBudgetExceeded

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
        from ..exceptions import BudgetContributor

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

    def build_summary(
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
        from src.domain.signals.schemas import DataQualityReport

        from ..signal_report.confluence_analyzer import calculate_confluence

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

            ticker_summary = self._build_ticker_summary(symbol, data, regime, timeframes)

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
        self.check_budget("tickers", {"tickers": ticker_summaries}, TICKERS_BUDGET_KB)

        # Market overview (for benchmarks like SPY, QQQ)
        market_overview = self._build_market_overview(regime_outputs)
        summary["market"] = market_overview

        # Check market budget
        self.check_budget("market", market_overview, MARKET_BUDGET_KB)

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
        self.check_budget("confluence", confluence_data, CONFLUENCE_BUDGET_KB)

        # Check data quality budget
        self.check_budget("data_quality", summary["run_data_quality"], DATA_QUALITY_BUDGET_KB)

        # Phase 3: Rule frequency computation
        rule_frequency_data = self._compute_rule_frequency(data, timeframes)
        summary["rule_frequency"] = rule_frequency_data

        # Add per-ticker signal_count for heatmap trending mode
        by_symbol = rule_frequency_data.get("by_symbol", {})
        buy_by_symbol = rule_frequency_data.get("buy_by_symbol", {})
        sell_by_symbol = rule_frequency_data.get("sell_by_symbol", {})
        for ticker in ticker_summaries:
            symbol = ticker.get("symbol", "")
            ticker["signal_count"] = by_symbol.get(symbol, 0)
            ticker["buy_signal_count"] = buy_by_symbol.get(symbol, 0)
            ticker["sell_signal_count"] = sell_by_symbol.get(symbol, 0)

        return summary

    def _compute_daily_change(self, df: pd.DataFrame, symbol: str) -> Optional[float]:
        """
        Compute daily change % from last two closes.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol for logging

        Returns:
            Daily change percentage rounded to 2 decimals, or None if insufficient data
        """
        if df is None or df.empty:
            return None
        if "close" not in df.columns or len(df) < 2:
            return None

        prev_close = df["close"].iloc[-2]
        curr_close = df["close"].iloc[-1]

        # Validate closes
        if pd.isna(prev_close) or pd.isna(curr_close):
            return None
        if prev_close <= 0:
            logger.warning(f"[{symbol}] Invalid prev_close for daily_change: {prev_close}")
            return None

        daily_change = ((curr_close - prev_close) / prev_close) * 100
        return float(round(daily_change, 2))

    def _build_ticker_summary(
        self,
        symbol: str,
        data: Dict[Tuple[str, str], pd.DataFrame],
        regime: Optional["RegimeOutput"],
        timeframes: Optional[List[str]] = None,
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

        # Compute daily_change_pct, volume, and close from DataFrame (for heatmap dashboard)
        summary["daily_change_pct"] = self._compute_daily_change(df, symbol)
        summary["volume"] = (
            int(df["volume"].iloc[-1])
            if df is not None and "volume" in df.columns and not pd.isna(df["volume"].iloc[-1])
            else None
        )
        # Close at top level for easy access by heatmap
        summary["close"] = (
            round(float(df["close"].iloc[-1]), 2)
            if df is not None and "close" in df.columns and not pd.isna(df["close"].iloc[-1])
            else None
        )

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
            is_valid, error_msg = validate_close_for_regime(regime_close, symbol, "ticker_summary")
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

        # Add DuckDB coverage stats (PR-03: Data provenance)
        try:
            from src.infrastructure.stores.duckdb_coverage_store import DuckDBCoverageStore

            store = DuckDBCoverageStore()
            data_stats = store.get_ticker_stats(symbol)
            summary["data_stats"] = data_stats
            store.close()
        except Exception as e:
            logger.debug(f"Could not load DuckDB stats for {symbol}: {e}")
            summary["data_stats"] = None

        # Phase 4.4: Add MTF confluence (alignment across 1h, 4h, 1d)
        from ..signal_report.confluence_analyzer import calculate_mtf_confluence

        tf_tuple = tuple(timeframes) if timeframes else ("1h", "4h", "1d")
        mtf_confluence = calculate_mtf_confluence(data, symbol, tf_tuple)
        summary["mtf_confluence"] = mtf_confluence

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

    def _compute_rule_frequency(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        timeframes: List[str],
        lookback_bars: int = 24,
    ) -> Dict[str, Any]:
        """
        Compute rule frequency from historical signal detection.

        Phase 3: This enables the "Trending" mode in heatmap visualization.

        Args:
            data: Dict mapping (symbol, timeframe) to DataFrame
            timeframes: List of timeframes to include
            lookback_bars: Number of bars to look back for frequency (default: 24)

        Returns:
            Dict with rule frequency data:
                - by_symbol: {symbol: total_count}
                - by_rule: {rule_name: total_count}
                - top_symbols: [(symbol, count), ...]
                - top_rules: [(rule, count), ...]
                - total_signals: int
                - lookback_bars: int
                - computed_at: ISO timestamp
        """
        from src.domain.signals.rules import ALL_RULES

        from ..signal_report.signal_detection import (
            aggregate_rule_frequency,
            detect_signals_with_frequency,
        )

        # Detect signals for each symbol/timeframe combination
        all_signals: Dict[str, List[Dict[str, Any]]] = {}

        for (symbol, timeframe), df in data.items():
            if timeframe not in timeframes:
                continue

            try:
                signals, _ = detect_signals_with_frequency(
                    df=df,
                    rules=ALL_RULES,
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_bars=lookback_bars,
                )
                key = f"{symbol}_{timeframe}"
                all_signals[key] = signals
            except Exception as e:
                logger.debug(f"Could not detect signals for {symbol}_{timeframe}: {e}")

        # Aggregate frequency across all symbols/timeframes
        frequency_data = aggregate_rule_frequency(all_signals)

        # Add metadata
        frequency_data["lookback_bars"] = lookback_bars
        frequency_data["computed_at"] = datetime.now().isoformat()

        return frequency_data
