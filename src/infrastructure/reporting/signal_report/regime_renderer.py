"""
Regime Renderer - Regime analysis sections for signal reports.

Renders methodology, decision tree, component analysis, quality, hysteresis,
turning point, optimization, and recommendations sections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from src.domain.services.regime import ParamProvenanceSet, RecommenderResult
    from src.domain.signals.indicators.base import Indicator
    from src.domain.signals.indicators.regime import RegimeOutput

logger = get_logger(__name__)


def compute_regime_outputs(
    data: Dict[Tuple[str, str], pd.DataFrame],
    indicators: List["Indicator"],
) -> Dict[str, "RegimeOutput"]:
    """
    Compute regime outputs for each (symbol, timeframe) pair.

    Args:
        data: Dict mapping (symbol, timeframe) to DataFrame
        indicators: List of indicators (should include regime_detector)

    Returns:
        Dict mapping "{symbol}_{timeframe}" to RegimeOutput.
        Also includes "{symbol}" key for 1d timeframe (backward compatibility).
    """
    from src.domain.signals.indicators.regime import RegimeDetectorIndicator, RegimeOutput

    regime_outputs: Dict[str, RegimeOutput] = {}

    # Find regime detector indicator
    regime_detector = None
    for ind in indicators:
        if isinstance(ind, RegimeDetectorIndicator):
            regime_detector = ind
            break

    if not regime_detector:
        logger.debug("No regime detector indicator found, skipping regime sections")
        return regime_outputs

    # Compute regime for each (symbol, timeframe) pair
    for (symbol, timeframe), df in data.items():
        if len(df) < regime_detector.minimum_bars:
            logger.debug(
                f"Skipping regime for {symbol}/{timeframe}: {len(df)} < {regime_detector.minimum_bars} bars"
            )
            continue

        try:
            # Use the indicator's calculate method to get all component values
            result_df = regime_detector.calculate(df, regime_detector.default_params)

            if result_df.empty:
                logger.debug(f"Skipping regime for {symbol}: empty result")
                continue

            # Get the last row which has all computed values
            last_row = result_df.iloc[-1]
            ohlc_row = df.iloc[-1]
            timestamp = df.index[-1] if hasattr(df.index[-1], "isoformat") else None

            # Build the flat state dict with all required values from result_df
            flat_state = {
                # OHLC
                "close": float(ohlc_row.get("close", 0)),
                "high": float(ohlc_row.get("high", 0)),
                "low": float(ohlc_row.get("low", 0)),
                "volume": float(ohlc_row.get("volume", 0)),
                # Component states (as strings from result_df)
                "trend_state": str(last_row.get("trend_state", "neutral")),
                "vol_state": str(last_row.get("vol_state", "vol_normal")),
                "chop_state": str(last_row.get("chop_state", "neutral")),
                "ext_state": str(last_row.get("ext_state", "neutral")),
                "iv_state": "na",  # IV handled at service level
                # Component values
                "ma20": float(last_row.get("ma20", 0)) if pd.notna(last_row.get("ma20")) else 0,
                "ma50": float(last_row.get("ma50", 0)) if pd.notna(last_row.get("ma50")) else 0,
                "ma200": (
                    float(last_row.get("ma200", 0)) if pd.notna(last_row.get("ma200")) else 0
                ),
                "ma50_slope": (
                    float(last_row.get("ma50_slope", 0))
                    if pd.notna(last_row.get("ma50_slope"))
                    else 0
                ),
                "atr20": (
                    float(last_row.get("atr20", 0)) if pd.notna(last_row.get("atr20")) else 0
                ),
                "atr_pct": (
                    float(last_row.get("atr_pct", 0)) if pd.notna(last_row.get("atr_pct")) else 0
                ),
                "atr_pct_63": (
                    float(last_row.get("atr_pct_63", 50))
                    if pd.notna(last_row.get("atr_pct_63"))
                    else 50
                ),
                "atr_pct_252": (
                    float(last_row.get("atr_pct_252", 50))
                    if pd.notna(last_row.get("atr_pct_252"))
                    else 50
                ),
                "chop": (float(last_row.get("chop", 50)) if pd.notna(last_row.get("chop")) else 50),
                "chop_pct_252": (
                    float(last_row.get("chop_pct_252", 50))
                    if pd.notna(last_row.get("chop_pct_252"))
                    else 50
                ),
                "ma20_crosses": (
                    int(last_row.get("ma20_crosses", 0))
                    if pd.notna(last_row.get("ma20_crosses"))
                    else 0
                ),
                "ext": float(last_row.get("ext", 0)) if pd.notna(last_row.get("ext")) else 0,
                "last_5_bar_high": (
                    float(last_row.get("last_5_bar_high", 0))
                    if pd.notna(last_row.get("last_5_bar_high"))
                    else float(ohlc_row.get("high", 0))
                ),
                "is_market_level": symbol.upper() in {"QQQ", "SPY", "IWM", "DIA"},
                # Phase 5: Composite scoring columns
                "composite_score": (
                    float(last_row.get("composite_score", 50))
                    if pd.notna(last_row.get("composite_score"))
                    else None
                ),
                "composite_trend": (
                    float(last_row.get("composite_trend", 0.5))
                    if pd.notna(last_row.get("composite_trend"))
                    else None
                ),
                "composite_trend_short": (
                    float(last_row.get("composite_trend_short", 0.5))
                    if pd.notna(last_row.get("composite_trend_short"))
                    else None
                ),
                "composite_momentum": (
                    float(last_row.get("composite_momentum", 0.5))
                    if pd.notna(last_row.get("composite_momentum"))
                    else None
                ),
                "composite_volatility": (
                    float(last_row.get("composite_volatility", 0.5))
                    if pd.notna(last_row.get("composite_volatility"))
                    else None
                ),
                "composite_macd_trend": (
                    float(last_row.get("composite_macd_trend", 0.5))
                    if pd.notna(last_row.get("composite_macd_trend"))
                    else None
                ),
                "composite_macd_momentum": (
                    float(last_row.get("composite_macd_momentum", 0.5))
                    if pd.notna(last_row.get("composite_macd_momentum"))
                    else None
                ),
            }

            # Compute full regime output with hysteresis
            output = regime_detector.update_with_hysteresis(
                symbol=symbol,
                state=flat_state,
                timestamp=timestamp,
                timeframe=timeframe,
            )
            # Store with timeframe-specific key
            key = f"{symbol}_{timeframe}"
            regime_outputs[key] = output
            # Also store under symbol-only key for 1d (backward compatibility)
            if timeframe == "1d":
                regime_outputs[symbol] = output
            logger.debug(f"Computed regime for {key}: {output.final_regime.value}")
        except Exception as e:
            logger.warning(f"Failed to compute regime for {symbol}/{timeframe}: {e}")

    return regime_outputs


def compute_param_analysis(
    data: Dict[Tuple[str, str], pd.DataFrame],
    indicators: List["Indicator"],
) -> Tuple[Dict[str, "ParamProvenanceSet"], Dict[str, "RecommenderResult"]]:
    """
    Compute parameter provenance and recommendations for each symbol.

    Uses the ParamRecommender to analyze threshold calibration and
    creates ParamProvenance to track parameter sources.

    Args:
        data: Dict mapping (symbol, timeframe) to DataFrame
        indicators: List of indicators (to get regime detector params)

    Returns:
        Tuple of (provenance_dict, recommendations_dict) mapping symbol to results
    """
    from src.domain.services.regime import (
        ParamProvenance,
        ParamProvenanceSet,
        ParamRecommender,
        ParamSource,
        RecommenderResult,
        get_regime_params,
    )
    from src.domain.signals.indicators.regime import RegimeDetectorIndicator

    provenance_dict: Dict[str, ParamProvenanceSet] = {}
    recommendations_dict: Dict[str, RecommenderResult] = {}

    # Find regime detector to get default params
    regime_detector = None
    for ind in indicators:
        if isinstance(ind, RegimeDetectorIndicator):
            regime_detector = ind
            break

    if not regime_detector:
        return provenance_dict, recommendations_dict

    # Create recommender instance
    recommender = ParamRecommender(lookback_days=63)

    # Process each symbol
    symbols_processed = set()
    for (symbol, timeframe), df in data.items():
        if symbol in symbols_processed:
            continue
        if len(df) < 63:  # Need minimum data for analysis
            continue

        try:
            # Get params for this symbol (may be symbol-specific or default)
            params = get_regime_params(symbol)

            # Create provenance from params
            provenance = ParamProvenance.from_params(
                params=params,
                symbol=symbol,
                source="default",
            )

            # Create param sources for each parameter (for detailed display)
            param_sources = {}
            for param_name, value in params.items():
                param_sources[param_name] = ParamSource(
                    param_name=param_name,
                    value=value,
                    source="default",
                    trained_on=None,
                )

            # Create full provenance set with param details
            provenance_set = ParamProvenanceSet(
                symbol=symbol,
                provenance=provenance,
                param_sources=param_sources,
            )
            provenance_dict[symbol] = provenance_set

            # Run recommender analysis
            result = recommender.analyze(
                symbol=symbol,
                ohlcv=df,
                current_params=params,
            )
            recommendations_dict[symbol] = result

            logger.debug(
                f"Param analysis for {symbol}: " f"has_recommendations={result.has_recommendations}"
            )
            symbols_processed.add(symbol)
        except Exception as e:
            logger.warning(f"Failed to compute param analysis for {symbol}: {e}")

    return provenance_dict, recommendations_dict


def render_regime_sections(
    regime_outputs: Optional[Dict[str, "RegimeOutput"]],
    provenance_dict: Optional[Dict[str, "ParamProvenanceSet"]] = None,
    recommendations_dict: Optional[Dict[str, "RecommenderResult"]] = None,
    theme: str = "dark",
) -> str:
    """
    Render regime analysis sections for all symbols with regime data.

    Returns HTML for methodology, decision tree, component analysis,
    quality, hysteresis, optimization, and recommendations sections
    (PR1 + PR2 + PR3).

    Each symbol's section has data-symbol attribute for JavaScript filtering.
    Only the selected symbol's section is shown (controlled by updateRegimeSection).

    Args:
        regime_outputs: Dict mapping symbol to RegimeOutput
        provenance_dict: Optional dict mapping symbol to ParamProvenanceSet
        recommendations_dict: Optional dict mapping symbol to RecommenderResult
        theme: Color theme ("dark" or "light")

    Returns:
        HTML string for regime analysis sections
    """
    from ..regime_report import (
        generate_components_4block_html,
        generate_composite_score_html,
        generate_decision_tree_html,
        generate_hysteresis_html,
        generate_methodology_html,
        generate_optimization_html,
        generate_quality_html,
        generate_recommendations_html,
        generate_regime_one_liner_html,
        generate_report_header_html,
        generate_turning_point_html,
    )

    if not regime_outputs:
        return ""

    provenance_dict = provenance_dict or {}
    recommendations_dict = recommendations_dict or {}

    # Build regime sections for each symbol
    sections_html = []

    for symbol, output in sorted(regime_outputs.items()):
        # Generate one-liner for this symbol
        one_liner = generate_regime_one_liner_html(output)

        # Phase 5: Generate composite score section (dedicated section)
        composite_score = generate_composite_score_html(output, theme)

        # Generate decision tree
        decision_tree = generate_decision_tree_html(output, theme)

        # Generate component analysis
        components = generate_components_4block_html(output, theme)

        # PR2: Generate quality and hysteresis sections
        quality = generate_quality_html(output, theme)
        hysteresis = generate_hysteresis_html(output, theme)

        # Phase 4: Generate turning point detection section
        turning_point = generate_turning_point_html(output, theme)

        # PR3: Generate optimization and recommendations sections with real data
        provenance_set = provenance_dict.get(symbol)
        recommendations_result = recommendations_dict.get(symbol)
        optimization = generate_optimization_html(
            provenance=None,
            provenance_set=provenance_set,
            theme=theme,
        )
        recommendations = generate_recommendations_html(
            result=recommendations_result,
            theme=theme,
        )

        # PR4: Generate report header with metadata
        report_header = generate_report_header_html(
            regime_output=output,
            provenance_set=provenance_set,
            recommendations_result=recommendations_result,
            theme=theme,
        )

        # Add data-symbol attribute for JavaScript filtering
        sections_html.append(f"""
        <div class="regime-symbol-section" id="regime-{symbol}" data-symbol="{symbol}" style="display: none;">
            {report_header}
            {one_liner}
            {composite_score}
            {decision_tree}
            {components}
            {hysteresis}
            {turning_point}
            {quality}
            {optimization}
            {recommendations}
        </div>
        """)

    # Wrap with methodology at the top
    methodology = generate_methodology_html(theme)

    return f"""
    <div class="regime-analysis-section">
        <h2 class="section-header" onclick="toggleSection('regime-content')">
            <span class="toggle-icon">▼</span> Regime Analysis
        </h2>
        <div id="regime-content" class="section-content">
            {methodology}
            <div class="regime-symbols-container" id="regime-symbols-container">
                {''.join(sections_html)}
            </div>
        </div>
    </div>
    """
