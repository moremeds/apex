"""
Signal Report Generator - Interactive HTML reports for signal analysis.

Generates self-contained HTML reports with:
- Symbol selector dropdown
- Timeframe toggle buttons
- Candlestick price charts with overlay indicators
- Separate subplots for oscillators (RSI, MACD, etc.)
- Auto-generated descriptions for indicators and rules
- Signal history showing when rules would have triggered
- Collapsible indicator sections for better readability
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logging_setup import get_logger

from ..divergence.cross_divergence import CrossIndicatorAnalyzer
from ..models import ConfluenceScore
from .description_generator import generate_indicator_description, generate_rule_description
from .regime_report import (
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

if TYPE_CHECKING:
    from src.domain.services.regime import ParamProvenanceSet, RecommenderResult
    from src.domain.services.regime.param_provenance import ParamProvenance

    from ..indicators.base import Indicator
    from ..indicators.regime import RegimeOutput
    from ..models import SignalRule

logger = get_logger(__name__)


# Timeframe ordering for consistent display
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}

# Indicator grouping for chart layout
# Overlays: Same Y-axis as price
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
# Bounded oscillators (0-100 or similar fixed range)
BOUNDED_OSCILLATORS = {"rsi", "stochastic", "kdj", "williams_r", "mfi", "cci", "adx"}
# Unbounded oscillators (MACD-style, centered around 0)
UNBOUNDED_OSCILLATORS = {"macd", "momentum", "roc", "cmf", "pvo", "force_index"}
# Volume indicators
VOLUME_INDICATORS = {"obv", "volume_profile", "vwma", "ease_of_movement", "chaikin_volatility"}


def derive_indicator_states(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Derive semantic indicator states from DataFrame's last row values.

    Translates raw indicator values into the state format expected by
    CrossIndicatorAnalyzer (e.g., RSI=35 → {'zone': 'oversold', 'value': 35}).

    Args:
        df: DataFrame with indicator columns

    Returns:
        Dict mapping indicator names to state dicts
    """
    if df.empty:
        return {}

    last_row = df.iloc[-1]
    states: Dict[str, Dict[str, Any]] = {}

    # RSI state derivation
    rsi_col = next((c for c in df.columns if c.lower().startswith("rsi_rsi")), None)
    if rsi_col and pd.notna(last_row[rsi_col]):
        rsi_val = float(last_row[rsi_col])
        zone = "oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else "neutral"
        states["rsi"] = {"value": rsi_val, "zone": zone}

    # MACD state derivation
    macd_col = next((c for c in df.columns if "macd_macd" in c.lower()), None)
    signal_col = next((c for c in df.columns if "macd_signal" in c.lower()), None)
    hist_col = next((c for c in df.columns if "macd_histogram" in c.lower()), None)
    if hist_col and pd.notna(last_row[hist_col]):
        hist_val = float(last_row[hist_col])
        macd_val = float(last_row[macd_col]) if macd_col and pd.notna(last_row[macd_col]) else 0
        signal_val = (
            float(last_row[signal_col]) if signal_col and pd.notna(last_row[signal_col]) else 0
        )
        # Detect cross from last two rows
        cross = "neutral"
        if len(df) >= 2:
            prev_macd = (
                df[macd_col].iloc[-2] if macd_col and pd.notna(df[macd_col].iloc[-2]) else None
            )
            prev_signal = (
                df[signal_col].iloc[-2]
                if signal_col and pd.notna(df[signal_col].iloc[-2])
                else None
            )
            if prev_macd is not None and prev_signal is not None:
                if prev_macd <= prev_signal and macd_val > signal_val:
                    cross = "bullish"
                elif prev_macd >= prev_signal and macd_val < signal_val:
                    cross = "bearish"
        states["macd"] = {
            "histogram": hist_val,
            "macd": macd_val,
            "signal": signal_val,
            "cross": cross,
        }

    # SuperTrend state derivation
    st_col = next((c for c in df.columns if "supertrend_direction" in c.lower()), None)
    if st_col and pd.notna(last_row[st_col]):
        direction = str(last_row[st_col]).lower()
        states["supertrend"] = {"direction": direction}
    else:
        # Infer from SuperTrend vs price
        st_val_col = next((c for c in df.columns if "supertrend_supertrend" in c.lower()), None)
        close_col = "close" if "close" in df.columns else None
        if (
            st_val_col
            and close_col
            and pd.notna(last_row[st_val_col])
            and pd.notna(last_row[close_col])
        ):
            st_val = float(last_row[st_val_col])
            close_val = float(last_row[close_col])
            direction = "bullish" if close_val > st_val else "bearish"
            states["supertrend"] = {"direction": direction, "value": st_val}

    # Bollinger Bands state derivation
    bb_upper = next((c for c in df.columns if "bollinger_bb_upper" in c.lower()), None)
    bb_lower = next((c for c in df.columns if "bollinger_bb_lower" in c.lower()), None)
    close_col = "close" if "close" in df.columns else None
    if bb_upper and bb_lower and close_col:
        if (
            pd.notna(last_row[bb_upper])
            and pd.notna(last_row[bb_lower])
            and pd.notna(last_row[close_col])
        ):
            upper = float(last_row[bb_upper])
            lower = float(last_row[bb_lower])
            close = float(last_row[close_col])
            if close <= lower:
                zone = "below_lower"
            elif close >= upper:
                zone = "above_upper"
            else:
                zone = "middle"
            states["bollinger"] = {"zone": zone, "upper": upper, "lower": lower}

    # KDJ state derivation
    k_col = next((c for c in df.columns if c.lower().startswith("kdj_k")), None)
    d_col = next((c for c in df.columns if c.lower().startswith("kdj_d")), None)
    if k_col and d_col and pd.notna(last_row[k_col]) and pd.notna(last_row[d_col]):
        k_val = float(last_row[k_col])
        d_val = float(last_row[d_col])
        zone = "oversold" if k_val < 20 else "overbought" if k_val > 80 else "neutral"
        # Detect cross
        cross = "neutral"
        if len(df) >= 2:
            prev_k = df[k_col].iloc[-2] if pd.notna(df[k_col].iloc[-2]) else None
            prev_d = df[d_col].iloc[-2] if pd.notna(df[d_col].iloc[-2]) else None
            if prev_k is not None and prev_d is not None:
                if prev_k <= prev_d and k_val > d_val:
                    cross = "bullish"
                elif prev_k >= prev_d and k_val < d_val:
                    cross = "bearish"
        states["kdj"] = {"k": k_val, "d": d_val, "zone": zone, "cross": cross}

    # ADX state derivation
    adx_col = next((c for c in df.columns if c.lower().startswith("adx_adx")), None)
    di_plus_col = next(
        (c for c in df.columns if "di_plus" in c.lower() or "di_p" in c.lower()), None
    )
    di_minus_col = next(
        (c for c in df.columns if "di_minus" in c.lower() or "di_m" in c.lower()), None
    )
    if di_plus_col and di_minus_col:
        if pd.notna(last_row[di_plus_col]) and pd.notna(last_row[di_minus_col]):
            di_plus = float(last_row[di_plus_col])
            di_minus = float(last_row[di_minus_col])
            adx_val = float(last_row[adx_col]) if adx_col and pd.notna(last_row[adx_col]) else 0
            states["adx"] = {"adx": adx_val, "di_plus": di_plus, "di_minus": di_minus}

    return states


def calculate_confluence(data: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[str, ConfluenceScore]:
    """
    Calculate confluence scores for all symbol/timeframe combinations.

    Args:
        data: Dict mapping (symbol, timeframe) to DataFrame

    Returns:
        Dict mapping "symbol_timeframe" to ConfluenceScore
    """
    analyzer = CrossIndicatorAnalyzer()
    confluence_scores: Dict[str, ConfluenceScore] = {}

    for (symbol, timeframe), df in data.items():
        key = f"{symbol}_{timeframe}"
        indicator_states = derive_indicator_states(df)

        if indicator_states:
            score = analyzer.analyze(symbol, timeframe, indicator_states)
            confluence_scores[key] = score
            logger.debug(
                f"Confluence calculated for {key}",
                extra={
                    "alignment_score": score.alignment_score,
                    "bullish": score.bullish_count,
                    "bearish": score.bearish_count,
                    "indicators_analyzed": list(indicator_states.keys()),
                },
            )

    return confluence_scores


def detect_historical_signals(
    df: pd.DataFrame,
    rules: List["SignalRule"],
    symbol: str,
    timeframe: str,
) -> List[Dict[str, Any]]:
    """
    Detect where signal rules would have triggered in historical data.

    Scans the DataFrame for cross events, threshold breaches, and state changes
    that match the rule conditions.

    Args:
        df: DataFrame with OHLCV and indicator columns
        rules: List of signal rules to check
        symbol: Trading symbol
        timeframe: Timeframe string

    Returns:
        List of detected signal events with timestamps and details
    """
    from ..models import ConditionType

    signals = []
    timestamps = df.index.tolist()

    for rule in rules:
        if not rule.enabled or timeframe not in rule.timeframes:
            continue

        indicator = rule.indicator.lower()
        cond = rule.condition_config

        # Match columns by indicator prefix
        prefix = f"{indicator}_"
        ind_cols = [c for c in df.columns if c.lower().startswith(prefix)]

        if rule.condition_type == ConditionType.CROSS_UP:
            line_a = cond.get("line_a", "")
            line_b = cond.get("line_b", "")
            col_a = next((c for c in ind_cols if line_a in c.lower()), None)
            col_b = next((c for c in ind_cols if line_b in c.lower()), None)

            if col_a and col_b:
                a_vals = df[col_a].values
                b_vals = df[col_b].values
                for i in range(1, len(df)):
                    if (
                        pd.notna(a_vals[i])
                        and pd.notna(b_vals[i])
                        and pd.notna(a_vals[i - 1])
                        and pd.notna(b_vals[i - 1])
                    ):
                        if a_vals[i - 1] <= b_vals[i - 1] and a_vals[i] > b_vals[i]:
                            signals.append(
                                {
                                    "timestamp": timestamps[i],
                                    "rule": rule.name,
                                    "direction": rule.direction.value,
                                    "indicator": rule.indicator,
                                    "message": rule.message_template.format(symbol=symbol),
                                    "value": float(a_vals[i]),
                                }
                            )

        elif rule.condition_type == ConditionType.CROSS_DOWN:
            line_a = cond.get("line_a", "")
            line_b = cond.get("line_b", "")
            col_a = next((c for c in ind_cols if line_a in c.lower()), None)
            col_b = next((c for c in ind_cols if line_b in c.lower()), None)

            if col_a and col_b:
                a_vals = df[col_a].values
                b_vals = df[col_b].values
                for i in range(1, len(df)):
                    if (
                        pd.notna(a_vals[i])
                        and pd.notna(b_vals[i])
                        and pd.notna(a_vals[i - 1])
                        and pd.notna(b_vals[i - 1])
                    ):
                        if a_vals[i - 1] >= b_vals[i - 1] and a_vals[i] < b_vals[i]:
                            signals.append(
                                {
                                    "timestamp": timestamps[i],
                                    "rule": rule.name,
                                    "direction": rule.direction.value,
                                    "indicator": rule.indicator,
                                    "message": rule.message_template.format(symbol=symbol),
                                    "value": float(a_vals[i]),
                                }
                            )

        elif rule.condition_type == ConditionType.THRESHOLD_CROSS_UP:
            field = cond.get("field", "value")
            threshold = cond.get("threshold")
            col = next((c for c in ind_cols if field in c.lower()), None)

            if col and threshold is not None:
                vals = df[col].values
                for i in range(1, len(df)):
                    if pd.notna(vals[i]) and pd.notna(vals[i - 1]):
                        if vals[i - 1] <= threshold < vals[i]:
                            signals.append(
                                {
                                    "timestamp": timestamps[i],
                                    "rule": rule.name,
                                    "direction": rule.direction.value,
                                    "indicator": rule.indicator,
                                    "message": rule.message_template.format(
                                        symbol=symbol, value=vals[i], threshold=threshold
                                    ),
                                    "value": float(vals[i]),
                                    "threshold": threshold,
                                }
                            )

        elif rule.condition_type == ConditionType.THRESHOLD_CROSS_DOWN:
            field = cond.get("field", "value")
            threshold = cond.get("threshold")
            col = next((c for c in ind_cols if field in c.lower()), None)

            if col and threshold is not None:
                vals = df[col].values
                for i in range(1, len(df)):
                    if pd.notna(vals[i]) and pd.notna(vals[i - 1]):
                        if vals[i - 1] >= threshold > vals[i]:
                            signals.append(
                                {
                                    "timestamp": timestamps[i],
                                    "rule": rule.name,
                                    "direction": rule.direction.value,
                                    "indicator": rule.indicator,
                                    "message": rule.message_template.format(
                                        symbol=symbol, value=vals[i], threshold=threshold
                                    ),
                                    "value": float(vals[i]),
                                    "threshold": threshold,
                                }
                            )

        # MACD signal/zero line cross detection
        if indicator == "macd":
            macd_col = next((c for c in df.columns if "macd_macd" in c.lower()), None)
            signal_col = next((c for c in df.columns if "macd_signal" in c.lower()), None)

            if macd_col and signal_col and "cross" not in rule.name.lower():
                macd_vals = df[macd_col].values
                signal_vals = df[signal_col].values
                for i in range(1, len(df)):
                    if all(
                        pd.notna(v)
                        for v in [
                            macd_vals[i],
                            macd_vals[i - 1],
                            signal_vals[i],
                            signal_vals[i - 1],
                        ]
                    ):
                        # Bullish cross
                        if macd_vals[i - 1] <= signal_vals[i - 1] and macd_vals[i] > signal_vals[i]:
                            if not any(
                                s["timestamp"] == timestamps[i] and "macd" in s["rule"].lower()
                                for s in signals
                            ):
                                signals.append(
                                    {
                                        "timestamp": timestamps[i],
                                        "rule": "macd_bullish_cross",
                                        "direction": "buy",
                                        "indicator": "macd",
                                        "message": f"{symbol} MACD crossed above signal line",
                                        "value": float(macd_vals[i]),
                                    }
                                )
                        # Bearish cross
                        elif (
                            macd_vals[i - 1] >= signal_vals[i - 1] and macd_vals[i] < signal_vals[i]
                        ):
                            if not any(
                                s["timestamp"] == timestamps[i] and "macd" in s["rule"].lower()
                                for s in signals
                            ):
                                signals.append(
                                    {
                                        "timestamp": timestamps[i],
                                        "rule": "macd_bearish_cross",
                                        "direction": "sell",
                                        "indicator": "macd",
                                        "message": f"{symbol} MACD crossed below signal line",
                                        "value": float(macd_vals[i]),
                                    }
                                )

    # Sort by timestamp
    signals.sort(key=lambda x: x["timestamp"])
    return signals


class SignalReportGenerator:
    """
    Generate interactive HTML reports for signal analysis.

    Uses Plotly for charts with proper subplots:
    - Price chart with overlay indicators (Bollinger, SuperTrend, etc.)
    - RSI subplot (0-100 scale)
    - MACD subplot (unbounded scale)
    - Volume subplot
    """

    def __init__(self, theme: str = "dark") -> None:
        self.theme = theme
        self._colors = self._get_theme_colors(theme)

    def generate(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        indicators: List["Indicator"],
        rules: List["SignalRule"],
        output_path: Path,
        regime_outputs: Optional[Dict[str, "RegimeOutput"]] = None,
    ) -> Path:
        """
        Generate combined HTML report with symbol selector.

        Args:
            data: Dict mapping (symbol, timeframe) to DataFrame with OHLCV + indicator columns
            indicators: List of computed indicators
            rules: List of signal rules
            output_path: Where to save HTML
            regime_outputs: Optional dict mapping symbol to RegimeOutput for regime sections

        Returns:
            Path to generated HTML file
        """
        symbols = sorted(set(sym for sym, tf in data.keys()))
        timeframes = sorted(
            set(tf for sym, tf in data.keys()),
            key=lambda x: TIMEFRAME_SECONDS.get(x, 0),
        )

        # Build chart data for JavaScript
        chart_data = self._build_chart_data(data)

        # Calculate confluence scores for each symbol/timeframe
        confluence_scores = calculate_confluence(data)
        confluence_data = {
            key: {
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
            for key, score in confluence_scores.items()
        }

        # Detect historical signals for each symbol/timeframe
        signal_history: Dict[str, List[Dict[str, Any]]] = {}
        for (symbol, timeframe), df in data.items():
            key = f"{symbol}_{timeframe}"
            signals = detect_historical_signals(df, rules, symbol, timeframe)
            signal_history[key] = signals
            if signals:
                logger.info(f"Detected {len(signals)} signals for {key}")

        # Build indicator and rule descriptions
        indicator_info = self._build_indicator_info(indicators, rules)

        # Compute regime outputs if not provided
        if regime_outputs is None:
            regime_outputs = self._compute_regime_outputs(data, indicators)

        # Compute parameter provenance and recommendations
        provenance_dict, recommendations_dict = self._compute_param_analysis(data, indicators)

        # Render HTML
        html = self._render_html(
            symbols=symbols,
            timeframes=timeframes,
            chart_data=chart_data,
            indicator_info=indicator_info,
            signal_history=signal_history,
            confluence_data=confluence_data,
            regime_outputs=regime_outputs,
            provenance_dict=provenance_dict,
            recommendations_dict=recommendations_dict,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        logger.info(f"Signal report generated: {output_path}")
        return output_path

    def _compute_regime_outputs(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        indicators: List["Indicator"],
    ) -> Dict[str, "RegimeOutput"]:
        """
        Compute regime outputs for each symbol using the regime detector indicator.

        Args:
            data: Dict mapping (symbol, timeframe) to DataFrame
            indicators: List of indicators (should include regime_detector)

        Returns:
            Dict mapping symbol to RegimeOutput
        """
        from ..indicators.regime import RegimeDetectorIndicator, RegimeOutput

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

        # Compute regime for each symbol (use daily timeframe preferentially)
        symbols_processed = set()
        for (symbol, timeframe), df in data.items():
            if symbol in symbols_processed:
                continue
            if len(df) < regime_detector.warmup_periods:
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
                        float(last_row.get("atr_pct", 0))
                        if pd.notna(last_row.get("atr_pct"))
                        else 0
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
                    "chop": (
                        float(last_row.get("chop", 50)) if pd.notna(last_row.get("chop")) else 50
                    ),
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
                }

                # Compute full regime output with hysteresis
                output = regime_detector.update_with_hysteresis(
                    symbol=symbol,
                    state=flat_state,
                    timestamp=timestamp,
                )
                regime_outputs[symbol] = output
                symbols_processed.add(symbol)
                logger.debug(f"Computed regime for {symbol}: {output.final_regime.value}")
            except Exception as e:
                logger.warning(f"Failed to compute regime for {symbol}: {e}")

        return regime_outputs

    def _compute_param_analysis(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        indicators: List["Indicator"],
    ) -> Tuple[Dict[str, "ParamProvenance"], Dict[str, "RecommenderResult"]]:
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

        from ..indicators.regime import RegimeDetectorIndicator

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
                    source="default",  # Currently all use defaults
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
                    f"Param analysis for {symbol}: "
                    f"has_recommendations={result.has_recommendations}"
                )
                symbols_processed.add(symbol)
            except Exception as e:
                logger.warning(f"Failed to compute param analysis for {symbol}: {e}")

        return provenance_dict, recommendations_dict

    def _get_theme_colors(self, theme: str) -> Dict[str, str]:
        if theme == "dark":
            return {
                "bg": "#0f172a",
                "card_bg": "#1e293b",
                "text": "#e2e8f0",
                "text_muted": "#94a3b8",
                "border": "#334155",
                "profit": "#22c55e",
                "loss": "#ef4444",
                "primary": "#3b82f6",
                "candle_up": "#22c55e",
                "candle_down": "#ef4444",
            }
        return {
            "bg": "#f8fafc",
            "card_bg": "#ffffff",
            "text": "#1e293b",
            "text_muted": "#64748b",
            "border": "#e2e8f0",
            "profit": "#16a34a",
            "loss": "#dc2626",
            "primary": "#2563eb",
            "candle_up": "#16a34a",
            "candle_down": "#dc2626",
        }

    def _build_chart_data(self, data: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[str, Any]:
        """Build chart data structure for JavaScript with indicator grouping."""
        chart_data = {}
        oscillator_names = BOUNDED_OSCILLATORS | UNBOUNDED_OSCILLATORS

        for (symbol, timeframe), df in data.items():
            key = f"{symbol}_{timeframe}"

            # Convert timestamps to ISO strings
            if hasattr(df.index, "strftime"):
                timestamps = df.index.strftime("%Y-%m-%dT%H:%M:%S").tolist()
            else:
                timestamps = [str(t) for t in df.index]

            # Base OHLCV data
            ohlcv = {
                name: df[name].tolist() if name in df.columns else []
                for name in ("open", "high", "low", "close", "volume")
            }
            chart_data[key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "bar_count": len(df),
                "timestamps": timestamps,
                **ohlcv,
                "overlays": {},
                "rsi": {},
                "macd": {},
                "oscillators": {},
                "volume_ind": {},
            }

            # Categorize indicator columns
            ohlcv_cols = {"open", "high", "low", "close", "volume", "timestamp"}
            for col in df.columns:
                if col.lower() in ohlcv_cols:
                    continue

                values = df[col].tolist()
                values = [None if pd.isna(v) else v for v in values]

                # Parse indicator name from prefixed column (e.g., "macd_histogram" → "macd")
                parts = col.split("_")
                ind_name = parts[0].lower() if parts else col.lower()

                # Route to appropriate subplot bucket
                if ind_name in OVERLAY_INDICATORS:
                    bucket = "overlays"
                elif ind_name == "rsi":
                    bucket = "rsi"
                elif ind_name == "macd":
                    bucket = "macd"
                elif ind_name in oscillator_names:
                    bucket = "oscillators"
                elif ind_name in VOLUME_INDICATORS:
                    bucket = "volume_ind"
                else:
                    bucket = "oscillators"
                chart_data[key][bucket][col] = values

        return chart_data

    def _build_indicator_info(
        self,
        indicators: List["Indicator"],
        rules: List["SignalRule"],
    ) -> List[Dict[str, Any]]:
        """Build indicator information with descriptions and linked rules."""
        rules_by_indicator: Dict[str, List[Dict[str, str]]] = {}
        for rule in rules:
            rules_by_indicator.setdefault(rule.indicator, []).append(
                {
                    "name": rule.name,
                    "description": generate_rule_description(rule),
                    "direction": rule.direction.value,
                    "timeframes": list(rule.timeframes),
                }
            )

        info_list = [
            {
                "name": ind.name,
                "category": ind.category.value,
                "description": generate_indicator_description(ind),
                "warmup_periods": ind.warmup_periods,
                "rules": rules_by_indicator.get(ind.name, []),
            }
            for ind in indicators
        ]
        info_list.sort(key=lambda x: (x["category"], x["name"]))
        return info_list

    def _render_html(
        self,
        symbols: List[str],
        timeframes: List[str],
        chart_data: Dict[str, Any],
        indicator_info: List[Dict[str, Any]],
        signal_history: Dict[str, List[Dict[str, Any]]],
        confluence_data: Dict[str, Dict[str, Any]],
        regime_outputs: Optional[Dict[str, "RegimeOutput"]] = None,
        provenance_dict: Optional[Dict[str, "ParamProvenanceSet"]] = None,
        recommendations_dict: Optional[Dict[str, "RecommenderResult"]] = None,
    ) -> str:
        self._colors
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        regime_outputs = regime_outputs or {}
        provenance_dict = provenance_dict or {}
        recommendations_dict = recommendations_dict or {}

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
{self._get_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Signal Analysis Report</h1>
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
                    {self._render_symbol_options(symbols)}
                </select>
            </div>
            <div class="control-group">
                <label>Timeframe</label>
                <div class="timeframe-buttons">
                    {self._render_timeframe_buttons(timeframes)}
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div id="main-chart"></div>
        </div>

        <div class="confluence-section">
            <h2 class="section-header" onclick="toggleSection('confluence-content')">
                <span class="toggle-icon">▼</span> Confluence Analysis
            </h2>
            <div id="confluence-content" class="section-content">
                <div id="confluence-panel"></div>
            </div>
        </div>

        {self._render_regime_sections(regime_outputs, provenance_dict, recommendations_dict)}

        <div class="signal-history-section">
            <h2 class="section-header" onclick="toggleSection('signal-history-content')">
                <span class="toggle-icon">▼</span> Signal History
            </h2>
            <div id="signal-history-content" class="section-content">
                <div id="signal-history-table"></div>
            </div>
        </div>

        <div class="indicators-section">
            <h2 class="section-header" onclick="toggleSection('indicators-content')">
                <span class="toggle-icon">▼</span> Indicators
            </h2>
            <div id="indicators-content" class="section-content collapsed">
                {self._render_indicator_cards(indicator_info)}
            </div>
        </div>
    </div>

    <script>
{self._get_scripts(chart_data, symbols, timeframes, signal_history, confluence_data)}
    </script>
</body>
</html>"""

    def _get_styles(self) -> str:
        c = self._colors
        return (
            f"""
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

.header h1 {{
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 8px;
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
    height: 900px;
}}

.confluence-section,
.signal-history-section,
.indicators-section {{
    background: {c['card_bg']};
    border-radius: 12px;
    border: 1px solid {c['border']};
    padding: 24px;
    margin-bottom: 24px;
}}

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
    background: linear-gradient(to right, {c['loss']} 0%, {c['text_muted']} 50%, {c['profit']} 100%);
    border-radius: 12px;
    position: relative;
    overflow: hidden;
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

.alignment-value.bullish {{ color: {c['profit']}; }}
.alignment-value.bearish {{ color: {c['loss']}; }}
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

.count-value.bullish {{ color: {c['profit']}; }}
.count-value.bearish {{ color: {c['loss']}; }}
.count-value.neutral {{ color: {c['text_muted']}; }}

.count-label {{
    font-size: 12px;
    color: {c['text_muted']};
    text-transform: uppercase;
}}

.divergence-list {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}

.divergence-item {{
    padding: 12px;
    background: {c['bg']};
    border-radius: 8px;
    font-size: 13px;
}}

.divergence-item .indicators {{
    font-weight: 600;
    margin-bottom: 4px;
}}

.divergence-item .reason {{
    color: {c['text_muted']};
    font-size: 12px;
}}

.no-divergences {{
    text-align: center;
    color: {c['text_muted']};
    padding: 24px;
    font-style: italic;
}}

.strongest-signal {{
    text-align: center;
    padding: 12px;
    background: {c['bg']};
    border-radius: 8px;
    margin-top: 8px;
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

.strongest-signal .value.bullish {{ color: {c['profit']}; }}
.strongest-signal .value.bearish {{ color: {c['loss']}; }}
.strongest-signal .value.neutral {{ color: {c['text_muted']}; }}

.section-header {{
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid {c['border']};
    cursor: pointer;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section-header:hover {{
    color: {c['primary']};
}}

.toggle-icon {{
    font-size: 12px;
    transition: transform 0.2s ease;
}}

.section-content.collapsed {{
    display: none;
}}

.section-content.collapsed + .section-header .toggle-icon {{
    transform: rotate(-90deg);
}}

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
    color: {c['profit']};
}}

.signal-badge.sell {{
    background: rgba(239, 68, 68, 0.2);
    color: {c['loss']};
}}

.signal-badge.alert {{
    background: rgba(59, 130, 246, 0.2);
    color: {c['primary']};
}}

.no-signals {{
    text-align: center;
    color: {c['text_muted']};
    padding: 24px;
    font-style: italic;
}}

.category-group {{
    margin-bottom: 24px;
}}

.category-title {{
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    color: {c['text_muted']};
    margin-bottom: 12px;
    padding: 8px 12px;
    background: {c['bg']};
    border-radius: 6px;
}}

.indicator-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 16px;
}}

.indicator-card {{
    padding: 16px;
    background: {c['bg']};
    border-radius: 8px;
    border: 1px solid {c['border']};
}}

.indicator-card h3 {{
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
    color: {c['primary']};
}}

.indicator-card .description {{
    font-size: 14px;
    color: {c['text_muted']};
    margin-bottom: 12px;
}}

.indicator-card .rules {{
    font-size: 13px;
}}

.indicator-card .rules h4 {{
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    color: {c['text_muted']};
    margin-bottom: 8px;
}}

.rule-item {{
    padding: 8px;
    background: {c['card_bg']};
    border-radius: 4px;
    margin-bottom: 4px;
}}

.rule-item .rule-name {{
    font-weight: 500;
}}

.rule-item .rule-desc {{
    font-size: 12px;
    color: {c['text_muted']};
}}

.direction-buy {{ color: {c['profit']}; }}
.direction-sell {{ color: {c['loss']}; }}
.direction-alert {{ color: {c['primary']}; }}

@media (max-width: 768px) {{
    .controls {{
        flex-direction: column;
        align-items: stretch;
    }}
    .timeframe-buttons {{
        flex-wrap: wrap;
    }}
    .indicator-cards {{
        grid-template-columns: 1fr;
    }}
}}

/* Regime Analysis Section */
.regime-analysis-section {{
    background: {c['card_bg']};
    border-radius: 12px;
    border: 1px solid {c['border']};
    padding: 24px;
    margin-bottom: 24px;
}}

.regime-symbol-section {{
    margin-bottom: 32px;
    padding-bottom: 24px;
    border-bottom: 1px solid {c['border']};
}}

.regime-symbol-section:last-child {{
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}}

.regime-symbol-header {{
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 16px;
    color: {c['primary']};
}}

.regime-symbols-container {{
    margin-top: 24px;
}}

/* CSS Variables for regime report */
:root {{
    --bg: {c['bg']};
    --card-bg: {c['card_bg']};
    --text: {c['text']};
    --text-muted: {c['text_muted']};
    --border: {c['border']};
    --header-bg: {c['bg']};
    --highlight-bg: {c['bg']};
    --code-bg: {c['bg']};
}}
"""
            + generate_regime_styles()
        )

    def _render_symbol_options(self, symbols: List[str]) -> str:
        return "\n".join(f'<option value="{s}">{s}</option>' for s in symbols)

    def _render_timeframe_buttons(self, timeframes: List[str]) -> str:
        return "\n".join(
            f'<button class="tf-btn{" active" if i == 0 else ""}" data-tf="{tf}" '
            f"onclick=\"selectTimeframe('{tf}', this)\">{tf}</button>"
            for i, tf in enumerate(timeframes)
        )

    def _render_rules(self, rules: List[Dict[str, Any]]) -> str:
        """Render rules section for an indicator card."""
        if not rules:
            return ""
        rule_items = "\n".join(
            f"""<div class="rule-item">
                <span class="rule-name direction-{rule['direction']}">{rule['name']}</span>
                <div class="rule-desc">{rule['description']}</div>
            </div>"""
            for rule in rules
        )
        return f"""<div class="rules"><h4>Rules</h4>{rule_items}</div>"""

    def _render_regime_sections(
        self,
        regime_outputs: Optional[Dict[str, "RegimeOutput"]],
        provenance_dict: Optional[Dict[str, "ParamProvenanceSet"]] = None,
        recommendations_dict: Optional[Dict[str, "RecommenderResult"]] = None,
    ) -> str:
        """
        Render regime analysis sections for all symbols with regime data.

        Returns HTML for methodology, decision tree, component analysis,
        quality, hysteresis, optimization, and recommendations sections
        (PR1 + PR2 + PR3).

        Each symbol's section has data-symbol attribute for JavaScript filtering.
        Only the selected symbol's section is shown (controlled by updateRegimeSection).
        """
        if not regime_outputs:
            return ""

        provenance_dict = provenance_dict or {}
        recommendations_dict = recommendations_dict or {}

        # Build regime sections for each symbol
        sections_html = []

        for symbol, output in sorted(regime_outputs.items()):
            # Generate one-liner for this symbol
            one_liner = generate_regime_one_liner_html(output)

            # Generate decision tree
            decision_tree = generate_decision_tree_html(output, self.theme)

            # Generate component analysis
            components = generate_components_4block_html(output, self.theme)

            # PR2: Generate quality and hysteresis sections
            quality = generate_quality_html(output, self.theme)
            hysteresis = generate_hysteresis_html(output, self.theme)

            # Phase 4: Generate turning point detection section
            turning_point = generate_turning_point_html(output, self.theme)

            # PR3: Generate optimization and recommendations sections with real data
            provenance_set = provenance_dict.get(symbol)
            recommendations_result = recommendations_dict.get(symbol)
            optimization = generate_optimization_html(
                provenance=None,
                provenance_set=provenance_set,
                theme=self.theme,
            )
            recommendations = generate_recommendations_html(
                result=recommendations_result,
                theme=self.theme,
            )

            # PR4: Generate report header with metadata
            report_header = generate_report_header_html(
                regime_output=output,
                provenance_set=provenance_set,
                recommendations_result=recommendations_result,
                theme=self.theme,
            )

            # Add data-symbol attribute for JavaScript filtering
            sections_html.append(
                f"""
            <div class="regime-symbol-section" id="regime-{symbol}" data-symbol="{symbol}" style="display: none;">
                {report_header}
                {one_liner}
                {decision_tree}
                {components}
                {hysteresis}
                {turning_point}
                {quality}
                {optimization}
                {recommendations}
            </div>
            """
            )

        # Wrap with methodology at the top
        methodology = generate_methodology_html(self.theme)

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

    def _render_indicator_cards(self, indicator_info: List[Dict[str, Any]]) -> str:
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for info in indicator_info:
            categories.setdefault(info["category"], []).append(info)

        category_order = ["momentum", "trend", "volatility", "volume", "pattern"]
        category_labels = {
            "momentum": "Momentum Indicators",
            "trend": "Trend Indicators",
            "volatility": "Volatility Indicators",
            "volume": "Volume Indicators",
            "pattern": "Pattern Indicators",
        }

        html_parts = []
        for cat in category_order:
            if cat not in categories:
                continue

            cards_html = []
            for ind in categories[cat]:
                rules_html = self._render_rules(ind["rules"])
                cards_html.append(
                    f"""
                    <div class="indicator-card">
                        <h3>{ind['name'].upper()}</h3>
                        <div class="description">{ind['description']}</div>
                        {rules_html}
                    </div>
                """
                )

            html_parts.append(
                f"""
                <div class="category-group">
                    <div class="category-title">{category_labels.get(cat, cat.title())}</div>
                    <div class="indicator-cards">
                        {''.join(cards_html)}
                    </div>
                </div>
            """
            )

        return "\n".join(html_parts)

    def _get_scripts(
        self,
        chart_data: Dict[str, Any],
        symbols: List[str],
        timeframes: List[str],
        signal_history: Dict[str, List[Dict[str, Any]]],
        confluence_data: Dict[str, Dict[str, Any]],
    ) -> str:
        data_json = json.dumps(chart_data, default=str)
        symbols_json = json.dumps(symbols)
        timeframes_json = json.dumps(timeframes)
        colors_json = json.dumps(self._colors)
        signals_json = json.dumps(signal_history, default=str)
        confluence_json = json.dumps(confluence_data, default=str)

        return f"""
const chartData = {data_json};
const symbols = {symbols_json};
const timeframes = {timeframes_json};
const colors = {colors_json};
const signalHistory = {signals_json};
const confluenceData = {confluence_json};

let currentSymbol = symbols[0] || '';
let currentTimeframe = timeframes[0] || '1d';

function getDataKey() {{
    return `${{currentSymbol}}_${{currentTimeframe}}`;
}}

function selectTimeframe(tf, btn) {{
    currentTimeframe = tf;
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    updateChart();
}}

function updateChart() {{
    currentSymbol = document.getElementById('symbol-select').value;
    const key = getDataKey();
    const data = chartData[key];

    if (!data) {{
        console.warn('No data for', key);
        return;
    }}

    renderMainChart(data);
    updateSignalHistoryTable();
    updateConfluencePanel();
    updateRegimeSection();
}}

function updateRegimeSection() {{
    // Hide all regime symbol sections
    const sections = document.querySelectorAll('.regime-symbol-section');
    sections.forEach(section => {{
        section.style.display = 'none';
    }});

    // Show the selected symbol's section
    const selectedSection = document.getElementById('regime-' + currentSymbol);
    if (selectedSection) {{
        selectedSection.style.display = 'block';
    }}
}}

function renderMainChart(data) {{
    // Fixed 4-row layout: Price (55%), RSI (15%), MACD (14%), Volume (10%)
    const traces = [];
    const hasData = (values) => values && !values.every(v => v === null);

    // For intraday charts, use index-based x-axis to avoid gaps
    // This bypasses Plotly rangebreaks bug with candlesticks (Issue #4795)
    const isIntraday = ['1m', '5m', '15m', '30m', '1h', '4h'].includes(data.timeframe);
    const xValues = isIntraday
        ? data.timestamps.map((_, i) => i)  // Use indices for intraday
        : data.timestamps;                   // Use timestamps for daily

    // Create a timestamp-to-index map for signal markers
    const tsToIdx = {{}};
    if (isIntraday) {{
        data.timestamps.forEach((ts, i) => {{ tsToIdx[ts] = i; }});
    }}

    // Row 1: Price candlesticks
    traces.push({{
        type: 'candlestick',
        x: xValues,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        name: 'Price',
        increasing: {{ line: {{ color: colors.candle_up }}, fillcolor: colors.candle_up }},
        decreasing: {{ line: {{ color: colors.candle_down }}, fillcolor: colors.candle_down }},
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
        const values = data.overlays[name];
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
    const rsiValues = data.rsi['rsi_rsi'];
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
            {{ value: 70, name: 'Overbought', color: colors.candle_down }},
            {{ value: 30, name: 'Oversold', color: colors.candle_up }},
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
    const macdHist = data.macd['macd_histogram'];
    if (hasData(macdHist)) {{
        const barColors = macdHist.map(v => v >= 0 ? colors.candle_up : colors.candle_down);
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
        const values = data.macd[key];
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
    if (data.volume && data.volume.length > 0) {{
        const volColors = data.close.map((c, i) => {{
            if (i === 0) return colors.text_muted;
            return c >= data.close[i-1] ? colors.candle_up : colors.candle_down;
        }});
        traces.push({{
            type: 'bar',
            x: xValues,
            y: data.volume,
            name: 'Volume',
            marker: {{ color: volColors, opacity: 0.5 }},
            xaxis: 'x',
            yaxis: 'y4',
        }});
    }}

    // Add signal markers on price chart
    const key = getDataKey();
    const signals = signalHistory[key] || [];
    const buySignals = signals.filter(s => s.direction === 'buy');
    const sellSignals = signals.filter(s => s.direction === 'sell');

    if (buySignals.length > 0) {{
        const buyData = buySignals.map(s => {{
            const idx = data.timestamps.findIndex(t => t === s.timestamp);
            if (idx < 0) return null;
            return {{
                x: isIntraday ? idx : s.timestamp,
                y: data.low[idx] * 0.995,
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
                    color: colors.candle_up,
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
            const idx = data.timestamps.findIndex(t => t === s.timestamp);
            if (idx < 0) return null;
            return {{
                x: isIntraday ? idx : s.timestamp,
                y: data.high[idx] * 1.005,
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
                    color: colors.candle_down,
                    line: {{ color: 'white', width: 1 }}
                }},
                hovertemplate: '%{{text}}<extra></extra>',
                text: sellData.map(d => d.rule),
                xaxis: 'x',
                yaxis: 'y',
            }});
        }}
    }}

    // Layout with 4 subplots (4% gaps between panels)
    // Hide gaps to create continuous chart visualization
    const isDaily = ['1d', '1w', '1D', '1W'].includes(data.timeframe);

    // For daily charts, use rangebreaks to hide weekends
    // For intraday, we use index-based x-axis (no rangebreaks needed - already continuous)
    let rangebreaks = [];
    if (isDaily) {{
        rangebreaks = [{{ bounds: ['sat', 'mon'] }}];
    }}

    // For intraday charts with index-based x-axis, create custom tick labels
    // Show ~15 evenly spaced labels with readable date/time format
    let tickvals = null;
    let ticktext = null;
    if (isIntraday && data.timestamps.length > 0) {{
        const n = data.timestamps.length;
        const step = Math.max(1, Math.floor(n / 15));
        tickvals = [];
        ticktext = [];
        for (let i = 0; i < n; i += step) {{
            tickvals.push(i);
            // Format timestamp as "MM/DD HH:mm"
            const ts = new Date(data.timestamps[i]);
            const month = String(ts.getUTCMonth() + 1).padStart(2, '0');
            const day = String(ts.getUTCDate()).padStart(2, '0');
            const hour = String(ts.getUTCHours()).padStart(2, '0');
            const min = String(ts.getUTCMinutes()).padStart(2, '0');
            ticktext.push(`${{month}}/${{day}} ${{hour}}:${{min}}`);
        }}
        // Always include last bar
        if (tickvals[tickvals.length - 1] !== n - 1) {{
            tickvals.push(n - 1);
            const ts = new Date(data.timestamps[n - 1]);
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

        // Shared X-axis configuration
        // For daily: use timestamps with rangebreaks
        // For intraday: use indices with custom tick labels (continuous chart)
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

        // Timezone annotation
        annotations: [{{
            text: 'All times displayed in UTC',
            xref: 'paper',
            yref: 'paper',
            x: 1,
            y: 1.02,
            xanchor: 'right',
            yanchor: 'bottom',
            showarrow: false,
            font: {{ size: 10, color: colors.text_muted }},
        }}],

        // Y1: Price (48% with 4% gap)
        yaxis: {{
            title: 'Price',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.52, 1.00],
            autorange: true,
        }},

        // Y2: RSI (12% with 4% gap)
        yaxis2: {{
            title: 'RSI',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.36, 0.48],
            range: [0, 100],
            dtick: 25,
        }},

        // Y3: MACD (12% with 4% gap)
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

function updateSignalHistoryTable() {{
    const key = getDataKey();
    const signals = signalHistory[key] || [];
    const container = document.getElementById('signal-history-table');

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

function updateConfluencePanel() {{
    const key = getDataKey();
    const confluence = confluenceData[key];
    const container = document.getElementById('confluence-panel');

    if (!confluence) {{
        container.innerHTML = '<div class="no-divergences">No confluence data available for this symbol/timeframe</div>';
        return;
    }}

    const alignmentPct = (confluence.alignment_score + 100) / 2;  // Convert -100..+100 to 0..100
    const alignmentClass = confluence.alignment_score > 20 ? 'bullish' : confluence.alignment_score < -20 ? 'bearish' : 'neutral';
    const strongestClass = confluence.strongest_signal === 'bullish' ? 'bullish' : confluence.strongest_signal === 'bearish' ? 'bearish' : 'neutral';

    let divergenceHtml = '';
    if (confluence.diverging_pairs && confluence.diverging_pairs.length > 0) {{
        divergenceHtml = confluence.diverging_pairs.slice(0, 5).map(p => `
            <div class="divergence-item">
                <div class="indicators">${{p.ind1}} ↔ ${{p.ind2}}</div>
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
                <div class="alignment-value ${{alignmentClass}}">${{confluence.alignment_score > 0 ? '+' : ''}}${{confluence.alignment_score}}</div>
                <div class="signal-counts">
                    <div class="count-item">
                        <div class="count-value bullish">▲ ${{confluence.bullish_count}}</div>
                        <div class="count-label">Bullish</div>
                    </div>
                    <div class="count-item">
                        <div class="count-value neutral">● ${{confluence.neutral_count}}</div>
                        <div class="count-label">Neutral</div>
                    </div>
                    <div class="count-item">
                        <div class="count-value bearish">▼ ${{confluence.bearish_count}}</div>
                        <div class="count-label">Bearish</div>
                    </div>
                </div>
                <div class="strongest-signal">
                    <div class="label">Strongest Signal</div>
                    <div class="value ${{strongestClass}}">${{confluence.strongest_signal ? confluence.strongest_signal.toUpperCase() : 'NONE'}}</div>
                </div>
            </div>
            <div class="divergence-list">
                <h4 style="margin-bottom: 12px; color: ${{colors.text_muted}}; font-size: 12px; text-transform: uppercase;">Diverging Indicators</h4>
                ${{divergenceHtml}}
            </div>
        </div>
    `;
}}

function toggleSection(arg) {{
    // Handle both string ID and DOM element (for regime report sections)
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
            indicator.style.transform = section.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
        }}
    }}
}}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {{
    updateChart();
    updateSignalHistoryTable();
    updateConfluencePanel();
    updateRegimeSection();
}});
"""
