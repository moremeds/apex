"""
Theme Styles - CSS generation for signal reports.

Generates dark and light theme CSS for interactive HTML reports.
"""

from __future__ import annotations

from typing import Dict


def get_theme_colors(theme: str) -> Dict[str, str]:
    """
    Get color scheme for a theme.

    Args:
        theme: Theme name ("dark" or "light")

    Returns:
        Dict of color values
    """
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


def get_styles(colors: Dict[str, str]) -> str:
    """
    Generate complete CSS for signal report.

    Args:
        colors: Theme color dictionary

    Returns:
        Complete CSS content string
    """
    from ..regime import generate_regime_styles

    c = colors
    return f"""
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

/* Rule Frequency Summary (Phase 3) */
.rule-frequency-summary {{
    margin-bottom: 24px;
    padding: 16px;
    background: {c['bg']};
    border-radius: 8px;
}}

.rule-freq-bars {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}

.rule-freq-item {{
    display: flex;
    flex-direction: column;
    gap: 4px;
}}

.rule-freq-label {{
    display: flex;
    justify-content: space-between;
    font-size: 12px;
}}

.rule-freq-label .rule-name {{
    font-weight: 500;
    color: {c['text']};
}}

.rule-freq-label .rule-count {{
    font-weight: 600;
}}

.rule-freq-label .rule-count.buy {{
    color: {c['profit']};
}}

.rule-freq-label .rule-count.sell {{
    color: {c['loss']};
}}

.rule-freq-label .rule-count.alert {{
    color: {c['primary']};
}}

.rule-freq-bar-bg {{
    height: 6px;
    background: {c['border']};
    border-radius: 3px;
    overflow: hidden;
}}

.rule-freq-bar {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}}

.rule-freq-bar.buy {{
    background: {c['profit']};
}}

.rule-freq-bar.sell {{
    background: {c['loss']};
}}

.rule-freq-bar.alert {{
    background: {c['primary']};
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
""" + generate_regime_styles()
