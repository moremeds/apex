"""
HTML Renderer - Builds HTML structure for signal reports.

Renders symbol options, timeframe buttons, indicator cards, and section structure.
"""

from __future__ import annotations

from typing import Any, Dict, List


def render_symbol_options(symbols: List[str]) -> str:
    """
    Render symbol dropdown options.

    Args:
        symbols: List of symbol names

    Returns:
        HTML string of option elements
    """
    return "\n".join(f'<option value="{s}">{s}</option>' for s in symbols)


def render_timeframe_buttons(timeframes: List[str]) -> str:
    """
    Render timeframe toggle buttons.

    Args:
        timeframes: List of timeframe strings

    Returns:
        HTML string of button elements
    """
    return "\n".join(
        f'<button class="tf-btn{" active" if i == 0 else ""}" data-tf="{tf}" '
        f"onclick=\"selectTimeframe('{tf}', this)\">{tf}</button>"
        for i, tf in enumerate(timeframes)
    )


def render_rules(rules: List[Dict[str, Any]]) -> str:
    """
    Render rules section for an indicator card.

    Args:
        rules: List of rule dictionaries

    Returns:
        HTML string for rules section
    """
    if not rules:
        return ""
    rule_items = "\n".join(f"""<div class="rule-item">
            <span class="rule-name direction-{rule['direction']}">{rule['name']}</span>
            <div class="rule-desc">{rule['description']}</div>
        </div>""" for rule in rules)
    return f"""<div class="rules"><h4>Rules</h4>{rule_items}</div>"""


def render_indicator_cards(indicator_info: List[Dict[str, Any]]) -> str:
    """
    Render indicator cards grouped by category.

    Args:
        indicator_info: List of indicator info dictionaries

    Returns:
        HTML string for indicator cards section
    """
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
            rules_html = render_rules(ind["rules"])
            cards_html.append(f"""
                <div class="indicator-card">
                    <h3>{ind['name'].upper()}</h3>
                    <div class="description">{ind['description']}</div>
                    {rules_html}
                </div>
            """)

        html_parts.append(f"""
            <div class="category-group">
                <div class="category-title">{category_labels.get(cat, cat.title())}</div>
                <div class="indicator-cards">
                    {''.join(cards_html)}
                </div>
            </div>
        """)

    return "\n".join(html_parts)
