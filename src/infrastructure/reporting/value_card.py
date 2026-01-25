"""
Value Card Rendering Utilities for Regime Reports.

Provides centralized helpers for:
- HTML escaping (security against XSS)
- Consistent metric card rendering
- Value formatting with units
- Section rendering with collapsible behavior

All HTML output functions use escape_html() to prevent XSS vulnerabilities.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, List, Optional


def escape_html(text: Any) -> str:
    """
    Escape HTML special characters (central helper).

    This is the ONLY function that should be used to escape user-provided
    or dynamic content before inserting into HTML. All rendering functions
    must use this to prevent XSS attacks.

    Args:
        text: Any value to escape (will be converted to str first)

    Returns:
        HTML-escaped string safe for insertion into HTML
    """
    return html.escape(str(text))


def format_value(value: float, unit: str = "", precision: int = 2) -> str:
    """
    Format numeric values consistently across the report.

    Args:
        value: The numeric value to format
        unit: Unit string ("%" or "ATR" or "" for dimensionless)
        precision: Decimal places to show

    Returns:
        Formatted string like "82.5%" or "1.23 ATR" or "145.20"
    """
    if unit == "%":
        return f"{value:.{precision}f}%"
    elif unit == "ATR":
        return f"{value:.{precision}f} ATR"
    else:
        return f"{value:.{precision}f}{escape_html(unit)}"


def format_currency(value: float, precision: int = 2) -> str:
    """Format as currency with dollar sign."""
    return f"${value:,.{precision}f}"


def format_percentage(value: float, precision: int = 2, multiply: bool = False) -> str:
    """
    Format as percentage.

    Args:
        value: The value to format
        precision: Decimal places
        multiply: If True, multiply by 100 (for 0.05 -> 5.00%)
    """
    if multiply:
        value = value * 100
    return f"{value:.{precision}f}%"


@dataclass
class ValueCard:
    """
    Structured data for rendering a metric card in the report.

    Provides consistent formatting and optional pass/fail badging
    for threshold comparisons.
    """

    name: str
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    unit: str = ""
    precision: int = 2
    sparkline_data: Optional[List[float]] = None

    def format_value_str(self) -> str:
        """Format the value with unit."""
        return format_value(self.value, self.unit, self.precision)

    def format_threshold_str(self) -> str:
        """Format the threshold with unit (if present)."""
        if self.threshold is None:
            return ""
        return format_value(self.threshold, self.unit, self.precision)


def render_value_card(card: ValueCard) -> str:
    """
    Render a single metric card to HTML (escaped).

    The card shows:
    - Label (name)
    - Value with optional threshold comparison
    - Pass/Fail badge if threshold is set

    Args:
        card: ValueCard instance with metric data

    Returns:
        HTML string for the card (safe - all content escaped)
    """
    badge = ""
    if card.passed is not None:
        cls = "pass" if card.passed else "fail"
        txt = "PASS" if card.passed else "FAIL"
        badge = f'<span class="badge {cls}">{txt}</span>'

    value_str = card.format_value_str()
    threshold_str = ""
    if card.threshold is not None:
        threshold_str = f" / {card.format_threshold_str()}"

    return f"""
    <div class="value-card">
        <div class="card-label">{escape_html(card.name)}</div>
        <div class="card-value">{escape_html(value_str)}{escape_html(threshold_str)} {badge}</div>
    </div>
    """


def render_section(
    title: str,
    body: str,
    collapsed: bool = True,
    icon: str = "",
    section_id: str = "",
) -> str:
    """
    Render a collapsible section with consistent styling.

    Args:
        title: Section title (will be escaped)
        body: Section body HTML (assumed already safe)
        collapsed: Whether section starts collapsed
        icon: Optional icon/emoji for the section header
        section_id: Optional HTML id attribute for the section

    Returns:
        HTML string for the collapsible section
    """
    collapse_class = "collapsed" if collapsed else ""
    id_attr = f'id="{escape_html(section_id)}"' if section_id else ""

    return f"""
    <div class="report-section {collapse_class}" {id_attr}>
        <div class="section-header" onclick="toggleSection(this)">
            <span class="section-icon">{escape_html(icon)}</span>
            <span class="section-title">{escape_html(title)}</span>
            <span class="collapse-indicator">▼</span>
        </div>
        <div class="section-body">
            {body}
        </div>
    </div>
    """


def render_one_liner_box(
    decision_regime: str,
    final_regime: str,
    pending_count: int = 0,
    entry_threshold: int = 0,
) -> str:
    """
    Render the UX one-liner showing decision vs final regime.

    This single line reduces 80% of user confusion by clearly showing
    when hysteresis is blocking a regime transition.

    Args:
        decision_regime: Raw tree output (e.g., "R0")
        final_regime: After hysteresis (e.g., "R1")
        pending_count: Current bars waiting for confirmation
        entry_threshold: Bars needed to confirm transition

    Returns:
        HTML string for the one-liner box
    """
    if decision_regime == final_regime:
        # No hysteresis blocking
        status_text = f"Regime: {escape_html(final_regime)}"
    else:
        # Hysteresis is blocking - show pending status
        status_text = (
            f"Decision Regime (pre-hysteresis): {escape_html(decision_regime)} | "
            f"Final Regime: {escape_html(final_regime)} "
            f"(pending {pending_count}/{entry_threshold} bars to confirm {escape_html(decision_regime)})"
        )

    return f"""
    <div class="one-liner-box">
        <div class="one-liner-content">{status_text}</div>
    </div>
    """


def render_info_row(label: str, value: str, value_class: str = "") -> str:
    """
    Render a single label-value row for info displays.

    Args:
        label: Row label (will be escaped)
        value: Row value (will be escaped)
        value_class: Optional CSS class for the value span

    Returns:
        HTML string for the row
    """
    class_attr = f' class="{escape_html(value_class)}"' if value_class else ""
    return f"""
    <div class="info-row">
        <span class="info-label">{escape_html(label)}</span>
        <span class="info-value"{class_attr}>{escape_html(value)}</span>
    </div>
    """


def render_badge(text: str, color: str, bg_color: str) -> str:
    """
    Render a colored badge.

    Args:
        text: Badge text (will be escaped)
        color: Text color (CSS)
        bg_color: Background color (CSS)

    Returns:
        HTML string for the badge
    """
    return f"""<span class="badge" style="color: {escape_html(color)}; background: {escape_html(bg_color)}">{escape_html(text)}</span>"""


def get_value_card_styles() -> str:
    """
    Get CSS styles for value cards and related components.

    Returns:
        CSS string to include in report
    """
    return """
    .value-card {
        padding: 12px;
        background: var(--card-bg);
        border-radius: 6px;
        border: 1px solid var(--border);
    }

    .card-label {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .card-value {
        font-size: 16px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .badge {
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .badge.pass {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }

    .badge.fail {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }

    .report-section {
        margin-bottom: 16px;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    .section-header {
        padding: 12px 16px;
        background: var(--header-bg);
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
        user-select: none;
    }

    .section-header:hover {
        background: var(--header-hover);
    }

    .section-icon {
        font-size: 16px;
    }

    .section-title {
        font-weight: 600;
        flex: 1;
    }

    .collapse-indicator {
        font-size: 12px;
        transition: transform 0.2s;
    }

    .report-section.collapsed .collapse-indicator {
        transform: rotate(-90deg);
    }

    .report-section.collapsed .section-body {
        display: none;
    }

    .section-body {
        padding: 16px;
    }

    .one-liner-box {
        padding: 16px;
        background: var(--highlight-bg);
        border: 2px solid var(--highlight-border);
        border-radius: 8px;
        margin-bottom: 16px;
        font-family: monospace;
    }

    .one-liner-content {
        font-size: 14px;
        font-weight: 500;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid var(--border);
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-label {
        color: var(--text-muted);
    }

    .info-value {
        font-weight: 500;
    }
    """
