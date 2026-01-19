"""
Regime Report Styles - CSS for all regime report sections.

Contains all CSS styles organized by section (PR1, PR2, PR3, Phase 4).
"""

from __future__ import annotations

from ..value_card import get_value_card_styles


def generate_regime_styles() -> str:
    """Generate CSS styles for regime report sections."""
    base_styles = """
    /* Report Header Section (PR4) */
    .report-header-section {
        margin-bottom: 20px;
        padding: 16px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
    }

    .report-header-content {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    .header-title-row {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }

    .header-symbol {
        font-size: 24px;
        font-weight: 700;
        color: var(--text);
    }

    .header-regime {
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
    }

    .header-timestamp {
        margin-left: auto;
        font-size: 13px;
        color: var(--text-muted);
    }

    .header-details {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 8px 24px;
        padding: 12px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
    }

    .header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
    }

    .header-label {
        color: var(--text-muted);
    }

    .header-value {
        font-weight: 500;
    }

    .header-value code {
        background: rgba(59, 130, 246, 0.1);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 11px;
    }

    .regime-dashboard {
        margin-bottom: 24px;
    }

    .regime-timestamp {
        font-size: 12px;
        color: var(--text-muted);
        margin-bottom: 16px;
    }

    .regime-level {
        margin-bottom: 24px;
    }

    .regime-level h3 {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 12px;
    }

    .regime-cards {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 12px;
    }

    .regime-card {
        padding: 16px;
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .regime-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .regime-card-symbol {
        font-size: 18px;
        font-weight: 600;
    }

    .regime-badge {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }

    .regime-card-details {
        font-size: 12px;
        color: var(--text-muted);
    }

    .action-badge {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .action-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }

    .action-table th {
        text-align: left;
        padding: 12px 8px;
        border-bottom: 2px solid var(--border);
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
    }

    .action-table td {
        padding: 10px 8px;
        border-bottom: 1px solid var(--border);
    }

    .component-breakdown {
        margin-bottom: 24px;
    }

    .component-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 16px;
    }

    .component-card {
        padding: 16px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .component-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    .component-card-title {
        font-weight: 600;
    }

    .component-state {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .component-metrics {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .component-metric {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
    }

    .metric-label {
        color: var(--text-muted);
    }

    .metric-value {
        font-weight: 500;
    }

    .alerts-section {
        margin-bottom: 24px;
    }

    .no-alerts {
        text-align: center;
        color: var(--text-muted);
        padding: 24px;
        font-style: italic;
    }

    .alert-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .alert-item {
        padding: 12px;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        display: flex;
        gap: 12px;
        align-items: center;
    }

    .alert-symbol {
        font-weight: 600;
        color: #ef4444;
    }

    .alert-message {
        color: var(--text);
    }

    .timeline-symbol {
        margin-bottom: 16px;
    }

    .timeline-symbol h3 {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .timeline-events {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .timeline-event {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 8px;
        background: var(--bg);
        border-radius: 4px;
    }

    .event-regime {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }

    .event-details {
        font-size: 11px;
        color: var(--text-muted);
    }

    .event-time {
        font-weight: 500;
    }
    """

    # Add new PR1 styles
    pr1_styles = """
    /* Methodology Section */
    .methodology-content {
        line-height: 1.6;
    }

    .methodology-content h3 {
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 16px;
        font-weight: 600;
    }

    .methodology-content p {
        margin-bottom: 12px;
    }

    .regime-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin: 16px 0;
    }

    .regime-item {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .regime-item.r0 { border-left: 4px solid #166534; }
    .regime-item.r1 { border-left: 4px solid #ca8a04; }
    .regime-item.r2 { border-left: 4px solid #dc2626; }
    .regime-item.r3 { border-left: 4px solid #2563eb; }

    .regime-code {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .regime-label {
        font-weight: 600;
        margin-bottom: 4px;
    }

    .regime-desc {
        font-size: 12px;
        color: var(--text-muted);
    }

    .priority-list {
        margin: 12px 0;
        padding-left: 24px;
    }

    .priority-list li {
        margin-bottom: 8px;
    }

    .component-table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 13px;
    }

    .component-table th,
    .component-table td {
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    .component-table th {
        background: var(--header-bg);
        font-weight: 600;
    }

    /* Decision Tree Section */
    .decision-tree-content {
        font-family: monospace;
    }

    .decision-result {
        padding: 16px;
        background: var(--highlight-bg);
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }

    .decision-result.same .label {
        color: var(--text-muted);
    }

    .decision-result.different {
        border: 2px solid #ca8a04;
    }

    .decision-result .separator {
        font-size: 18px;
    }

    .decision-result .pending-info {
        font-size: 12px;
        color: #ca8a04;
    }

    .check-group {
        margin-bottom: 16px;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    .check-group.matched {
        border-color: #22c55e;
    }

    .check-group.skipped {
        opacity: 0.6;
    }

    .check-header {
        padding: 12px 16px;
        background: var(--header-bg);
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .check-priority {
        font-weight: 600;
        color: var(--text-muted);
    }

    .check-title {
        font-weight: 600;
        flex: 1;
    }

    .check-status {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }

    .check-group.matched .check-status {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }

    .check-group.skipped .check-status {
        background: rgba(148, 163, 184, 0.2);
        color: #94a3b8;
    }

    .check-rules {
        padding: 12px 16px;
    }

    .rule-item {
        padding: 8px 0;
        font-size: 13px;
        border-bottom: 1px solid var(--border);
    }

    .rule-item:last-child {
        border-bottom: none;
    }

    .rule-main {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .rule-details {
        margin-left: 28px;
        margin-top: 4px;
    }

    .rule-icon {
        width: 20px;
        text-align: center;
    }

    .rule-item.pass .rule-icon { color: #22c55e; }
    .rule-item.fail .rule-icon { color: #ef4444; }

    .rule-id {
        width: 180px;
        font-weight: 500;
        font-family: monospace;
        font-size: 12px;
    }

    .rule-desc {
        flex: 1;
        color: var(--text-muted);
    }

    .rule-status {
        width: 50px;
        font-size: 11px;
        font-weight: 600;
    }

    .rule-item.pass .rule-status { color: #22c55e; }
    .rule-item.fail .rule-status { color: #ef4444; }

    .rule-evidence {
        font-size: 12px;
        color: var(--text-muted);
        font-family: monospace;
    }

    .rule-threshold {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 4px;
        padding: 6px 10px;
        background: var(--bg);
        border-radius: 4px;
        font-size: 12px;
        font-family: monospace;
    }

    .threshold-metric {
        color: var(--text-muted);
    }

    .threshold-actual {
        font-weight: 600;
        color: var(--text);
    }

    .threshold-op {
        color: var(--text-muted);
    }

    .threshold-value {
        font-weight: 600;
        color: #3b82f6;
    }

    .threshold-gap {
        color: var(--text-muted);
        font-size: 11px;
    }

    .rule-item.pass .threshold-actual { color: #22c55e; }
    .rule-item.fail .threshold-actual { color: #ef4444; }

    /* Counterfactual Section */
    .counterfactual-section {
        margin-top: 24px;
        padding-top: 16px;
        border-top: 2px dashed var(--border);
    }

    .counterfactual-section h4 {
        margin-bottom: 12px;
        font-size: 14px;
    }

    .counterfactual-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 12px;
    }

    .counterfactual-item {
        padding: 12px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .cf-header {
        font-weight: 600;
        margin-bottom: 8px;
    }

    .cf-condition {
        font-size: 13px;
        padding: 4px 0;
    }

    .cf-current {
        font-size: 11px;
        color: var(--text-muted);
    }

    /* Component 4-Block */
    .components-4block {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 16px;
    }

    .component-4block {
        padding: 16px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .block-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--border);
    }

    .block-title {
        font-weight: 700;
        font-size: 14px;
    }

    .block-state {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }

    .block-section {
        margin-bottom: 12px;
    }

    .section-label {
        font-size: 10px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    .section-content {
        font-size: 13px;
    }

    .section-content code {
        font-size: 11px;
        background: var(--code-bg);
        padding: 2px 4px;
        border-radius: 4px;
    }

    .values-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }

    .value-item {
        display: flex;
        flex-direction: column;
    }

    .value-label {
        font-size: 11px;
        color: var(--text-muted);
    }

    .value-num {
        font-weight: 600;
        font-size: 14px;
    }
    """

    # PR2 styles for Quality and Hysteresis sections
    pr2_styles = """
    /* Quality Section */
    .quality-content {
        padding: 8px 0;
    }

    .quality-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }

    .quality-item {
        padding: 12px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .quality-label {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .quality-value {
        font-size: 14px;
        font-weight: 600;
    }

    .quality-value.ok { color: #22c55e; }
    .quality-value.warn { color: #ca8a04; }

    .validity-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }

    .validity-table th,
    .validity-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    .validity-table th {
        font-weight: 600;
        color: var(--text-muted);
        font-size: 11px;
        text-transform: uppercase;
    }

    .validity-table .ok { color: #22c55e; }
    .validity-table .na { color: #ca8a04; }
    .validity-table .issue { color: var(--text-muted); font-size: 12px; }

    .fallback-alert {
        padding: 16px;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .fallback-header {
        font-weight: 700;
        color: #ef4444;
        margin-bottom: 8px;
    }

    .fallback-reason,
    .fallback-exception,
    .fallback-result {
        font-size: 13px;
        margin-top: 4px;
    }

    /* Hysteresis Section */
    .hysteresis-content {
        padding: 8px 0;
    }

    .hysteresis-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .hysteresis-status.stable {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .hysteresis-status.accumulating {
        background: rgba(202, 138, 4, 0.1);
        border: 1px solid rgba(202, 138, 4, 0.3);
    }

    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }

    .hysteresis-status.stable .status-indicator {
        background: #22c55e;
    }

    .hysteresis-status.accumulating .status-indicator {
        background: #ca8a04;
    }

    .status-text {
        font-weight: 600;
        font-size: 14px;
    }

    .hysteresis-status.stable .status-text { color: #22c55e; }
    .hysteresis-status.accumulating .status-text { color: #ca8a04; }

    .state-grid {
        display: grid;
        gap: 4px;
        margin-bottom: 20px;
    }

    .state-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0;
        border-bottom: 1px solid var(--border);
    }

    .state-label {
        min-width: 150px;
        color: var(--text-muted);
        font-size: 13px;
    }

    .state-value {
        font-weight: 600;
        font-size: 14px;
    }

    .state-value.muted {
        color: var(--text-muted);
        font-weight: normal;
    }

    .pending-progress {
        font-size: 12px;
        color: #ca8a04;
    }

    .transition-reason {
        padding: 12px;
        background: var(--bg);
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 16px;
    }

    .transition-reason h4 {
        font-size: 14px;
        margin-bottom: 8px;
    }

    .reason-text {
        color: var(--text-muted);
    }

    .hysteresis-rules {
        margin-top: 16px;
    }

    .hysteresis-rules h4 {
        font-size: 14px;
        margin-bottom: 12px;
    }

    .no-rules {
        color: var(--text-muted);
        font-style: italic;
    }

    .hysteresis-rule {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 0;
        font-size: 13px;
        border-bottom: 1px solid var(--border);
    }

    .hysteresis-rule .rule-icon {
        width: 20px;
        text-align: center;
    }

    .hysteresis-rule.pass .rule-icon { color: #22c55e; }
    .hysteresis-rule.fail .rule-icon { color: #ef4444; }

    .hysteresis-rule .rule-id {
        width: 200px;
        font-weight: 500;
    }

    .hysteresis-rule .rule-desc {
        flex: 1;
        color: var(--text-muted);
    }

    .hysteresis-rule .rule-evidence {
        font-size: 11px;
        color: var(--text-muted);
        max-width: 250px;
    }

    .last-transition {
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid var(--border);
        font-size: 12px;
        color: var(--text-muted);
    }
    """

    # PR3 styles for Optimization and Recommendations sections
    pr3_styles = """
    /* Optimization Section */
    .optimization-content {
        padding: 8px 0;
    }

    .provenance-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px;
        background: var(--highlight-bg);
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .provenance-id {
        font-family: monospace;
        font-size: 16px;
        font-weight: 600;
    }

    .provenance-source {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }

    .provenance-source.symbol-specific {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }

    .provenance-source.group {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
    }

    .provenance-source.default {
        background: rgba(148, 163, 184, 0.2);
        color: #94a3b8;
    }

    .param-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-bottom: 20px;
    }

    .param-table th,
    .param-table td {
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    .param-table th {
        font-weight: 600;
        color: var(--text-muted);
        font-size: 11px;
        text-transform: uppercase;
        background: var(--header-bg);
    }

    .param-source {
        font-size: 11px;
        padding: 2px 6px;
        border-radius: 3px;
    }

    .validation-section {
        margin-top: 20px;
        padding-top: 16px;
        border-top: 1px solid var(--border);
    }

    .validation-section h4 {
        font-size: 14px;
        margin-bottom: 12px;
    }

    .validation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
    }

    .validation-item {
        padding: 12px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .validation-label {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .validation-value {
        font-size: 14px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .validation-value.pass { color: #22c55e; }
    .validation-value.fail { color: #ef4444; }

    /* Recommendations Section */
    .recommendations-content {
        padding: 8px 0;
    }

    .no-recommendations {
        padding: 20px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        text-align: center;
    }

    .no-recommendations .header {
        font-size: 16px;
        font-weight: 600;
        color: #22c55e;
        margin-bottom: 12px;
    }

    .no-recommendations .reason {
        font-size: 13px;
        color: var(--text-muted);
    }

    .no-recommendations .metrics {
        display: flex;
        justify-content: center;
        gap: 24px;
        margin-top: 12px;
        font-size: 12px;
    }

    .no-provenance {
        text-align: center;
        padding: 20px;
        color: var(--text-muted);
    }

    /* Current params table */
    .current-params {
        margin-top: 20px;
        text-align: left;
    }

    .current-params h4 {
        margin-bottom: 12px;
        font-size: 14px;
        color: var(--text-muted);
    }

    .params-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }

    .params-table th, .params-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    .params-table th {
        background: rgba(255, 255, 255, 0.05);
        font-weight: 500;
    }

    .params-table code {
        background: rgba(59, 130, 246, 0.1);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 12px;
    }

    /* Analysis details section */
    .analysis-details {
        margin-top: 24px;
        padding: 16px;
        background: rgba(59, 130, 246, 0.05);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        text-align: left;
    }

    .analysis-details h4 {
        margin-bottom: 16px;
        font-size: 14px;
        color: #3b82f6;
    }

    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-bottom: 16px;
    }

    .analysis-section {
        padding: 12px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 6px;
    }

    .analysis-title {
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 12px;
        color: var(--text);
    }

    .analysis-table {
        width: 100%;
        font-size: 12px;
        border-collapse: collapse;
    }

    .analysis-table td {
        padding: 6px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .analysis-table td:first-child {
        color: var(--text-muted);
        width: 45%;
    }

    .analysis-table td:last-child {
        font-family: monospace;
    }

    .analysis-methodology {
        font-size: 11px;
        color: var(--text-muted);
        padding: 12px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 4px;
        line-height: 1.5;
    }

    .recommendation-card {
        padding: 16px;
        background: rgba(202, 138, 4, 0.1);
        border: 1px solid rgba(202, 138, 4, 0.3);
        border-radius: 8px;
        margin-bottom: 16px;
    }

    .recommendation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .recommendation-param {
        font-family: monospace;
        font-size: 14px;
        font-weight: 600;
    }

    .recommendation-change {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .recommendation-current {
        font-size: 16px;
        color: var(--text-muted);
    }

    .recommendation-arrow {
        font-size: 14px;
        color: #ca8a04;
    }

    .recommendation-suggested {
        font-size: 16px;
        font-weight: 600;
        color: #ca8a04;
    }

    .recommendation-delta {
        font-size: 12px;
        padding: 2px 6px;
        background: rgba(202, 138, 4, 0.2);
        border-radius: 4px;
        color: #ca8a04;
    }

    .recommendation-confidence {
        font-size: 12px;
        color: var(--text-muted);
    }

    .recommendation-reason {
        font-size: 13px;
        margin-bottom: 12px;
        padding: 8px;
        background: var(--bg);
        border-radius: 4px;
    }

    .recommendation-evidence {
        font-size: 12px;
        color: var(--text-muted);
    }

    .evidence-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 8px;
        margin-top: 8px;
    }

    .evidence-item {
        display: flex;
        justify-content: space-between;
    }

    .manual-review-badge {
        padding: 4px 8px;
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 12px;
    }
    """

    # Phase 4: Turning Point Detection styles
    turning_point_styles = """
    /* Turning Point Detection Section */
    .turning-point-content {
        padding: 16px;
    }

    .turning-point-placeholder {
        text-align: center;
        padding: 24px;
    }

    .turning-point-placeholder .muted {
        color: var(--text-muted);
        font-size: 13px;
    }

    .tp-summary {
        margin-bottom: 16px;
    }

    .tp-state-row, .tp-confidence-row, .tp-meta-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }

    .tp-state-badge {
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .confidence-bar-container {
        flex: 1;
        height: 24px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        position: relative;
        overflow: hidden;
        max-width: 200px;
    }

    .confidence-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .confidence-value {
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 12px;
        font-weight: 600;
        color: #ffffff;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }

    .tp-version {
        font-family: monospace;
        font-size: 11px;
        color: var(--text-muted);
    }

    .tp-inference {
        font-size: 11px;
        color: var(--text-muted);
    }

    .tp-inference.fast {
        color: #16a34a;
    }

    .tp-inference.slow {
        color: #ca8a04;
    }

    .tp-gating {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    }

    .tp-gating.warning {
        background: rgba(220, 38, 38, 0.15);
        border: 1px solid rgba(220, 38, 38, 0.3);
    }

    .tp-gating.success {
        background: rgba(22, 163, 74, 0.15);
        border: 1px solid rgba(22, 163, 74, 0.3);
    }

    .gating-icon {
        font-size: 18px;
    }

    .gating-text {
        font-size: 13px;
    }

    .tp-features {
        margin-top: 16px;
    }

    .tp-features h4 {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
        color: var(--text-muted);
    }

    .feature-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .feature-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 4px;
    }

    .feature-name {
        font-family: monospace;
        font-size: 12px;
    }

    .feature-contrib {
        font-family: monospace;
        font-size: 12px;
        font-weight: 600;
    }

    /* Backtest Metrics Section */
    .tp-backtest {
        margin-top: 20px;
        padding: 16px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .tp-backtest h4 {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
        color: var(--text-muted);
    }

    .backtest-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }

    .backtest-model {
        padding: 12px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 6px;
    }

    .model-label {
        display: block;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }

    .metric-name {
        font-size: 12px;
        color: var(--text-muted);
    }

    .metric-value {
        font-family: monospace;
        font-size: 12px;
        font-weight: 600;
    }

    .backtest-meta {
        margin-top: 12px;
        font-size: 11px;
        color: var(--text-muted);
        text-align: right;
    }

    .backtest-meta a {
        color: #3b82f6;
        text-decoration: none;
    }

    .backtest-meta a:hover {
        text-decoration: underline;
    }
    """

    return (
        base_styles
        + pr1_styles
        + pr2_styles
        + pr3_styles
        + turning_point_styles
        + get_value_card_styles()
    )
