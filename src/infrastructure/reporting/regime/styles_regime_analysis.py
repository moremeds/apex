"""
Regime Analysis CSS styles - Methodology and Decision Tree sections.
"""

from __future__ import annotations

REGIME_ANALYSIS_STYLES = """
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

# Validation Status styles for Quality and Hysteresis sections
