"""
Parameters CSS styles - Optimization and Recommendations sections.
"""

from __future__ import annotations

PARAMETERS_STYLES = """
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

# Turning Point Detection styles
