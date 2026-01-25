"""
Validation Status CSS styles - Quality and Hysteresis sections.
"""

from __future__ import annotations

VALIDATION_STATUS_STYLES = """
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

# Parameters styles for Optimization and Recommendations sections
