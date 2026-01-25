"""
Base CSS styles for regime reports - header section and common components.
"""

from __future__ import annotations

BASE_STYLES = """
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
