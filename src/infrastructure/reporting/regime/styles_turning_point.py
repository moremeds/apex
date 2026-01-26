"""
Turning Point Detection CSS styles.
"""

from __future__ import annotations

TURNING_POINT_STYLES = """
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
