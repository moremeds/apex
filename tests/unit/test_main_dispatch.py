"""Tests for main.py argument parsing."""

from __future__ import annotations

from unittest.mock import patch


def test_parse_args_default_runs_api():
    """With no --mode, main() runs the API server (the production image's `python main.py`)."""
    from main import parse_args

    with patch("sys.argv", ["main.py"]):
        args = parse_args()
    assert args.mode is None


def test_parse_args_backward_compat_mode():
    """--mode backtest still parses for backward compat."""
    from main import parse_args

    with patch(
        "sys.argv",
        [
            "main.py",
            "--mode",
            "backtest",
            "--strategy",
            "trend_pulse",
            "--symbols",
            "SPY",
            "--start",
            "2025-01-01",
            "--end",
            "2025-06-30",
        ],
    ):
        args = parse_args()
    assert args.mode == "backtest"
    assert args.strategy == "trend_pulse"
