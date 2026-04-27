"""Tests for main.py service dispatch."""

from __future__ import annotations

from unittest.mock import patch


def test_parse_args_service_flag():
    """--service flag parses correctly."""
    from main import parse_args

    with patch("sys.argv", ["main.py", "--service", "api"]):
        args = parse_args()
    assert args.service == "api"


def test_parse_args_service_default():
    """Default --service is 'all'."""
    from main import parse_args

    with patch("sys.argv", ["main.py"]):
        args = parse_args()
    assert args.service == "all"


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
