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


def test_main_without_mode_delegates_to_api_server():
    """The image's `python main.py` must boot exactly like `python -m src.api.server`."""
    import main as entry

    with patch("sys.argv", ["main.py"]), patch("src.api.server.main") as server_main:
        entry.main()
    server_main.assert_called_once_with()


def test_api_server_runs_one_process_despite_web_concurrency(monkeypatch):
    """app.state holds per-process hub/subscription/job state; uvicorn must not fork workers."""
    from src.api import server

    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    with patch("uvicorn.run") as run:
        server.main()
    assert run.call_args.kwargs["workers"] == 1
