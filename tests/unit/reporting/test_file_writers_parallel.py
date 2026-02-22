"""Tests for parallel file writes in package builder."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd

from src.infrastructure.reporting.package.file_writers import (
    _write_one_data_file,
    write_data_files,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Patch target for the lazy import of get_strategy_params inside write_data_files.
# Because it uses `from src.domain.strategy.param_loader import get_strategy_params`
# at call time, we must patch the canonical location.
_GET_STRATEGY_PARAMS = "src.domain.strategy.param_loader.get_strategy_params"


def _make_test_df(n_bars: int = 200) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with DatetimeIndex."""
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(100, 100 + n_bars),
            "high": range(101, 101 + n_bars),
            "low": range(99, 99 + n_bars),
            "close": range(100, 100 + n_bars),
            "volume": [1_000_000] * n_bars,
        },
        index=dates,
    )


def _make_test_data(
    symbols: List[str] | None = None,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Build a {(symbol, '1d'): df} dict for testing."""
    if symbols is None:
        symbols = ["AAPL", "SPY", "MSFT"]
    return {(sym, "1d"): _make_test_df() for sym in symbols}


def _strip_generated_at(file_path: Path) -> dict:
    """Load JSON, drop the volatile generated_at field, return the dict."""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    data.pop("generated_at", None)
    return data


def _noop(*args, **kwargs):  # type: ignore[no-untyped-def]
    return []


def _patch_history_fns():
    """Context manager that stubs out the four heavy indicator history functions."""
    return patch.multiple(
        "src.infrastructure.reporting.package.file_writers",
        _compute_dual_macd_history_for_key=_noop,
        _compute_trend_pulse_history_for_key=_noop,
        _compute_regime_flex_history=_noop,
        _compute_sector_pulse_history=_noop,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSequentialWrites:
    """write_data_files with max_workers=1 (sequential path)."""

    def test_sequential_writes_files(self) -> None:
        """Sequential mode creates one JSON file per (symbol, timeframe) key."""
        data = _make_test_data(["AAPL", "SPY"])

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with _patch_history_fns():
                keys = write_data_files(
                    data=data,
                    indicators=[],
                    rules=[],
                    data_dir=data_dir,
                    max_workers=1,
                )

            assert sorted(keys) == ["AAPL_1d", "SPY_1d"]
            assert (data_dir / "AAPL_1d.json").exists()
            assert (data_dir / "SPY_1d.json").exists()


class TestParallelWrites:
    """write_data_files with max_workers>1 (parallel path)."""

    def test_parallel_writes_files(self) -> None:
        """Parallel mode creates the same set of files as sequential."""
        data = _make_test_data(["AAPL", "SPY"])

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with _patch_history_fns(), patch(_GET_STRATEGY_PARAMS, return_value={}):
                keys = write_data_files(
                    data=data,
                    indicators=[],
                    rules=[],
                    data_dir=data_dir,
                    max_workers=4,
                )

            assert sorted(keys) == ["AAPL_1d", "SPY_1d"]
            assert (data_dir / "AAPL_1d.json").exists()
            assert (data_dir / "SPY_1d.json").exists()


class TestSequentialParallelParity:
    """Sequential and parallel paths must produce byte-identical JSON (except generated_at)."""

    def test_sequential_parallel_parity(self) -> None:
        data = _make_test_data(["AAPL", "SPY", "MSFT"])

        seq_dir = tempfile.mkdtemp()
        par_dir = tempfile.mkdtemp()

        # Sequential run
        with _patch_history_fns():
            seq_keys = write_data_files(
                data=data,
                indicators=[],
                rules=[],
                data_dir=Path(seq_dir),
                max_workers=1,
            )

        # Parallel run
        with _patch_history_fns(), patch(_GET_STRATEGY_PARAMS, return_value={}):
            par_keys = write_data_files(
                data=data,
                indicators=[],
                rules=[],
                data_dir=Path(par_dir),
                max_workers=4,
            )

        assert sorted(seq_keys) == sorted(par_keys)

        for key in seq_keys:
            seq_data = _strip_generated_at(Path(seq_dir) / f"{key}.json")
            par_data = _strip_generated_at(Path(par_dir) / f"{key}.json")
            assert seq_data == par_data, f"Parity mismatch for {key}"


class TestJsonFormat:
    """Verify that data files are compact JSON (no indent)."""

    def test_json_no_indent(self) -> None:
        """Output JSON must be single-line (no indent=2 formatting)."""
        data = _make_test_data(["AAPL"])

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with _patch_history_fns():
                write_data_files(
                    data=data,
                    indicators=[],
                    rules=[],
                    data_dir=data_dir,
                    max_workers=1,
                )

            raw = (data_dir / "AAPL_1d.json").read_text(encoding="utf-8")

            # Compact JSON is a single line (no indent whitespace).
            lines = raw.strip().split("\n")
            assert len(lines) == 1, f"Expected single-line compact JSON, got {len(lines)} lines"

            # Round-trip parse sanity check.
            parsed = json.loads(raw)
            assert parsed["symbol"] == "AAPL"
            assert parsed["timeframe"] == "1d"


class TestParallelErrorHandling:
    """Parallel write failures raise RuntimeError (fail-fast for CI)."""

    def test_parallel_error_raises(self) -> None:
        """When any file write fails in parallel mode, RuntimeError is raised."""
        import pytest

        data = _make_test_data(["AAPL", "BAD", "SPY"])

        original_write_one = _write_one_data_file

        def _failing_write_one(symbol, timeframe, df, rules, data_dir, tz):
            if symbol == "BAD":
                raise RuntimeError("Simulated write failure for BAD")
            return original_write_one(symbol, timeframe, df, rules, data_dir, tz)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with (
                _patch_history_fns(),
                patch(
                    "src.infrastructure.reporting.package.file_writers._write_one_data_file",
                    side_effect=_failing_write_one,
                ),
                patch(_GET_STRATEGY_PARAMS, return_value={}),
                pytest.raises(RuntimeError, match="Failed to write 1 data file"),
            ):
                write_data_files(
                    data=data,
                    indicators=[],
                    rules=[],
                    data_dir=data_dir,
                    max_workers=4,
                )


class TestParallelPrewarmsStrategyCache:
    """Parallel mode must pre-warm get_strategy_params before spawning threads."""

    def test_parallel_prewarms_strategy_cache(self) -> None:
        data = _make_test_data(["AAPL"])

        mock_get_params = MagicMock(return_value={})

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with _patch_history_fns(), patch(_GET_STRATEGY_PARAMS, mock_get_params):
                write_data_files(
                    data=data,
                    indicators=[],
                    rules=[],
                    data_dir=data_dir,
                    max_workers=4,
                )

        # The pre-warm loop calls get_strategy_params for each of 4 strategies.
        expected_names = {"trend_pulse", "regime_flex", "sector_pulse", "rsi_mean_reversion"}
        called_names = {call.args[0] for call in mock_get_params.call_args_list}
        assert expected_names.issubset(
            called_names
        ), f"Expected pre-warm for {expected_names}, got {called_names}"

    def test_sequential_does_not_prewarm(self) -> None:
        """Sequential mode (max_workers=1) should NOT call get_strategy_params for pre-warming."""
        data = _make_test_data(["AAPL"])

        mock_get_params = MagicMock(return_value={})

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with _patch_history_fns(), patch(_GET_STRATEGY_PARAMS, mock_get_params):
                write_data_files(
                    data=data,
                    indicators=[],
                    rules=[],
                    data_dir=data_dir,
                    max_workers=1,
                )

        # Sequential path never enters the pre-warm block.
        mock_get_params.assert_not_called()


class TestWriteOneDataFile:
    """Direct unit tests for the per-file writer."""

    def test_write_one_data_file_creates_json(self) -> None:
        df = _make_test_df(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with _patch_history_fns():
                key = _write_one_data_file("TEST", "1d", df, [], data_dir, "US/Eastern")

            assert key == "TEST_1d"
            fp = data_dir / "TEST_1d.json"
            assert fp.exists()

            payload = json.loads(fp.read_text(encoding="utf-8"))
            assert payload["symbol"] == "TEST"
            assert payload["timeframe"] == "1d"
            assert payload["bar_count"] == 50
            assert "generated_at" in payload
            assert isinstance(payload["chart_data"], dict)

    def test_write_one_data_file_schema(self) -> None:
        """All expected top-level keys are present."""
        df = _make_test_df(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            with _patch_history_fns():
                _write_one_data_file("QQQ", "1d", df, [], data_dir, "US/Eastern")

            payload = json.loads((data_dir / "QQQ_1d.json").read_text(encoding="utf-8"))
            expected_keys = {
                "symbol",
                "timeframe",
                "generated_at",
                "bar_count",
                "chart_data",
                "signals",
                "dual_macd_history",
                "trend_pulse_history",
                "regime_flex_history",
                "sector_pulse_history",
            }
            assert set(payload.keys()) == expected_keys
