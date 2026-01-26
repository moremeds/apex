"""
Unit tests for PackageBuilder and SnapshotBuilder (PR-02).

Tests the directory-based package format with lazy loading.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.infrastructure.reporting.package import (
    PACKAGE_FORMAT_VERSION,
    PackageBuilder,
    PackageManifest,
)
from src.infrastructure.reporting.snapshot_builder import (
    SNAPSHOT_VERSION,
    SnapshotBuilder,
    SnapshotDiff,
)


@pytest.fixture
def sample_data() -> Dict[Tuple[str, str], pd.DataFrame]:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    data = {}
    for symbol in ["AAPL", "SPY"]:
        df = pd.DataFrame(
            {
                "open": [100 + i for i in range(100)],
                "high": [105 + i for i in range(100)],
                "low": [95 + i for i in range(100)],
                "close": [102 + i for i in range(100)],
                "volume": [1000000 + i * 1000 for i in range(100)],
            },
            index=dates,
        )
        data[(symbol, "1d")] = df

    return data


@pytest.fixture
def sample_data_with_indicators() -> Dict[Tuple[str, str], pd.DataFrame]:
    """Create sample OHLCV data with indicator columns for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    data = {}
    for symbol in ["AAPL", "SPY"]:
        df = pd.DataFrame(
            {
                "open": [100 + i for i in range(100)],
                "high": [105 + i for i in range(100)],
                "low": [95 + i for i in range(100)],
                "close": [102 + i for i in range(100)],
                "volume": [1000000 + i * 1000 for i in range(100)],
                # Overlay indicators
                "bollinger_bb_upper": [108 + i for i in range(100)],
                "bollinger_bb_middle": [103 + i for i in range(100)],
                "bollinger_bb_lower": [98 + i for i in range(100)],
                "supertrend_supertrend": [101 + i for i in range(100)],
                # RSI
                "rsi_rsi": [50 + (i % 30) for i in range(100)],
                # MACD
                "macd_macd": [0.5 + (i % 10) * 0.1 for i in range(100)],
                "macd_signal": [0.3 + (i % 10) * 0.1 for i in range(100)],
                "macd_histogram": [0.2 + (i % 5) * 0.1 for i in range(100)],
            },
            index=dates,
        )
        data[(symbol, "1d")] = df

    return data


@pytest.fixture
def mock_regime_outputs() -> Dict[str, MagicMock]:
    """Create mock regime outputs for testing."""
    outputs = {}

    for symbol in ["AAPL", "SPY"]:
        mock = MagicMock()
        mock.final_regime.value = "R0"
        mock.regime_name = "Healthy Uptrend"
        mock.confidence = 85
        mock.regime_changed = False
        mock.component_states.trend_state.value = "trend_up"
        mock.component_states.vol_state.value = "vol_normal"
        mock.component_states.chop_state.value = "trending"
        mock.component_states.ext_state.value = "neutral"
        mock.component_states.to_dict.return_value = {
            "trend_state": "trend_up",
            "vol_state": "vol_normal",
            "chop_state": "trending",
            "ext_state": "neutral",
        }
        mock.component_values.close = 200.0
        mock.component_values.atr_pct_63 = 45.0
        mock.component_values.chop_pct_252 = 38.0
        mock.component_values.ext = 0.5
        mock.turning_point = None

        # Mock to_dict() for _build_ticker_summary
        mock.to_dict.return_value = {
            "final_regime": "R0",
            "regime_name": "Healthy Uptrend",
            "confidence": 85,
            "regime_changed": False,
            "decision_regime": "R0",
            "component_states": {
                "trend_state": "trend_up",
                "vol_state": "vol_normal",
                "chop_state": "trending",
                "ext_state": "neutral",
                "iv_state": "na",
            },
            "component_values": {
                "close": 200.0,
                "ma20": 195.0,
                "ma50": 190.0,
                "ma200": 180.0,
                "atr_pct_63": 45.0,
                "chop_pct_252": 38.0,
                "ext": 0.5,
            },
            "derived_metrics": {},
            "transition": {
                "bars_in_current": 10,
                "pending_count": 0,
                "entry_threshold": 3,
            },
            "quality": {
                "warmup_ok": True,
                "warmup_bars_available": 300,
                "warmup_bars_needed": 252,
                "fallback_active": False,
                "fallback_reason": None,
            },
            "rules_fired_decision": [],
            "turning_point": None,
            "data_window": {"start": "2024-01-01", "end": "2024-04-09"},
            "asof_ts": "2024-04-09T16:00:00",
        }
        outputs[symbol] = mock

    return outputs


@pytest.fixture
def temp_dir() -> None:
    """Create temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


class TestPackageBuilder:
    """Tests for PackageBuilder."""

    def test_init_default_theme(self) -> None:
        """Test default theme initialization."""
        builder = PackageBuilder()
        assert builder.theme == "dark"

    def test_init_light_theme(self) -> None:
        """Test light theme initialization."""
        builder = PackageBuilder(theme="light")
        assert builder.theme == "light"

    def test_build_creates_directory_structure(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that build creates the expected directory structure."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        manifest = builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        # Verify manifest is returned
        assert manifest.version is not None
        # Verify directory structure
        assert output_dir.exists()
        assert (output_dir / "report.html").exists()  # Main report (heatmap would be index.html)
        assert (output_dir / "assets").is_dir()
        assert (output_dir / "assets" / "styles.css").exists()
        assert (output_dir / "assets" / "app.js").exists()
        assert (output_dir / "data").is_dir()
        assert (output_dir / "data" / "summary.json").exists()
        assert (output_dir / "snapshots").is_dir()
        assert (output_dir / "snapshots" / "payload_snapshot.json").exists()
        assert (output_dir / "manifest.json").exists()

    def test_build_creates_data_files(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that per-symbol data files are created."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        # Check data files exist
        assert (output_dir / "data" / "AAPL_1d.json").exists()
        assert (output_dir / "data" / "SPY_1d.json").exists()

        # Verify data file content
        with open(output_dir / "data" / "AAPL_1d.json") as f:
            data = json.load(f)
            assert data["symbol"] == "AAPL"
            assert data["timeframe"] == "1d"
            assert data["bar_count"] == 100
            assert "chart_data" in data

    def test_build_summary_contains_symbols(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that summary.json contains symbol list."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        with open(output_dir / "data" / "summary.json") as f:
            summary = json.load(f)
            assert "AAPL" in summary["symbols"]
            assert "SPY" in summary["symbols"]
            assert summary["symbol_count"] == 2
            assert summary["timeframe_count"] == 1

    def test_build_manifest_version(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that manifest contains correct version."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
            assert manifest["version"] == PACKAGE_FORMAT_VERSION
            assert manifest["total_data_files"] == 2

    def test_build_with_regime_outputs(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        mock_regime_outputs: Dict[str, MagicMock],
        temp_dir: Path,
    ) -> None:
        """Test building with regime outputs."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
            regime_outputs=mock_regime_outputs,
        )

        # Verify summary includes regime data
        with open(output_dir / "data" / "summary.json") as f:
            summary = json.load(f)
            assert "tickers" in summary
            # Check AAPL ticker has regime info
            aapl_ticker = next(t for t in summary["tickers"] if t["symbol"] == "AAPL")
            assert aapl_ticker["regime"] == "R0"
            assert aapl_ticker["confidence"] == 85

    def test_build_index_html_valid(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that report.html is valid HTML."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        html = (output_dir / "report.html").read_text()
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert 'href="assets/styles.css"' in html
        assert 'src="assets/app.js"' in html

    def test_chart_data_includes_structured_indicators(
        self,
        sample_data_with_indicators: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that per-symbol JSON includes structured indicator data for overlays."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data_with_indicators,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        # Verify data file content structure
        with open(output_dir / "data" / "AAPL_1d.json") as f:
            data = json.load(f)
            chart_data = data["chart_data"]

            # Check structured indicator buckets exist
            assert "overlays" in chart_data
            assert "rsi" in chart_data
            assert "macd" in chart_data

            # Check Bollinger Bands are in overlays
            assert "bollinger_bb_upper" in chart_data["overlays"]
            assert "bollinger_bb_middle" in chart_data["overlays"]
            assert "bollinger_bb_lower" in chart_data["overlays"]
            assert "supertrend_supertrend" in chart_data["overlays"]

            # Check RSI is in rsi bucket
            assert "rsi_rsi" in chart_data["rsi"]

            # Check MACD components are in macd bucket
            assert "macd_macd" in chart_data["macd"]
            assert "macd_signal" in chart_data["macd"]
            assert "macd_histogram" in chart_data["macd"]

    def test_summary_includes_confluence(
        self,
        sample_data_with_indicators: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that summary.json includes confluence data."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data_with_indicators,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        with open(output_dir / "data" / "summary.json") as f:
            summary = json.load(f)

            # Confluence section should exist
            assert "confluence" in summary

            # Check structure for AAPL_1d if confluence was calculated
            # Note: May be empty if no indicator states derived
            confluence = summary["confluence"]
            assert isinstance(confluence, dict)

    def test_summary_includes_full_regime(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        mock_regime_outputs: Dict[str, MagicMock],
        temp_dir: Path,
    ) -> None:
        """Test that summary.json includes full regime info (regime, confidence, components)."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
            regime_outputs=mock_regime_outputs,
        )

        with open(output_dir / "data" / "summary.json") as f:
            summary = json.load(f)
            tickers = summary["tickers"]

            # Find AAPL ticker
            aapl_ticker = next(t for t in tickers if t["symbol"] == "AAPL")

            # Check full regime data is present
            assert "regime" in aapl_ticker
            assert aapl_ticker["regime"] == "R0"

            assert "confidence" in aapl_ticker
            assert aapl_ticker["confidence"] == 85

            assert "component_states" in aapl_ticker
            components = aapl_ticker["component_states"]
            assert "trend_state" in components
            assert "vol_state" in components
            assert "chop_state" in components
            assert "ext_state" in components

    def test_javascript_contains_required_functions(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that app.js contains required rendering functions."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        js_content = (output_dir / "assets" / "app.js").read_text()

        # Check for required functions (PR-02 feature parity)
        assert "updateRegimeSection" in js_content
        assert "updateSignalHistoryTable" in js_content
        assert "updateConfluencePanel" in js_content
        assert "renderMainChart" in js_content

        # Check for autorange in yaxis config (critical fix)
        assert "autorange: true" in js_content

        # Check for multi-subplot layout
        assert "yaxis2" in js_content  # RSI
        assert "yaxis3" in js_content  # MACD
        assert "yaxis4" in js_content  # Volume

    def test_html_contains_all_sections(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that report.html contains all required sections."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        html = (output_dir / "report.html").read_text()

        # Check for confluence section
        assert "confluence-content" in html
        assert "Confluence Analysis" in html

        # Check for regime section
        assert "regime-content" in html
        assert "Regime Analysis" in html

        # Check for signal history section
        assert "signals-content" in html
        assert "Signal History" in html

    def test_css_contains_all_styles(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        temp_dir: Path,
    ) -> None:
        """Test that styles.css contains styles for all sections."""
        builder = PackageBuilder()
        output_dir = temp_dir / "test_package"

        builder.build(
            data=sample_data,
            indicators=[],
            rules=[],
            output_dir=output_dir,
        )

        css_content = (output_dir / "assets" / "styles.css").read_text()

        # Check for confluence styles
        assert ".confluence-panel" in css_content
        assert ".alignment-meter" in css_content
        assert ".alignment-bar" in css_content

        # Check for regime styles
        assert ".regime-dashboard" in css_content
        assert ".regime-badge" in css_content

        # Check for signal table styles
        assert ".signal-table" in css_content
        assert ".signal-badge" in css_content


class TestSnapshotBuilder:
    """Tests for SnapshotBuilder."""

    def test_build_basic_snapshot(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
    ) -> None:
        """Test building a basic snapshot."""
        builder = SnapshotBuilder()
        snapshot = builder.build(
            data=sample_data,
            regime_outputs={},
            symbols=["AAPL", "SPY"],
            timeframes=["1d"],
        )

        assert snapshot["version"] == SNAPSHOT_VERSION
        assert "created_at" in snapshot
        assert snapshot["inventory"]["symbols"] == ["AAPL", "SPY"]
        assert snapshot["inventory"]["timeframes"] == ["1d"]

    def test_build_with_regime_outputs(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
        mock_regime_outputs: Dict[str, MagicMock],
    ) -> None:
        """Test snapshot includes regime data."""
        builder = SnapshotBuilder()
        snapshot = builder.build(
            data=sample_data,
            regime_outputs=mock_regime_outputs,
            symbols=["AAPL", "SPY"],
            timeframes=["1d"],
        )

        assert "AAPL" in snapshot["regimes"]
        assert snapshot["regimes"]["AAPL"]["regime"] == "R0"
        assert snapshot["regimes"]["AAPL"]["confidence"] == 85

    def test_build_bar_counts(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
    ) -> None:
        """Test snapshot includes bar counts."""
        builder = SnapshotBuilder()
        snapshot = builder.build(
            data=sample_data,
            regime_outputs={},
            symbols=["AAPL", "SPY"],
            timeframes=["1d"],
        )

        assert snapshot["bar_counts"]["AAPL_1d"] == 100
        assert snapshot["bar_counts"]["SPY_1d"] == 100

    def test_build_content_hash(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
    ) -> None:
        """Test snapshot has content hash."""
        builder = SnapshotBuilder()
        snapshot = builder.build(
            data=sample_data,
            regime_outputs={},
            symbols=["AAPL", "SPY"],
            timeframes=["1d"],
        )

        assert "content_hash" in snapshot
        assert len(snapshot["content_hash"]) == 16  # SHA256 truncated to 16 chars

    def test_quick_compare_identical(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
    ) -> None:
        """Test quick compare returns True for identical snapshots."""
        builder = SnapshotBuilder()
        snapshot1 = builder.build(
            data=sample_data,
            regime_outputs={},
            symbols=["AAPL", "SPY"],
            timeframes=["1d"],
        )
        # Build again with same data
        snapshot2 = builder.build(
            data=sample_data,
            regime_outputs={},
            symbols=["AAPL", "SPY"],
            timeframes=["1d"],
        )

        assert builder.quick_compare(snapshot1, snapshot2)


class TestSnapshotDiff:
    """Tests for SnapshotDiff."""

    def test_diff_no_changes(
        self,
        sample_data: Dict[Tuple[str, str], pd.DataFrame],
    ) -> None:
        """Test diff with no changes."""
        builder = SnapshotBuilder()
        snapshot1 = builder.build(
            data=sample_data,
            regime_outputs={},
            symbols=["AAPL", "SPY"],
            timeframes=["1d"],
        )
        snapshot2 = builder.build(
            data=sample_data,
            regime_outputs={},
            symbols=["AAPL", "SPY"],
            timeframes=["1d"],
        )

        diff = builder.diff(snapshot1, snapshot2)
        assert not diff.has_changes

    def test_diff_symbol_added(self) -> None:
        """Test diff detects added symbol."""
        builder = SnapshotBuilder()

        old_snapshot = {
            "inventory": {"symbols": ["AAPL"]},
            "regimes": {},
            "metrics": {},
            "bar_counts": {},
        }
        new_snapshot = {
            "inventory": {"symbols": ["AAPL", "SPY"]},
            "regimes": {},
            "metrics": {},
            "bar_counts": {},
        }

        diff = builder.diff(old_snapshot, new_snapshot)
        assert diff.has_changes
        assert "SPY" in diff.symbols_added

    def test_diff_symbol_removed(self) -> None:
        """Test diff detects removed symbol."""
        builder = SnapshotBuilder()

        old_snapshot = {
            "inventory": {"symbols": ["AAPL", "SPY"]},
            "regimes": {},
            "metrics": {},
            "bar_counts": {},
        }
        new_snapshot = {
            "inventory": {"symbols": ["AAPL"]},
            "regimes": {},
            "metrics": {},
            "bar_counts": {},
        }

        diff = builder.diff(old_snapshot, new_snapshot)
        assert diff.has_changes
        assert "SPY" in diff.symbols_removed

    def test_diff_regime_change(self) -> None:
        """Test diff detects regime change."""
        builder = SnapshotBuilder()

        old_snapshot = {
            "inventory": {"symbols": ["AAPL"]},
            "regimes": {"AAPL": {"regime": "R0"}},
            "metrics": {},
            "bar_counts": {},
        }
        new_snapshot = {
            "inventory": {"symbols": ["AAPL"]},
            "regimes": {"AAPL": {"regime": "R2"}},
            "metrics": {},
            "bar_counts": {},
        }

        diff = builder.diff(old_snapshot, new_snapshot)
        assert diff.has_changes
        assert "AAPL" in diff.regime_changes
        assert diff.regime_changes["AAPL"]["old"] == "R0"
        assert diff.regime_changes["AAPL"]["new"] == "R2"

    def test_diff_bar_count_change(self) -> None:
        """Test diff detects bar count change."""
        builder = SnapshotBuilder()

        old_snapshot = {
            "inventory": {"symbols": ["AAPL"]},
            "regimes": {},
            "metrics": {},
            "bar_counts": {"AAPL_1d": 100},
        }
        new_snapshot = {
            "inventory": {"symbols": ["AAPL"]},
            "regimes": {},
            "metrics": {},
            "bar_counts": {"AAPL_1d": 101},
        }

        diff = builder.diff(old_snapshot, new_snapshot)
        assert diff.has_changes
        assert "AAPL_1d" in diff.bar_count_changes

    def test_diff_summary(self) -> None:
        """Test diff summary output."""
        diff = SnapshotDiff(
            symbols_added=("SPY",),
            symbols_removed=(),
            regime_changes={"AAPL": {"old": "R0", "new": "R2"}},
            metric_changes={},
            bar_count_changes={},
        )

        summary = diff.summary()
        assert "SPY" in summary
        assert "AAPL" in summary
        assert "R0" in summary
        assert "R2" in summary


class TestPackageManifest:
    """Tests for PackageManifest."""

    def test_manifest_to_dict(self) -> None:
        """Test manifest serialization."""
        manifest = PackageManifest(
            version="1.0",
            created_at="2024-01-01T00:00:00",
            symbols=("AAPL", "SPY"),
            timeframes=("1d",),
            total_data_files=2,
            summary_size_kb=50.5,
            theme="dark",
        )

        data = manifest.to_dict()
        assert data["version"] == "1.0"
        assert data["symbols"] == ["AAPL", "SPY"]
        assert data["total_data_files"] == 2
        assert data["summary_size_kb"] == 50.5
