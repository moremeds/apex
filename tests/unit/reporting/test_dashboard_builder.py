"""Tests for the Cloudflare dashboard builder and data transformer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.infrastructure.reporting.dashboard.builder import DashboardBuilder
from src.infrastructure.reporting.dashboard.data_transformer import (
    DataTransformer,
    TransformConfig,
    _trim_symbol_data,
)


@pytest.fixture
def sample_symbol_json() -> Dict[str, Any]:
    """600-bar per-symbol JSON matching file_writers output."""
    n = 600
    timestamps = [f"2024-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}T16:00:00" for i in range(n)]
    return {
        "symbol": "AAPL",
        "timeframe": "1d",
        "generated_at": "2024-12-01T16:00:00",
        "bar_count": n,
        "chart_data": {
            "timestamps": timestamps,
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [105.0 + i * 0.1 for i in range(n)],
            "low": [95.0 + i * 0.1 for i in range(n)],
            "close": [102.0 + i * 0.1 for i in range(n)],
            "volume": [1_000_000] * n,
            "overlays": {
                "ema_ema_fast": [None] * 20 + [100.0 + i * 0.1 for i in range(n - 20)],
                "ema_ema_slow": [None] * 50 + [100.0 + i * 0.1 for i in range(n - 50)],
            },
            "rsi": {
                "rsi_rsi": [None] * 14 + [50.0 + (i % 40) for i in range(n - 14)],
            },
            "oscillators": {
                "adx_adx": [None] * 28 + [20.0 + (i % 30) for i in range(n - 28)],
            },
        },
        "signals": [
            {
                "timestamp": timestamps[i],
                "rule": "macd_bullish_cross",
                "direction": "buy",
                "indicator": "macd",
                "value": 1.5,
            }
            for i in range(0, n, 20)  # 30 signals spread across bars
        ],
        "dual_macd_history": [
            {"date": f"2024-{12 - i // 30:02d}-{28 - i % 28:02d}", "trend_state": "IMPROVING"}
            for i in range(100)
        ],
        "trend_pulse_history": [
            {"date": f"2024-{12 - i // 30:02d}-{28 - i % 28:02d}", "swing_signal": "NONE"}
            for i in range(80)
        ],
        "regime_flex_history": [
            {
                "date": f"2024-{12 - i // 30:02d}-{28 - i % 28:02d}",
                "regime": "R0",
                "target_exposure": 100.0,
                "signal": "HOLD",
            }
            for i in range(60)
        ],
        "sector_pulse_history": [
            {"date": f"2024-{12 - i // 30:02d}-{28 - i % 28:02d}", "signal": "HOLD"}
            for i in range(40)
        ],
    }


@pytest.fixture
def source_dir(tmp_path: Path, sample_symbol_json: Dict, sample_summary_data: Dict) -> Path:
    """Set up a minimal source directory mimicking out/signals/."""
    src = tmp_path / "signals"
    data_dir = src / "data"
    data_dir.mkdir(parents=True)

    # Write manifest
    manifest = {"version": "1.0", "symbols": ["AAPL", "SPY"], "timeframes": ["1d"]}
    (src / "manifest.json").write_text(json.dumps(manifest))

    # Write summary
    (data_dir / "summary.json").write_text(json.dumps(sample_summary_data))

    # Write per-symbol JSONs
    (data_dir / "AAPL_1d.json").write_text(json.dumps(sample_symbol_json))

    spy = sample_symbol_json.copy()
    spy["symbol"] = "SPY"
    (data_dir / "SPY_1d.json").write_text(json.dumps(spy))

    # Write score_history
    (data_dir / "score_history.json").write_text(json.dumps({"snapshots": []}))

    # Write indicators
    (data_dir / "indicators.json").write_text(json.dumps({"indicators": []}))

    return src


@pytest.fixture
def source_dir_with_regime(source_dir: Path) -> Path:
    """Source directory with regime HTML files."""
    regime_dir = source_dir / "data" / "regime"
    regime_dir.mkdir()
    (regime_dir / "SPY_1d.html").write_text("<div>SPY regime</div>")
    (regime_dir / "AAPL_1d.html").write_text("<div>AAPL regime</div>")
    return source_dir


# =============================================================================
# Test Cases
# =============================================================================


class TestImportSmoke:
    def test_importable(self):
        """DashboardBuilder is importable from the package."""
        from src.infrastructure.reporting.dashboard import DashboardBuilder

        assert DashboardBuilder is not None


class TestBuilderValidation:
    def test_fails_fast_if_source_missing(self, tmp_path: Path):
        """Builder raises FileNotFoundError if source_dir doesn't exist."""
        builder = DashboardBuilder(source_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="Source directory not found"):
            builder.build(output_dir=tmp_path / "out")

    def test_fails_fast_if_data_missing(self, tmp_path: Path):
        """Builder raises FileNotFoundError if source data dir is missing."""
        src = tmp_path / "signals"
        src.mkdir()
        builder = DashboardBuilder(source_dir=src)
        with pytest.raises(FileNotFoundError, match="Source data directory not found"):
            builder.build(output_dir=tmp_path / "out")


class TestBarTrimming:
    def test_trims_to_500_bars_default(self, sample_symbol_json: Dict):
        """600-bar input trimmed to 500 bars (new default)."""
        config = TransformConfig(source_dir=Path("."))
        result = _trim_symbol_data(sample_symbol_json, config)

        assert config.max_bars == 500
        assert result["bar_count"] == 500
        assert len(result["chart_data"]["timestamps"]) == 500

    def test_trims_to_custom_limit(self, sample_symbol_json: Dict):
        """600-bar input trimmed to custom limit."""
        config = TransformConfig(max_bars=200, source_dir=Path("."))
        result = _trim_symbol_data(sample_symbol_json, config)

        assert result["bar_count"] == 200
        assert len(result["chart_data"]["timestamps"]) == 200
        assert len(result["chart_data"]["open"]) == 200
        assert len(result["chart_data"]["close"]) == 200
        assert len(result["chart_data"]["volume"]) == 200

    def test_parallel_arrays_aligned(self, sample_symbol_json: Dict):
        """All parallel arrays in chart_data have the same length after trim."""
        config = TransformConfig(max_bars=200, source_dir=Path("."))
        result = _trim_symbol_data(sample_symbol_json, config)

        expected = 200
        cd = result["chart_data"]
        for key in ("timestamps", "open", "high", "low", "close", "volume"):
            assert len(cd[key]) == expected, f"{key} length mismatch"

        for group_key in ("overlays", "rsi", "oscillators"):
            for ind_key, arr in cd.get(group_key, {}).items():
                assert len(arr) == expected, f"{group_key}.{ind_key} length mismatch"

    def test_no_trim_if_under_limit(self, sample_symbol_json: Dict):
        """Data under max_bars is not modified."""
        config = TransformConfig(max_bars=1000, source_dir=Path("."))
        result = _trim_symbol_data(sample_symbol_json, config)
        assert result["bar_count"] == 600


class TestSignalFiltering:
    def test_signals_outside_window_dropped(self, sample_symbol_json: Dict):
        """Signals before the trimmed window are removed."""
        config = TransformConfig(max_bars=200, source_dir=Path("."))
        result = _trim_symbol_data(sample_symbol_json, config)

        # Original has 30 signals; after trim only those in last 200 bars remain
        cutoff = result["chart_data"]["timestamps"][0]
        for sig in result["signals"]:
            assert sig["timestamp"] >= cutoff

    def test_signals_inside_window_kept(self, sample_symbol_json: Dict):
        """Signals within the trimmed window are preserved."""
        config = TransformConfig(max_bars=200, source_dir=Path("."))
        result = _trim_symbol_data(sample_symbol_json, config)
        assert len(result["signals"]) > 0


class TestHistoryCapping:
    def test_history_capped_at_60_default(self, sample_symbol_json: Dict):
        """Strategy histories exceeding max (60 default) are capped."""
        config = TransformConfig(source_dir=Path("."))
        result = _trim_symbol_data(sample_symbol_json, config)

        assert config.max_history_entries == 60
        assert len(result["dual_macd_history"]) == 60  # was 100
        assert len(result["trend_pulse_history"]) == 60  # was 80
        assert len(result["regime_flex_history"]) == 60  # was 60, exactly at cap
        assert len(result["sector_pulse_history"]) == 40  # was 40, under cap

    def test_history_capped_custom(self, sample_symbol_json: Dict):
        """Strategy histories capped at custom limit."""
        config = TransformConfig(max_bars=600, max_history_entries=50, source_dir=Path("."))
        result = _trim_symbol_data(sample_symbol_json, config)

        assert len(result["dual_macd_history"]) == 50
        assert len(result["trend_pulse_history"]) == 50
        assert len(result["regime_flex_history"]) == 50
        assert len(result["sector_pulse_history"]) == 40  # under cap


class TestRegimeHTMLCopy:
    def test_regime_html_copied(self, source_dir_with_regime: Path, tmp_path: Path):
        """Regime pre-rendered HTML files are copied to output."""
        config = TransformConfig(source_dir=source_dir_with_regime)
        output = tmp_path / "out"
        DataTransformer(config).transform(output)

        regime_dir = output / "data" / "regime"
        assert regime_dir.is_dir()
        assert (regime_dir / "SPY_1d.html").exists()
        assert (regime_dir / "AAPL_1d.html").exists()

    def test_no_regime_dir_no_error(self, source_dir: Path, tmp_path: Path):
        """Missing regime dir doesn't cause an error."""
        config = TransformConfig(source_dir=source_dir)
        output = tmp_path / "out"
        # Should not raise
        DataTransformer(config).transform(output)
        assert not (output / "data" / "regime").exists()


class TestScreenerMerge:
    def test_merge_with_both_sources(self, tmp_path: Path, source_dir: Path):
        """Screener merge produces both momentum and pead sections."""
        mom_path = tmp_path / "momentum.json"
        pead_path = tmp_path / "pead.json"
        mom_path.write_text(json.dumps({"candidates": [{"symbol": "KLAC"}]}))
        pead_path.write_text(json.dumps({"candidates": []}))

        config = TransformConfig(
            source_dir=source_dir,
            momentum_path=mom_path,
            pead_path=pead_path,
        )
        output = tmp_path / "out"
        DataTransformer(config).transform(output)

        screeners = json.loads((output / "data" / "screeners.json").read_text())
        assert screeners["momentum"] is not None
        assert screeners["momentum"]["candidates"][0]["symbol"] == "KLAC"
        assert screeners["pead"] is not None

    def test_missing_source_gives_null(self, source_dir: Path, tmp_path: Path):
        """Missing screener files produce null sections, no crash."""
        config = TransformConfig(
            source_dir=source_dir,
            momentum_path=tmp_path / "nonexistent.json",
            pead_path=tmp_path / "also_nonexistent.json",
        )
        output = tmp_path / "out"
        DataTransformer(config).transform(output)

        screeners = json.loads((output / "data" / "screeners.json").read_text())
        assert screeners["momentum"] is None
        assert screeners["pead"] is None


class TestFullBuild:
    def test_produces_correct_structure(self, source_dir: Path, tmp_path: Path):
        """Full build produces index.html, assets, data directory."""
        output = tmp_path / "site"
        builder = DashboardBuilder(source_dir=source_dir, max_bars=100)
        manifest = builder.build(output_dir=output)

        assert (output / "index.html").exists()
        assert (output / "assets" / "app.js").exists()
        assert (output / "assets" / "styles.css").exists()
        assert (output / "data" / "summary.json").exists()
        assert (output / "data" / "screeners.json").exists()
        assert (output / "_headers").exists()
        assert (output / ".nojekyll").exists()

        assert manifest.file_count > 0
        assert manifest.total_size_bytes > 0
        assert len(manifest.symbols) == 2

    def test_no_ts_files_in_output(self, source_dir: Path, tmp_path: Path):
        """Output dir has .js files (compiled from .ts), no .ts files."""
        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        ts_files = list((output / "assets").rglob("*.ts"))
        assert ts_files == [], f"Found .ts files in output: {ts_files}"

        # JS files must exist
        assert (output / "assets" / "app.js").exists()
        assert (output / "assets" / "charts.js").exists()

    def test_no_types_or_tsconfig_in_output(self, source_dir: Path, tmp_path: Path):
        """types/ directory and tsconfig.json are removed from output."""
        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        assert not (output / "assets" / "types").exists()
        assert not (output / "assets" / "tsconfig.json").exists()

    def test_html_contains_all_pages(self, source_dir: Path, tmp_path: Path):
        """HTML shell contains all 5 page sections and import map."""
        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        html = (output / "index.html").read_text()
        for page in ("overview", "signals", "screeners", "regime", "backtest"):
            assert f'id="page-{page}"' in html

        assert "importmap" in html
        assert "lightweight-charts" in html

    def test_overview_has_etf_dashboard_containers(self, source_dir: Path, tmp_path: Path):
        """Overview page has ETF dashboard, controls, stats, treemap containers."""
        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        html = (output / "index.html").read_text()
        assert 'id="overview-etf-dashboard"' in html
        assert 'id="overview-controls"' in html
        assert 'id="overview-stats"' in html
        assert 'id="overview-heatmap"' in html

    def test_overview_has_no_movers_alerts(self, source_dir: Path, tmp_path: Path):
        """Overview page does not contain legacy movers/alerts sections."""
        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        html = (output / "index.html").read_text()
        assert "overview-movers" not in html
        assert "overview-alerts" not in html
        assert "overview-benchmarks" not in html

    def test_signals_has_chart_container(self, source_dir: Path, tmp_path: Path):
        """Signals page has LC v5 multi-pane chart wrapper and sections div."""
        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        html = (output / "index.html").read_text()
        assert 'id="signals-chart-wrapper"' in html
        assert 'id="signals-chart-main"' in html
        assert 'id="signals-sections"' in html

    def test_html_contains_plotly(self, source_dir: Path, tmp_path: Path):
        """HTML shell contains Plotly script tag for treemap + backtest charts."""
        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        html = (output / "index.html").read_text()
        assert "plotly" in html.lower()

    def test_html_contains_backtest_content_container(self, source_dir: Path, tmp_path: Path):
        """HTML has backtest-content container for dynamic tab rendering."""
        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        html = (output / "index.html").read_text()
        assert 'id="backtest-content"' in html
        assert 'id="backtest-empty"' in html

    def test_cf_headers_content(self, source_dir: Path, tmp_path: Path):
        """CF _headers file contains correct cache and security headers."""
        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        headers = (output / "_headers").read_text()
        assert "/data/*" in headers
        assert "Cache-Control" in headers
        assert "X-Content-Type-Options: nosniff" in headers
        assert "X-Frame-Options: DENY" in headers


class TestSourceImmutability:
    def test_source_files_unchanged(self, source_dir: Path, tmp_path: Path):
        """Source files are not modified during build."""
        # Record file contents before build
        src_data = source_dir / "data" / "AAPL_1d.json"
        original = src_data.read_text()

        output = tmp_path / "site"
        DashboardBuilder(source_dir=source_dir).build(output_dir=output)

        # Verify source unchanged
        assert src_data.read_text() == original
