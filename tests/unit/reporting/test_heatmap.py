"""
Tests for heatmap module — color functions, data models, extractors,
Plotly data builder, ETF dashboard sparklines.
"""

from __future__ import annotations

import pytest

from src.infrastructure.reporting.heatmap.etf_dashboard import (
    _build_regime_badge,
    _render_sparkline_svg,
    build_etf_dashboard,
    get_etf_display_name,
    render_etf_card_html,
)
from src.infrastructure.reporting.heatmap.extractors import (
    extract_alignment_score,
    extract_composite_score,
    extract_regime,
)
from src.infrastructure.reporting.heatmap.model import (
    ETFCardData,
    HeatmapModel,
    RegimeColor,
    SectorGroup,
    TreemapNode,
    get_alignment_color,
    get_daily_change_color,
    get_regime_color,
    get_rule_frequency_color,
    get_rule_frequency_direction_color,
    get_score_gradient_color,
)
from src.infrastructure.reporting.heatmap.plotly_data import build_plotly_data

# =============================================================================
# Color functions (pure math)
# =============================================================================


class TestGetRegimeColor:
    def test_r0(self) -> None:
        assert get_regime_color("R0") == RegimeColor.R0.value

    def test_r1(self) -> None:
        assert get_regime_color("R1") == RegimeColor.R1.value

    def test_r2(self) -> None:
        assert get_regime_color("R2") == RegimeColor.R2.value

    def test_r3(self) -> None:
        assert get_regime_color("R3") == RegimeColor.R3.value

    def test_none(self) -> None:
        assert get_regime_color(None) == RegimeColor.UNKNOWN.value

    def test_unknown_string(self) -> None:
        assert get_regime_color("INVALID") == RegimeColor.UNKNOWN.value

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("HEALTHY", RegimeColor.R0.value),
            ("BULLISH", RegimeColor.R0.value),
            ("CHOPPY", RegimeColor.R1.value),
            ("EXTENDED", RegimeColor.R1.value),
            ("RISK_OFF", RegimeColor.R2.value),
            ("BEARISH", RegimeColor.R2.value),
            ("REBOUND", RegimeColor.R3.value),
            ("RECOVERY", RegimeColor.R3.value),
        ],
    )
    def test_aliases(self, alias: str, expected: str) -> None:
        assert get_regime_color(alias) == expected


class TestGetDailyChangeColor:
    def test_none_returns_unknown(self) -> None:
        assert get_daily_change_color(None) == RegimeColor.UNKNOWN.value

    def test_zero_change(self) -> None:
        color = get_daily_change_color(0.0)
        # Should be close to neutral (200, 220, 180) -> #c8dcb4
        assert color.startswith("#")
        assert len(color) == 7

    def test_positive_change_green(self) -> None:
        color = get_daily_change_color(5.0)
        # At +5%, should be greenish
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        assert g > r  # Green dominates

    def test_negative_change_red(self) -> None:
        color = get_daily_change_color(-5.0)
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        assert r > g  # Red dominates

    def test_clamping_positive(self) -> None:
        """Values > 5% clamped to 5%."""
        color_at_5 = get_daily_change_color(5.0)
        color_at_10 = get_daily_change_color(10.0)
        assert color_at_5 == color_at_10

    def test_clamping_negative(self) -> None:
        """Values < -5% clamped to -5%."""
        color_at_neg5 = get_daily_change_color(-5.0)
        color_at_neg10 = get_daily_change_color(-10.0)
        assert color_at_neg5 == color_at_neg10


class TestGetAlignmentColor:
    def test_none_returns_unknown(self) -> None:
        assert get_alignment_color(None) == RegimeColor.UNKNOWN.value

    def test_positive_alignment(self) -> None:
        color = get_alignment_color(50.0)
        assert color.startswith("#22")  # Green family

    def test_negative_alignment(self) -> None:
        color = get_alignment_color(-50.0)
        assert color.endswith("4444")  # Red family

    def test_zero_alignment(self) -> None:
        color = get_alignment_color(0.0)
        # Should be base green: #22b45e (g=180)
        assert color.startswith("#22")

    def test_clamping(self) -> None:
        color_at_100 = get_alignment_color(100.0)
        color_at_200 = get_alignment_color(200.0)
        assert color_at_100 == color_at_200


class TestGetScoreGradientColor:
    def test_score_0_is_red(self) -> None:
        color = get_score_gradient_color(0.0)
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        assert r > g  # Red dominates at score 0

    def test_score_100_is_green(self) -> None:
        color = get_score_gradient_color(100.0)
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        assert g > r  # Green dominates at score 100

    def test_score_50_amber(self) -> None:
        color = get_score_gradient_color(50.0)
        # At 50, hue = 45° (amber/yellow)
        assert color.startswith("#")

    def test_clamping(self) -> None:
        assert get_score_gradient_color(-10) == get_score_gradient_color(0)
        assert get_score_gradient_color(110) == get_score_gradient_color(100)


class TestGetRuleFrequencyColor:
    def test_zero_signals(self) -> None:
        assert get_rule_frequency_color(0, 10) == "#444444"

    def test_zero_max_count(self) -> None:
        assert get_rule_frequency_color(5, 0) == "#444444"

    @pytest.mark.parametrize(
        "count,expected",
        [
            (1, "#88cc88"),
            (2, "#88cc88"),
            (3, "#ffcc44"),
            (4, "#ffcc44"),
            (5, "#ff8844"),
            (7, "#ff8844"),
            (8, "#ff4444"),
            (15, "#ff4444"),
        ],
    )
    def test_thresholds(self, count: int, expected: str) -> None:
        assert get_rule_frequency_color(count, 20) == expected


class TestGetRuleFrequencyDirectionColor:
    def test_no_signals(self) -> None:
        assert get_rule_frequency_direction_color(0, 0) == "#444444"

    def test_balanced(self) -> None:
        assert get_rule_frequency_direction_color(3, 3) == "#6b7280"

    def test_bullish(self) -> None:
        color = get_rule_frequency_direction_color(5, 1)
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        assert g > r  # Green = bullish

    def test_bearish(self) -> None:
        color = get_rule_frequency_direction_color(1, 5)
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        assert r > g  # Red = bearish


# =============================================================================
# Data models
# =============================================================================


class TestTreemapNode:
    def test_to_dict(self) -> None:
        node = TreemapNode(
            symbol="AAPL",
            label="AAPL",
            parent="sector_XLK",
            value=2_800_000_000_000,
            color="#22c55e",
            regime="R0",
            daily_change_pct=1.5,
        )
        d = node.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["parent"] == "sector_XLK"
        assert d["regime"] == "R0"

    def test_default_fields(self) -> None:
        node = TreemapNode(symbol="X", label="X", parent="", value=1.0, color="#000")
        d = node.to_dict()
        assert d["signal_count"] == 0
        assert d["composite_score"] is None


class TestETFCardData:
    def test_direction_class_neutral(self) -> None:
        card = ETFCardData(symbol="SPY", display_name="S&P 500", category="market")
        assert card.direction_class == "hm-direction-neutral"

    def test_direction_class_bullish(self) -> None:
        card = ETFCardData(
            symbol="SPY",
            display_name="S&P 500",
            category="market",
            buy_signal_count=5,
            sell_signal_count=2,
        )
        assert card.direction_class == "hm-direction-bullish"

    def test_direction_class_bearish(self) -> None:
        card = ETFCardData(
            symbol="SPY",
            display_name="S&P 500",
            category="market",
            buy_signal_count=1,
            sell_signal_count=4,
        )
        assert card.direction_class == "hm-direction-bearish"

    def test_direction_class_equal(self) -> None:
        card = ETFCardData(
            symbol="SPY",
            display_name="S&P 500",
            category="market",
            buy_signal_count=3,
            sell_signal_count=3,
        )
        assert card.direction_class == "hm-direction-neutral"


class TestHeatmapModel:
    def test_get_all_nodes(self) -> None:
        node = TreemapNode(
            symbol="AAPL", label="AAPL", parent="sector_XLK", value=1.0, color="#000"
        )
        sector = SectorGroup(sector_id="XLK", sector_name="Technology", stocks=[node])
        model = HeatmapModel(sectors=[sector])
        nodes = model.get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0].symbol == "AAPL"

    def test_to_dict(self) -> None:
        model = HeatmapModel(symbol_count=5, regime_distribution={"R0": 3, "R1": 2})
        d = model.to_dict()
        assert d["symbol_count"] == 5
        assert d["regime_distribution"]["R0"] == 3


# =============================================================================
# Extractors
# =============================================================================


class TestExtractRegime:
    def test_direct_field(self) -> None:
        assert extract_regime({"regime": "R0"}) == "R0"

    def test_nested_field(self) -> None:
        assert extract_regime({"regime_output": {"regime": "R1"}}) == "R1"

    def test_missing(self) -> None:
        assert extract_regime({}) is None
        assert extract_regime({"regime": None}) is None
        assert extract_regime({"regime": ""}) is None


class TestExtractAlignmentScore:
    def test_direct(self) -> None:
        assert extract_alignment_score({"alignment_score": 75.0}) == 75.0

    def test_nested(self) -> None:
        assert extract_alignment_score({"confluence": {"alignment_score": -50.0}}) == -50.0

    def test_missing(self) -> None:
        assert extract_alignment_score({}) is None


class TestExtractCompositeScore:
    def test_present(self) -> None:
        assert extract_composite_score({"composite_score_avg": 72.5}) == 72.5

    def test_missing(self) -> None:
        assert extract_composite_score({}) is None


# =============================================================================
# build_plotly_data
# =============================================================================


class TestBuildPlotlyData:
    def test_structural_invariants(self) -> None:
        node1 = TreemapNode(symbol="AAPL", label="AAPL", parent="XLK", value=100.0, color="#22c55e")
        node2 = TreemapNode(symbol="MSFT", label="MSFT", parent="XLK", value=80.0, color="#22c55e")
        sector = SectorGroup(sector_id="XLK", sector_name="Technology", stocks=[node1, node2])
        model = HeatmapModel(sectors=[sector])
        data = build_plotly_data(model)

        # All arrays same length
        assert len(data["ids"]) == len(data["labels"])
        assert len(data["ids"]) == len(data["parents"])
        assert len(data["ids"]) == len(data["values"])
        assert len(data["ids"]) == len(data["colors"])

        # Root node exists
        assert "root" in data["ids"]
        root_idx = data["ids"].index("root")
        assert data["parents"][root_idx] == ""

        # Root value = sum of children
        assert data["values"][root_idx] == pytest.approx(180.0)

    def test_empty_model(self) -> None:
        model = HeatmapModel()
        data = build_plotly_data(model)
        assert "root" in data["ids"]
        # Just root node
        assert len(data["ids"]) == 1

    def test_zero_value_stocks_get_minimum(self) -> None:
        """Stocks with value=0 get value=1.0 as fallback."""
        node = TreemapNode(symbol="TINY", label="TINY", parent="XLK", value=0.0, color="#000")
        sector = SectorGroup(sector_id="XLK", sector_name="Tech", stocks=[node])
        model = HeatmapModel(sectors=[sector])
        data = build_plotly_data(model)
        stock_idx = data["ids"].index("stock_TINY")
        assert data["values"][stock_idx] == 1.0


# =============================================================================
# ETF Dashboard
# =============================================================================


class TestSparklineSVG:
    def test_insufficient_points(self) -> None:
        assert _render_sparkline_svg([50.0]) == ""
        assert _render_sparkline_svg([]) == ""

    def test_uptrend_green(self) -> None:
        svg = _render_sparkline_svg([30.0, 40.0, 50.0, 60.0, 70.0])
        assert "svg" in svg
        assert "#10b981" in svg  # green

    def test_downtrend_red(self) -> None:
        svg = _render_sparkline_svg([70.0, 60.0, 50.0, 40.0, 30.0])
        assert "#ef4444" in svg  # red

    def test_flat_gray(self) -> None:
        svg = _render_sparkline_svg([50.0, 51.0, 50.0, 51.0])
        assert "#94a3b8" in svg  # gray

    def test_contains_polyline(self) -> None:
        svg = _render_sparkline_svg([30.0, 50.0, 70.0])
        assert "polyline" in svg
        assert "circle" in svg


class TestGetETFDisplayName:
    def test_from_cap_result(self) -> None:
        cap = type("Cap", (), {"short_name": "SPDR S&P 500"})()
        assert get_etf_display_name("SPY", cap) == "SPDR S&P 500"

    def test_market_etf_fallback(self) -> None:
        assert get_etf_display_name("SPY", None) == "S&P 500"

    def test_sector_etf_fallback(self) -> None:
        assert get_etf_display_name("XLK", None) == "Technology"

    def test_other_etf_fallback(self) -> None:
        assert get_etf_display_name("GLD", None) == "Gold"

    def test_unknown_returns_symbol(self) -> None:
        assert get_etf_display_name("UNKNOWN_ETF", None) == "UNKNOWN_ETF"


class TestBuildETFDashboard:
    def test_builds_categories(self) -> None:
        tickers = [
            {"symbol": "SPY", "regime": "R0", "close": 500.0, "daily_change_pct": 0.5},
            {"symbol": "QQQ", "regime": "R1", "close": 400.0},
        ]
        dashboard = build_etf_dashboard(tickers, {}, {})
        assert "market_indices" in dashboard

    def test_missing_ticker_handled(self) -> None:
        """ETF not in tickers still gets a card with defaults."""
        dashboard = build_etf_dashboard([], {}, {})
        assert "market_indices" in dashboard
        cards = dashboard["market_indices"]
        assert len(cards) == 4  # SPY, QQQ, IWM, DIA
        assert cards[0].regime is None  # No data available

    def test_signal_counts_passed(self) -> None:
        tickers = [{"symbol": "SPY"}]
        dashboard = build_etf_dashboard(
            tickers,
            {},
            {},
            buy_counts_by_symbol={"SPY": 5},
            sell_counts_by_symbol={"SPY": 3},
        )
        spy_card = dashboard["market_indices"][0]
        assert spy_card.buy_signal_count == 5
        assert spy_card.sell_signal_count == 3


# =============================================================================
# Score Badge Rendering
# =============================================================================


class TestScoreBadgeRendering:
    """Verify gradient badges for composite scores vs discrete class fallback."""

    def test_score_45_gets_inline_gradient(self) -> None:
        """Card with composite_score=45 → inline style with gradient color."""
        card = ETFCardData(
            symbol="SPY",
            display_name="S&P 500",
            category="market_indices",
            composite_score=45.0,
            regime="R1",
        )
        html = render_etf_card_html(card, "large")
        # Should have inline style, not hm-regime-r1 class
        assert "style=" in html
        assert "rgba(" in html
        assert "hm-regime-r1" not in html

    def test_score_85_different_color_from_45(self) -> None:
        """Cards with different scores get different gradient colors."""
        card_45 = ETFCardData(
            symbol="SPY", display_name="S&P 500", category="m", composite_score=45.0
        )
        card_85 = ETFCardData(
            symbol="QQQ", display_name="Nasdaq", category="m", composite_score=85.0
        )
        _, attrs_45 = _build_regime_badge(card_45)
        _, attrs_85 = _build_regime_badge(card_85)
        # Different scores → different inline styles
        assert attrs_45 != attrs_85
        # Both should have inline styles
        assert "rgba(" in attrs_45
        assert "rgba(" in attrs_85

    def test_no_score_falls_back_to_css_class(self) -> None:
        """Card with composite_score=None → discrete hm-regime-* CSS class."""
        card = ETFCardData(
            symbol="SPY",
            display_name="S&P 500",
            category="market_indices",
            regime="R0",
            composite_score=None,
        )
        html = render_etf_card_html(card, "compact")
        assert "hm-regime-r0" in html
        assert "rgba(" not in html

    def test_gradient_badge_in_mini_style(self) -> None:
        """Gradient badge works in mini card style too."""
        card = ETFCardData(
            symbol="GLD", display_name="Gold", category="commodities", composite_score=60.0
        )
        html = render_etf_card_html(card, "mini")
        assert "style=" in html
        assert "rgba(" in html

    def test_no_regime_no_score(self) -> None:
        """No regime and no score → hm-regime-unknown class."""
        card = ETFCardData(
            symbol="X", display_name="X", category="m", regime=None, composite_score=None
        )
        _, attrs = _build_regime_badge(card)
        assert "hm-regime-unknown" in attrs
