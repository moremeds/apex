"""
Heatmap Builder for Signal Reports.

PR-C Deliverable: Builds treemap/heatmap visualization for the signal landing page.

Three-Layer Architecture:
1. build_heatmap_model() - Transforms summary + caps into HeatmapModel (pure data)
2. render_heatmap_html() - Renders HeatmapModel to HTML with embedded data
3. heatmap.js - Frontend Plotly rendering with interactive toggles

Usage:
    from src.domain.signals.reporting.heatmap_builder import HeatmapBuilder

    builder = HeatmapBuilder(market_cap_service)
    model = builder.build_heatmap_model(summary_data, universe_config)
    html = builder.render_heatmap_html(model, output_dir)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.services.market_cap_service import MarketCapService
from src.utils.logging_setup import get_logger

from .heatmap_model import (
    MARKET_ETFS,
    SECTOR_MAPPING,
    ColorMetric,
    HeatmapModel,
    SectorGroup,
    SizeMetric,
    TreemapNode,
    get_alignment_color,
    get_daily_change_color,
    get_regime_color,
)

logger = get_logger(__name__)

# Stock to sector mapping (simplified - can be extended with external data)
# Format: symbol -> sector_etf
STOCK_SECTOR_MAP: Dict[str, str] = {
    # Technology (XLK)
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AVGO": "XLK",
    "ORCL": "XLK",
    "CRM": "XLK",
    "AMD": "XLK",
    "ADBE": "XLK",
    "CSCO": "XLK",
    "ACN": "XLK",
    "IBM": "XLK",
    "INTC": "XLK",
    "QCOM": "XLK",
    "TXN": "XLK",
    "MU": "XLK",
    # Communication Services (XLC)
    "GOOGL": "XLC",
    "GOOG": "XLC",
    "META": "XLC",
    "NFLX": "XLC",
    "DIS": "XLC",
    "CMCSA": "XLC",
    "T": "XLC",
    "VZ": "XLC",
    "TMUS": "XLC",
    # Consumer Discretionary (XLY)
    "AMZN": "XLY",
    "TSLA": "XLY",
    "HD": "XLY",
    "MCD": "XLY",
    "NKE": "XLY",
    "LOW": "XLY",
    "SBUX": "XLY",
    "TJX": "XLY",
    "BKNG": "XLY",
    # Financials (XLF)
    "BRK.B": "XLF",
    "JPM": "XLF",
    "V": "XLF",
    "MA": "XLF",
    "BAC": "XLF",
    "WFC": "XLF",
    "GS": "XLF",
    "MS": "XLF",
    "AXP": "XLF",
    "BLK": "XLF",
    "C": "XLF",
    "SCHW": "XLF",
    # Health Care (XLV)
    "UNH": "XLV",
    "JNJ": "XLV",
    "LLY": "XLV",
    "ABBV": "XLV",
    "MRK": "XLV",
    "PFE": "XLV",
    "TMO": "XLV",
    "ABT": "XLV",
    "DHR": "XLV",
    "BMY": "XLV",
    "AMGN": "XLV",
    "CVS": "XLV",
    # Consumer Staples (XLP)
    "PG": "XLP",
    "KO": "XLP",
    "PEP": "XLP",
    "COST": "XLP",
    "WMT": "XLP",
    "PM": "XLP",
    "MDLZ": "XLP",
    "CL": "XLP",
    # Energy (XLE)
    "XOM": "XLE",
    "CVX": "XLE",
    "COP": "XLE",
    "SLB": "XLE",
    "EOG": "XLE",
    "MPC": "XLE",
    "PSX": "XLE",
    "VLO": "XLE",
    # Industrials (XLI)
    "GE": "XLI",
    "CAT": "XLI",
    "RTX": "XLI",
    "HON": "XLI",
    "UNP": "XLI",
    "BA": "XLI",
    "DE": "XLI",
    "LMT": "XLI",
    "UPS": "XLI",
    "MMM": "XLI",
    # Materials (XLB)
    "LIN": "XLB",
    "APD": "XLB",
    "SHW": "XLB",
    "ECL": "XLB",
    "NEM": "XLB",
    "FCX": "XLB",
    "DD": "XLB",
    # Real Estate (XLRE)
    "PLD": "XLRE",
    "AMT": "XLRE",
    "EQIX": "XLRE",
    "CCI": "XLRE",
    "PSA": "XLRE",
    "SPG": "XLRE",
    # Utilities (XLU)
    "NEE": "XLU",
    "DUK": "XLU",
    "SO": "XLU",
    "D": "XLU",
    "AEP": "XLU",
    "EXC": "XLU",
    "SRE": "XLU",
}


class HeatmapBuilder:
    """
    Builds heatmap visualization from summary data and market caps.

    This builder follows the three-layer architecture:
    1. Model building (pure data transformation)
    2. HTML rendering (template-based)
    3. Frontend rendering (deferred to heatmap.js)
    """

    def __init__(
        self,
        market_cap_service: Optional[MarketCapService] = None,
        size_metric: SizeMetric = SizeMetric.MARKET_CAP,
        color_metric: ColorMetric = ColorMetric.REGIME,
    ) -> None:
        """
        Initialize heatmap builder.

        Args:
            market_cap_service: Service for market cap lookups (creates default if None)
            size_metric: How to size rectangles (market_cap, volume, or equal)
            color_metric: How to color rectangles (regime, daily_change, alignment_score)
        """
        self._cap_service = market_cap_service or MarketCapService()
        self._size_metric = size_metric
        self._color_metric = color_metric

    def build_heatmap_model(
        self,
        summary_data: Dict[str, Any],
        manifest: Optional[Dict[str, Any]] = None,
    ) -> HeatmapModel:
        """
        Build HeatmapModel from summary.json data.

        Args:
            summary_data: Parsed summary.json content
            manifest: Optional manifest.json for report URLs

        Returns:
            HeatmapModel ready for rendering
        """
        tickers = summary_data.get("tickers", [])
        if not tickers:
            logger.warning("No tickers in summary data")
            return HeatmapModel(generated_at=datetime.now())

        # Extract all symbols
        symbols = [t.get("symbol") for t in tickers if t.get("symbol")]

        # Get market caps
        cap_results = self._cap_service.get_market_caps(symbols)

        # Build report URL map from manifest
        report_urls: Dict[str, str] = {}
        if manifest and "symbol_reports" in manifest:
            report_urls = manifest["symbol_reports"]

        # Classify symbols
        market_etf_nodes: List[TreemapNode] = []
        sector_etf_nodes: List[TreemapNode] = []
        stocks_by_sector: Dict[str, List[TreemapNode]] = {
            sector_id: [] for sector_id in SECTOR_MAPPING.keys()
        }
        other_stocks: List[TreemapNode] = []

        # Track statistics
        regime_counts: Dict[str, int] = {}
        cap_missing_count = 0

        for ticker in tickers:
            symbol = ticker.get("symbol")
            if not symbol:
                continue

            # Extract regime from ticker data
            regime = self._extract_regime(ticker)
            if regime:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Get market cap
            cap_result = cap_results.get(symbol)
            market_cap = cap_result.market_cap if cap_result else 0.0
            cap_missing = cap_result.cap_missing if cap_result else True
            if cap_missing:
                cap_missing_count += 1

            # Determine color based on metric
            color = self._get_node_color(ticker, regime)

            # Calculate size value
            size_value = self._get_size_value(ticker, market_cap)

            # Build node
            node = TreemapNode(
                symbol=symbol,
                label=symbol,
                parent="",  # Will be set based on classification
                value=size_value,
                color=color,
                regime=regime,
                daily_change_pct=ticker.get("daily_change_pct"),
                market_cap=market_cap if not cap_missing else None,
                market_cap_missing=cap_missing,
                close_price=ticker.get("close"),
                alignment_score=self._extract_alignment_score(ticker),
                volume=ticker.get("volume"),
                report_url=report_urls.get(symbol),
            )

            # Classify node
            if symbol in MARKET_ETFS:
                node.parent = "Market"
                market_etf_nodes.append(node)
            elif symbol in SECTOR_MAPPING:
                node.parent = "Sectors"
                sector_etf_nodes.append(node)
            elif symbol in STOCK_SECTOR_MAP:
                sector = STOCK_SECTOR_MAP[symbol]
                node.parent = sector
                stocks_by_sector[sector].append(node)
            else:
                # Unknown sector - put in "Other"
                node.parent = "Other"
                other_stocks.append(node)

        # Build sector groups
        sectors: List[SectorGroup] = []
        for sector_id, (sector_name, gics_code) in SECTOR_MAPPING.items():
            stocks = stocks_by_sector.get(sector_id, [])
            if stocks:
                sectors.append(
                    SectorGroup(
                        sector_id=sector_id,
                        sector_name=sector_name,
                        gics_sector=gics_code,
                        stocks=stocks,
                    )
                )

        # Add "Other" sector if there are unclassified stocks
        if other_stocks:
            sectors.append(
                SectorGroup(
                    sector_id="Other",
                    sector_name="Other",
                    gics_sector=None,
                    stocks=other_stocks,
                )
            )

        return HeatmapModel(
            market_etfs=market_etf_nodes,
            sector_etfs=sector_etf_nodes,
            sectors=sectors,
            generated_at=datetime.now(),
            size_metric=self._size_metric,
            color_metric=self._color_metric,
            symbol_count=len(symbols),
            cap_missing_count=cap_missing_count,
            regime_distribution=regime_counts,
        )

    def render_heatmap_html(
        self,
        model: HeatmapModel,
        output_dir: Path,
        template_name: str = "heatmap.html",
    ) -> str:
        """
        Render HeatmapModel to HTML page.

        Args:
            model: HeatmapModel to render
            output_dir: Directory where HTML will be written
            template_name: Output filename

        Returns:
            HTML content as string
        """
        # Build Plotly-compatible data structure
        plotly_data = self._build_plotly_data(model)

        # Generate HTML with embedded data
        html = self._render_template(model, plotly_data, output_dir)

        return html

    def _extract_regime(self, ticker: Dict[str, Any]) -> Optional[str]:
        """Extract regime string from ticker data."""
        # Try different possible locations for regime
        regime = ticker.get("regime")
        if regime:
            return regime

        # Check nested regime_output
        regime_output = ticker.get("regime_output", {})
        if isinstance(regime_output, dict):
            return regime_output.get("regime")

        return None

    def _extract_alignment_score(self, ticker: Dict[str, Any]) -> Optional[float]:
        """Extract alignment score from ticker data."""
        # Try different possible locations
        score = ticker.get("alignment_score")
        if score is not None:
            return float(score)

        confluence = ticker.get("confluence", {})
        if isinstance(confluence, dict):
            score = confluence.get("alignment_score")
            if score is not None:
                return float(score)

        return None

    def _get_node_color(self, ticker: Dict[str, Any], regime: Optional[str]) -> str:
        """Determine node color based on configured metric."""
        if self._color_metric == ColorMetric.REGIME:
            return get_regime_color(regime)
        elif self._color_metric == ColorMetric.DAILY_CHANGE:
            return get_daily_change_color(ticker.get("daily_change_pct"))
        elif self._color_metric == ColorMetric.ALIGNMENT_SCORE:
            return get_alignment_color(self._extract_alignment_score(ticker))
        else:
            return get_regime_color(regime)

    def _get_size_value(self, ticker: Dict[str, Any], market_cap: float) -> float:
        """Determine size value based on configured metric."""
        if self._size_metric == SizeMetric.MARKET_CAP:
            return market_cap if market_cap > 0 else 1.0  # Fallback to 1.0
        elif self._size_metric == SizeMetric.VOLUME:
            volume = ticker.get("volume", 0)
            return float(volume) if volume > 0 else 1.0
        else:  # EQUAL
            return 1.0

    def _build_plotly_data(self, model: HeatmapModel) -> Dict[str, Any]:
        """
        Build Plotly treemap data structure.

        Plotly treemap requires:
        - One root node with parent=""
        - All other nodes must connect back to root through parent chain
        - branchvalues="total" means parent values = sum of children

        CRITICAL: With branchvalues="total", parent nodes MUST have values equal
        to the sum of their children, otherwise the treemap renders empty.

        Returns dict with:
        - ids: List of unique identifiers
        - labels: List of display labels
        - parents: List of parent identifiers
        - values: List of size values
        - colors: List of colors
        - customdata: List of additional metadata for tooltips
        """
        ids: List[str] = []
        labels: List[str] = []
        parents: List[str] = []
        values: List[float] = []
        colors: List[str] = []
        customdata: List[Dict[str, Any]] = []

        # Track parent-child relationships for value aggregation
        # Map: parent_id -> list of (child_index, child_value)
        parent_children: Dict[str, List[int]] = {}

        def add_node(
            node_id: str,
            label: str,
            parent: str,
            value: float,
            color: str,
            data: Dict[str, Any],
        ) -> int:
            """Add a node and return its index."""
            idx = len(ids)
            ids.append(node_id)
            labels.append(label)
            parents.append(parent)
            values.append(value)
            colors.append(color)
            customdata.append(data)

            # Track parent-child relationship
            if parent:
                if parent not in parent_children:
                    parent_children[parent] = []
                parent_children[parent].append(idx)

            return idx

        # Single root node - ALL other nodes must trace back to this
        add_node("root", "Signal Universe", "", 0, "#1a1a2e", {"type": "root"})

        # Market ETFs category (under root)
        if model.market_etfs:
            add_node("cat_market", "Market ETFs", "root", 0, "#374151", {"type": "category"})

            for etf in model.market_etfs:
                add_node(
                    f"etf_{etf.symbol}",
                    etf.label,
                    "cat_market",
                    etf.value if etf.value > 0 else 1.0,  # Ensure positive value
                    etf.color,
                    etf.to_dict(),
                )

        # Sector ETFs category (under root) - if any
        if model.sector_etfs:
            add_node("cat_sectors", "Sector ETFs", "root", 0, "#374151", {"type": "category"})

            for etf in model.sector_etfs:
                add_node(
                    f"setf_{etf.symbol}",
                    etf.label,
                    "cat_sectors",
                    etf.value if etf.value > 0 else 1.0,  # Ensure positive value
                    etf.color,
                    etf.to_dict(),
                )

        # Sector groups with stocks (under root)
        for sector in model.sectors:
            # Sector container - directly under root
            sector_id = f"sector_{sector.sector_id}"
            add_node(
                sector_id,
                sector.sector_name,
                "root",
                0,  # Will be calculated from children
                "#4b5563",
                {"type": "sector", "sector_id": sector.sector_id},
            )

            # Stocks in sector
            for stock in sector.stocks:
                add_node(
                    f"stock_{stock.symbol}",
                    stock.label,
                    sector_id,
                    stock.value if stock.value > 0 else 1.0,  # Ensure positive value
                    stock.color,
                    stock.to_dict(),
                )

        # === CRITICAL: Calculate parent values from children ===
        # With branchvalues="total", parent values must equal sum of children.
        # We need to aggregate bottom-up: sectors -> categories -> root

        # Build id->index map
        id_to_idx: Dict[str, int] = {node_id: i for i, node_id in enumerate(ids)}

        # Calculate parent values bottom-up
        def calculate_parent_value(parent_id: str) -> float:
            """Recursively calculate parent value as sum of children."""
            if parent_id not in parent_children:
                return 0.0

            total = 0.0
            for child_idx in parent_children[parent_id]:
                child_id = ids[child_idx]
                child_value = values[child_idx]

                # If child has its own children, recursively calculate
                if child_id in parent_children:
                    child_value = calculate_parent_value(child_id)
                    values[child_idx] = child_value

                total += child_value

            return total

        # Calculate for root (which triggers all descendants)
        root_value = calculate_parent_value("root")
        values[0] = root_value  # Set root value

        logger.debug(f"Heatmap data built: {len(ids)} nodes, root_value={root_value:.2f}")

        return {
            "ids": ids,
            "labels": labels,
            "parents": parents,
            "values": values,
            "colors": colors,
            "customdata": customdata,
        }

    def _render_template(
        self,
        model: HeatmapModel,
        plotly_data: Dict[str, Any],
        output_dir: Path,
    ) -> str:
        """Render the HTML template with embedded data."""
        # Embedded model data for frontend
        model_json = json.dumps(model.to_dict(), indent=2)
        plotly_json = json.dumps(plotly_data)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Heatmap - {model.generated_at.strftime('%Y-%m-%d') if model.generated_at else 'N/A'}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }}
        .header {{
            padding: 20px;
            background: #16213e;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        .header .meta {{
            font-size: 0.875rem;
            color: #9ca3af;
        }}
        .controls {{
            padding: 15px 20px;
            background: #1f2937;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .control-group label {{
            font-size: 0.875rem;
            color: #9ca3af;
        }}
        .control-group select {{
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #374151;
            background: #111827;
            color: #fff;
            font-size: 0.875rem;
            cursor: pointer;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            margin-left: auto;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        #heatmap {{
            width: 100%;
            height: calc(100vh - 140px);
            min-height: 500px;
        }}
        .stats {{
            padding: 10px 20px;
            background: #111827;
            display: flex;
            gap: 30px;
            font-size: 0.75rem;
            color: #9ca3af;
        }}
        .stat {{
            display: flex;
            gap: 6px;
        }}
        .stat-value {{
            color: #fff;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Signal Heatmap</h1>
        <div class="meta">
            Generated: {model.generated_at.strftime('%Y-%m-%d %H:%M') if model.generated_at else 'N/A'}
        </div>
    </div>

    <div class="controls">
        <div class="control-group">
            <label>Color by:</label>
            <select id="colorMetric">
                <option value="regime" {'selected' if model.color_metric == ColorMetric.REGIME else ''}>Regime (R0-R3)</option>
                <option value="daily_change" {'selected' if model.color_metric == ColorMetric.DAILY_CHANGE else ''}>Daily Change</option>
                <option value="alignment" {'selected' if model.color_metric == ColorMetric.ALIGNMENT_SCORE else ''}>Alignment Score</option>
            </select>
        </div>
        <div class="control-group">
            <label>Size by:</label>
            <select id="sizeMetric">
                <option value="market_cap" {'selected' if model.size_metric == SizeMetric.MARKET_CAP else ''}>Market Cap</option>
                <option value="volume" {'selected' if model.size_metric == SizeMetric.VOLUME else ''}>Volume</option>
                <option value="equal" {'selected' if model.size_metric == SizeMetric.EQUAL else ''}>Equal Weight</option>
            </select>
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #22c55e;"></div>
                <span>R0 Healthy</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #eab308;"></div>
                <span>R1 Choppy</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ef4444;"></div>
                <span>R2 Risk-Off</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #3b82f6;"></div>
                <span>R3 Rebound</span>
            </div>
        </div>
    </div>

    <div class="stats">
        <div class="stat">
            <span>Symbols:</span>
            <span class="stat-value">{model.symbol_count}</span>
        </div>
        <div class="stat">
            <span>Missing Caps:</span>
            <span class="stat-value">{model.cap_missing_count}</span>
        </div>
        {''.join(f'<div class="stat"><span>{r}:</span><span class="stat-value">{c}</span></div>' for r, c in sorted(model.regime_distribution.items()))}
    </div>

    <div id="heatmap"></div>

    <script>
        // Embedded model data
        const modelData = {model_json};
        const plotlyData = {plotly_json};

        // Report URL mapping for click navigation
        const reportUrls = {{}};
        modelData.market_etfs.forEach(e => {{ if (e.report_url) reportUrls[e.symbol] = e.report_url; }});
        modelData.sector_etfs.forEach(e => {{ if (e.report_url) reportUrls[e.symbol] = e.report_url; }});
        modelData.sectors.forEach(s => {{
            s.stocks.forEach(stock => {{ if (stock.report_url) reportUrls[stock.symbol] = stock.report_url; }});
        }});

        // Debug: Check if Plotly loaded
        console.log('Plotly loaded:', typeof Plotly !== 'undefined');

        // Determine scale based on size metric (Bug 1 fix: metric-aware normalization)
        const sizeMetric = modelData.size_metric;
        const getScale = (metric) => metric === 'volume' ? 1e6 : metric === 'equal' ? 1 : 1e9;
        let currentScale = getScale(sizeMetric);
        const normalizedValues = plotlyData.values.map(v => v / currentScale);
        console.log('Data loaded - ids:', plotlyData.ids.length, 'values:', normalizedValues, 'scale:', currentScale);

        // Build parent-children map once for parent recalculation (Bug 2 fix)
        function buildChildrenMap(ids, parents) {{
            const children = {{}};
            parents.forEach((parent, i) => {{
                if (parent) {{
                    if (!children[parent]) children[parent] = [];
                    children[parent].push(i);
                }}
            }});
            return children;
        }}

        const childrenMap = buildChildrenMap(plotlyData.ids, plotlyData.parents);

        // Recalculate parent values bottom-up (Bug 2 fix)
        function recalculateParents(values, children, ids) {{
            function sumChildren(nodeId) {{
                const idx = ids.indexOf(nodeId);
                if (idx === -1) return 0;
                if (!children[nodeId]) return values[idx];

                let sum = 0;
                for (const childIdx of children[nodeId]) {{
                    const childId = ids[childIdx];
                    sum += sumChildren(childId);
                }}
                values[idx] = sum;
                return sum;
            }}
            sumChildren('root');
        }}

        // Debug: Check container dimensions
        const container = document.getElementById('heatmap');
        console.log('Container dimensions:', container.offsetWidth, 'x', container.offsetHeight);

        // Color schemes
        function getRegimeColor(regime) {{
            const colors = {{'R0': '#22c55e', 'R1': '#eab308', 'R2': '#ef4444', 'R3': '#3b82f6'}};
            return colors[regime] || '#9ca3af';
        }}

        function getDailyChangeColor(pct) {{
            if (pct === null || pct === undefined) return '#9ca3af';
            const clamped = Math.max(-5, Math.min(5, pct));
            if (clamped >= 0) {{
                const intensity = Math.floor(200 - (clamped / 5) * 100);
                return `rgb(${{intensity}}, 197, 94)`;
            }} else {{
                const intensity = Math.floor(200 + (clamped / 5) * 100);
                return `rgb(244, ${{intensity}}, ${{intensity}})`;
            }}
        }}

        function getAlignmentColor(score) {{
            if (score === null || score === undefined) return '#9ca3af';
            const clamped = Math.max(-100, Math.min(100, score));
            if (clamped >= 0) {{
                const g = Math.floor(180 + (clamped / 100) * 40);
                return `rgb(34, ${{g}}, 94)`;
            }} else {{
                const r = Math.floor(200 + (clamped / 100) * 44);
                return `rgb(${{r}}, 68, 68)`;
            }}
        }}

        function updateColors(metric) {{
            const newColors = plotlyData.customdata.map((d, i) => {{
                if (!d || !d.symbol) return plotlyData.colors[i];
                if (metric === 'regime') return getRegimeColor(d.regime);
                if (metric === 'daily_change') return getDailyChangeColor(d.daily_change_pct);
                if (metric === 'alignment') return getAlignmentColor(d.alignment_score);
                return plotlyData.colors[i];
            }});
            Plotly.restyle('heatmap', {{'marker.colors': [newColors]}});
        }}

        function updateSizes(metric) {{
            const scale = getScale(metric);
            currentScale = scale;

            // Copy original values to avoid mutating
            const newValues = [...plotlyData.values];

            // Update leaf node values based on metric
            plotlyData.customdata.forEach((d, i) => {{
                if (d && d.symbol) {{
                    // Only update leaf nodes (nodes with a symbol)
                    if (metric === 'market_cap') {{
                        newValues[i] = (d.market_cap || 1) / scale;
                    }} else if (metric === 'volume') {{
                        newValues[i] = (d.volume || 1) / scale;
                    }} else {{
                        newValues[i] = 1;  // equal weight
                    }}
                }}
            }});

            // Bug 2 fix: Recalculate parent values bottom-up
            recalculateParents(newValues, childrenMap, plotlyData.ids);

            // Update hovertemplate unit suffix
            const unitSuffix = metric === 'volume' ? 'M' : metric === 'equal' ? '' : 'B';
            const newTemplate = '<b>%{{label}}</b><br>' +
                'Regime: %{{customdata.regime}}<br>' +
                'Close: $%{{customdata.close_price}}<br>' +
                'Daily: %{{customdata.daily_change_pct}}%<br>' +
                'Value: %{{value:,.2f}}' + unitSuffix + '<extra></extra>';

            Plotly.restyle('heatmap', {{'values': [newValues], 'hovertemplate': newTemplate}});
        }}

        // Initial render
        const trace = {{
            type: 'treemap',
            ids: plotlyData.ids,
            labels: plotlyData.labels,
            parents: plotlyData.parents,
            values: normalizedValues,
            customdata: plotlyData.customdata,
            marker: {{
                colors: plotlyData.colors,
                line: {{ width: 1, color: '#1a1a2e' }}
            }},
            textinfo: 'label',
            textfont: {{ size: 14, color: '#fff' }},
            // Hover template - customdata fields accessed via customdata.field
            // Note: Plotly shows empty string for null values
            // Bug 1 fix: Dynamic unit suffix based on size metric
            hovertemplate: '<b>%{{label}}</b><br>' +
                'Regime: %{{customdata.regime}}<br>' +
                'Close: $%{{customdata.close_price}}<br>' +
                'Daily: %{{customdata.daily_change_pct}}%<br>' +
                'Value: %{{value:,.2f}}' + (sizeMetric === 'volume' ? 'M' : sizeMetric === 'equal' ? '' : 'B') + '<extra></extra>',
            pathbar: {{ visible: true, textfont: {{ size: 12 }} }},
            branchvalues: 'total',
            maxdepth: 3  // Show root -> category -> items
        }};

        const layout = {{
            margin: {{ t: 30, l: 10, r: 10, b: 10 }},
            paper_bgcolor: '#1a1a2e',
            font: {{ color: '#fff' }},
            // Explicit sizing to prevent empty render
            autosize: true,
            height: Math.max(500, window.innerHeight - 140)
        }};

        const config = {{
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
            responsive: true
        }};

        console.log('Calling Plotly.newPlot with', plotlyData.ids.length, 'nodes, root_value:', normalizedValues[0]);
        Plotly.newPlot('heatmap', [trace], layout, config).then(function(gd) {{
            console.log('Plotly render complete!');

            // CRITICAL: .on() is a Plotly method added AFTER newPlot completes
            // Click handler for navigation - must be inside .then()
            gd.on('plotly_click', function(data) {{
                if (data && data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    const symbol = point.label;
                    const url = reportUrls[symbol];
                    console.log('Clicked:', symbol, 'URL:', url);
                    if (url) {{
                        window.location.href = url;
                    }}
                }}
            }});

        }}).catch(function(err) {{
            console.error('Plotly render error:', err);
        }});

        // Control handlers (these use standard DOM events, not Plotly)
        document.getElementById('colorMetric').addEventListener('change', function(e) {{
            updateColors(e.target.value);
        }});

        document.getElementById('sizeMetric').addEventListener('change', function(e) {{
            updateSizes(e.target.value);
        }});

        // Handle window resize
        window.addEventListener('resize', function() {{
            Plotly.relayout('heatmap', {{
                height: Math.max(500, window.innerHeight - 140)
            }});
        }});
    </script>
</body>
</html>"""

        return html

    def save_heatmap(
        self,
        model: HeatmapModel,
        output_dir: Path,
        filename: str = "heatmap.html",
    ) -> Path:
        """
        Build and save heatmap HTML to file.

        Args:
            model: HeatmapModel to render
            output_dir: Output directory
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        html = self.render_heatmap_html(model, output_dir)

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Saved heatmap to {output_path}")
        return output_path
