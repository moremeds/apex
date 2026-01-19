"""
Regime Methodology - Educational explanation of the regime system.

Provides the methodology section that helps users understand how
regime classification works.
"""

from __future__ import annotations

from ..value_card import render_section


def generate_methodology_html(theme: str = "dark") -> str:
    """
    Generate the Methodology section explaining how regime classification works.

    This is educational content that helps users understand the system.
    """
    body = """
    <div class="methodology-content">
        <h3>Overview</h3>
        <p>The regime classification system uses a <strong>priority-based decision tree</strong>
        to classify market conditions into one of four regimes:</p>

        <div class="regime-grid">
            <div class="regime-item r0">
                <div class="regime-code">R0</div>
                <div class="regime-label">Healthy Uptrend</div>
                <div class="regime-desc">Full trading allowed. TrendUp + NormalVol + Trending.</div>
            </div>
            <div class="regime-item r1">
                <div class="regime-code">R1</div>
                <div class="regime-label">Choppy/Extended</div>
                <div class="regime-desc">Reduced frequency, wider spreads. TrendUp but Choppy OR Overbought.</div>
            </div>
            <div class="regime-item r2">
                <div class="regime-code">R2</div>
                <div class="regime-label">Risk-Off</div>
                <div class="regime-desc">No new positions (veto). TrendDown OR (HighVol + close&lt;MA50) OR IV_HIGH.</div>
            </div>
            <div class="regime-item r3">
                <div class="regime-code">R3</div>
                <div class="regime-label">Rebound Window</div>
                <div class="regime-desc">Small defined-risk only. HighVol + Oversold + structural confirm.</div>
            </div>
        </div>

        <h3>Priority Order</h3>
        <p>The decision tree evaluates regimes in strict priority order (highest to lowest):</p>
        <ol class="priority-list">
            <li><strong>R2 (Risk-Off)</strong> - Veto power, always checked first</li>
            <li><strong>R3 (Rebound)</strong> - Only if NOT in active downtrend + structural confirm</li>
            <li><strong>R1 (Choppy)</strong> - Only if NOT in strong trend acceleration</li>
            <li><strong>R0 (Healthy)</strong> - Default when conditions are favorable</li>
        </ol>
        <p>If no regime conditions are fully met, the system defaults to R1 (Choppy).</p>

        <h3>Hysteresis (Stability)</h3>
        <p>To prevent whipsaw transitions, the system uses <strong>hysteresis</strong>:</p>
        <ul>
            <li><strong>Entry threshold:</strong> Bars needed to confirm a new regime</li>
            <li><strong>Exit threshold:</strong> Minimum bars before leaving current regime</li>
        </ul>
        <p>This is why <code>decision_regime</code> (raw tree output) may differ from
        <code>final_regime</code> (after hysteresis).</p>

        <h3>Components</h3>
        <table class="component-table">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Inputs</th>
                    <th>States</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Trend</td>
                    <td>Close, MA50, MA200, MA50 slope</td>
                    <td>UP, DOWN, NEUTRAL</td>
                </tr>
                <tr>
                    <td>Volatility</td>
                    <td>ATR20, ATR percentile (63d, 252d)</td>
                    <td>HIGH, NORMAL, LOW</td>
                </tr>
                <tr>
                    <td>Choppiness</td>
                    <td>CHOP index, CHOP percentile, MA20 crosses</td>
                    <td>CHOPPY, TRENDING, NEUTRAL</td>
                </tr>
                <tr>
                    <td>Extension</td>
                    <td>(Close - MA20) / ATR20</td>
                    <td>OVERBOUGHT, OVERSOLD, SLIGHTLY_HIGH, SLIGHTLY_LOW, NEUTRAL</td>
                </tr>
                <tr>
                    <td>IV (market only)</td>
                    <td>VIX/VXN percentile (63d)</td>
                    <td>HIGH, ELEVATED, NORMAL, LOW, NA</td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    return render_section(
        title="Methodology",
        body=body,
        collapsed=True,
        icon="📖",
        section_id="methodology-section",
    )
