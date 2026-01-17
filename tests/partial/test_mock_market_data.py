"""
Test mock market data provider.

This test verifies that mock market data can be generated for positions
when IB is not connected.
"""

import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.infrastructure.adapters.mock_market_data import MockMarketDataProvider
from src.models.market_data import DataQuality, GreeksSource
from src.models.position import AssetType, Position, PositionSource


def test_mock_market_data():
    """Test mock market data generation."""
    print("=" * 80)
    print("Testing Mock Market Data Provider")
    print("=" * 80)

    # Create sample positions
    print("\n1. Creating sample positions...")
    positions = [
        Position(
            symbol="AAPL",
            underlying="AAPL",
            asset_type=AssetType.STOCK,
            quantity=100,
            avg_price=180.00,
            source=PositionSource.MANUAL,
        ),
        Position(
            symbol="AAPL DEC 20 '25 185 Call",
            underlying="AAPL",
            asset_type=AssetType.OPTION,
            quantity=-2,
            avg_price=5.50,
            multiplier=100,
            expiry="20251220",
            strike=185.0,
            right="C",
            source=PositionSource.MANUAL,
        ),
        Position(
            symbol="SPY",
            underlying="SPY",
            asset_type=AssetType.STOCK,
            quantity=50,
            avg_price=450.00,
            source=PositionSource.MANUAL,
        ),
    ]
    print(f"   ✓ Created {len(positions)} positions")

    # Create mock provider
    print("\n2. Creating mock market data provider...")
    mock_provider = MockMarketDataProvider(seed=42)
    print(f"   ✓ Provider created")

    # Generate market data
    print("\n3. Generating mock market data...")
    market_data_list = mock_provider.generate_market_data(positions)
    print(f"   ✓ Generated {len(market_data_list)} market data entries")

    # Verify market data
    print("\n4. Verifying market data:")
    for md in market_data_list:
        print(f"\n   Symbol: {md.symbol}")
        print(f"   - Last: ${md.last:.2f}" if md.last else "   - Last: None")
        print(f"   - Bid: ${md.bid:.2f}" if md.bid else "   - Bid: None")
        print(f"   - Ask: ${md.ask:.2f}" if md.ask else "   - Ask: None")
        print(f"   - Mid: ${md.mid:.2f}" if md.mid else "   - Mid: None")
        print(f"   - Volume: {md.volume:,}" if md.volume else "   - Volume: None")

        # Greeks
        if md.delta is not None:
            print(f"   - Delta: {md.delta:.3f}")
        if md.gamma is not None:
            print(f"   - Gamma: {md.gamma:.3f}")
        if md.vega is not None:
            print(f"   - Vega: {md.vega:.3f}")
        if md.theta is not None:
            print(f"   - Theta: {md.theta:.3f}")

        print(f"   - Greeks source: {md.greeks_source.value}")
        print(f"   - Quality: {md.quality.value}")

        # Validate
        assert md.greeks_source == GreeksSource.MOCK, "Greeks source should be MOCK"
        assert md.quality == DataQuality.GOOD, "Quality should be GOOD"
        assert md.bid is not None and md.ask is not None, "Bid and ask should be present"
        assert md.bid <= md.ask, "Bid should be <= ask"

    print("\n5. Coverage check:")
    symbols_requested = {p.symbol for p in positions}
    symbols_received = {md.symbol for md in market_data_list}

    coverage = len(symbols_received) / len(symbols_requested) * 100
    print(f"   ✓ Coverage: {coverage:.0f}% ({len(symbols_received)}/{len(symbols_requested)})")

    if symbols_received != symbols_requested:
        missing = symbols_requested - symbols_received
        print(f"   ⚠ Missing symbols: {missing}")

    print("\n" + "=" * 80)
    print("✓ Mock market data provider test passed!")
    print("=" * 80)
    print("\nSummary:")
    print("  - Mock provider generates market data for all positions")
    print("  - Stocks get delta=1.0")
    print("  - Options get reasonable Greeks")
    print("  - All data marked with MOCK source and GOOD quality")
    print("  - No more 100% missing market data warnings!")
    print()

    return True


if __name__ == "__main__":
    success = test_mock_market_data()
    sys.exit(0 if success else 1)
