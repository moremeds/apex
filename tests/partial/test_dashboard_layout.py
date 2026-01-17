"""Test script to visualize the dashboard layout with sample data."""

import time
from datetime import datetime

from src.infrastructure.monitoring import ComponentHealth, HealthStatus
from src.models.market_data import DataQuality, GreeksSource, MarketData
from src.models.position import AssetType, Position, PositionSource
from src.models.risk_snapshot import RiskSnapshot
from src.tui.dashboard import TerminalDashboard


def create_sample_positions():
    """Create sample positions for testing."""
    positions = []

    # GLD positions
    positions.append(
        Position(
            symbol="GLD",
            underlying="GLD",
            asset_type=AssetType.STOCK,
            quantity=-3,
            avg_price=373.00,
            source=PositionSource.IB,
        )
    )

    positions.append(
        Position(
            symbol="GLD NOV 24 '25 373 Put",
            underlying="GLD",
            asset_type=AssetType.OPTION,
            quantity=-3,
            avg_price=1.848,
            multiplier=100,
            expiry="20251124",
            strike=373.0,
            right="P",
            source=PositionSource.IB,
        )
    )

    # QQQ positions
    positions.append(
        Position(
            symbol="QQQ",
            underlying="QQQ",
            asset_type=AssetType.STOCK,
            quantity=279.4413,
            avg_price=595.13,
            source=PositionSource.IB,
        )
    )

    positions.append(
        Position(
            symbol="QQQ DEC 05 '25 587 Put",
            underlying="QQQ",
            asset_type=AssetType.OPTION,
            quantity=-1,
            avg_price=10.426,
            multiplier=100,
            expiry="20251205",
            strike=587.0,
            right="P",
            source=PositionSource.IB,
        )
    )

    positions.append(
        Position(
            symbol="QQQ JAN 16 '26 594.78 Put",
            underlying="QQQ",
            asset_type=AssetType.OPTION,
            quantity=-1,
            avg_price=23.38,
            multiplier=100,
            expiry="20260116",
            strike=594.78,
            right="P",
            source=PositionSource.IB,
        )
    )

    # SPY position
    positions.append(
        Position(
            symbol="SPY",
            underlying="SPY",
            asset_type=AssetType.STOCK,
            quantity=0.0658,
            avg_price=662.66,
            source=PositionSource.IB,
        )
    )

    # VOO position
    positions.append(
        Position(
            symbol="VOO",
            underlying="VOO",
            asset_type=AssetType.STOCK,
            quantity=9.3128,
            avg_price=609.28,
            source=PositionSource.IB,
        )
    )

    return positions


def create_sample_market_data():
    """Create sample market data."""
    market_data = {}

    # GLD
    market_data["GLD"] = MarketData(
        symbol="GLD",
        last=374.95,
        bid=374.90,
        ask=375.00,
        mid=374.95,
        delta=1.0,
        timestamp=datetime.now(),
        greeks_source=GreeksSource.IBKR,
        quality=DataQuality.GOOD,
    )

    market_data["GLD NOV 24 '25 373 Put"] = MarketData(
        symbol="GLD NOV 24 '25 373 Put",
        last=1.848,
        bid=1.84,
        ask=1.86,
        mid=1.85,
        delta=-0.397,
        gamma=0.077,
        vega=0.13,
        theta=-0.347,
        timestamp=datetime.now(),
        greeks_source=GreeksSource.IBKR,
        quality=DataQuality.GOOD,
    )

    # QQQ
    market_data["QQQ"] = MarketData(
        symbol="QQQ",
        last=595.13,
        bid=595.10,
        ask=595.16,
        mid=595.13,
        delta=1.0,
        timestamp=datetime.now(),
        greeks_source=GreeksSource.IBKR,
        quality=DataQuality.GOOD,
    )

    market_data["QQQ DEC 05 '25 587 Put"] = MarketData(
        symbol="QQQ DEC 05 '25 587 Put",
        last=10.426,
        bid=10.40,
        ask=10.45,
        mid=10.425,
        delta=-0.44,
        gamma=0.01,
        vega=0.46,
        theta=-0.42,
        timestamp=datetime.now(),
        greeks_source=GreeksSource.IBKR,
        quality=DataQuality.GOOD,
    )

    market_data["QQQ JAN 16 '26 594.78 Put"] = MarketData(
        symbol="QQQ JAN 16 '26 594.78 Put",
        last=23.38,
        bid=23.35,
        ask=23.41,
        mid=23.38,
        delta=-0.50,
        gamma=0.01,
        vega=0.92,
        theta=-0.17,
        timestamp=datetime.now(),
        greeks_source=GreeksSource.IBKR,
        quality=DataQuality.GOOD,
    )

    # SPY
    market_data["SPY"] = MarketData(
        symbol="SPY",
        last=662.66,
        bid=662.60,
        ask=662.72,
        mid=662.66,
        delta=1.0,
        timestamp=datetime.now(),
        greeks_source=GreeksSource.IBKR,
        quality=DataQuality.GOOD,
    )

    # VOO
    market_data["VOO"] = MarketData(
        symbol="VOO",
        last=609.28,
        bid=609.20,
        ask=609.36,
        mid=609.28,
        delta=1.0,
        timestamp=datetime.now(),
        greeks_source=GreeksSource.IBKR,
        quality=DataQuality.GOOD,
    )

    return market_data


def create_sample_snapshot():
    """Create sample risk snapshot."""
    return RiskSnapshot(
        timestamp=datetime.now(),
        total_unrealized_pnl=-4057,
        total_daily_pnl=1523,
        portfolio_delta=503,
        portfolio_gamma=-25,
        portfolio_vega=-176,
        portfolio_theta=162,
        total_gross_notional=500000,
        total_net_notional=450000,
        max_underlying_notional=222720,
        max_underlying_symbol="QQQ",
        concentration_pct=0.4454,
        margin_utilization=0.35,
        buying_power=250000,
    )


def create_sample_health():
    """Create sample health statuses."""
    return [
        ComponentHealth(
            component_name="ib_adapter",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
        ),
        ComponentHealth(
            component_name="file_loader",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
        ),
        ComponentHealth(
            component_name="market_data_coverage",
            status=HealthStatus.DEGRADED,
            last_check=datetime.now(),
            message="2 positions missing market data",
        ),
        ComponentHealth(
            component_name="snapshot_freshness",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
        ),
    ]


def main():
    """Run dashboard test."""
    # Create sample data
    positions = create_sample_positions()
    market_data = create_sample_market_data()
    snapshot = create_sample_snapshot()
    breaches = []  # No breaches for now
    health = create_sample_health()

    # Initialize dashboard
    config = {"show_positions": True}
    dashboard = TerminalDashboard(config)

    # Start dashboard
    dashboard.start()

    try:
        # Update with sample data
        dashboard.update(snapshot, breaches, health, positions, market_data)

        # Keep running for 10 seconds
        print("Dashboard running... (will stop after 10 seconds)")
        time.sleep(10)

    finally:
        dashboard.stop()
        print("Dashboard stopped")


if __name__ == "__main__":
    main()
