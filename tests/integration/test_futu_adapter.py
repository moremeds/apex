"""
Simple integration test for Futu adapter.

Requires Futu OpenD to be running locally.
Run with: python -m pytest tests/integration/test_futu_adapter.py -v -s
Or directly: python tests/integration/test_futu_adapter.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.infrastructure.adapters import FutuAdapter


async def test_connection_only():
    """Test just the connection to Futu OpenD."""
    adapter = FutuAdapter(
        host="127.0.0.1",
        port=11111,
        security_firm="FUTUSECURITIES",
        trd_env="REAL",
        filter_trading_market="US",
    )

    try:
        print("\n" + "=" * 60)
        print("Testing connection to Futu OpenD...")
        print("=" * 60)
        await adapter.connect()
        print(f"✓ Connected! Account ID: {adapter._acc_id}")
        print(f"✓ Connection test passed!")
        return True

    except ConnectionError as e:
        print(f"\n✗ Connection failed: {e}")
        return False

    finally:
        await adapter.disconnect()
        print("Disconnected from Futu OpenD")


async def test_fetch_positions_and_account():
    """Test fetching positions and account info from Futu OpenD."""
    adapter = FutuAdapter(
        host="127.0.0.1",
        port=11111,
        security_firm="FUTUSECURITIES",
        trd_env="REAL",
        filter_trading_market="US",
    )

    try:
        print("\n" + "=" * 60)
        print("Connecting to Futu OpenD...")
        print("=" * 60)
        await adapter.connect()
        print(f"✓ Connected! Account ID: {adapter._acc_id}")

        # Fetch positions
        print("\n" + "-" * 60)
        print("Fetching positions...")
        print("-" * 60)
        try:
            positions = await adapter.fetch_positions()
            print(f"✓ Found {len(positions)} positions\n")

            for pos in positions:
                print(f"  {pos.asset_type.value:8} | {pos.symbol:20} | "
                      f"Qty: {pos.quantity:>8.0f} | AvgPrice: ${pos.avg_price:>10.2f}")
                if pos.asset_type.value == "OPTION":
                    print(f"           └─ {pos.underlying} {pos.expiry} "
                          f"${pos.strike} {pos.right}")
        except Exception as e:
            error_msg = str(e)
            if "disclaimer" in error_msg.lower():
                print(f"⚠ Positions fetch skipped - disclaimer not agreed")
                print(f"  Please visit the Futu URL to agree to disclaimer first")
            else:
                raise

        # Fetch account info
        print("\n" + "-" * 60)
        print("Fetching account info...")
        print("-" * 60)
        try:
            account = await adapter.fetch_account_info()
            print(f"✓ Account info retrieved\n")
            print(f"  Net Liquidation:    ${account.net_liquidation:>15,.2f}")
            print(f"  Total Cash:         ${account.total_cash:>15,.2f}")
            print(f"  Buying Power:       ${account.buying_power:>15,.2f}")
            print(f"  Margin Used:        ${account.margin_used:>15,.2f}")
            print(f"  Margin Available:   ${account.margin_available:>15,.2f}")
            print(f"  Unrealized P&L:     ${account.unrealized_pnl:>15,.2f}")
            print(f"  Realized P&L:       ${account.realized_pnl:>15,.2f}")
        except Exception as e:
            error_msg = str(e)
            if "disclaimer" in error_msg.lower():
                print(f"⚠ Account info fetch skipped - disclaimer not agreed {error_msg}")
                print(f"  Please visit the Futu URL to agree to disclaimer first")
            else:
                raise

        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)

    except ConnectionError as e:
        print(f"\n✗ Connection failed: {e}")
        print("\nMake sure:")
        print("  1. Futu OpenD is running")
        print("  2. The port (11111) is correct")
        print("  3. You have logged in to Futu OpenD")
        return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await adapter.disconnect()
        print("\nDisconnected from Futu OpenD")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Futu Adapter Integration Test")
    print("=" * 60)

    # First test connection
    conn_ok = asyncio.run(test_connection_only())

    if conn_ok:
        # Then test full data fetch
        success = asyncio.run(test_fetch_positions_and_account())
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)
