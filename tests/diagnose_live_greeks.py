"""
Diagnose live Greeks and option data retrieval from IBKR.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


from ib_async import IB, Option


async def diagnose_live_greeks() -> None:
    """Test live Greeks retrieval from IBKR."""
    print("=" * 80)
    print("LIVE GREEKS DIAGNOSTIC")
    print("=" * 80)

    ib = IB()

    try:
        # Connect to IBKR
        print("\n1. Connecting to IBKR...")
        await ib.connectAsync("127.0.0.1", 4002, clientId=999)  # Paper trading port
        print("   ✓ Connected to IBKR")

        # Create option contract - TSLA 360C Dec 19
        print("\n2. Creating option contract...")
        contract = Option(
            symbol="TSLA",
            lastTradeDateOrContractMonth="20251219",
            strike=360.0,
            right="C",
            exchange="SMART",
            multiplier="100",
            currency="USD",
        )
        print(f"   Contract: {contract}")

        # Qualify contract
        print("\n3. Qualifying contract...")
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            print("   ✗ Failed to qualify contract!")
            return

        contract = qualified[0]
        print(f"   ✓ Qualified: {contract.localSymbol}")

        # Request market data WITHOUT Greeks (baseline)
        print("\n4. Testing WITHOUT Greeks (tick type '')...")
        ticker_no_greeks = ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(3)  # Wait for data

        print(f"   Bid: {ticker_no_greeks.bid}")
        print(f"   Ask: {ticker_no_greeks.ask}")
        print(f"   Last: {ticker_no_greeks.last}")
        print(f"   IV: {ticker_no_greeks.impliedVolatility}")
        print(f"   modelGreeks: {ticker_no_greeks.modelGreeks}")

        ib.cancelMktData(contract)
        await asyncio.sleep(1)

        # Request market data WITH Greeks (tick type 106)
        print("\n5. Testing WITH Greeks (tick type '106')...")
        ticker_with_greeks = ib.reqMktData(contract, "106", False, False)
        print("   Requested tick 106, waiting for data...")

        # Wait and check periodically
        for i in range(10):
            await asyncio.sleep(1)
            print(f"\n   After {i+1}s:")
            print(f"     Bid: {ticker_with_greeks.bid}")
            print(f"     Ask: {ticker_with_greeks.ask}")
            print(f"     Last: {ticker_with_greeks.last}")
            print(f"     IV: {ticker_with_greeks.impliedVolatility}")

            if ticker_with_greeks.modelGreeks:
                greeks = ticker_with_greeks.modelGreeks
                print(f"     modelGreeks FOUND:")
                print(f"       Delta: {greeks.delta}")
                print(f"       Gamma: {greeks.gamma}")
                print(f"       Vega: {greeks.vega}")
                print(f"       Theta: {greeks.theta}")
                print(f"       undPrice: {greeks.undPrice}")
                break
            else:
                print(f"     modelGreeks: None")

        if not ticker_with_greeks.modelGreeks:
            print("\n   ✗ Greeks NOT received after 10 seconds!")
            print("\n   Possible reasons:")
            print("     - Market is closed")
            print("     - Option has no liquidity")
            print("     - IBKR subscription doesn't include Greeks")
            print("     - Need to wait longer for Greeks to populate")

        # Show full ticker object
        print(f"\n6. Full ticker object:")
        print(f"   {ticker_with_greeks}")

        ib.cancelMktData(contract)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("\n7. Disconnecting...")
        ib.disconnect()
        print("   ✓ Disconnected")

    print("\n" + "=" * 80)
    print("Diagnostic complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(diagnose_live_greeks())
