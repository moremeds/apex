"""Update market caps and upload to R2 meta/market_caps.json.

Reads universe.yaml symbols, fetches market caps via yfinance,
and uploads the result to R2 for dashboard consumption.

Usage:
    python scripts/r2_market_caps.py
    python scripts/r2_market_caps.py --universe config/universe.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Update market caps in R2")
    parser.add_argument(
        "--universe",
        type=str,
        default="config/universe.yaml",
        help="Path to universe.yaml (default: config/universe.yaml)",
    )
    args = parser.parse_args()

    # Load universe symbols
    from src.services.market_cap_service import MarketCapService, load_universe_symbols

    universe_path = Path(args.universe)
    if not universe_path.is_absolute():
        universe_path = PROJECT_ROOT / universe_path

    if not universe_path.exists():
        logger.error("Universe file not found: %s", universe_path)
        return 1

    symbols = load_universe_symbols(universe_path)
    if not symbols:
        logger.error("No symbols found in universe file")
        return 1
    logger.info("Loaded %d symbols from %s", len(symbols), args.universe)

    # Update market caps (sync — fetches from yfinance)
    svc = MarketCapService()
    results = svc.update_market_caps(symbols)
    success = sum(1 for r in results.values() if not r.cap_missing)
    logger.info("Market caps: %d/%d successful", success, len(symbols))

    # Upload to R2
    try:
        from src.infrastructure.adapters.r2.client import R2Client

        client = R2Client()
        caps = svc.get_all_cached_caps()
        if not caps:
            logger.error("No market caps after update")
            return 1

        client.put_json("meta/market_caps.json", caps)
        logger.info("Uploaded %d market caps to R2 meta/market_caps.json", len(caps))
    except (ValueError, ImportError) as e:
        logger.error("R2 upload failed (credentials not configured): %s", e)
        return 1
    except Exception as e:
        logger.error("R2 upload failed: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
