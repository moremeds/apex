"""
Futu security code parsing utilities.

Parses Futu code formats to extract asset details.
"""

import re
from typing import Optional, Tuple

from ....models.position import AssetType


def parse_futu_code(
    code: str,
) -> Tuple[AssetType, str, str, Optional[str], Optional[float], Optional[str]]:
    """
    Parse Futu security code to extract asset details.

    Futu code formats:
    - Stock: "US.AAPL", "HK.00700"
    - Option: "US.AAPL240119C190000" (underlying + YYMMDD + C/P + strike*1000)

    Args:
        code: Futu security code string.

    Returns:
        Tuple of (asset_type, symbol, underlying, expiry, strike, right)
    """
    # Remove market prefix (e.g., "US.", "HK.")
    if "." in code:
        market, ticker = code.split(".", 1)
    else:
        ticker = code

    # Check if it's an option (has date and C/P in the format)
    # Option format: SYMBOL + YYMMDD + C/P + STRIKE*1000
    # Example: AAPL240119C190000 (AAPL Jan 19, 2024 Call $190)
    option_pattern = r"^([A-Z]+)(\d{6})([CP])(\d+)$"
    match = re.match(option_pattern, ticker)

    if match:
        underlying = match.group(1)
        date_str = match.group(2)  # YYMMDD
        right = match.group(3)  # C or P
        strike_raw = match.group(4)

        # Convert YYMMDD to YYYYMMDD
        year = int(date_str[:2])
        year_full = 2000 + year if year < 50 else 1900 + year
        expiry = f"{year_full}{date_str[2:]}"

        # Strike is stored as strike * 1000
        strike = float(strike_raw) / 1000.0

        return (
            AssetType.OPTION,
            ticker,  # Full option symbol
            underlying,
            expiry,
            strike,
            right,
        )
    else:
        # It's a stock
        return (
            AssetType.STOCK,
            ticker,
            ticker,
            None,
            None,
            None,
        )
