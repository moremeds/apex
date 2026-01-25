#!/usr/bin/env python
"""
Generate validation universe with reproducible, rule-based sampling.

Avoids hand-picking "obviously trending" or "obviously choppy" names.
Uses stratified sampling by market cap tier with sector coverage.

Usage:
    python scripts/generate_validation_universe.py --seed 42 --output config/validation/

Output:
    config/universe.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


@dataclass
class UniverseGenerationConfig:
    """Reproducible universe generation config."""

    seed: int = 42
    version: str = "v1.0"

    # Size targets
    total_symbols: int = 200
    holdout_pct: float = 0.30  # 30% holdout

    # Stratification
    large_cap_pct: float = 0.40
    mid_cap_pct: float = 0.40
    small_cap_pct: float = 0.20

    # Sector coverage (all 11 GICS)
    min_per_sector: int = 5


# Pre-defined symbol pools by market cap tier
# In production, these would be dynamically pulled from a data source
LARGE_CAP_SYMBOLS: List[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "META", "AVGO", "ORCL", "CSCO", "ADBE",
    "CRM", "ACN", "INTC", "AMD", "QCOM", "TXN", "IBM", "INTU", "NOW", "AMAT",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "MDT", "ISRG", "GILD", "VRTX", "SYK", "CVS", "CI", "HUM", "ELV",
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SCHW",
    "BLK", "C", "USB", "PNC", "TFC", "COF", "CME", "ICE", "MCO", "SPGI",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG",
    "MAR", "HLT", "DHI", "LEN", "PHM", "F", "GM", "ORLY", "AZO", "ROST",
    # Communication Services
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "WBD",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ", "CL", "KMB",
    "EL", "GIS", "K", "SYY", "STZ", "ADM", "HSY", "KHC", "MKC", "CHD",
    # Industrials
    "UNP", "HON", "UPS", "RTX", "BA", "CAT", "DE", "GE", "LMT", "MMM",
    "EMR", "ITW", "NSC", "CSX", "NOC", "GD", "FDX", "WM", "ETN", "PH",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD",
    "HES", "DVN", "HAL", "KMI", "WMB", "FANG", "OKE", "TRGP", "LNG", "BKR",
    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "PPG",
    "VMC", "MLM", "CTVA", "ALB", "IFF", "FMC", "CE", "CF", "MOS", "EMN",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL", "EXC", "WEC", "ED",
    "PEG", "ES", "AWK", "DTE", "ETR", "FE", "AEE", "CMS", "PPL", "ATO",
    # Real Estate
    "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
]

MID_CAP_SYMBOLS: List[str] = [
    # Technology
    "CDNS", "ANSS", "SNPS", "MCHP", "KEYS", "SWKS", "FTNT", "PANW", "ZS", "CRWD",
    "NET", "DDOG", "MDB", "SNOW", "PLTR", "OKTA", "TEAM", "ZM", "DOCU", "WDAY",
    # Healthcare
    "DXCM", "IDXX", "ALGN", "ILMN", "HOLX", "IQV", "MTD", "RMD", "TFX", "TECH",
    "EXAS", "BIO", "PODD", "JAZZ", "NBIX", "MRNA", "BNTX", "SGEN", "BMRN", "ALNY",
    # Financials
    "TROW", "CINF", "CBOE", "NDAQ", "MSCI", "FDS", "HBAN", "KEY", "CFG", "RF",
    "ALLY", "FITB", "MTB", "ZION", "SIVB", "WAL", "FRC", "EWBC", "CMA", "BOKF",
    # Consumer Discretionary
    "ULTA", "POOL", "WSM", "RH", "DPZ", "TXRH", "PENN", "CZR", "WYNN", "LVS",
    "MGM", "HAS", "MAT", "DRI", "YUM", "SBAC", "EL", "TPR", "CPRI", "GRMN",
    # Industrials
    "ODFL", "EXPD", "JBHT", "XPO", "CHRW", "SAIA", "LSTR", "HUBG", "WERN", "RXO",
    "URI", "PWR", "J", "TRMB", "BR", "ROK", "GNRC", "MIDD", "TT", "IR",
    # Energy
    "MRO", "APA", "CTRA", "PR", "MTDR", "AR", "RRC", "EQT", "CNX", "SM",
    "NOV", "CHK", "XEC", "CPE", "PDCE", "WTI", "LPI", "CRK", "DK", "HFC",
    # Materials
    "BALL", "PKG", "IP", "WRK", "SEE", "AVY", "RS", "CMC", "STLD", "X",
    "AA", "CLF", "ATI", "KALU", "HCC", "CRS", "TS", "ZEUS", "CENX", "ARNC",
    # Consumer Staples
    "BF.B", "MNST", "CLX", "SJM", "CAG", "CPB", "HRL", "COKE", "SAM", "LW",
    "THS", "HAIN", "INGR", "BGS", "USFD", "PFGC", "SPTN", "KR", "ACI", "GO",
]

SMALL_CAP_SYMBOLS: List[str] = [
    # Technology
    "LITE", "CIEN", "VIAV", "COHU", "ACLS", "AEHR", "FORM", "ONTO", "IPGP", "MKSI",
    "CRUS", "SLAB", "DIOD", "SMTC", "POWI", "VSH", "SGH", "AMKR", "AOSL", "AAOI",
    # Healthcare
    "NEOG", "UFPT", "SEM", "AMPH", "XRAY", "HSIC", "PDCO", "CNMD", "GMED", "ESTA",
    "ICUI", "HZNP", "UTHR", "RARE", "ARGX", "IONS", "SRPT", "ALKS", "EXEL", "FOLD",
    # Financials
    "OZK", "FFIN", "PNFP", "FNB", "GBCI", "UBSI", "HOPE", "WAFD", "TOWN", "WSFS",
    "CATY", "HTLF", "SBCF", "FFBC", "FRME", "SFBS", "PFBC", "FCF", "RNST", "SASR",
    # Consumer Discretionary
    "BOOT", "SHAK", "WING", "BJRI", "DENN", "JACK", "CHUY", "PLAY", "RUTH", "CAKE",
    "BLMN", "EAT", "TACO", "ARCO", "WEN", "RRGB", "LOCO", "QSR", "PZZA", "NDLS",
    # Industrials
    "HEES", "HLIO", "WWD", "RBC", "JBT", "GGG", "FELE", "DY", "MWA", "TILE",
    "NPO", "HAYW", "SPXC", "POWL", "EE", "ATRO", "ESAB", "GVA", "PRLB", "HNI",
    # Materials
    "RGLD", "WPM", "FNV", "MAG", "PAAS", "HL", "EXK", "SILV", "AG", "GOLD",
    "KGC", "BTG", "AUY", "IAG", "DRD", "HMY", "AU", "SSRM", "CDE", "USAS",
    # Energy
    "TELL", "GEVO", "BE", "PLUG", "FCEL", "BLDP", "REX", "GPOR", "CDEV", "VTLE",
    "WTI", "SD", "CPE", "HPK", "ESTE", "NOG", "PVAC", "CRK", "PDCE", "SWN",
]

# GICS Sectors for validation
GICS_SECTORS: Dict[str, List[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO", "ADBE", "CRM", "AMD", "INTC"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY"],
    "Financials": ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "C"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "EA", "CHTR"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ", "CL", "KMB"],
    "Industrials": ["UNP", "HON", "UPS", "RTX", "BA", "CAT", "DE", "GE", "LMT", "MMM"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "PPG"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL", "EXC", "WEC", "ED"],
    "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
}


def generate_universe(config: UniverseGenerationConfig) -> Dict:
    """
    Generate universe using rules, not hand-picking.

    Process:
    1. Stratify by market cap tier
    2. Ensure sector coverage
    3. Random sample within strata (seeded)
    4. Split train/holdout (seeded)

    Args:
        config: Generation configuration

    Returns:
        Dictionary with training_universe, holdout_universe, and metadata
    """
    random.seed(config.seed)

    # Deduplicate and SORT symbol pools for determinism
    large_cap = sorted(set(LARGE_CAP_SYMBOLS))
    mid_cap = sorted(set(MID_CAP_SYMBOLS))
    small_cap = sorted(set(SMALL_CAP_SYMBOLS))

    # Calculate target counts
    n_large = int(config.total_symbols * config.large_cap_pct)
    n_mid = int(config.total_symbols * config.mid_cap_pct)
    n_small = config.total_symbols - n_large - n_mid

    # Ensure sector coverage first (iterate in sorted order for determinism)
    sector_representatives: List[str] = []
    for sector in sorted(GICS_SECTORS.keys()):
        symbols = GICS_SECTORS[sector]
        available = sorted([s for s in symbols if s in large_cap or s in mid_cap])
        if available:
            selected = random.sample(available, min(config.min_per_sector, len(available)))
            sector_representatives.extend(selected)

    # Deduplicate sector reps while preserving order
    seen: Set[str] = set()
    unique_sector_reps: List[str] = []
    for s in sector_representatives:
        if s not in seen:
            seen.add(s)
            unique_sector_reps.append(s)
    sector_representatives = unique_sector_reps

    # Sample from each stratum, excluding already-selected sector reps
    large_remaining = sorted([s for s in large_cap if s not in sector_representatives])
    mid_remaining = sorted([s for s in mid_cap if s not in sector_representatives])
    small_remaining = sorted([s for s in small_cap if s not in sector_representatives])

    # Adjust counts for sector representatives already selected
    large_from_sectors = len([s for s in sector_representatives if s in large_cap])
    mid_from_sectors = len([s for s in sector_representatives if s in mid_cap])

    n_large_additional = max(0, n_large - large_from_sectors)
    n_mid_additional = max(0, n_mid - mid_from_sectors)

    # Sample additional symbols
    selected = list(sector_representatives)
    selected.extend(random.sample(large_remaining, min(n_large_additional, len(large_remaining))))
    selected.extend(random.sample(mid_remaining, min(n_mid_additional, len(mid_remaining))))
    selected.extend(random.sample(small_remaining, min(n_small, len(small_remaining))))

    # Deduplicate while preserving order
    seen = set()
    unique_selected: List[str] = []
    for s in selected:
        if s not in seen:
            seen.add(s)
            unique_selected.append(s)
    selected = unique_selected

    # Shuffle (deterministic with seed)
    random.shuffle(selected)

    # Truncate to target size if we have too many
    if len(selected) > config.total_symbols:
        selected = selected[: config.total_symbols]

    # Split train/holdout
    holdout_size = int(len(selected) * config.holdout_pct)
    holdout_universe = selected[:holdout_size]
    training_universe = selected[holdout_size:]

    return {
        "training_universe": sorted(training_universe),
        "holdout_universe": sorted(holdout_universe),
        "generation_config": {
            "seed": config.seed,
            "version": config.version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_symbols": len(selected),
            "training_count": len(training_universe),
            "holdout_count": len(holdout_universe),
            "large_cap_pct": config.large_cap_pct,
            "mid_cap_pct": config.mid_cap_pct,
            "small_cap_pct": config.small_cap_pct,
        },
    }


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate validation universe with reproducible sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="Total number of symbols (default: 200)",
    )
    parser.add_argument(
        "--holdout-pct",
        type=float,
        default=0.30,
        help="Holdout percentage (default: 0.30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/validation/",
        help="Output directory (default: config/validation/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    # Create config
    config = UniverseGenerationConfig(
        seed=args.seed,
        total_symbols=args.total,
        holdout_pct=args.holdout_pct,
    )

    print("=" * 60)
    print("VALIDATION UNIVERSE GENERATION")
    print("=" * 60)
    print(f"Seed: {config.seed}")
    print(f"Target symbols: {config.total_symbols}")
    print(f"Holdout %: {config.holdout_pct:.0%}")
    print()

    # Generate universe
    universe = generate_universe(config)

    print(f"Generated:")
    print(f"  Training: {len(universe['training_universe'])} symbols")
    print(f"  Holdout:  {len(universe['holdout_universe'])} symbols")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write YAML
    output_path = output_dir / "regime_universe.yaml"
    with open(output_path, "w") as f:
        yaml.dump(universe, f, default_flow_style=False, sort_keys=False)

    print(f"Output written to: {output_path}")

    if args.verbose:
        print()
        print("Training Universe:")
        for i, sym in enumerate(universe["training_universe"][:20]):
            print(f"  {sym}", end="")
            if (i + 1) % 10 == 0:
                print()
        if len(universe["training_universe"]) > 20:
            print(f"  ... and {len(universe['training_universe']) - 20} more")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
