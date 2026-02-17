#!/usr/bin/env python3
"""
Add chain length groups to commercial phthalates CSV.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources" / "phthalate_lists"

# Manual chain group assignments based on longest carbon backbone
CHAIN_GROUPS = {
    "DMP": "1-3",    # Methyl = C1
    "DEP": "1-3",    # Ethyl = C2
    "DPrP": "1-3",   # Propyl = C3
    "DBP": "4-6",    # Butyl = C4
    "DIBP": "4-6",   # Isobutyl = C4
    "DnPP": "4-6",   # Pentyl = C5
    "DIPP": "4-6",   # Isopentyl = C5
    "DnHP": "4-6",   # Hexyl = C6
    "BBP": "4-6",    # Benzyl butyl = C4 longest (butyl side)
    "DnHpP": "7-8",  # Heptyl = C7
    "DIHP": "7-8",   # Isoheptyl = C7
    "DEHP": "7-8",   # 2-ethylhexyl = C8 total, C6 longest chain
    "DnOP": "7-8",   # Octyl = C8
    "DIOP": "7-8",   # Isooctyl = C8
    "DINP": "9+",    # Isononyl = C9
    "DIDP": "9+",    # Isodecyl = C10
    "DPHP": "9+",    # 2-propylheptyl = C10 total
    "DIUP": "9+",    # Isoundecyl = C11
    "DTDP": "9+",    # Isotridecyl = C13
    "DCHP": "4-6",   # Cyclohexyl = C6
}


def main():
    """Add chain groups to commercial CSV."""
    csv_path = RESOURCES_DIR / "commercial_top20.csv"
    df = pd.read_csv(csv_path)

    # Add chain_length_group column
    df["chain_length_group"] = df["abbreviation"].map(CHAIN_GROUPS)

    # Check for any missing mappings
    missing = df[df["chain_length_group"].isna()]["abbreviation"].tolist()
    if missing:
        logger.warning(f"Missing chain group mappings for: {missing}")

    # Save updated CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Updated {csv_path} with chain_length_group column")

    # Summary
    logger.info("\nChain length distribution:")
    for group in ["1-3", "4-6", "7-8", "9+"]:
        count = (df["chain_length_group"] == group).sum()
        logger.info(f"  {group}: {count}")


if __name__ == "__main__":
    main()
