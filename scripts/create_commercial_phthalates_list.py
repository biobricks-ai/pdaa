#!/usr/bin/env python3
"""
Create a CSV of the 20 most commercially important phthalates.

Based on synthesis of multiple industry sources including market reports,
production volume data, and regulatory assessments (EPA, CDC NHANES, ATSDR).

Key sources:
- DEHP: 3.24 million tonnes global production (2018)
- DINP: 28.6% market share (2021), largest individual share
- Combined DINP + DEHP = 75%+ of phthalate market
- High MW (C8-C13): DEHP, DINP, DIDP, DPHP, DnOP - primary plasticizers
- Low MW (C1-C6): DMP, DEP, DBP, DIBP, BBP - solvents, cosmetics
- EPA Priority: BBP, DBP, DCHP, DEHP, DIBP
- EU REACH: Above + DINP, DIDP, DnOP
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "resources" / "phthalate_lists"

# Commercial phthalates ranked by production volume and commercial importance
# Ranking based on synthesis of market data, production volumes, and regulatory lists
COMMERCIAL_PHTHALATES = [
    # Rank 1-2: Dominant high MW plasticizers (75%+ of market)
    {
        "rank": 1,
        "abbreviation": "DEHP",
        "name": "Di(2-ethylhexyl) phthalate",
        "smiles": "CCCCC(CC)COC(=O)c1ccccc1C(=O)OCC(CC)CCCC",
        "commercial_category": "High MW plasticizer",
        "production_volume": "3.24M tonnes (2018)",
        "market_notes": "Historically #1, being phased out but still dominant"
    },
    {
        "rank": 2,
        "abbreviation": "DINP",
        "name": "Diisononyl phthalate",
        "smiles": "CC(C)CCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCC(C)C",
        "commercial_category": "High MW plasticizer",
        "production_volume": "28.6% market share (2021)",
        "market_notes": "Largest individual market share, DEHP replacement"
    },
    # Rank 3-5: Other major high MW plasticizers
    {
        "rank": 3,
        "abbreviation": "DIDP",
        "name": "Diisodecyl phthalate",
        "smiles": "CC(C)CCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCC(C)C",
        "commercial_category": "High MW plasticizer",
        "production_volume": "Major (est. >100k tonnes/yr)",
        "market_notes": "DEHP replacement, EU REACH listed"
    },
    {
        "rank": 4,
        "abbreviation": "DnOP",
        "name": "Di-n-octyl phthalate",
        "smiles": "CCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCC",
        "commercial_category": "High MW plasticizer",
        "production_volume": "Major (DOP market: $2.56B in 2023)",
        "market_notes": "EU REACH listed, primary plasticizer"
    },
    {
        "rank": 5,
        "abbreviation": "DPHP",
        "name": "Di(2-propylheptyl) phthalate",
        "smiles": "CCCCCCC(CCC)COC(=O)c1ccccc1C(=O)OCC(CCC)CCCCCC",
        "commercial_category": "High MW plasticizer",
        "production_volume": "Growing (DEHP replacement)",
        "market_notes": "Non-ortho phthalate alternative gaining market share"
    },
    # Rank 6-10: Important low-medium MW phthalates (regulatory priority, specialized uses)
    {
        "rank": 6,
        "abbreviation": "DBP",
        "name": "Di-n-butyl phthalate",
        "smiles": "CCCCOC(=O)c1ccccc1C(=O)OCCCC",
        "commercial_category": "Low MW solvent/plasticizer",
        "production_volume": "Major (most common low MW)",
        "market_notes": "EPA Priority, EU REACH, widely used in adhesives/coatings"
    },
    {
        "rank": 7,
        "abbreviation": "BBP",
        "name": "Benzyl butyl phthalate",
        "smiles": "CCCCOC(=O)c1ccccc1C(=O)OCc2ccccc2",
        "commercial_category": "Medium MW plasticizer",
        "production_volume": "Moderate",
        "market_notes": "EPA Priority, EU REACH, flooring applications"
    },
    {
        "rank": 8,
        "abbreviation": "DIBP",
        "name": "Diisobutyl phthalate",
        "smiles": "CC(C)COC(=O)c1ccccc1C(=O)OCC(C)C",
        "commercial_category": "Low MW solvent/plasticizer",
        "production_volume": "Growing (DBP substitute)",
        "market_notes": "EPA Priority, EU REACH, DBP replacement"
    },
    {
        "rank": 9,
        "abbreviation": "DEP",
        "name": "Diethyl phthalate",
        "smiles": "CCOC(=O)c1ccccc1C(=O)OCC",
        "commercial_category": "Low MW solvent",
        "production_volume": "Major (cosmetics, personal care)",
        "market_notes": "CDC NHANES biomonitoring, cosmetics/fragrances"
    },
    {
        "rank": 10,
        "abbreviation": "DMP",
        "name": "Dimethyl phthalate",
        "smiles": "COC(=O)c1ccccc1C(=O)OC",
        "commercial_category": "Low MW solvent",
        "production_volume": "Moderate (specialty applications)",
        "market_notes": "Insect repellents, plastics manufacturing"
    },
    # Rank 11-15: Additional commercial phthalates with moderate use
    {
        "rank": 11,
        "abbreviation": "DIOP",
        "name": "Diisooctyl phthalate",
        "smiles": "CC(C)CCCCCOC(=O)c1ccccc1C(=O)OCCCCCC(C)C",
        "commercial_category": "High MW plasticizer",
        "production_volume": "Moderate",
        "market_notes": "Alternative to DEHP in some applications"
    },
    {
        "rank": 12,
        "abbreviation": "DnHP",
        "name": "Di-n-hexyl phthalate",
        "smiles": "CCCCCCOC(=O)c1ccccc1C(=O)OCCCCCC",
        "commercial_category": "Medium MW plasticizer",
        "production_volume": "Moderate",
        "market_notes": "Specialty plasticizer applications"
    },
    {
        "rank": 13,
        "abbreviation": "DCHP",
        "name": "Dicyclohexyl phthalate",
        "smiles": "O=C(OC1CCCCC1)c2ccccc2C(=O)OC3CCCCC3",
        "commercial_category": "Specialty plasticizer",
        "production_volume": "Lower volume specialty",
        "market_notes": "EPA Priority (regulatory focus), limited commercial use"
    },
    {
        "rank": 14,
        "abbreviation": "DTDP",
        "name": "Diisotridecyl phthalate",
        "smiles": "CC(C)CCCCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCCCC(C)C",
        "commercial_category": "High MW plasticizer",
        "production_volume": "Moderate (specialty)",
        "market_notes": "Very high MW, specialty flexible PVC"
    },
    {
        "rank": 15,
        "abbreviation": "DPrP",
        "name": "Di-n-propyl phthalate",
        "smiles": "CCCOC(=O)c1ccccc1C(=O)OCCC",
        "commercial_category": "Low MW solvent",
        "production_volume": "Moderate",
        "market_notes": "Research standard, limited commercial use"
    },
    # Rank 16-20: Additional phthalates for structural coverage
    {
        "rank": 16,
        "abbreviation": "DIPP",
        "name": "Diisopentyl phthalate",
        "smiles": "CC(C)CCOC(=O)c1ccccc1C(=O)OCCC(C)C",
        "commercial_category": "Medium MW plasticizer",
        "production_volume": "Moderate",
        "market_notes": "Also called diisoamyl phthalate"
    },
    {
        "rank": 17,
        "abbreviation": "DnPP",
        "name": "Di-n-pentyl phthalate",
        "smiles": "CCCCCOC(=O)c1ccccc1C(=O)OCCCCC",
        "commercial_category": "Medium MW plasticizer",
        "production_volume": "Moderate",
        "market_notes": "Specialty applications"
    },
    {
        "rank": 18,
        "abbreviation": "DIHP",
        "name": "Diisoheptyl phthalate",
        "smiles": "CC(C)CCCCOC(=O)c1ccccc1C(=O)OCCCCC(C)C",
        "commercial_category": "High MW plasticizer",
        "production_volume": "Moderate",
        "market_notes": "Historically important (1990s references)"
    },
    {
        "rank": 19,
        "abbreviation": "DnHpP",
        "name": "Di-n-heptyl phthalate",
        "smiles": "CCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCC",
        "commercial_category": "Medium MW plasticizer",
        "production_volume": "Lower volume",
        "market_notes": "Research/analytical standard"
    },
    {
        "rank": 20,
        "abbreviation": "DIUP",
        "name": "Diisoundecyl phthalate",
        "smiles": "CC(C)CCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCC(C)C",
        "commercial_category": "High MW plasticizer",
        "production_volume": "Lower volume specialty",
        "market_notes": "High MW specialty applications"
    },
]


def main():
    """Create commercial phthalates CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(COMMERCIAL_PHTHALATES)

    output_path = OUTPUT_DIR / "commercial_top20.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Created commercial phthalates list: {output_path}")
    logger.info(f"Total phthalates: {len(df)}")

    # Summary by category
    logger.info("\nBy commercial category:")
    for category, count in df['commercial_category'].value_counts().items():
        logger.info(f"  {category}: {count}")


if __name__ == "__main__":
    main()
