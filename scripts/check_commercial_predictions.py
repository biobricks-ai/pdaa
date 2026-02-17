#!/usr/bin/env python3
"""
Check which commercial phthalates have predictions in the activity matrix.
"""

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
RESOURCES_DIR = PROJECT_ROOT / "resources"


def main():
    """Check predictions for commercial phthalates."""
    # Load activity matrix
    activity_path = CACHE_DIR / "entity_similarity" / "activity_matrix_filled.parquet"
    activity_df = pd.read_parquet(activity_path)
    logger.info(f"Loaded activity matrix with {len(activity_df)} compounds")

    # Load commercial phthalates
    commercial_path = RESOURCES_DIR / "phthalate_lists" / "commercial_top20.csv"
    commercial_df = pd.read_csv(commercial_path)
    logger.info(f"Loaded {len(commercial_df)} commercial phthalates")

    # Check which have predictions
    results = []
    for _, row in commercial_df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            logger.warning(f"Invalid SMILES for {row['abbreviation']}")
            continue

        inchi = Chem.MolToInchi(mol)
        has_predictions = inchi in activity_df.index

        results.append({
            "rank": row["rank"],
            "abbreviation": row["abbreviation"],
            "name": row["name"],
            "has_predictions": has_predictions,
            "mean_activity": activity_df.loc[inchi].mean() if has_predictions else None
        })

    results_df = pd.DataFrame(results)

    # Summary
    n_with_predictions = results_df["has_predictions"].sum()
    n_total = len(results_df)
    logger.info(f"\n{n_with_predictions}/{n_total} commercial phthalates have predictions")

    # Show which ones have predictions
    logger.info("\nWith predictions:")
    for _, row in results_df[results_df["has_predictions"]].iterrows():
        logger.info(f"  Rank {row['rank']:2d}: {row['abbreviation']:6s} - {row['name']}")

    logger.info("\nWithout predictions:")
    for _, row in results_df[~results_df["has_predictions"]].iterrows():
        logger.info(f"  Rank {row['rank']:2d}: {row['abbreviation']:6s} - {row['name']}")


if __name__ == "__main__":
    main()
