#!/usr/bin/env python3
"""
Classify assays in the activity matrix by experimental type (in vivo, in vitro, in silico).

Classification is based on assay title keywords and patterns since explicit
test system metadata is not available in the primary data files.
"""

import logging
from pathlib import Path
import pandas as pd
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"


def classify_assay(title: str) -> tuple[str, str]:
    """
    Classify an assay by experimental system and subtype.

    Classification hierarchy:
    1. In vivo: Whole organism studies
    2. In vitro: Studies outside living organisms
       - Cell-based: Using intact cells
       - Biochemical: Cell-free (purified proteins/enzymes/receptors)
    3. In silico: Computational predictions
    4. Unknown: Cannot determine from title

    Parameters
    ----------
    title : str
        Assay title

    Returns
    -------
    tuple[str, str]
        (primary_type, subtype) where:
        - primary_type: 'in vivo', 'in vitro', 'in silico', 'unknown'
        - subtype: 'cell-based', 'biochemical', or empty string
    """
    title_lower = title.lower()

    # In vivo markers - whole organism studies
    in_vivo_markers = [
        'zebrafish', 'embryo', 'tadpole', 'frog', 'xenopus',
        'whole organism', 'in vivo', 'animal', 'mouse', 'rat',
        'mammalian', 'oral', 'developmental', 'terato', 'larvae',
        'hpf',  # hours post fertilization
        'dpf',  # days post fertilization
    ]

    # In vitro markers - cell-based assays
    in_vitro_markers = [
        'cell', 'hepg2', 'hek293', 'mcf-7', 'h295r', 'sf9',
        'bsk', 'hela', 'mda-mb', 'mda kb', 'mdakb', 'cho',
        'vm7', 'bg1', 'casm3c', 'kf3ct', 'be3c', '3c', '4h',
        'cell line', 'cell-based', 'cellular', 'cytotoxicity',
        'proliferation', 'viability', 'cell viability',
        'in vitro', 'culture',
    ]

    # Biochemical markers - protein/enzyme assays
    biochemical_markers = [
        'binding', 'receptor', 'ligand', 'reporter', 'luciferase',
        'enzyme', 'kinase', 'protein', 'biochemical', 'substrate',
        'inhibitor', 'agonist', 'antagonist', 'ic50', 'ec50',
        'radioligand', 'fluorescence', 'luminescence', 'bla',
        'fret', 'tr-fret', 'beta-lactamase', 'diaphorase',
        'coactivator', 'cofactor', 'transactivation',
    ]

    # Computational/in silico markers (rare in experimental datasets)
    computational_markers = [
        'in silico', 'prediction', 'qsar', 'model', 'computational',
        'virtual', 'docking', 'simulated',
    ]

    # Check in order of specificity
    # 1. In silico/Computational (most specific, rarest)
    if any(marker in title_lower for marker in computational_markers):
        return 'in silico', ''

    # 2. In vivo (whole organism)
    if any(marker in title_lower for marker in in_vivo_markers):
        return 'in vivo', ''

    # 3. In vitro - determine subtype
    has_cell_markers = any(marker in title_lower for marker in in_vitro_markers)
    has_biochem_markers = any(marker in title_lower for marker in biochemical_markers)

    if has_cell_markers and has_biochem_markers:
        # Both markers present - cell-based takes precedence (likely a cell-based reporter assay)
        return 'in vitro', 'cell-based'
    elif has_cell_markers:
        return 'in vitro', 'cell-based'
    elif has_biochem_markers:
        return 'in vitro', 'biochemical'

    # 4. Unknown
    return 'unknown', ''


def main():
    """Classify all assays in the activity matrix."""
    # Load activity matrix
    activity_path = CACHE_DIR / "entity_similarity" / "activity_matrix_filled.parquet"
    if not activity_path.exists():
        logger.error(f"Activity matrix not found at {activity_path}")
        return

    df = pd.read_parquet(activity_path)
    logger.info(f"Loaded activity matrix with {df.shape[1]} assays")

    # Classify each assay
    assay_titles = df.columns.tolist()
    classifications = []

    for title in assay_titles:
        # Clean title (remove leading/trailing quotes and spaces)
        clean_title = title.strip().strip('"').strip()
        primary_type, subtype = classify_assay(clean_title)
        classifications.append({
            'title': clean_title,
            'primary_type': primary_type,
            'subtype': subtype
        })

    results_df = pd.DataFrame(classifications)

    # Calculate statistics
    primary_counts = results_df['primary_type'].value_counts()
    total = len(results_df)

    logger.info("\n" + "="*70)
    logger.info("ASSAY CLASSIFICATION SUMMARY")
    logger.info("="*70)
    logger.info("PRIMARY CLASSIFICATION:")
    logger.info("-"*70)

    for primary_type in ['in vivo', 'in vitro', 'in silico', 'unknown']:
        count = primary_counts.get(primary_type, 0)
        percentage = (count / total) * 100
        logger.info(f"  {primary_type:12s}: {count:4d} ({percentage:5.1f}%)")

    logger.info("-"*70)
    logger.info(f"  {'TOTAL':12s}: {total:4d} (100.0%)")
    logger.info("="*70)

    # Show in vitro breakdown
    in_vitro_df = results_df[results_df['primary_type'] == 'in vitro']
    if len(in_vitro_df) > 0:
        logger.info("\nIN VITRO SUBTYPE BREAKDOWN:")
        logger.info("-"*70)
        subtype_counts = in_vitro_df['subtype'].value_counts()
        in_vitro_total = len(in_vitro_df)

        for subtype in ['cell-based', 'biochemical']:
            count = subtype_counts.get(subtype, 0)
            percentage = (count / in_vitro_total) * 100
            percentage_of_total = (count / total) * 100
            logger.info(f"  {subtype:12s}: {count:4d} ({percentage:5.1f}% of in vitro, {percentage_of_total:5.1f}% of total)")

        logger.info("-"*70)
        logger.info(f"  {'TOTAL':12s}: {in_vitro_total:4d} (100.0% of in vitro)")
        logger.info("="*70)

    # Save detailed classification
    output_path = CACHE_DIR / "assay_classification.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nDetailed classification saved to {output_path}")

    # Show examples from each category
    logger.info("\nEXAMPLES FROM EACH CATEGORY:")
    logger.info("="*70)

    # In vivo examples
    examples = results_df[results_df['primary_type'] == 'in vivo']['title'].head(3)
    if len(examples) > 0:
        logger.info(f"\nIN VIVO:")
        for i, example in enumerate(examples, 1):
            logger.info(f"  {i}. {example[:75]}{'...' if len(example) > 75 else ''}")

    # In vitro - cell-based examples
    examples = results_df[(results_df['primary_type'] == 'in vitro') &
                          (results_df['subtype'] == 'cell-based')]['title'].head(3)
    if len(examples) > 0:
        logger.info(f"\nIN VITRO (Cell-based):")
        for i, example in enumerate(examples, 1):
            logger.info(f"  {i}. {example[:75]}{'...' if len(example) > 75 else ''}")

    # In vitro - biochemical examples
    examples = results_df[(results_df['primary_type'] == 'in vitro') &
                          (results_df['subtype'] == 'biochemical')]['title'].head(3)
    if len(examples) > 0:
        logger.info(f"\nIN VITRO (Biochemical/Cell-free):")
        for i, example in enumerate(examples, 1):
            logger.info(f"  {i}. {example[:75]}{'...' if len(example) > 75 else ''}")

    # In silico examples
    examples = results_df[results_df['primary_type'] == 'in silico']['title'].head(3)
    if len(examples) > 0:
        logger.info(f"\nIN SILICO (Computational):")
        for i, example in enumerate(examples, 1):
            logger.info(f"  {i}. {example[:75]}{'...' if len(example) > 75 else ''}")

    # Unknown examples
    examples = results_df[results_df['primary_type'] == 'unknown']['title'].head(3)
    if len(examples) > 0:
        logger.info(f"\nUNKNOWN:")
        for i, example in enumerate(examples, 1):
            logger.info(f"  {i}. {example[:75]}{'...' if len(example) > 75 else ''}")


if __name__ == "__main__":
    main()
