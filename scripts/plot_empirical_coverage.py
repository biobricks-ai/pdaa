#!/usr/bin/env python3
"""
Create empirical data coverage heatmap showing which chemical-assay pairs
have training data from ICE/ToxCast vs. model predictions.

Generates:
1. Dot-matrix heatmap of empirical coverage
2. Side-by-side comparison: empirical vs predictions
3. Coverage statistics report
"""

import logging
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = CACHE_DIR / "empirical_coverage"


def map_dtxsid_to_inchi(dtxsid: str) -> str:
    """
    Map DTXSID to InChI using CompTox Dashboard API.

    Parameters
    ----------
    dtxsid : str
        DSSTox Substance Identifier (e.g., DTXSID7020182)

    Returns
    -------
    str
        Standard InChI or None if not found
    """
    import requests

    try:
        # CompTox API endpoint
        url = f"https://comptox.epa.gov/dashboard-api/ccdapp2/chemical-detail/search/by-dtxsid/{dtxsid}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            inchi = data.get('inchi', None)
            return inchi
        else:
            return None

    except Exception as e:
        logger.warning(f"Failed to map {dtxsid}: {e}")
        return None


def build_dtxsid_to_inchi_cache(
    dtxsids: Set[str],
    cache_path: Path
) -> Dict[str, str]:
    """
    Build or load DTXSID → InChI mapping cache.

    Parameters
    ----------
    dtxsids : set of str
        All DTXSID values to map
    cache_path : Path
        Path to cache CSV file

    Returns
    -------
    dict
        Mapping {dtxsid: inchi}
    """
    # Load existing cache if available
    if cache_path.exists():
        logger.info(f"Loading DTXSID→InChI cache from {cache_path}")
        cache_df = pd.read_csv(cache_path)
        mapping = dict(zip(cache_df['dtxsid'], cache_df['inchi']))

        # Find DTXSIDs not in cache
        missing = dtxsids - set(mapping.keys())
        logger.info(f"Cache contains {len(mapping)} mappings, {len(missing)} missing")
    else:
        mapping = {}
        missing = dtxsids
        logger.info(f"No cache found, will map {len(missing)} DTXSIDs")

    # Map missing DTXSIDs
    if missing:
        logger.info("Mapping DTXSIDs to InChI via CompTox API...")
        from tqdm import tqdm
        import time

        new_mappings = []
        for dtxsid in tqdm(missing, desc="Mapping DTXSIDs"):
            inchi = map_dtxsid_to_inchi(dtxsid)
            if inchi:
                mapping[dtxsid] = inchi
                new_mappings.append({'dtxsid': dtxsid, 'inchi': inchi})

            # Rate limiting
            time.sleep(0.1)

        # Update cache
        if new_mappings:
            new_df = pd.DataFrame(new_mappings)
            if cache_path.exists():
                existing_df = pd.read_csv(cache_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            combined_df.to_csv(cache_path, index=False)
            logger.info(f"Updated cache with {len(new_mappings)} new mappings")

    return mapping


def create_empirical_coverage_matrix(
    empirical_pairs: pd.DataFrame,
    activity_matrix: pd.DataFrame,
    dtxsid_to_inchi: Dict[str, str]
) -> Tuple[np.ndarray, pd.Index, pd.Index]:
    """
    Create binary matrix showing empirical data coverage.

    Parameters
    ----------
    empirical_pairs : DataFrame
        Columns: dtxsid, assay_token
    activity_matrix : DataFrame
        Rows = InChI, Columns = assay titles
    dtxsid_to_inchi : dict
        DTXSID → InChI mapping

    Returns
    -------
    coverage_matrix : ndarray (bool)
        Binary matrix (chemicals × assays)
    chemicals : Index
        Chemical InChI identifiers (rows)
    assays : Index
        Assay titles (columns)
    """
    # Get chemicals and assays from activity matrix
    chemicals = activity_matrix.index
    assays = activity_matrix.columns

    # Build mapping from assay token to column index
    # Need to map token → title
    # Load token mapping
    token_map_path = CACHE_DIR / "entity_similarity" / "token_title_mapping.csv"
    if token_map_path.exists():
        token_map_df = pd.read_csv(token_map_path)
        token_to_title = dict(zip(token_map_df['token'].astype(str), token_map_df['title']))
    else:
        logger.warning("Token mapping file not found, will use direct matching")
        token_to_title = {}

    # Create index mappings
    chem_to_idx = {inchi: i for i, inchi in enumerate(chemicals)}
    assay_to_idx = {title: i for i, title in enumerate(assays)}

    # Build coverage matrix
    n_chem = len(chemicals)
    n_assay = len(assays)
    coverage = np.zeros((n_chem, n_assay), dtype=bool)

    matched_pairs = 0
    unmatched_chem = set()
    unmatched_assay = set()

    for _, row in empirical_pairs.iterrows():
        dtxsid = row['dtxsid']
        assay_token = str(row['assay_token'])

        # Map DTXSID to InChI
        inchi = dtxsid_to_inchi.get(dtxsid)
        if inchi is None or inchi not in chem_to_idx:
            unmatched_chem.add(dtxsid)
            continue

        # Map token to assay title
        assay_title = token_to_title.get(assay_token)
        if assay_title is None or assay_title not in assay_to_idx:
            unmatched_assay.add(assay_token)
            continue

        # Mark as having empirical data
        i = chem_to_idx[inchi]
        j = assay_to_idx[assay_title]
        coverage[i, j] = True
        matched_pairs += 1

    logger.info(f"Matched {matched_pairs} empirical pairs to activity matrix")
    logger.info(f"Unmatched chemicals: {len(unmatched_chem)}")
    logger.info(f"Unmatched assays: {len(unmatched_assay)}")

    return coverage, chemicals, assays


def plot_empirical_coverage(
    coverage_matrix: np.ndarray,
    chemicals: pd.Index,
    assays: pd.Index,
    output_path: Path
):
    """
    Create dot-matrix heatmap showing empirical coverage.

    Parameters
    ----------
    coverage_matrix : ndarray (bool)
        Binary matrix (chemicals × assays)
    chemicals : Index
        Chemical InChI identifiers (rows)
    assays : Index
        Assay titles (columns)
    output_path : Path
        Where to save figure
    """
    n_chem, n_assay = coverage_matrix.shape
    total_pairs = n_chem * n_assay
    empirical_count = coverage_matrix.sum()
    coverage_pct = (empirical_count / total_pairs) * 100

    logger.info(f"Empirical coverage: {empirical_count:,} / {total_pairs:,} ({coverage_pct:.2f}%)")

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12), dpi=150)

    # Get coordinates of empirical data points
    y_coords, x_coords = np.where(coverage_matrix)

    ax.scatter(
        x_coords, y_coords,
        s=0.5,  # Very small dots
        c='black',
        alpha=0.7,
        rasterized=True  # Faster rendering
    )

    ax.set_xlim(-0.5, n_assay - 0.5)
    ax.set_ylim(n_chem - 0.5, -0.5)  # Invert y-axis to match heatmap convention
    ax.set_xlabel('Assays', fontsize=16)
    ax.set_ylabel('Chemicals', fontsize=16)
    ax.set_title(
        f'Empirical Training Data Coverage (ICE/ToxCast)\n'
        f'{empirical_count:,} chemical-assay pairs ({coverage_pct:.2f}% of {total_pairs:,})',
        fontsize=18,
        fontweight='bold'
    )

    # Add grid for visual reference
    ax.grid(False)
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved empirical coverage plot to {output_path}")


def plot_comparison(
    empirical_matrix: np.ndarray,
    prediction_matrix: pd.DataFrame,
    output_path: Path
):
    """
    Create side-by-side comparison of empirical vs predictions.

    Parameters
    ----------
    empirical_matrix : ndarray (bool)
        Binary empirical coverage matrix
    prediction_matrix : DataFrame
        Continuous prediction values
    output_path : Path
        Where to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12), dpi=150)

    # Panel A: Empirical coverage (dots)
    n_chem, n_assay = empirical_matrix.shape
    y, x = np.where(empirical_matrix)

    ax1.scatter(x, y, s=0.3, c='black', alpha=0.6, rasterized=True)
    ax1.set_xlim(-0.5, n_assay - 0.5)
    ax1.set_ylim(n_chem - 0.5, -0.5)
    ax1.set_xlabel('Assays', fontsize=14)
    ax1.set_ylabel('Chemicals', fontsize=14)
    ax1.set_title('A) Empirical Training Data\n(ICE/ToxCast Observations)', fontsize=16, fontweight='bold')
    ax1.set_facecolor('white')

    # Panel B: Predictions (heatmap)
    sns.heatmap(
        prediction_matrix,
        ax=ax2,
        cmap='viridis',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Predicted Activity', 'shrink': 0.8},
        vmin=0,
        vmax=1
    )
    ax2.set_title('B) ToxTransformer Predictions\n(Model Outputs)', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved comparison plot to {output_path}")


def main():
    """Main entry point."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load empirical pairs
    empirical_pairs_path = CACHE_DIR / "ice_empirical_pairs.csv"
    if not empirical_pairs_path.exists():
        logger.error(f"Empirical pairs file not found: {empirical_pairs_path}")
        logger.error("Run download_ice_empirical_data.py first")
        return

    empirical_pairs = pd.read_csv(empirical_pairs_path)
    logger.info(f"Loaded {len(empirical_pairs)} empirical chemical-assay pairs")

    # Load activity matrix
    activity_matrix_path = CACHE_DIR / "entity_similarity" / "activity_matrix_filled.parquet"
    activity_matrix = pd.read_parquet(activity_matrix_path)
    logger.info(f"Loaded activity matrix: {activity_matrix.shape[0]} chemicals × {activity_matrix.shape[1]} assays")

    # Build DTXSID → InChI mapping
    dtxsids = set(empirical_pairs['dtxsid'].dropna())
    cache_path = CACHE_DIR / "dtxsid_to_inchi_cache.csv"
    dtxsid_to_inchi = build_dtxsid_to_inchi_cache(dtxsids, cache_path)

    logger.info(f"DTXSID→InChI mapping: {len(dtxsid_to_inchi)} chemicals")

    # Create empirical coverage matrix
    coverage, chemicals, assays = create_empirical_coverage_matrix(
        empirical_pairs,
        activity_matrix,
        dtxsid_to_inchi
    )

    # Plot empirical coverage
    coverage_plot_path = OUTPUT_DIR / "empirical_coverage_heatmap.png"
    plot_empirical_coverage(coverage, chemicals, assays, coverage_plot_path)

    # Plot comparison
    comparison_plot_path = OUTPUT_DIR / "empirical_vs_predictions.png"
    plot_comparison(coverage, activity_matrix, comparison_plot_path)

    # Save coverage matrix for later analysis
    coverage_df = pd.DataFrame(
        coverage.astype(int),
        index=chemicals,
        columns=assays
    )
    coverage_matrix_path = OUTPUT_DIR / "empirical_coverage_matrix.parquet"
    coverage_df.to_parquet(coverage_matrix_path)
    logger.info(f"Saved coverage matrix to {coverage_matrix_path}")

    # Generate coverage statistics report
    report_path = OUTPUT_DIR / "coverage_statistics.txt"
    with open(report_path, 'w') as f:
        n_chem, n_assay = coverage.shape
        total_pairs = n_chem * n_assay
        empirical_count = coverage.sum()
        coverage_pct = (empirical_count / total_pairs) * 100

        f.write("EMPIRICAL DATA COVERAGE STATISTICS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total chemicals in study: {n_chem:,}\n")
        f.write(f"Total assays in study: {n_assay:,}\n")
        f.write(f"Possible chemical-assay pairs: {total_pairs:,}\n\n")

        f.write(f"Empirical pairs from ICE: {empirical_count:,}\n")
        f.write(f"Coverage percentage: {coverage_pct:.2f}%\n\n")

        # Coverage by chemical
        chem_coverage = coverage.sum(axis=1)
        f.write(f"Chemicals with >0 empirical assays: {(chem_coverage > 0).sum():,}\n")
        f.write(f"Mean assays per chemical: {chem_coverage.mean():.1f}\n")
        f.write(f"Median assays per chemical: {np.median(chem_coverage):.1f}\n")
        f.write(f"Max assays for any chemical: {chem_coverage.max()}\n\n")

        # Coverage by assay
        assay_coverage = coverage.sum(axis=0)
        f.write(f"Assays with >0 empirical chemicals: {(assay_coverage > 0).sum():,}\n")
        f.write(f"Mean chemicals per assay: {assay_coverage.mean():.1f}\n")
        f.write(f"Median chemicals per assay: {np.median(assay_coverage):.1f}\n")
        f.write(f"Max chemicals for any assay: {assay_coverage.max()}\n")

    logger.info(f"Saved coverage statistics to {report_path}")


if __name__ == "__main__":
    main()
