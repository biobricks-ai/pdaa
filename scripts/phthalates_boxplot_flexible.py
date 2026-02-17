#!/usr/bin/env python3
"""
Generate boxplot of phthalates by chain length from any CSV file.

This script creates a boxplot showing mean activity values (MAV) for phthalates
from a specified CSV file, grouped by longest carbon backbone length.

Usage:
    python phthalates_boxplot_flexible.py <csv_file> [--output-name <name>]

CSV Requirements:
    - Must have columns: abbreviation, name, smiles
    - Optional: chain_length_group (if not present, will be calculated)

Output: cache/wide_groupings/<output_name>_boxplot.png
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = CACHE_DIR / "wide_groupings"

# Plot styling
BOX_COLOR = "#7eb8da"
POINT_COLOR = "#2c5f7c"
MEAN_COLOR = "#e07b54"
GROUP_ORDER = ["1-3", "4-6", "7-8", "9+"]


def get_longest_carbon_chain(mol: Chem.Mol) -> int:
    """Calculate the longest carbon chain in a molecule."""
    from rdkit.Chem import Descriptors
    # Use the number of carbons in the longest chain
    # For phthalates, we need to find the longest alkyl chain on the ester groups

    # Simple approach: count carbons in each potential chain
    max_chain = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            # Use atom's position in the structure
            # For phthalates, side chains are attached to oxygen
            pass

    # Alternative: use total carbon count as proxy
    # For symmetric phthalates: (total_C - 8) / 2 gives chain length
    # 8 carbons in the phthalate core (benzene + 2 carboxyl)
    total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    # For symmetric phthalates
    if total_carbons >= 8:
        chain_carbons = (total_carbons - 8) / 2
        return int(chain_carbons)

    return 0


def assign_chain_group(smiles: str) -> str:
    """Assign chain length group based on SMILES structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"

    longest_chain = get_longest_carbon_chain(mol)

    if longest_chain <= 3:
        return "1-3"
    elif longest_chain <= 6:
        return "4-6"
    elif longest_chain <= 8:
        return "7-8"
    else:
        return "9+"


def load_activity_data() -> pd.DataFrame:
    """Load the activity matrix from parquet."""
    activity_path = CACHE_DIR / "entity_similarity" / "activity_matrix_filled.parquet"
    return pd.read_parquet(activity_path)


def normalize_group_format(group: str) -> str:
    """Normalize chain group format to '1-3', '4-6', '7-8', '9+' format."""
    if pd.isna(group):
        return group
    # Remove 'C' prefix if present (e.g., "C1-3" -> "1-3")
    group = str(group).strip()
    if group.startswith('C'):
        group = group[1:]
    return group


def get_phthalate_activities(phthalates_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Look up activity values for each phthalate.

    Returns DataFrame with abbrev, name, group, and mean_activity columns.
    """
    results = []
    for _, row in phthalates_df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            logger.warning(f"Invalid SMILES for {row['abbreviation']}")
            continue

        inchi = Chem.MolToInchi(mol)
        if inchi in activity_df.index:
            mean_activity = activity_df.loc[inchi].mean()

            # Get chain group from CSV if available, otherwise calculate
            if "chain_length_group" in row and pd.notna(row["chain_length_group"]):
                group = normalize_group_format(row["chain_length_group"])
            else:
                group = assign_chain_group(row["smiles"])

            results.append({
                "abbrev": row["abbreviation"],
                "name": row["name"],
                "group": group,
                "mean_activity": mean_activity
            })
        else:
            logger.warning(f"{row['abbreviation']} not found in activity matrix")

    return pd.DataFrame(results)


def get_box_obstacle_points(ax, n_points_per_edge: int = 10) -> tuple[list, list]:
    """
    Extract points along boxplot edges to use as obstacles for text placement.

    Returns lists of x and y coordinates for points along box edges, whiskers,
    and median lines.
    """
    obstacle_x = []
    obstacle_y = []

    # Get all patches (boxes) - these are PathPatch objects
    for patch in ax.patches:
        # Get the path vertices (corners of the box)
        path = patch.get_path()
        vertices = path.vertices

        if len(vertices) >= 4:
            # Get bounding box from vertices
            x_coords = vertices[:, 0]
            y_coords = vertices[:, 1]
            x0, x1 = x_coords.min(), x_coords.max()
            y0, y1 = y_coords.min(), y_coords.max()

            # Add points along all four edges
            # Bottom edge
            for x in np.linspace(x0, x1, n_points_per_edge):
                obstacle_x.append(x)
                obstacle_y.append(y0)
            # Top edge
            for x in np.linspace(x0, x1, n_points_per_edge):
                obstacle_x.append(x)
                obstacle_y.append(y1)
            # Left edge
            for y in np.linspace(y0, y1, n_points_per_edge):
                obstacle_x.append(x0)
                obstacle_y.append(y)
            # Right edge
            for y in np.linspace(y0, y1, n_points_per_edge):
                obstacle_x.append(x1)
                obstacle_y.append(y)

    # Get lines (whiskers, medians, caps)
    for line in ax.lines:
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        if len(xdata) >= 2:
            # Add points along the line
            for x, y in zip(
                np.linspace(xdata[0], xdata[-1], n_points_per_edge),
                np.linspace(ydata[0], ydata[-1], n_points_per_edge)
            ):
                obstacle_x.append(x)
                obstacle_y.append(y)

    return obstacle_x, obstacle_y


def create_boxplot(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str,
    do_stat_tests: bool = False,
) -> None:
    """Create and save the boxplot figure.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: abbrev, name, group, mean_activity
    output_path : Path
        Path to save the figure
    title : str
        Title for the plot
    do_stat_tests : bool, default False
        If True, create 3-panel figure with statistical tests (Conover-Iman, Cliff's δ)
    """

    # Statistical tests setup if requested
    if do_stat_tests:
        import scikit_posthocs as sp
        from cliffs_delta import cliffs_delta

        # Create 3-panel mosaic layout
        fig, axdict = plt.subplot_mosaic(
            [['box', 'box'],             # top row: boxplot spans both columns
             ['p',   'delta']],          # bottom row: p-values | effect sizes
            figsize=(14, 10),
            constrained_layout=True
        )
        ax = axdict['box']

        # Prepare group labels for statistical tests
        # Filter to only groups that have data
        present_groups = [g for g in GROUP_ORDER if (results_df['group'] == g).any()]

        # Run Conover-Iman pairwise comparisons
        p_mat = sp.posthoc_conover(
            results_df,
            val_col="mean_activity",
            group_col="group",
            p_adjust="holm"
        )

        # Reorder to match GROUP_ORDER
        ordered_groups = [g for g in GROUP_ORDER if g in p_mat.index]
        p_mat = p_mat.loc[ordered_groups, ordered_groups]

        # Clean matrix helper (remove top row and right column, mask upper triangle)
        def clean_matrix(mat: pd.DataFrame):
            mat_cleaned = mat.iloc[1:, :-1]
            mask = np.triu(np.ones_like(mat_cleaned, dtype=bool), k=1)
            return {'data': mat_cleaned, 'mask': mask}

        # Plot p-value heatmap (panel B)
        sns.heatmap(
            **clean_matrix(p_mat),
            annot=True,
            fmt=".2g",
            cmap="viridis_r",
            cbar_kws={"label": "p (adj)"},
            vmin=0,
            vmax=1,
            ax=axdict['p'],
        )
        axdict['p'].set_title("Conover-Iman pairwise comparisons")
        axdict['p'].set_ylabel("")
        axdict['p'].set_xlabel("")

        # Calculate Cliff's delta effect sizes
        effect = np.full((len(ordered_groups), len(ordered_groups)), np.nan)
        for i, gi in enumerate(ordered_groups):
            ai = results_df.loc[results_df.group == gi, "mean_activity"]
            for j, gj in enumerate(ordered_groups):
                if i < j:
                    aj = results_df.loc[results_df.group == gj, "mean_activity"]
                    d, _ = cliffs_delta(ai, aj)
                    effect[i, j] = d
                    effect[j, i] = -d

        eff_df = pd.DataFrame(effect, index=ordered_groups, columns=ordered_groups)

        # Plot effect size heatmap (panel C)
        sns.heatmap(
            **clean_matrix(eff_df),
            annot=True,
            fmt=".2f",
            cmap="bwr",
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Cliff's δ"},
            ax=axdict['delta'],
        )
        axdict['delta'].set_title("Effect-size matrix (Cliff's δ)")
        axdict['delta'].set_ylabel("")
        axdict['delta'].set_xlabel("")

        # Rotate heatmap tick labels
        for ax_hm in (axdict['p'], axdict['delta']):
            plt.setp(ax_hm.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax_hm.get_yticklabels(), rotation=0)
    else:
        _fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot without outlier markers
    sns.boxplot(
        data=results_df,
        x="group",
        y="mean_activity",
        order=GROUP_ORDER,
        color=BOX_COLOR,
        showfliers=False,
        ax=ax,
        linewidth=1.5,
        width=0.6
    )

    # Extract box edges as obstacles for text placement
    obstacle_x, obstacle_y = get_box_obstacle_points(ax, n_points_per_edge=10)

    # Overlay individual points with labels
    np.random.seed(42)
    texts = []
    x_positions = []
    y_positions = []

    for i, group in enumerate(GROUP_ORDER):
        group_data = results_df[results_df["group"] == group].copy()
        n_points = len(group_data)

        if n_points == 0:
            continue

        # Generate jitter
        x_jitter = np.random.uniform(-0.15, 0.15, n_points)

        for j, (_, row) in enumerate(group_data.iterrows()):
            # Set DIPP jitter to 0 (it's often an outlier, keep centered)
            if row["abbrev"] == "DIPP":
                x_jitter[j] = 0

            x_pos = i + x_jitter[j]
            y_pos = row["mean_activity"]

            ax.scatter(
                x_pos, y_pos,
                color=POINT_COLOR,
                s=45,
                zorder=5,
                alpha=0.85,
                edgecolors='white',
                linewidth=0.5
            )

            x_positions.append(x_pos)
            y_positions.append(y_pos)

            txt = ax.annotate(
                row["abbrev"],
                (x_pos, y_pos),
                fontsize=8,
                alpha=0.9,
                color='#333333'
            )
            texts.append(txt)

    # Add mean markers and collect their positions as obstacles
    means = results_df.groupby("group")["mean_activity"].mean().reindex(GROUP_ORDER)
    mean_x = []
    mean_y = []
    for i, mean in enumerate(means):
        if pd.notna(mean):
            ax.scatter(
                i, mean,
                marker="D",
                s=70,
                color=MEAN_COLOR,
                zorder=6,
                edgecolors=MEAN_COLOR,
                linewidth=0,
                label="Mean" if i == 0 else ""
            )
            mean_x.append(i)
            mean_y.append(mean)

    # Combine data points, box obstacles, and mean markers for text avoidance
    all_obstacle_x = x_positions + obstacle_x + mean_x
    all_obstacle_y = y_positions + obstacle_y + mean_y

    # Adjust text positions to avoid overlaps
    adjust_text(
        texts,
        x=all_obstacle_x,
        y=all_obstacle_y,
        ax=ax,
        arrowprops=dict(arrowstyle='-', color='#888888', lw=0.5),
        expand_points=(1.5, 1.5),
        expand_text=(1.2, 1.2),
        force_text=(0.5, 0.5),
        force_points=(0.5, 0.5),
        lim=200
    )

    ax.set_title(title, fontsize=14, fontweight='medium')
    ax.set_xlabel("Longest Carbon Backbone", fontsize=11)
    ax.set_ylabel("MAV", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add panel labels if doing statistical tests
    if do_stat_tests:
        fig_labels = {
            'box':   'A)',   # top boxplot
            'p':     'B)',   # Conover-Iman heatmap
            'delta': 'C)'    # Cliff's δ heatmap
        }

        for key, lab in fig_labels.items():
            panel_ax = axdict[key]
            panel_ax.text(
                -0.05, 1.05, lab,
                transform=panel_ax.transAxes,
                fontsize=14,
                fontweight='bold',
                va='top',
                ha='right'
            )

    if not do_stat_tests:
        plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved figure to {output_path}")


def main():
    """Main entry point."""
    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Generate phthalate boxplot from CSV")
    parser.add_argument("csv_file", help="Path to CSV file with phthalates (relative to project root or absolute)")
    parser.add_argument("--output-name", help="Base name for output file (default: derived from CSV filename)")
    parser.add_argument("--title", help="Plot title (default: derived from CSV filename)")
    parser.add_argument("--stats", action="store_true", help="Include statistical test panels (Conover-Iman, Cliff's δ)")

    args = parser.parse_args()

    # Resolve CSV path
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    # Determine output name from CSV filename if not provided
    if args.output_name:
        output_name = args.output_name
    else:
        # Extract name from CSV filename
        # e.g., "commercial_top20.csv" -> "commercial_top20"
        output_name = csv_path.stem

    # Determine title
    if args.title:
        title = args.title
    else:
        # Create a nice title from the output name
        # e.g., "commercial_top20" -> "Activity of Commercial Top 20 Phthalates"
        title_base = output_name.replace("_", " ").title()
        title = f"Activity of {title_base} Phthalates by Longest Carbon Backbone"

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading phthalates from {csv_path}")
    phthalates_df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(phthalates_df)} phthalates")

    logger.info("Loading activity matrix...")
    activity_df = load_activity_data()

    # Get phthalate activities
    logger.info("Looking up phthalate activities...")
    results_df = get_phthalate_activities(phthalates_df, activity_df)
    logger.info(f"Found {len(results_df)} phthalates with activity data")

    if len(results_df) == 0:
        logger.error("No phthalates with activity data found!")
        return

    # Create plot
    suffix = "_combined" if args.stats else "_boxplot"
    output_path = OUTPUT_DIR / f"{output_name}{suffix}.png"
    create_boxplot(results_df, output_path, title, do_stat_tests=args.stats)

    # Print summary statistics
    logger.info("Summary statistics:")
    for group in GROUP_ORDER:
        subset = results_df[results_df["group"] == group]
        if len(subset) > 0:
            logger.info(f"  {group}: mean={subset['mean_activity'].mean():.4f}, n={len(subset)}")


if __name__ == "__main__":
    main()
