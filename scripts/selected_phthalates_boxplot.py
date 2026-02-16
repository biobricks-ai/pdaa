#!/usr/bin/env python3
"""
Generate boxplot of selected phthalates by chain length.

This script creates a boxplot showing mean activity values (MAV) for a curated
set of 18 common phthalates, grouped by longest carbon backbone length.

Output: cache/wide_groupings/selected_phthalates_boxplot.png
"""

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

# Selected phthalates with SMILES and chain length groups
PHTHALATES = {
    # C1-3
    "DMP": {"smiles": "COC(=O)c1ccccc1C(=O)OC", "group": "1-3", "name": "Dimethyl phthalate"},
    "DEP": {"smiles": "CCOC(=O)c1ccccc1C(=O)OCC", "group": "1-3", "name": "Diethyl phthalate"},
    "DPrP": {"smiles": "CCCOC(=O)c1ccccc1C(=O)OCCC", "group": "1-3", "name": "Di-n-propyl phthalate"},
    # C4-6
    "DBP": {"smiles": "CCCCOC(=O)c1ccccc1C(=O)OCCCC", "group": "4-6", "name": "Di-n-butyl phthalate"},
    "DIBP": {"smiles": "CC(C)COC(=O)c1ccccc1C(=O)OCC(C)C", "group": "4-6", "name": "Diisobutyl phthalate"},
    "DnPP": {"smiles": "CCCCCOC(=O)c1ccccc1C(=O)OCCCCC", "group": "4-6", "name": "Di-n-pentyl phthalate"},
    "DIPP": {"smiles": "CC(C)CCOC(=O)c1ccccc1C(=O)OCCC(C)C", "group": "4-6", "name": "Diisopentyl phthalate"},
    "DnHP": {"smiles": "CCCCCCOC(=O)c1ccccc1C(=O)OCCCCCC", "group": "4-6", "name": "Di-n-hexyl phthalate"},
    # C7-8
    "DnHpP": {"smiles": "CCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCC", "group": "7-8", "name": "Di-n-heptyl phthalate"},
    "DIHP": {"smiles": "CC(C)CCCCOC(=O)c1ccccc1C(=O)OCCCCC(C)C", "group": "7-8", "name": "Diisoheptyl phthalate"},
    "DEHP": {"smiles": "CCCCC(CC)COC(=O)c1ccccc1C(=O)OCC(CC)CCCC", "group": "7-8", "name": "Di(2-ethylhexyl) phthalate"},
    "DnOP": {"smiles": "CCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCC", "group": "7-8", "name": "Di-n-octyl phthalate"},
    "DIOP": {"smiles": "CC(C)CCCCCOC(=O)c1ccccc1C(=O)OCCCCCC(C)C", "group": "7-8", "name": "Diisooctyl phthalate"},
    # C9+
    "DINP": {"smiles": "CC(C)CCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCC(C)C", "group": "9+", "name": "Diisononyl phthalate"},
    "DnNP": {"smiles": "CCCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCC", "group": "9+", "name": "Di-n-nonyl phthalate"},
    "DIDP": {"smiles": "CC(C)CCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCC(C)C", "group": "9+", "name": "Diisodecyl phthalate"},
    "DPHP": {"smiles": "CCCCCCC(CCC)COC(=O)c1ccccc1C(=O)OCC(CCC)CCCCCC", "group": "9+", "name": "Di(2-propylheptyl) phthalate"},
    "DIUP": {"smiles": "CC(C)CCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCC(C)C", "group": "9+", "name": "Diisoundecyl phthalate"},
    "DTDP": {"smiles": "CC(C)CCCCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCCCC(C)C", "group": "9+", "name": "Diisotridecyl phthalate"},
}

# Plot styling
BOX_COLOR = "#7eb8da"
POINT_COLOR = "#2c5f7c"
MEAN_COLOR = "#e07b54"
GROUP_ORDER = ["1-3", "4-6", "7-8", "9+"]


def load_activity_data() -> pd.DataFrame:
    """Load the activity matrix from parquet."""
    activity_path = CACHE_DIR / "entity_similarity" / "activity_matrix_filled.parquet"
    return pd.read_parquet(activity_path)


def get_phthalate_activities(activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Look up activity values for each selected phthalate.

    Returns DataFrame with abbrev, name, group, and mean_activity columns.
    """
    results = []
    for abbrev, data in PHTHALATES.items():
        mol = Chem.MolFromSmiles(data["smiles"])
        if mol is None:
            logger.warning(f"Invalid SMILES for {abbrev}")
            continue

        inchi = Chem.MolToInchi(mol)
        if inchi in activity_df.index:
            mean_activity = activity_df.loc[inchi].mean()
            results.append({
                "abbrev": abbrev,
                "name": data["name"],
                "group": data["group"],
                "mean_activity": mean_activity
            })
        else:
            logger.warning(f"{abbrev} not found in activity matrix")

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


def create_boxplot(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create and save the boxplot figure."""
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

        # Generate jitter
        x_jitter = np.random.uniform(-0.15, 0.15, n_points)

        for j, (_, row) in enumerate(group_data.iterrows()):
            # Set DIPP jitter to 0 (it's an outlier, keep centered)
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

    ax.set_title("Activity of Selected Phthalates by Longest Carbon Backbone", fontsize=14, fontweight='medium')
    ax.set_xlabel("Longest Carbon Backbone", fontsize=11)
    ax.set_ylabel("MAV", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved figure to {output_path}")


def main():
    """Main entry point."""
    import warnings
    warnings.filterwarnings('ignore')

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading activity matrix...")
    activity_df = load_activity_data()

    # Get phthalate activities
    logger.info("Looking up phthalate activities...")
    results_df = get_phthalate_activities(activity_df)
    logger.info(f"Found {len(results_df)} phthalates with activity data")

    # Create plot
    output_path = OUTPUT_DIR / "selected_phthalates_boxplot.png"
    create_boxplot(results_df, output_path)

    # Print summary statistics
    logger.info("Summary statistics:")
    for group in GROUP_ORDER:
        subset = results_df[results_df["group"] == group]
        logger.info(f"  {group}: mean={subset['mean_activity'].mean():.4f}, n={len(subset)}")


if __name__ == "__main__":
    main()
