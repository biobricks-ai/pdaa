import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns

def get_activity_df(cachedir: str | Path) -> pd.DataFrame:
    """Load the activity matrix from a parquet file."""
    # Define the path to the parquet file
    cachedir = Path('cache/entity_similarity')
    activity_df_path = cachedir / 'activity_matrix_filled.parquet'

    # Load the activity matrix from the parquet file
    activity_df = pd.read_parquet(activity_df_path)

    return activity_df

def smiles_to_inchi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchi = Chem.MolToInchi(mol)
    return inchi

def inchi_to_smiles(inchi):
    mol = Chem.MolFromInchi(inchi)
    smiles = Chem.MolToSmiles(mol)
    return smiles

def styled_heatmap(matrix, *,
    dpi=600,
    fontcolor='black',
    linecolor='black',
    show_dendro=False,
    row_group_size: int = 1,
    outdir: Path,
    heatmap_y_label: str = 'Diester Phthalates'
):
    """
    Wrapper around seaborn.clustermap with the same visual
    tweaks used in build_heatmap.py (_generate_heatmap).
    - matrix: rows = substances, cols = ICE assays
    - row_group_size: int, number of contiguous rows to average together (default 1)
    """
    # --- Per-row or grouped-row mean activity bar chart -----------------
    from scipy.special import softmax
    row_means = matrix.mean(axis=1)
    matrix['row_means'] = row_means
    matrix.sort_values(by='row_means', inplace=True, ascending=False)
    # Cluster only columns; we already ordered rows
    g = sns.clustermap(
        matrix,
        # square=True,       # ← force equal-sized cells
        cbar_kws={'drawedges': False},  # disable seaborn’s built-in bar
        cmap='viridis',
        row_cluster=False, col_cluster=True,
        xticklabels=False, yticklabels=False,
        linecolor=linecolor,
        # linewidths=0.5,
        figsize=(18, 9),
        cbar_pos=(0.95, 0.3, 0.02, 0.4),
        dendrogram_ratio=(0.10, 0.05),
        tree_kws={'linewidths': 0.5}
    )
    # Remove any stray colorbar
    if hasattr(g, 'cax') and g.cax:
        g.cax.remove()

    # Create a new colorbar on its own axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(g.ax_heatmap)

    # Determine group centers, heights, and averaged values
    if row_group_size > 1:
        centers, heights, values = [], [], []
        n_rows = len(matrix)
        for start in range(0, n_rows, row_group_size):
            end = min(start + row_group_size, n_rows)
            group_len = end - start
            centers.append(start + group_len / 2)          # center of the group
            heights.append(group_len)                      # bar spans the group
            values.append(matrix['row_means'].iloc[start:end].mean())
    else:
        centers  = np.arange(len(matrix)) + 0.5
        heights  = 1.0
        values   = matrix['row_means'].values

    bar_ax = divider.append_axes("right", size="6%", pad=0.2, sharey=g.ax_heatmap)
    # Normalize values to [0, 1] for colormap mapping
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('viridis')
    bar_colors = cmap(norm(values))

    bar_ax.barh(
        centers,
        values,
        height=heights,
        color=bar_colors,
        edgecolor='none',
        align='center'
    )
    bar_ax.set_ylim(g.ax_heatmap.get_ylim())       # lock vertical span

    # Add ticks and tick labels
    bar_ax.set_yticks(centers)
    bar_ax.set_yticklabels(matrix.index, fontsize=10)
    bar_ax.tick_params(axis='y', length=0)
    # --------------------------------------------------------------------


    # # --- Per-row mean activity bar chart -------------------------------
    # row_means = matrix.mean(axis=1)
    # # Share the y-axis so bars line up perfectly with heat-map rows
    # bar_ax = divider.append_axes("right", size="6%", pad=0.2, sharey=g.ax_heatmap)
    # bar_ax.barh(np.arange(len(row_means)) + 0.5,   # center on each row
    #             row_means.values,
    #             height=1.0,
    #             # color='darkgray',
    #             color='black',
    #             # edgecolor=linecolor,
    #             edgecolor='none',
    #             align='center')
    # bar_ax.set_ylim(g.ax_heatmap.get_ylim())       # lock vertical span

    # bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    bar_ax.set_xlabel('MAV', fontsize=14)
    # -------------------------------------------------------------------

    cax = divider.append_axes("right", size="2%", pad=0.6)

    # expose the divider so callers can add more axes without destroying the layout
    g.divider = divider
    sm  = plt.cm.ScalarMappable(
        cmap='viridis',
        norm=plt.Normalize(
            vmin=0,
            vmax=1,
        )
    )
    sm.set_array([])
    cb = g.figure.colorbar(sm, cax=cax)
    cb.set_label('Activity Score', fontsize=18, labelpad=10)
    cb.ax.tick_params(labelsize=14)

    # Hide col dendrogram but keep clustering
    g.ax_row_dendrogram.set_visible(show_dendro)
    g.ax_col_dendrogram.set_visible(show_dendro)

    g.ax_heatmap.set_xlabel('DART or ED Assays', color=fontcolor, fontsize=20)
    g.ax_heatmap.set_ylabel(heatmap_y_label, color=fontcolor, fontsize=20)
    if not show_dendro:  # no room on the left
        g.ax_heatmap.yaxis.set_label_position('left')

    # Label the colorbar
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.set_label('Activity Score', fontsize=20, labelpad=10)

    # Tidy up margins so nothing is clipped
    g.figure.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.05)

    plt.savefig(outdir / 'activity_matrix_heatmap.png', dpi=dpi, bbox_inches='tight')

    return g  # caller can add arrows, bars, etc.