import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from rdkit.Chem import AllChem, Descriptors, Descriptors3D

import sys
sys.path.append('./')  # so utility scripts can be found
from stages.utils.pdaa import is_phthalate, longest_carbon_backbone

def classify_isomer(mol: AllChem.Mol) -> int:
    """
    Classify the isomer type of a phthalate molecule based on its structure.

    Parameters
    ----------
    mol : AllChem.Mol
        Molecule already validated as a phthalate.

    Returns
    -------
    int
        0 for ortho, 1 for iso (meta), 2 for tere (para) phthalate.
    """
    possible_modes = ('ortho_phthalate', 'meta_phthalate', 'para_phthalate')
    for i, mode in enumerate(possible_modes):
        if is_phthalate(mol, modes=(mode,)):
            return i

def get_branching_ratio(mol) -> float:
    """
    Calculate the branching ratio of a molecule.

    Parameters
    ----------
    mol : AllChem.Mol
        Molecule to calculate the branching ratio for.

    Returns
    -------
    float
        Branching ratio of the molecule.
    """
    # list all carbons
    carbons = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    if len(carbons) == 0:
        return 0.0

    # Carbons with more than 2 heavy neighbors
    branch_count = 0
    for atom in carbons:
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_neighbors) > 2:
            branch_count += 1
    
    return branch_count / len(carbons)

def get_descriptors(mol: AllChem.Mol, *, use_phthalate_set: bool = True) -> dict:
    """Calculate descriptors for a given molecule."""
    feats = {
        'MolWt'            : Descriptors.MolWt(mol),  # molecular weight
        'cLogP'            : Descriptors.MolLogP(mol),  # octanol-water partition coefficient
        'TPSA'             : Descriptors.TPSA(mol),  # topological polar surface area
        'RotB'             : Descriptors.NumRotatableBonds(mol),  # number of rotatable bonds
        'MolMR'            : Descriptors.MolMR(mol),  # molar refractivity
        'Fsp3'             : Descriptors.FractionCSP3(mol),  # fraction of sp3 hybridized carbons
        'Kappa1'           : Descriptors.Kappa1(mol),  # Kappa shape index 1
        'Kappa2'           : Descriptors.Kappa2(mol),  # Kappa shape index 2
        'Kappa3'           : Descriptors.Kappa3(mol),  # Kappa shape index 3
        'BranchingRatio'   : get_branching_ratio(mol),  # branching ratio
    }
    if use_phthalate_set:
        # Phthalate-specific descriptors
        feats['LongestCarbonBackbone'] = longest_carbon_backbone(mol),  # longest carbon side chain length
        feats['Isomer'] = classify_isomer(mol)  # 0=ortho,1=iso,2=tere
    # 3-D shape (needs conformer)
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    feats['Rgyr'] = Descriptors3D.RadiusOfGyration(mol)
    # custom: side-chain length & branching (sketch)
    return feats

def z_scale_df(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize the dataframe by columns."""
    return (df - df.mean())/df.std()

def get_linear_model(X: pd.DataFrame, Y: pd.DataFrame):
    """
    Fit a linear regression model to the data.

    Parameters
    ----------
    X : pd.DataFrame
        Features (descriptors).
    Y : pd.DataFrame
        Target (activities).

    Returns
    -------
    ols : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted linear regression model.
    marginal_r2 : dict
        Dictionary with marginal R² values for each descriptor.
        Keys are descriptor names, values are R² values.
    """
    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # model.fit(X, Y)
    # return model
    
    # from sklearn.metrics import r2_score

    y_mean = Y.mean(axis=1)
    y_mean_z = (y_mean - y_mean.mean()) / y_mean.std()  # z-score the mean activity
    X_lin = sm.add_constant(X)              # X came from z_scale_df(descriptor_df)
    ols = sm.OLS(y_mean_z, X_lin).fit()
    print(ols.summary())

    r2 = ols.rsquared
    r2_adj = ols.rsquared_adj

    print(
        f"Linear model explains {r2*100:.1f}% of the variance "
        f"({r2_adj*100:.1f}% adjusted)."
    )

    # marginal_r2 = {}
    # for col in X.columns:
    #     mod = sm.OLS(y_mean, sm.add_constant(X[[col]])).fit()
    #     marginal_r2[col] = mod.rsquared
    #     print(f"{100*mod.rsquared:.3f}% variance explained by {col}")

    return (
        ols
        # marginal_r2,
    )

def compute_vifs(X: pd.DataFrame, *, add_intercept: bool = False) -> pd.DataFrame:
    """
    Calculate VIF for each column in a descriptor matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Columns are descriptors, rows are compounds (already z-scaled is ideal).
    add_intercept : bool, default False
        - If True, appends a constant column before computing VIFs.
        - Most chem-descriptor sets don't need the intercept; set to True only
          if you plan to include one in later regression models.

    Returns
    -------
    pd.DataFrame
        Two columns:
        - 'descriptor': original column names
        - 'VIF': variance inflation factor (≥ 1; > 10 is a red flag)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # - Guard against perfectly constant descriptors (std == 0)
    constant_cols = X.columns[X.std() == 0]
    if len(constant_cols):
        raise ValueError(f"Constant descriptors detected: {list(constant_cols)}")

    XX = X.copy()
    if add_intercept:
        XX = XX.assign(_intercept_=1.0)

    # - Compute VIF for each column; statsmodels needs ndarray input
    vifs = [
        variance_inflation_factor(XX.values, idx)
        for idx in range(XX.shape[1])
    ]

    res = pd.DataFrame({
        'descriptor': XX.columns,
        'VIF': vifs
    })

    # - If an intercept was added, drop it from the result
    if add_intercept:
        res = res.query("descriptor != '_intercept_'").reset_index(drop=True)

    return res.sort_values('VIF', ascending=False).reset_index(drop=True)

def get_activity_df(cachedir: str | Path) -> pd.DataFrame:
    """Load the activity matrix from a parquet file."""
    # Define the path to the parquet file
    cachedir = Path('cache/entity_similarity')
    activity_df_path = cachedir / 'activity_matrix_filled.parquet'

    # Load the activity matrix from the parquet file
    activity_df = pd.read_parquet(activity_df_path)

    return activity_df

def smiles_to_inchi(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    inchi = AllChem.MolToInchi(mol)
    return inchi

def inchi_to_smiles(inchi):
    mol = AllChem.MolFromInchi(inchi)
    smiles = AllChem.MolToSmiles(mol)
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