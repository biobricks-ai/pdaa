import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from skmisc.loess import loess               # pip install scikit-misc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA               # Principal Component Analysis

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import logging
import re

from rdkit.Chem import (
    AllChem,
    Descriptors,
    Descriptors3D,
    DataStructs,
    rdFingerprintGenerator as rfg
)

import sys
sys.path.append('./')  # so utility scripts can be found
from stages.utils.pdaa import is_phthalate, longest_carbon_backbone

def PCA_plot(
        X: pd.DataFrame,
        Y: pd.DataFrame,
    ) -> plt.Axes:
    """
    Perform PCA on the descriptor matrix and plot the first two components.

    Parameters
    ----------
    X : pd.DataFrame
        Descriptor matrix (rows = compounds, columns = descriptors).
    Y : pd.DataFrame
        Activity matrix (rows = compounds, columns = assays).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis containing the PCA plot.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA               # Principal Component Analysis
    from sklearn.linear_model import LinearRegression
    from mpl_toolkits.mplot3d import Axes3D              # registers the 3-D projection

    # Standardise BOTH X and y so units don’t distort slopes
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    y = Y.mean(axis=1).values  # mean activity per compound

    X_z = scaler_X.fit_transform(X)
    y_z = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # ------------------------------------------------------------------
    # 1.  Principal-component projection
    # ------------------------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    PCs = pca.fit_transform(X_z)                 # PCs[:,0] = PC1, PCs[:,1] = PC2

    # ------------------------------------------------------------------
    # 2.  Fit linear model in PC space
    # ------------------------------------------------------------------
    reg = LinearRegression()
    reg.fit(PCs, y_z)                            # slope in rotated coordinates

    # ------------------------------------------------------------------
    # 3.  Visualise scatter + regression plane
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(21, 15))
    ax  = fig.add_subplot(111, projection='3d')

    # --- 3-D scatter: each point = one compound
    ax.scatter(PCs[:, 0], PCs[:, 1], y_z,
            c=y_z, cmap='viridis', s=18, alpha=0.8, linewidth=0)

    # --- regression plane
    xx, yy = np.meshgrid(np.linspace(PCs[:,0].min(), PCs[:,0].max(), 25),
                        np.linspace(PCs[:,1].min(), PCs[:,1].max(), 25))
    zz = reg.intercept_ + reg.coef_[0]*xx + reg.coef_[1]*yy
    ax.plot_surface(xx, yy, zz, alpha=0.25, color='lightgrey', rstride=1, cstride=1, linewidth=0)

    # --- cosmetics
    # ax.set_xlabel(f'PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    # ax.set_ylabel(f'PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax.set_xlabel(f'PC1', fontsize=25, labelpad=15)
    ax.set_ylabel(f'PC2', fontsize=25, labelpad=15)
    ax.set_zlabel('MAV (z-score)', fontsize=25, labelpad=10)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    # ax.zaxis.set_rotate_label(False)

    ax.view_init(elev=22, azim=-38)

    # ax.set_title('Linear Trend in PC Space')
    plt.tight_layout()
    plt.show()

    return ax


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

def get_descriptors(
        mol: AllChem.Mol, *,
        use_phthalate_set: bool = True,
        use_general_set: bool = False,
        use_vectors: bool = False,
) -> dict:
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

    if use_general_set or use_vectors:
        from rdkit.Chem import (
            rdMolDescriptors as rdmd,
            MolSurf,
        )
    
    if use_general_set:
        # General descriptors
        general_feats = {
            'NumHAcceptors'    : Descriptors.NumHAcceptors(mol),  # number of hydrogen bond acceptors
            'NumHDonors'       : Descriptors.NumHDonors(mol),  # number of hydrogen bond donors
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),  # number of valence electrons
            'NumAromaticRings' : Descriptors.NumAromaticRings(mol),  # number of aromatic rings
            'NumHalogens'      : Descriptors.fr_halogen(mol),  # number of halogen atoms
            'PBF'              : rdmd.CalcPBF(mol),  # planarity
            'Spher'            : rdmd.CalcSpherocityIndex(mol),  # spherocity index
            'Chi0'             : Descriptors.Chi0(mol),  # chi index 0
            'Chi1'             : Descriptors.Chi1(mol),  # chi index 1
        }
        feats.update(general_feats)

    if use_vectors:
        # ---------- PEOE-VSA block ----------
        try:
            # 1) vector form
            peoe_vec = rdmd.CalcPEOE_VSA(mol)      # tuple of 14 floats
        except AttributeError:
            # 2) fall-back: call each bin function explicitly
            peoe_vec = [getattr(MolSurf, f"PEOE_VSA{i}")(mol) for i in range(1, 15)]
        feats.update({f'peoe_vsa_{i+1}': v
                    for i, v in enumerate(peoe_vec)})
        
        # ---------- SlogP block ----------
        try:
            # 1) vector form
            slogp_vec = rdmd.CalcSlogP_VSA(mol)    # tuple of floats
        except AttributeError:
            # 2) fall-back: call each bin function explicitly
            slogp_vec = [getattr(MolSurf, f"SlogP_VSA{i}")(mol) for i in range(1, 13)]

        feats.update({f'slogp_vsa_{i+1}': v
                    for i, v in enumerate(slogp_vec)})

    return feats

def z_scale_df(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize the dataframe by columns."""
    return (df - df.mean())/df.std()

def zscore_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Z-score each assay column (ddof=0). Drop zero-variance assays.

    Parameters
    ----------
    df : pd.DataFrame
        Raw activity matrix.

    Returns
    -------
    z : pd.DataFrame
        Column-standardized matrix with zero-variance assays removed.
    dropped : list of str
        Assay names that were dropped due to zero variance.
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=0)

    zero_var = stds[stds == 0.0].index.tolist()
    if zero_var:
        logging.warning("Dropping %d zero-variance assays.", len(zero_var))

    keep = stds.index.difference(zero_var)
    if len(keep) == 0:
        raise ValueError("All assays have zero variance after standardization; nothing to cluster.")

    z = (df[keep] - means[keep]) / stds[keep]
    # For numerical stability (should not happen with std>0)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z, zero_var

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

    if Y.ndim > 1:
        y_mean = Y.mean(axis=1)
    else:
        y_mean = Y

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

def get_activity_df(cachedir: str | Path = Path('cache/entity_similarity')) -> pd.DataFrame:
    """Load the activity matrix from a parquet file."""
    # Define the path to the parquet file
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

def loess_ci(x, y, span=0.3, x_grid=None, level=0.95):
    x_grid = np.linspace(x.min(), x.max(), 200) if x_grid is None else x_grid
    model   = loess(x, y, span=span, degree=1)
    model.fit()
    pred    = model.predict(x_grid, stderror=True)
    # conf    = pred.confidence(level=level)
    conf    = pred.confidence(alpha=1 - level)
    return x_grid, pred.values, conf.lower, conf.upper

def plot_activity_features(descriptor_df: pd.DataFrame, activity_df: pd.DataFrame, *, linear_model = None, outdir: Path):
    """
    Plot the activity features against the descriptors.

    Parameters
    ----------
    descriptor_df : pd.DataFrame
        DataFrame containing descriptors.
    activity_df : pd.DataFrame
        DataFrame containing activities.
    """
    # key_descriptors = [
    #     'MolWt', 'cLogP', 'RotB',
    #     'LongestCarbonBackbone', 'BranchingRatio'
    # ]
    if linear_model is None:
        key_descriptors = [
            'Rgyr', 'RotB', 'BranchingRatio',
            'Fsp3', 'Kappa1', 'TPSA',
            'MolWt', 'cLogP', 'Isomer',
        ]
    else:
        key_descriptors = [
            'Rgyr', 'RotB', 'BranchingRatio',
            'Fsp3', 'LinearModel', 'Kappa1',
            'MolWt', 'cLogP', 'Isomer',
        ]
    descriptors_to_labels = {
        'MolWt': 'Molecular Weight [g/mol]',
        'cLogP': 'cLogP',
        'TPSA': 'TPSA [Å²]',
        'RotB': 'Number of Rotatable Bonds',
        'MolMR': 'Molar Refractivity [cm³/mol]',
        'Fsp3': 'Fraction of sp³ Carbons',
        'Kappa1': 'Kappa Shape Index 1',
        'Kappa2': 'Kappa Shape Index 2',
        'Kappa3': 'Kappa Shape Index 3',
        'LongestCarbonBackbone': 'Longest Carbon Backbone',
        'BranchingRatio': 'Branching Ratio',
        'Isomer': 'Isomer Type (0=ortho, 1=iso, 2=tere)',
        'Rgyr': 'Radius of Gyration [Å]',
        'LinearModel': 'Linear Model Prediction',
    }
    # is_discrete = [False, False, True, True, False]  # whether the descriptor is discrete

    # Create a figure with subplots for each descriptor
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()
    Y = activity_df.mean(axis=1)  # mean activity across all assays
    for i, descriptor in enumerate(key_descriptors):
        ax = axes[i]
        x = descriptor_df[descriptor]
        if descriptor in ['LinearModel', 'Isomer']:
            if descriptor == 'LinearModel':
                sns.regplot(
                    x=x,
                    y=Y,
                    fit_reg=True,
                    ci=95,
                    scatter_kws={
                        'alpha': 0.5,
                        'edgecolors': 'white',
                        'color': 'green'
                    },
                    line_kws={'color': 'black', 'lw': 2},                    
                    ax=ax
                )
                # Set xtick steps to 0.05
                import matplotlib.ticker as mticker
                ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
            else:
                sns.regplot(
                    x=x,
                    y=Y,
                    fit_reg=True,
                    ci=95,
                    scatter_kws={
                        'alpha': 0.5,
                        'edgecolors': 'white',
                    },
                    line_kws={'color': 'black', 'lw': 2},
                    ax=ax
                )
            
        else:
            sns.regplot(
                x=x,
                y=Y,
                # lowess=True,
                # robust=True,
                fit_reg=False,
                # ci=95,
                scatter_kws={'alpha': 0.5, 'edgecolors': 'white'},
                # line_kws={'color': 'black', 'lw': 2},
                ax=ax
            )
            xg, curve, lo, hi = loess_ci(x.values, Y.values, span=0.5)
            ax.fill_between(xg, lo, hi, color='grey', alpha=0.25, zorder=1)
            ax.plot(xg, curve, color="black", lw=2, zorder=2)
        

        # limit the x-axis range to exclude outliers
        Q1 = x.quantile(0.25)
        Q3 = x.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Set x-axis limits if the data are outside the bounds
        if x.min() > lower_bound:
            lower_bound = None  # no need to set lower bound if all values are above it
        if x.max() < upper_bound:
            upper_bound = None  # no need to set upper bound if all values are below it
        if (lower_bound is not None) or (upper_bound is not None):
            ax.set_xlim(lower_bound, upper_bound)

        # # Plot each activity against the descriptor
        # for activity in activity_df.columns:
            # ax.scatter(
            #     descriptor_df[descriptor],
            #     activity_df[activity],
            #     # label=activity,
            #     alpha=0.5
            # )
        
        # ax.set_title(f"Activity vs. {descriptor}")
        # ax.set_xlabel(descriptor)
        ax.set_xlabel(descriptors_to_labels[descriptor])
        ax.set_ylabel("Mean Activity Value")

    # Remove any empty subplots
    for j in range(len(key_descriptors), len(axes)):
        fig.delaxes(axes[j])

    plt.savefig(outdir / "activity_by_descriptors_CI.png")
    plt.show()

def pca_variance_ratio(X: pd.DataFrame) -> PCA:

    # Standardise X so units don’t distort slopes
    scaler_X = StandardScaler()

    X_z = scaler_X.fit_transform(X)

    # Principal Component Analysis
    pca = PCA(random_state=42)
    pca.fit_transform(X_z)
    print("Explained variance ratio by each PC:")
    cumulative_variance = 0.0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative_variance += var
        print(f"PC{i+1}: {var:.2%}, cumulative: {cumulative_variance:.2%}")

    return pca

def remove_high_vif_descriptors(
        X: pd.DataFrame,
        vif_threshold: float = 10.0,
        to_print: bool = True,
) -> pd.DataFrame:
    """
    Remove descriptors with high VIF iteratively until all VIFs are below a threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Descriptor matrix (rows = compounds, columns = descriptors).
    vif_threshold : float, default 10.0
        Threshold for variance inflation factor (VIF).

    Returns
    -------
    pd.DataFrame
        Descriptor matrix with high VIF descriptors removed.
    """
    # drop any columns with NaNs
    X = X.dropna(axis=1, how='any')

    # drop any constant columns
    constant_cols = X.columns[X.std() == 0]
    X.drop(columns=constant_cols, inplace=True)

    # Automatically remove the highest VIF descriptors until all VIFs are below a threshold
    while True:
        vif_table = compute_vifs(X)
        high_vif = vif_table[vif_table['VIF'] > vif_threshold]
        
        if high_vif.empty:
            break
        
        # Remove the descriptor with the highest VIF
        descriptor_to_remove = high_vif.loc[high_vif['VIF'].idxmax(), 'descriptor']
        if to_print:
            print(f"Removing descriptor '{descriptor_to_remove}' with VIF {high_vif['VIF'].max()}")
        X.drop(columns=[descriptor_to_remove], inplace=True)

    return X

# def inchis_to_morgan_df(
#         df: pd.DataFrame,
#         *,
#         inchi_col: str = 'inchi',     # name of the column holding InChI strings
#         radius: int = 2,              # ECFP-4 by default
#         n_bits: int = 2048,           # fingerprint length
#         use_features: bool = True,    # use pharmacophore-type features
# ) -> pd.DataFrame:
#     """
#     Convert a DataFrame of InChI strings into a DataFrame of Morgan-fingerprint bits.

#     Returns a DataFrame with the same index (rows with invalid InChI are dropped)
#     and columns named fp_0 ... fp_<n_bits-1>, each holding 0/1 integers.
#     """
#     bit_arrays = []
#     valid_idx = []

#     # Fetch InChIs either from a column or the index
#     inchis = df[inchi_col] if inchi_col in df.columns else df.index.to_series()

#     for idx, inchi in inchis.items():
#         mol = AllChem.MolFromInchi(inchi, sanitize=True, removeHs=True)
#         if mol is None:
#             # Skip rows that cannot be parsed
#             continue
#         # fp = AllChem.GetMorganFingerprintAsBitVect(
#         #     mol,
#         #     radius=radius,
#         #     nBits=n_bits,
#         #     useFeatures=use_features,
#         # )
#         morgan_gen = rfg.GetMorganGenerator(
#             radius=radius,
#             fpSize=n_bits,
#             useFeatures=use_features,
#         )
#         arr = np.zeros((n_bits,), dtype=np.uint8)
#         DataStructs.ConvertToNumpyArray(morgan_gen, arr)
#         bit_arrays.append(arr)
#         valid_idx.append(idx)

#     fp_df = pd.DataFrame(
#         data=np.vstack(bit_arrays),
#         index=valid_idx,
#         columns=[f'fp_{i}' for i in range(n_bits)],
#         dtype=np.uint8,
#     )

#     return fp_df

def inchis_to_morgan_df(
        df: pd.DataFrame,
        *,
        inchi_col: str | None = 'inchi',   # set to None if InChIs are in the index
        radius: int = 2,                   # ECFP‑4 (radius 2)
        n_bits: int = 2048,
) -> pd.DataFrame:
    """
    Convert InChI strings to a DataFrame of Morgan‑fingerprint bits
    using RDKit's newer FingerprintGenerator API.
    """
    # -------- generator setup (done once) -----------------------------------
    morgan_gen = rfg.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        # includeChirality=False,
    )

    # -------- pull InChIs ----------------------------------------------------
    if inchi_col and inchi_col in df.columns:          # InChIs in a column
        inchis = df[inchi_col]
    else:                                              # InChIs in the index
        inchis = df.index.to_series()

    bit_arrays, valid_idx = [], []

    for idx, inchi in inchis.items():
        mol = AllChem.MolFromInchi(inchi, sanitize=True, removeHs=True)
        if mol is None:
            continue
        fp = morgan_gen.GetFingerprint(mol)            # new API call
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        bit_arrays.append(arr)
        valid_idx.append(idx)

    return pd.DataFrame(
        np.vstack(bit_arrays),
        index=valid_idx,
        columns=[f'fp_{i}' for i in range(n_bits)],
        dtype=np.uint8,
    )

def perform_clustering(X, y, *, method='kmeans', n=2, print_tag='', var_name='logRBA', with_mean=True, plot_clusters=False):
    if method == 'kmeans':
        kwargs = {
            'k': KMeans(n_clusters=n, n_init=30, init='k-means++', random_state=0)
        }
    elif method == 'gmm':
        kwargs = {
            'gmm': GaussianMixture(
                n_components=n,
                covariance_type='diag',
                random_state=0,
                init_params='kmeans',
                weights_init=[0.5, 0.5]                # <-- forces 50 / 50 prior
            )
        }
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    pipe = general_pipe(kwargs, with_mean=with_mean)
    general_clustering(pipe, X, y, print_tag=print_tag, var_name=var_name, plot_clusters=plot_clusters)

# convencience functions for clustering
def kmeans_clustering(X, y, *, n_clusters=2, print_tag='', var_name='logRBA', with_mean=True, plot_clusters=False):
    perform_clustering(X, y, method='kmeans', n=n_clusters, print_tag=print_tag, var_name=var_name, with_mean=with_mean, plot_clusters=plot_clusters)
def Gaussian_mixture_clustering(X, y, *, n_components=2, print_tag='', var_name='logRBA', with_mean=False, plot_clusters=False):
    perform_clustering(X, y, method='gmm', n=n_components, print_tag=print_tag, var_name=var_name, with_mean=with_mean, plot_clusters=plot_clusters)

def general_pipe(kwargs, *, with_mean=True):
    """
    General function to create a pipeline for clustering.
    """
    steps = [('scale', StandardScaler(with_mean=with_mean))]   # if False, keep sparsity structure
    steps.extend(kwargs.items())
    pipe = Pipeline(steps)
    
    return pipe

def general_clustering(pipe, X, y, *, print_tag='', var_name='logRBA', plot_clusters=False):
    """
    General clustering function that can be used with any clustering pipeline.
    """
    cluster_labels = pipe.fit_predict(X)
    print('Silhouette', print_tag + ':',
        silhouette_score(X, cluster_labels))

    # Visual sanity‑check: does one cluster skew toward low log RBA?
    df = pd.DataFrame({var_name: y, 'cluster': cluster_labels})
    print(df.groupby('cluster')[var_name].describe())

    # if plot_clusters:
    #     _, ax = plt.subplots(figsize=(10, 6))
    #     sns.scatterplot(
    #         x=X.iloc[:, 0], y=X.iloc[:, 1],
    #         hue=cluster_labels, palette='viridis',
    #         ax=ax, s=50, alpha=0.7, edgecolor='w'
    #     )
    #     ax.set_title(f'Clustering with {print_tag} labels')
    #     ax.set_xlabel(X.columns[0])
    #     ax.set_ylabel(X.columns[1])
    #     plt.legend(title='Cluster', loc='upper right')
    #     plt.tight_layout()
    #     plt.show()
    if plot_clusters:
        # Use PCA to reduce X to two components for plotting
        pca = PCA(n_components=2, random_state=0)
        X_pca = pca.fit_transform(X)
        _, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1],
            hue=cluster_labels, palette='viridis',
            ax=ax, s=50, alpha=0.7, edgecolor='w'
        )
        ax.set_title(f'Clustering with {print_tag} labels (PCA axes)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.legend(title='Cluster', loc='upper right')
        plt.tight_layout()
        plt.show()

def comma_remove(s):
    return s.replace(',', '')

def get_endpoint_series(df, endpoint, *, p_conversion = False, sentinel_threshold=-10):
    # filter for desired endpoint
    endpoint_series = df.loc[df['EndpointName'] == endpoint, ['inchi', 'EndpointValue']]
    endpoint_series.rename(columns={'EndpointValue': endpoint}, inplace=True)
    # set the InChI as index
    endpoint_series.set_index('inchi', inplace=True)
    endpoint_series = endpoint_series[endpoint_series.index.notna()]
    # clean the endpoint column
    endpoint_series[endpoint] = endpoint_series[endpoint].apply(comma_remove)
    # convert endpoint values to numeric
    endpoint_series[endpoint] = pd.to_numeric(endpoint_series[endpoint], errors='coerce')
    # convert to pIC50, pKi, etc. if applicable
    if p_conversion:
        endpoint_series[endpoint] = -np.log10(endpoint_series[endpoint])
    # replace inf with NaN
    endpoint_series = endpoint_series.replace([np.inf, -np.inf], np.nan)
    # set sentinel values to NaN
    if sentinel_threshold is not None:
        endpoint_series[endpoint_series < sentinel_threshold] = np.nan
    # drop rows with NaN in endpoint value
    endpoint_series = endpoint_series.dropna()
    # for duplicate InChIs, take the mean of endpoint values
    endpoint_series = endpoint_series.groupby(endpoint_series.index).mean()

    return endpoint_series

# ----------------------------- PCA diagnostics ----------------------------- #

def pca_broken_stick_diagnostic(Xz: pd.DataFrame) -> None:
    """
    Log a broken-stick diagnostic on the assay correlation spectrum.

    For p assays, eigenvalue proportions (PCA on assay correlation matrix) are
    compared to the broken-stick expectation. We log how many components exceed
    the null and show the top few proportions vs. the null.
    """
    p = Xz.shape[1]
    if p < 2:
        logging.info("Broken-stick diagnostic skipped (p<2).")
        return

    R = np.corrcoef(Xz.values, rowvar=False)
    # Eigenvalues of a correlation matrix sum to p
    evals = np.linalg.eigvalsh(R)  # ascending
    props = evals[::-1] / float(p)  # descending proportions

    # Broken-stick expected proportions b_k
    # b_k = (1/p) * sum_{i=k}^p (1/i)
    inv = 1.0 / np.arange(1, p + 1, dtype=float)
    csum = np.cumsum(inv[::-1])[::-1]  # sums from k..p
    broken = csum / float(p)

    k_keep = int(np.sum(props > broken))
    top = min(5, p)
    pairs = " ; ".join([f"{i+1}:{props[i]:.3f}>{broken[i]:.3f}" if props[i] > broken[i]
                        else f"{i+1}:{props[i]:.3f}≤{broken[i]:.3f}" for i in range(top)])
    logging.info("Broken-stick diagnostic: components above null = %d (of %d); top comps (prop vs. null): %s",
                 k_keep, p, pairs)
    
    return k_keep

# def clean_title(title: str) -> str:
#     """
#     Clean a title string by removing unwanted characters and normalizing spaces.
#     """
#     # Remove unwanted characters and normalize spaces
#     cleaned_title = re.sub(r'[^\w\s]', ' ', title)  # replace punctuation with spaces
#     cleaned_title = re.sub(r'_', ' ', cleaned_title)  # replace underscores with spaces
#     cleaned_title = re.sub(r'\s+', ' ', cleaned_title)  # normalize spaces
#     cleaned_title = cleaned_title.strip()  # remove leading/trailing spaces
#     cleaned_title = cleaned_title.lower()  # Convert to lowercase for consistency
#     return cleaned_title

import unicodedata

# Map common Greek letters to names; extend if you see others in your data
_GREEK_MAP = str.maketrans({
    'α':'alpha','β':'beta','γ':'gamma','δ':'delta','ε':'epsilon','ζ':'zeta','η':'eta','θ':'theta',
    'ι':'iota','κ':'kappa','λ':'lambda','μ':'mu','ν':'nu','ξ':'xi','ο':'omicron','π':'pi','ρ':'rho',
    'σ':'sigma','ς':'sigma','τ':'tau','υ':'upsilon','φ':'phi','χ':'chi','ψ':'psi','ω':'omega',
    'Α':'alpha','Β':'beta','Γ':'gamma','Δ':'delta','Ε':'epsilon','Ζ':'zeta','Η':'eta','Θ':'theta',
    'Ι':'iota','Κ':'kappa','Λ':'lambda','Μ':'mu','Ν':'nu','Ξ':'xi','Ο':'omicron','Π':'pi','Ρ':'rho',
    'Σ':'sigma','Τ':'tau','Υ':'upsilon','Φ':'phi','Χ':'chi','Ψ':'psi','Ω':'omega',
})

# # Conservative roman-numeral normalizer (standalone tokens I..X only)
# _ROMAN_RE  = re.compile(r'\b(i{1,3}|iv|v|vi{0,3}|ix|x)\b', re.IGNORECASE)
# _ROMAN_MAP = {'i':'1','ii':'2','iii':'3','iv':'4','v':'5','vi':'6','vii':'7','viii':'8','ix':'9','x':'10'}
_ROMAN_TOKEN_RE = re.compile(r'(?<!\w)([IVXLCDMivxlcdm]{2,})(?!\w)')

def roman_to_int(s: str) -> int:
    """Converts a Roman numeral string to its integer equivalent.

    Args:
        s: The Roman numeral string (e.g., "MCMXCIV").

    Returns:
        The integer equivalent of the Roman numeral.
    """
    roman_values = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
        'i': 1,
        'v': 5,
        'x': 10,
        'l': 50,
        'c': 100,
        'd': 500,
        'm': 1000,
    }

    total = 0
    prev_value = 0  # To handle subtractive cases (e.g., IV, IX)

    # Iterate through the Roman numeral string in reverse
    for char in reversed(s):
        current_value = roman_values[char]

        # If the current value is less than the previous value, it's a subtractive case
        if current_value < prev_value:
            total -= current_value
        else:
            total += current_value

        prev_value = current_value

    return total

def _roman_to_arabic(text: str) -> str:
    # return _ROMAN_RE.sub(lambda m: _ROMAN_MAP[m.group(0).lower()], text)
    # Replace each standalone Roman token with its Arabic value using roman_to_int.
    def _repl(m: re.Match) -> str:
        token = m.group(1)
        try:
            return str(roman_to_int(token))
        except KeyError:
            # Non-Roman character slipped through; leave as-is.
            return token
    return _ROMAN_TOKEN_RE.sub(_repl, text)

def clean_title(title: str) -> str:
    """
    Normalize assay titles for equality matching across data sources.

    Steps:
    - Unicode NFKC fold to canonicalize look-alike chars (e.g., micro sign).
    - Replace Greek letters with names (β→beta, κ→kappa, μ→mu, …).
    - Optionally normalize standalone roman numerals (II→2).
    - Replace punctuation/underscores with spaces; collapse whitespace; lowercase.
    """
    x = unicodedata.normalize('NFKC', str(title))
    x = x.translate(_GREEK_MAP)
    x = _roman_to_arabic(x)
    x = re.sub(r'[^\w\s]', ' ', x)   # punctuation → spaces
    x = re.sub(r'_', ' ', x)         # underscores → spaces
    x = re.sub(r'\s+', ' ', x)       # collapse runs of whitespace
    return x.strip().lower()


def get_assay_strength(fullpred: List[Dict], category_label: str ='endocrine disruption', to_clean: bool = False) -> List[Tuple[str, float]]:
    """
    Get the strength of the specified category from the fullpred predictions.
    """
    assay_strength = []

    if to_clean:
        # Clean the assay titles for consistency
        f = lambda title: clean_title(title)
    else:
        f = lambda title: title

    for fp in fullpred:
        prop = fp['property']
        categories = prop['categories']  # list of dicts

        for category in categories:
            if category['category'] == category_label:
                assay_strength.append(
                    (f(prop['title']), category['strength'])
                )
                break

    return assay_strength  # can cast to set or dict if needed