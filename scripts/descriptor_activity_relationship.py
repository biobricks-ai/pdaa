import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit.Chem import AllChem
from skmisc.loess import loess               # pip install scikit-misc
from tqdm import tqdm
import statsmodels.api as sm

import sys
sys.path.append('./')  # so utility scripts can be found
# from stages.utils.pdaa import is_phthalate, longest_carbon_backbone
from scripts.utils.helpers import (
    get_activity_df,
    styled_heatmap,
    z_scale_df,
    get_linear_model,
    get_descriptors,
    compute_vifs,
)

# savepath for figures
fig_path = Path('cache/descriptors')

def Pearson_correlation_heatmap(X: pd.DataFrame, Y: pd.DataFrame, draw_heatmap: bool = False) -> pd.DataFrame:
    """
    Calculate the Pearson correlation between each descriptor in X and each activity in Y.

    Parameters
    ----------
    X : pd.DataFrame
        Columns are descriptors, rows are compounds.
    Y : pd.DataFrame
        Columns are activities, rows are compounds.

    Returns
    -------
    pd.DataFrame
        DataFrame with descriptors as index and activities as columns.
    """
    xy   = pd.concat([X, Y], axis=1)                 # same index, side-by-side
    rho  = xy.corr(method='spearman')                # full (p+d) × (p+d) matrix
    rho  = rho.loc[X.columns, Y.columns]             # slice to p × d block

    # get the norm of earch row (descriptor)
    descriptor_norm = np.linalg.norm(rho, axis=1)
    # make a new dataframe with the norm as a new column
    descriptor_norm_df = pd.DataFrame({
        'descriptor': rho.index,
        'norm': descriptor_norm
    }).set_index('descriptor')
    # sort by norm
    descriptor_norm_df = descriptor_norm_df.sort_values('norm', ascending=False)

    print("Descriptor norms:")
    print(descriptor_norm_df)

    if draw_heatmap:
        

        # # rho = your (descriptor × assay) DataFrame, already filled with Spearman ρ
        # fig, ax = plt.subplots(figsize=(10, 4))

        # cax = ax.imshow(rho, aspect="auto", interpolation="none")   # heatmap

        # # y-axis → descriptors
        # ax.set_yticks(range(len(rho.index)))
        # ax.set_yticklabels(rho.index)

        # # x-axis → hide assay labels
        # ax.set_xticks([])                  # remove ticks & labels
        # ax.set_xlabel("Assays")

        # # colour bar
        # fig.colorbar(cax, ax=ax, label="Spearman ρ")

        # ax.set_title("Descriptor-vs.-Assay Correlation Heatmap")
        # plt.tight_layout()
        # plt.show()

        g = sns.clustermap(
            rho,
            cmap='vlag',                # diverging palette
            center=0,                   # center the colormap at 0
            row_cluster=True,
            col_cluster=True,
            yticklabels=True,           # keep descriptor labels
            xticklabels=False,          # hide assay labels
            figsize=(10, 4),
            cbar_kws={'label': 'Spearman ρ'}
        )
        g.ax_heatmap.set_xlabel("Assays")
        plt.title('Descriptor-vs.-Assay Correlation Heatmap', pad=40)
        plt.show()

    return rho

def get_oob_score(X: pd.DataFrame, Y: pd.DataFrame) -> float:
    """
    Fit a Random Forest model to the data and return the out-of-bag score.

    Parameters
    ----------
    X : pd.DataFrame
        Features (descriptors).
    Y : pd.DataFrame
        Target (activities).

    Returns
    -------
    float
        Out-of-bag score of the Random Forest model.
    """
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=500, oob_score=True, n_jobs=-1)
    rf.fit(X, Y)
    print(f"Random Forest OOB Score: {rf.oob_score_}")

def plot_activity_scatter_lcb(descriptor_df: pd.DataFrame, activity_df: pd.DataFrame):
    """
    Plot the descriptors against the longest carbon backbone (LCB) and activities.

    Parameters
    ----------
    descriptor_df : pd.DataFrame
        DataFrame containing descriptors.
    activity_df : pd.DataFrame
        DataFrame containing activities.
    """

    # ------------------------------------------------------------------
    # 1. Prepare the plotting frame
    # ------------------------------------------------------------------
    compound_mean = activity_df.mean(axis=1)                       # mean activity per chemical
    plot_df = (
        descriptor_df[['LongestCarbonBackbone']]
        .assign(mean_activity=compound_mean)
    )

    # Sort LCB levels so the x-axis is ordered numerically
    lcb_levels = sorted(plot_df['LongestCarbonBackbone'].unique())
    x_pos      = {lcb: i for i, lcb in enumerate(lcb_levels)}
    jitter     = 0.25                                              # horizontal jitter half-width

    # ------------------------------------------------------------------
    # 2. Build the figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)

    for lcb in lcb_levels:
        group = plot_df.loc[plot_df.LongestCarbonBackbone == lcb, 'mean_activity']
        x     = np.random.uniform(-jitter, jitter, size=len(group)) + x_pos[lcb]

        ax.scatter(x, group, s=40, alpha=0.7, edgecolors='white')

        # horizontal bar for the group-mean of compounds’ means
        bar_y = group.mean()
        ax.hlines(bar_y, x_pos[lcb] - 0.4, x_pos[lcb] + 0.4,
                lw=3, color='black', zorder=3)

    # ------------------------------------------------------------------
    # 3. Cosmetics
    # ------------------------------------------------------------------
    ax.set_xticks(range(len(lcb_levels)))
    ax.set_xticklabels(lcb_levels)
    ax.set_xlabel("Longest Carbon Backbone (number of atoms)")
    ax.set_ylabel("Mean Activity Value")
    # ax.set_title("Mean Activity by Longest Carbon Backbone")
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_activity_boxplot_lcb_isomer(
        descriptor_df: pd.DataFrame,
        activity_df: pd.DataFrame,
        *,
        lcb_min: int = 0,
        lcb_max: int = 6,
        ax: plt.Axes | None = None,
        palette: str | None = "Set2",
        show_points: bool = False,
        point_jitter: float = 0.15,
        point_kwargs: dict | None = None,
        do_stat_tests: bool = False,
        outdir: Path,
    ) -> plt.Axes:
    """
    Draw a grouped boxplot of mean activity values.

    Groups:
      - X-axis: Longest Carbon Backbone (LCB) length, capped at `lcb_max` (default 6)
      - Hue: Isomer class (0=ortho, 1=iso, 2=tere)

    Parameters
    ----------
    descriptor_df : DataFrame
        Must contain 'LongestCarbonBackbone' and 'Isomer' columns.
    activity_df   : DataFrame
        Rows = compounds, columns = assays; numeric activity values.
    lcb_max       : int, default 6
        Highest LCB value to display (inclusive).
    ax            : matplotlib axis (optional)
        If None, a new figure/axis is created.
    palette       : seaborn palette name or list of colors.
    show_points   : bool, default False
        Overlay jittered individual points when True.
    point_jitter  : float, default 0.15
        Half-width of horizontal jitter applied to points.
    point_kwargs  : dict, default None
        Extra kwargs forwarded to ax.scatter for points.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis containing the rendered plot.
    """
    # ------------------------------------------------------------
    # 1. Assemble a tidy frame with compound-level mean activities
    # ------------------------------------------------------------
    compound_mean = activity_df.mean(axis=1, skipna=True)  # mean per compound
    df = (
        descriptor_df[['LongestCarbonBackbone', 'Isomer']]
        .assign(mean_activity=compound_mean)
        .dropna(subset=['LongestCarbonBackbone', 'Isomer'])
    )

    # # keep only LCB ≤ lcb_max
    # df = df[df['LongestCarbonBackbone'] <= lcb_max]

    # # keep only LCB ≥ lcb_min
    # df = df[df['LongestCarbonBackbone'] >= lcb_min]

    # cap any LCBs beyond lcb_max at a single overflow bin (e.g. 7)
    overflow_val = lcb_max + 1
    df.loc[df['LongestCarbonBackbone'] > lcb_max, 'LongestCarbonBackbone'] = overflow_val
    overflow_str = f"{overflow_val}+"
  
    # cap any LCBs below lcb_min at a single underflow bin (e.g. 0)
    underflow_val = lcb_min - 1
    df.loc[df['LongestCarbonBackbone'] < lcb_min, 'LongestCarbonBackbone'] = underflow_val
    underflow_str = f"{underflow_val}−"

    # ensure numeric, ordered categories
    df['LongestCarbonBackbone'] = df['LongestCarbonBackbone'].astype(int)
    df['Isomer'] = df['Isomer'].astype(int)

    # group the data by LCB and isomer to perform statistical difference tests
    def get_tick_label(x, min_val=lcb_min, max_val=lcb_max):
        """Format tick labels for LCB values."""
        if x < min_val:
            return underflow_str
        elif x > max_val:
            return overflow_str
        else:
            return str(x)
        
    isomer_labels = {0: 'ortho', 1: 'iso', 2: 'tere'}
    def group_isomer(x):
        return isomer_labels.get(x, 'unknown')
        
    if do_stat_tests:
        from scipy.stats import kruskal
        import scikit_posthocs as sp
        # import pingouin as pg
        from cliffs_delta import cliffs_delta

        fig, axdict = plt.subplot_mosaic(
            [['box', 'box'],             # top row: the boxplot spans both columns
            ['p',   'delta']],          # bottom row: Dunn–Holm | Cliff’s Δ
            figsize=(12, 10),            # tweak size as you like
            constrained_layout=True     # auto-tight layout
        )

        df['group'] = df['LongestCarbonBackbone'].apply(get_tick_label) + "_" + df['Isomer'].apply(group_isomer)
        groups = [d["mean_activity"].values for _, d in df.groupby("group")]
        group_labels = df["group"].unique()
        
        H, p_kw = kruskal(*groups)
        print(f"Kruskal-Wallis test: H={H:.3f}, p={p_kw:.3g}")

        p_mat = sp.posthoc_dunn(
            df,
            val_col="mean_activity",
            group_col="group",
            p_adjust="holm"
        )
        # p_mat.index = group_labels; p_mat.columns = group_labels   # nice ordering
        order = sorted(p_mat.index, key=lambda s: (s.startswith('5'), s))
        p_mat = p_mat.loc[order, order]

        mask = np.triu(np.ones_like(p_mat, dtype=bool))   # show lower triangle only
        sns.heatmap(
            p_mat,
            mask=mask,
            annot=True,
            fmt=".2g",
            cmap="viridis_r",
            cbar_kws={"label": "p (adj)"},
            vmin=0, vmax=1,
            ax=axdict['p'],
        )
        axdict['p'].set_title("Dunn-Holm pairwise comparisons")
        axdict['p'].set_ylabel("") ; axdict['p'].set_xlabel("")
        # plt.savefig(outdir / "Dunn_Holm.png")
        # plt.show()

        effect = np.full(p_mat.shape, np.nan)
        for i, gi in enumerate(group_labels):
            for j, gj in enumerate(group_labels):
                if i < j:
                    # d = pg.cliffs_delta(
                    d, _ = cliffs_delta(
                        df.loc[df.group==gi, "mean_activity"],
                        df.loc[df.group==gj, "mean_activity"],
                        # eftype="cliffs"
                    )
                    effect[i, j] = effect[j, i] = d

        eff_df = pd.DataFrame(effect, index=order, columns=order)
        mask = np.triu(np.ones_like(eff_df, dtype=bool))   # hide upper triangle
        sns.heatmap(
            eff_df, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            cbar_kws={"label": "Cliff's δ"},
            ax=axdict['delta'],
        )
        axdict['delta'].set_title("Effect-size matrix (Cliff's δ)")
        # plt.savefig(outdir / "Cliffs.png")
        # plt.show()


    lcb_order    = sorted(df['LongestCarbonBackbone'].unique())  # 1 … lcb_max
        
    tick_labels = [get_tick_label(x) for x in lcb_order]

    isomer_order = [0, 1, 2]                                     # ortho, iso, tere
    

    # ------------------------------------------------------------
    # 2. Create the plot
    # ------------------------------------------------------------
    if do_stat_tests:
        ax = axdict['box']
    else:
        _, ax = plt.subplots(figsize=(14, 6), dpi=120)

    # seaborn handles grouped boxplots in one line
    sns.boxplot(
        data=df,
        x='LongestCarbonBackbone',
        y='mean_activity',
        hue='Isomer',
        order=lcb_order,
        hue_order=isomer_order,
        palette=palette,
        width=0.8,
        showfliers=False,
        ax=ax,
    )

    # ------------------------------------------------------------
    # 3. Optional overlay of individual compound points
    # ------------------------------------------------------------
    if show_points:
        default_pts = dict(s=20, alpha=0.5, edgecolors='white')
        default_pts.update(point_kwargs or {})

        # Map categorical positions to numeric centres
        pos_map = {lcb: i for i, lcb in enumerate(lcb_order)}
        hue_offsets = {0: -0.20, 1: 0.0, 2: 0.20}               # shift iso groups

        xs = (
            df['LongestCarbonBackbone'].map(pos_map)
            + df['Isomer'].map(hue_offsets)
            + np.random.uniform(-point_jitter, point_jitter, size=len(df))
        )
        ax.scatter(xs, df['mean_activity'], **default_pts)

    # ------------------------------------------------------------
    # 4. Cosmetics
    # ------------------------------------------------------------
    ax.set_xlabel("Longest Carbon Backbone (number of atoms)")
    ax.set_ylabel("Mean Activity Value")
    # ax.set_title("Mean Activity by LCB and Isomer (≤ C6)")

    ax.set_xticklabels(tick_labels)

    # Replace legend labels with human-friendly isomer names
    handles, labels = ax.get_legend_handles_labels()
    labels = [isomer_labels[int(lbl)] for lbl in labels]
    ax.legend(handles, labels, title="Isomer", frameon=False)

    ax.spines[['top', 'right']].set_visible(False)
    # plt.tight_layout()

    if do_stat_tests:
        fig_labels = {
            'box'  : 'A)',   # top span
            'p'    : 'B)',   # Dunn–Holm heat-map
            'delta': 'C)'    # Cliff’s Δ heat-map
        }

        for key, lab in fig_labels.items():
            ax = axdict[key]
            ax.text(-0.05, 1.05, lab, transform=ax.transAxes,      # just outside upper-left
                    fontsize=14, fontweight='bold', va='top', ha='right')

        plt.savefig(outdir / "combined_lcb_binary.png")

        ax = axdict

    plt.show()
    return ax

def show_C0_mols(descriptor_df: pd.DataFrame):
    """
    Show the molecules with the longest carbon backbone (LCB) of 0.

    Parameters
    ----------
    descriptor_df : pd.DataFrame
        DataFrame containing descriptors.
    """
    from rdkit.Chem import Draw
    # Get the molecules with LCB of 0
    inchi_list = descriptor_df[descriptor_df['LongestCarbonBackbone'] == 0].index

    # Convert InChI strings to RDKit Mol objects, then save as images
    img_path = Path('cache/descriptors/mols')
    img_path.mkdir(parents=True, exist_ok=True)

    for inchi in inchi_list:
        mol = AllChem.MolFromInchi(inchi)
        if mol is not None:
            # Save as image
            img = Draw.MolToImage(mol, size=(300, 300))
            fname = f"{inchi.replace('/', '_')}.png"
            img.save(img_path / fname)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process activity matrix for entity similarity.")
    parser.add_argument('--cachedir', type=str, default='cache/entity_similarity',
                        help='Directory to cache the activity matrix.')
    parser.add_argument('--outdir', type=str, default='cache/descriptors',
                        help='Directory to cache the descriptors.')
    parser.add_argument('--descriptor_plots', action='store_true',
                        help='Plot the activity features against the descriptors.')
    parser.add_argument('--heatmap', action='store_true',
                        help='Draw a heatmap of the descriptor vs. activity correlation.')
    parser.add_argument('--lcb_plots', action='store_true',
                        help='Plot the activity vs. longest carbon backbone (LCB) and isomer type.')
    parser.add_argument('--linear_plot', action='store_true',
                        help='Plot the linear regression model of descriptors vs. activity.')
    parser.add_argument('--no_linear_descriptor', action='store_true',
                        help='Do not use the linear regression model as a descriptor.')
    parser.add_argument('--cluster_heatmap', action='store_true',
                        help='Plot the hierarchical clustered heatmap for compounds and assays.')
    # parser.add_argument('--normalize', action='store_true',
    #                     help='Normalize the activity matrix.')
    # parser.add_argument('--z_score', action='store_true',
    #                     help='Calculate z-scores of the activity matrix by column.')
    args = parser.parse_args()

    cachedir = Path(args.cachedir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    activity_df = get_activity_df(cachedir)

    if args.cluster_heatmap:
        g = styled_heatmap(activity_df, outdir=outdir, row_group_size = 1)        
        plt.show()
    quit()

    # Convert the 'title' column to RDKit Mol objects
    mol_list = [AllChem.AddHs(AllChem.MolFromInchi(s)) for s in tqdm(activity_df.index, desc="Converting InChIs to RDKit Mol objects")]

    # Calculate descriptors for each molecule
    descriptor_parquet = outdir / 'descriptors.parquet'
    if descriptor_parquet.exists():
        print(f"Loading existing descriptors from {descriptor_parquet}")
        descriptor_df = pd.read_parquet(descriptor_parquet)
    else:
        descriptor_vectors = [get_descriptors(mol) for mol in tqdm(mol_list, desc="Calculating descriptors")]
        descriptor_df = pd.DataFrame(descriptor_vectors, index=activity_df.index)
        descriptor_df.to_parquet(descriptor_parquet)

    # manually dropping descriptors with high VIFs
    descriptor_df_cp = descriptor_df.copy()
    descriptor_df = descriptor_df.drop(columns=[
        'MolWt',
        'MolMR',
        'Kappa2',
        'Kappa3',
        'cLogP',
    ])

    if args.lcb_plots:
        # plot the trend with longest carbon backbone
        # plot_activity_scatter_lcb(descriptor_df, activity_df)
        plot_activity_boxplot_lcb_isomer(descriptor_df, activity_df, outdir=outdir)
        plot_activity_boxplot_lcb_isomer(descriptor_df, activity_df, lcb_min=6, lcb_max=5, outdir=outdir, do_stat_tests=True)
        # plot_activity_boxplot_lcb_isomer(descriptor_df, activity_df, lcb_max=100, lcb_min=7)
        # show_C0_mols(descriptor_df)

    # Data preprocessing
    X = z_scale_df(descriptor_df)
    Y = z_scale_df(activity_df)

    # Compute variance inflation factors (VIFs) to check for multicollinearity
    vif_table = compute_vifs(X)
    print("VIF Table:")
    print(vif_table)
    
    for descriptor in vif_table['descriptor']:
        if vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0] > 10:
            print(f"Warning: High VIF detected for descriptor '{descriptor}' (VIF={vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0]}). Consider removing it.")

    # Fit a linear regression model to the data
    # ols, marginal_r2 = get_linear_model(X, Y)
    ols = get_linear_model(X, Y)

    if args.descriptor_plots:
        # Plot the activity features against the descriptors
        # This will create a scatter plot for each descriptor against the mean activity
        # and a LOESS curve to show the trend.
        print("Plotting activity features against descriptors...")
        if args.no_linear_descriptor:
            plot_activity_features(descriptor_df_cp, activity_df, linear_model=None, outdir=outdir)
        else:
            y_mean = activity_df.mean(axis=1)  # mean activity across all assays
            # add linear model predictions as a new descriptor, unscaled
            descriptor_df_cp['LinearModel'] = ols.predict(sm.add_constant(X))*y_mean.std() + y_mean.mean() 
            plot_activity_features(descriptor_df_cp, activity_df, linear_model=ols, outdir=outdir)

    if args.linear_plot:
        # Plot the linear regression model
        # This will show the relationship between the descriptors and the mean activity
        print("Plotting linear regression model...")
        PCA_plot(X, Y)

    # Quick Pearson/Spearman heat-map
    rho = Pearson_correlation_heatmap(X, Y, draw_heatmap=args.heatmap)

    # isolate the "Isomer" row to check if activities line up with chemical intuition
    isomer_row = rho.loc['Isomer']
    # save to a CSV file
    isomer_row.to_csv(outdir / 'isomer_correlation.csv')

    # TODO: PLS regression to find the most predictive descriptors
    rf = get_oob_score(X, Y)
    
    # explainer = shap.TreeExplainer(rf)
    # import shap
    # shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X, plot_type="bar", max_display=20)
    # TODO: random forest regression for nonlinear relationships 
    # TODO: summarize the results in a report or visualization

    
