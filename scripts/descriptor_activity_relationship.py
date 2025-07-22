import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from statsmodels.stats.outliers_influence import variance_inflation_factor
from skmisc.loess import loess               # pip install scikit-misc
from tqdm import tqdm

import sys
sys.path.append('./')
from stages.utils.pdaa import is_phthalate, longest_carbon_backbone

def z_scale_df(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize the dataframe by columns."""
    # mean = np.mean(matrix, axis=0)
    # std_dev = np.std(matrix, axis=0)
    return (df - df.mean())/df.std()

def classify_isomer(mol: AllChem.Mol) -> int:
    """
    Classify the isomer type of a phthalate molecule based on its structure.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
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

def get_branching_ratio(mol: AllChem.Mol) -> float:
    """
    Calculate the branching ratio of a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
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

def get_descriptors(mol):
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
        'LongestCarbonBackbone': longest_carbon_backbone(mol),  # longest carbon side chain length
        'BranchingRatio'   : get_branching_ratio(mol),  # branching ratio
    }
    # 3-D shape (needs conformer)
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    feats['Rgyr'] = Descriptors3D.RadiusOfGyration(mol)
    # custom: side-chain length & branching (sketch)
    feats['Isomer'] = classify_isomer(mol)  # 0=ortho,1=iso,2=tere
    return feats

def get_activity_df(cachedir: str | Path) -> pd.DataFrame:
    """Load the activity matrix from a parquet file."""
    # Define the path to the parquet file
    cachedir = Path('cache/entity_similarity')
    activity_df_path = cachedir / 'activity_matrix_filled.parquet'

    # Load the activity matrix from the parquet file
    activity_df = pd.read_parquet(activity_df_path)

    return activity_df

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

        # ax.set_title("Descriptor-vs-Assay Correlation Heatmap")
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
        plt.title('Descriptor-vs-Assay Correlation Heatmap', pad=40)
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
        lcb_max: int = 6,
        lcb_min: int = 0,
        ax: plt.Axes | None = None,
        palette: str | None = "Set2",
        show_points: bool = False,
        point_jitter: float = 0.15,
        point_kwargs: dict | None = None,
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
    # cap any LCBs beyond lcb_max at a single overflow bin (e.g. 7)
    overflow_val = lcb_max + 1          # 7 when lcb_max == 6
    df.loc[df['LongestCarbonBackbone'] > lcb_max, 'LongestCarbonBackbone'] = overflow_val

    # keep only LCB ≥ lcb_min
    df = df[df['LongestCarbonBackbone'] >= lcb_min]

    # ensure numeric, ordered categories
    df['LongestCarbonBackbone'] = df['LongestCarbonBackbone'].astype(int)
    df['Isomer'] = df['Isomer'].astype(int)

    lcb_order    = sorted(df['LongestCarbonBackbone'].unique())  # 1 … lcb_max
    tick_labels = [str(x) if x <= lcb_max else f"{x}+" for x in lcb_order]

    isomer_order = [0, 1, 2]                                     # ortho, iso, tere
    isomer_labels = {0: 'ortho', 1: 'iso', 2: 'tere'}

    # ------------------------------------------------------------
    # 2. Create the plot
    # ------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6), dpi=120)

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
    plt.tight_layout()
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

def plot_activity_features(descriptor_df: pd.DataFrame, activity_df: pd.DataFrame):
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
    key_descriptors = [
        'Rgyr', 
        'RotB', 'BranchingRatio', 'Fsp3',
        'Kappa1', 'TPSA', 
    ]
    # is_discrete = [False, False, True, True, False]  # whether the descriptor is discrete

    # Create a figure with subplots for each descriptor
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    Y = activity_df.mean(axis=1)  # mean activity across all assays
    for i, descriptor in enumerate(key_descriptors):
        ax = axes[i]
        x = descriptor_df[descriptor]
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

        # # Plot each activity against the descriptor
        # for activity in activity_df.columns:
            # ax.scatter(
            #     descriptor_df[descriptor],
            #     activity_df[activity],
            #     # label=activity,
            #     alpha=0.5
            # )
        
        # ax.set_title(f"Activity vs. {descriptor}")
        ax.set_xlabel(descriptor)
        ax.set_ylabel("Activity")

    # Remove any empty subplots
    for j in range(len(key_descriptors), len(axes)):
        fig.delaxes(axes[j])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process activity matrix for entity similarity.")
    parser.add_argument('--cachedir', type=str, default='cache/entity_similarity',
                        help='Directory to cache the activity matrix.')
    parser.add_argument('--outdir', type=str, default='cache/descriptors',
                        help='Directory to cache the descriptors.')
    parser.add_argument('--heatmap', action='store_true',
                        help='Draw a heatmap of the descriptor vs activity correlation.')
    parser.add_argument('--lcb_plots', action='store_true',
                        help='Plot the activity vs. longest carbon backbone (LCB) and isomer type.')
    # parser.add_argument('--normalize', action='store_true',
    #                     help='Normalize the activity matrix.')
    # parser.add_argument('--z_score', action='store_true',
    #                     help='Calculate z-scores of the activity matrix by column.')
    args = parser.parse_args()

    cachedir = Path(args.cachedir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    activity_df = get_activity_df(cachedir)

    # # Convert the DataFrame to a NumPy array
    # activity_array = activity_df.to_numpy()

    # # subtract the mean by column
    # activity_array -= np.mean(activity_array, axis=0)
    # # optionally calculate z-scores of the activity matrix by column
    # if args.z_score:
    #     activity_array = activity_array/np.std(activity_array, axis=0)
    #     if args.normalize:
    #         # give a warning that only one of normalize or z-score should be used
    #         print("Warning: Both --normalize and --z_score are set. Only z-score will be applied.")
    # optionally normalize the activity matrix by row
    # elif args.normalize:
    #     activity_array /= np.linalg.norm(activity_array, axis=1, keepdims=True)

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

    plot_activity_features(descriptor_df, activity_df)

    # manually dropping descriptors with high VIFs
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
        plot_activity_boxplot_lcb_isomer(descriptor_df, activity_df)
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

    
