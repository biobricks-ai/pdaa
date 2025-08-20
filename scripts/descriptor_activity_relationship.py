import argparse
from typing import List, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit.Chem import AllChem
# from skmisc.loess import loess               # pip install scikit-misc
from tqdm import tqdm
import statsmodels.api as sm

import sys
sys.path.append('./')  # so utility scripts can be found
# from stages.utils.pdaa import is_phthalate, longest_carbon_backbone
from scripts.utils.helpers import (
    get_activity_df,
    styled_heatmap,
    # z_scale_df,
    zscore_columns,
    get_linear_model,
    get_descriptors,
    compute_vifs,
    plot_activity_features,
    remove_high_vif_descriptors,
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

# def plot_activity_boxplot_lcb_isomer(
#         descriptor_df: pd.DataFrame,
#         activity_df: pd.DataFrame,
#         *,
#         lcb_min: int = 0,
#         lcb_max: int = 6,
#         ax: plt.Axes | None = None,
#         palette: str | None = "Set2",
#         show_points: bool = False,
#         point_jitter: float = 0.15,
#         point_kwargs: dict | None = None,
#         do_stat_tests: bool = False,
#         outdir: Path,
#     ) -> plt.Axes:
#     """
#     Draw a grouped boxplot of mean activity values.

#     Groups:
#       - X-axis: Longest Carbon Backbone (LCB) length, capped at `lcb_max` (default 6)
#       - Hue: Isomer class (0=ortho, 1=iso, 2=tere)

#     Parameters
#     ----------
#     descriptor_df : DataFrame
#         Must contain 'LongestCarbonBackbone' and 'Isomer' columns.
#     activity_df   : DataFrame
#         Rows = compounds, columns = assays; numeric activity values.
#     lcb_max       : int, default 6
#         Highest LCB value to display (inclusive).
#     ax            : matplotlib axis (optional)
#         If None, a new figure/axis is created.
#     palette       : seaborn palette name or list of colors.
#     show_points   : bool, default False
#         Overlay jittered individual points when True.
#     point_jitter  : float, default 0.15
#         Half-width of horizontal jitter applied to points.
#     point_kwargs  : dict, default None
#         Extra kwargs forwarded to ax.scatter for points.

#     Returns
#     -------
#     ax : matplotlib.axes.Axes
#         Axis containing the rendered plot.
#     """
#     # ------------------------------------------------------------
#     # 1. Assemble a tidy frame with compound-level mean activities
#     # ------------------------------------------------------------
#     compound_mean = activity_df.mean(axis=1, skipna=True)  # mean per compound
#     df = (
#         descriptor_df[['LongestCarbonBackbone', 'Isomer']]
#         .assign(mean_activity=compound_mean)
#         .dropna(subset=['LongestCarbonBackbone', 'Isomer'])
#     )

#     # # keep only LCB ≤ lcb_max
#     # df = df[df['LongestCarbonBackbone'] <= lcb_max]

#     # # keep only LCB ≥ lcb_min
#     # df = df[df['LongestCarbonBackbone'] >= lcb_min]

#     # cap any LCBs beyond lcb_max at a single overflow bin (e.g. 7)
#     overflow_val = lcb_max + 1
#     df.loc[df['LongestCarbonBackbone'] > lcb_max, 'LongestCarbonBackbone'] = overflow_val
#     overflow_str = f"{overflow_val}+"
  
#     # cap any LCBs below lcb_min at a single underflow bin (e.g. 0)
#     underflow_val = lcb_min - 1
#     df.loc[df['LongestCarbonBackbone'] < lcb_min, 'LongestCarbonBackbone'] = underflow_val
#     underflow_str = f"{underflow_val}−"

#     # ensure numeric, ordered categories
#     df['LongestCarbonBackbone'] = df['LongestCarbonBackbone'].astype(int)
#     df['Isomer'] = df['Isomer'].astype(int)

#     # group the data by LCB and isomer to perform statistical difference tests
#     def get_tick_label(x, min_val=lcb_min, max_val=lcb_max):
#         """Format tick labels for LCB values."""
#         if x < min_val:
#             return underflow_str
#         elif x > max_val:
#             return overflow_str
#         else:
#             return str(x)
        
#     isomer_labels = {0: 'ortho', 1: 'iso', 2: 'tere'}
#     def group_isomer(x):
#         return isomer_labels.get(x, 'unknown')
        
#     if do_stat_tests:
#         from scipy.stats import kruskal
#         import scikit_posthocs as sp
#         # import pingouin as pg
#         from cliffs_delta import cliffs_delta

#         fig, axdict = plt.subplot_mosaic(
#             [['box', 'box'],             # top row: the boxplot spans both columns
#             ['p',   'delta']],          # bottom row: Dunn–Holm | Cliff’s Δ
#             figsize=(12, 10),            # tweak size as you like
#             constrained_layout=True     # auto-tight layout
#         )

#         df['group'] = df['LongestCarbonBackbone'].apply(get_tick_label) + "_" + df['Isomer'].apply(group_isomer)
#         groups = [d["mean_activity"].values for _, d in df.groupby("group")]
#         group_labels = df["group"].unique()
        
#         H, p_kw = kruskal(*groups)
#         print(f"Kruskal-Wallis test: H={H:.3f}, p={p_kw:.3g}")

#         p_mat = sp.posthoc_dunn(
#             df,
#             val_col="mean_activity",
#             group_col="group",
#             p_adjust="holm"
#         )
#         # p_mat.index = group_labels; p_mat.columns = group_labels   # nice ordering
#         order = sorted(p_mat.index, key=lambda s: (s.startswith('5'), s))
#         p_mat = p_mat.loc[order, order]

#         mask = np.triu(np.ones_like(p_mat, dtype=bool))   # show lower triangle only
#         sns.heatmap(
#             p_mat,
#             mask=mask,
#             annot=True,
#             fmt=".2g",
#             cmap="viridis_r",
#             cbar_kws={"label": "p (adj)"},
#             vmin=0, vmax=1,
#             ax=axdict['p'],
#         )
#         axdict['p'].set_title("Dunn-Holm pairwise comparisons")
#         axdict['p'].set_ylabel("") ; axdict['p'].set_xlabel("")
#         # plt.savefig(outdir / "Dunn_Holm.png")
#         # plt.show()

#         effect = np.full(p_mat.shape, np.nan)
#         for i, gi in enumerate(group_labels):
#             for j, gj in enumerate(group_labels):
#                 if i < j:
#                     # d = pg.cliffs_delta(
#                     d, _ = cliffs_delta(
#                         df.loc[df.group==gi, "mean_activity"],
#                         df.loc[df.group==gj, "mean_activity"],
#                         # eftype="cliffs"
#                     )
#                     effect[i, j] = effect[j, i] = d

#         eff_df = pd.DataFrame(effect, index=order, columns=order)
#         mask = np.triu(np.ones_like(eff_df, dtype=bool))   # hide upper triangle
#         sns.heatmap(
#             eff_df, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
#             cbar_kws={"label": "Cliff's δ"},
#             ax=axdict['delta'],
#         )
#         axdict['delta'].set_title("Effect-size matrix (Cliff's δ)")
#         # plt.savefig(outdir / "Cliffs.png")
#         # plt.show()


#     lcb_order    = sorted(df['LongestCarbonBackbone'].unique())  # 1 … lcb_max
        
#     tick_labels = [get_tick_label(x) for x in lcb_order]

#     isomer_order = [0, 1, 2]                                     # ortho, iso, tere
    

#     # ------------------------------------------------------------
#     # 2. Create the plot
#     # ------------------------------------------------------------
#     if do_stat_tests:
#         ax = axdict['box']
#     else:
#         _, ax = plt.subplots(figsize=(14, 6), dpi=120)

#     # seaborn handles grouped boxplots in one line
#     sns.boxplot(
#         data=df,
#         x='LongestCarbonBackbone',
#         y='mean_activity',
#         hue='Isomer',
#         order=lcb_order,
#         hue_order=isomer_order,
#         palette=palette,
#         width=0.8,
#         showfliers=False,
#         ax=ax,
#     )

#     # ------------------------------------------------------------
#     # 3. Optional overlay of individual compound points
#     # ------------------------------------------------------------
#     if show_points:
#         default_pts = dict(s=20, alpha=0.5, edgecolors='white')
#         default_pts.update(point_kwargs or {})

#         # Map categorical positions to numeric centres
#         pos_map = {lcb: i for i, lcb in enumerate(lcb_order)}
#         hue_offsets = {0: -0.20, 1: 0.0, 2: 0.20}               # shift iso groups

#         xs = (
#             df['LongestCarbonBackbone'].map(pos_map)
#             + df['Isomer'].map(hue_offsets)
#             + np.random.uniform(-point_jitter, point_jitter, size=len(df))
#         )
#         ax.scatter(xs, df['mean_activity'], **default_pts)

#     # ------------------------------------------------------------
#     # 4. Cosmetics
#     # ------------------------------------------------------------
#     ax.set_xlabel("Longest Carbon Backbone (number of atoms)")
#     ax.set_ylabel("Mean Activity Value")
#     # ax.set_title("Mean Activity by LCB and Isomer (≤ C6)")

#     ax.set_xticklabels(tick_labels)

#     # Replace legend labels with human-friendly isomer names
#     handles, labels = ax.get_legend_handles_labels()
#     labels = [isomer_labels[int(lbl)] for lbl in labels]
#     ax.legend(handles, labels, title="Isomer", frameon=False)

#     ax.spines[['top', 'right']].set_visible(False)
#     # plt.tight_layout()

#     if do_stat_tests:
#         fig_labels = {
#             'box'  : 'A)',   # top span
#             'p'    : 'B)',   # Dunn–Holm heat-map
#             'delta': 'C)'    # Cliff’s Δ heat-map
#         }

#         for key, lab in fig_labels.items():
#             ax = axdict[key]
#             ax.text(-0.05, 1.05, lab, transform=ax.transAxes,      # just outside upper-left
#                     fontsize=14, fontweight='bold', va='top', ha='right')

#         plt.savefig(outdir / "combined_lcb_binary.png")

#         ax = axdict

#     plt.show()
#     return ax

def plot_activity_boxplot_lcb_isomer(
        descriptor_df: pd.DataFrame,
        activity_df: pd.DataFrame,
        *,
        range_sets : List[Set] = None,
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

    if range_sets is None:
        # cap any LCBs below lcb_min at a single underflow bin (e.g. 0)
        underflow_val = lcb_min - 1
        low_set = set(range(df['LongestCarbonBackbone'].min(), underflow_val + 1))

        # cap any LCBs beyond lcb_max at a single overflow bin (e.g. 7)
        overflow_val = lcb_max + 1
        high_set = set(range(overflow_val, df['LongestCarbonBackbone'].max() + 1))
    
        # make a set for each other LCB value
        range_sets = [{i} for i in range(lcb_min, lcb_max + 1)]
        # add other sets in the correct order
        range_sets = [low_set] + range_sets + [high_set]

        # remove any empty sets
        range_sets = [s for s in range_sets if s]
        # remove singletons with invalid values
        range_sets = [
            s for s in range_sets
            if not (
                len(s) == 1 and (next(iter(s)) not in df['LongestCarbonBackbone'].unique())
            )
        ]

    # loop through rows to get sets containing the LCB value,
    # duplicating the row if multiple sets match
    df['lcb_group'] = pd.Series(dtype='object')  # initialize a new column for groups
    new_rows = []
    rows_to_drop = []
    for i, row in df.iterrows():
        lcb = row['LongestCarbonBackbone']
        matching_sets = [s for s in range_sets if lcb in s]
        if not matching_sets:
            df.at[i, 'lcb_group'] = None
        elif len(matching_sets) == 1:
            df.at[i, 'lcb_group'] = tuple(matching_sets[0])
        else:
            # multiple sets match, so we need to create a new row for each
            for s in matching_sets:
                new_row = row.copy()
                new_row['lcb_group'] = tuple(s)
                new_rows.append(new_row)
            rows_to_drop.append(i)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    if rows_to_drop:
        df = df.drop(rows_to_drop)

    # get the string representation of the set for tick labels
    def set_to_str(s):
        """Convert a set to a string representation."""
        if s is None:
            return "unknown"
        elif len(s) == 1:
            return str(next(iter(s)))  # pulls the single value from the set
        elif s == set(range(min(s), max(s) + 1)):
            if min(s) <= 1:
                return f"{max(s)}−"
            elif max(s) >= df['LongestCarbonBackbone'].max():
                return f"{min(s)}+"
            return f"{min(s)}–{max(s)}"
        else:
            return str(sorted(s))
        
    set_labels = {tuple(s): set_to_str(s) for s in range_sets}
    # Convert tuple categories to string labels to avoid pandas MultiIndex issues
    df['lcb_group'] = df['lcb_group'].map(lambda t: set_labels.get(t, "unknown"))
    
    # Keep an explicit string label order for plotting
    order_labels = [set_labels[tuple(s)] for s in range_sets]

    # Normalize isomer to a name column and keep working off that (handles missing "iso")
    isomer_labels = {0: 'ortho', 1: 'iso', 2: 'tere'}
    if np.issubdtype(df['Isomer'].dtype, np.number):
        df['IsomerName'] = df['Isomer'].map(isomer_labels).fillna('unknown')
    else:
        df['IsomerName'] = (
            df['Isomer'].astype(str).str.lower()
              .map({'o': 'ortho', 'ortho': 'ortho',
                    'i': 'iso',   'iso': 'iso',
                    't': 'tere',  'tere': 'tere'})
              .fillna('unknown')
        )
        
    isomer_labels = {0: 'ortho', 1: 'iso', 2: 'tere'}
    def group_isomer(x):
        return isomer_labels.get(x, 'unknown')
        
    if do_stat_tests:
        from scipy.stats import kruskal
        import scikit_posthocs as sp
        # import pingouin as pg
        from cliffs_delta import cliffs_delta

        # def remove_masked_values(mat: pd.DataFrame, mask) -> pd.DataFrame:
        #     rows_to_keep, cols_to_keep = np.where(~mask)
        #     rows_to_keep = np.unique(rows_to_keep)
        #     cols_to_keep = np.unique(cols_to_keep)
        #     mat_filtered = mat.iloc[rows_to_keep, cols_to_keep]
        #     return mat_filtered

        def clean_matrix(mat: pd.DataFrame):
            # first remove the top row and right column
            # mat_cleaned = mat.iloc[:-1, :-1]
            mat_cleaned = mat.iloc[1:, :-1]
            # mask upper triangle
            mask = np.triu(np.ones_like(mat_cleaned, dtype=bool), k=1)

            d = {'data': mat_cleaned, 'mask': mask}

            return d

        fig, axdict = plt.subplot_mosaic(
            [['box', 'box'],             # top row: the boxplot spans both columns
            ['p',   'delta']],          # bottom row: Dunn–Holm | Cliff’s Δ
            figsize=(12, 10),            # tweak size as you like
            constrained_layout=True     # auto-tight layout
        )

        # Use normalized names so this works when only a subset (e.g., ortho/tere) is present
        df['group'] = df['lcb_group'] + "_" + df['IsomerName']

        groups = [d["mean_activity"].values for _, d in df.groupby("group")]
        group_labels = df["group"].unique()
        
        H, p_kw = kruskal(*groups)
        print(f"Kruskal-Wallis test: H={H:.3f}, p={p_kw:.3g}")

        # p_mat = sp.posthoc_dunn(
        p_mat = sp.posthoc_conover(
            df,
            val_col="mean_activity",
            group_col="group",
            p_adjust="holm"
        )
        # p_mat.index = group_labels; p_mat.columns = group_labels   # nice ordering
        order = sorted(p_mat.index, key=lambda s: (s.startswith('5'), s))
        p_mat = p_mat.loc[order, order]

        # mask = np.triu(np.ones_like(p_mat, dtype=bool))   # show lower triangle only
        sns.heatmap(
            # remove_masked_values(p_mat, mask),
            **clean_matrix(p_mat),
            # p_mat,
            # mask=mask,
            annot=True,
            fmt=".2g",
            cmap="viridis_r",
            cbar_kws={"label": "p (adj)"},
            vmin=0, vmax=1,
            ax=axdict['p'],
        )
        # axdict['p'].set_title("Dunn-Holm pairwise comparisons")
        axdict['p'].set_title("Conover-Iman pairwise comparisons")
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
        # 
        # mask = np.triu(np.ones_like(eff_df, dtype=bool))   # hide upper triangle
        sns.heatmap(
            # remove_masked_values(eff_df, mask),
            **clean_matrix(eff_df),
            # eff_df,
            # mask=mask,
            annot=True, fmt=".2f", cmap="coolwarm", center=0,
            cbar_kws={"label": "Cliff's δ"},
            ax=axdict['delta'],
        )
        axdict['delta'].set_title("Effect-size matrix (Cliff's δ)")
        # plt.savefig(outdir / "Cliffs.png")
        # plt.show()

    # Determine isomer order dynamically (works for a subset like ['ortho','tere'])
    canonical = ['ortho', 'iso', 'tere', 'unknown']
    present = list(dict.fromkeys(df['IsomerName']))  # preserve appearance order
    isomer_order = [lvl for lvl in canonical if lvl in set(present)]

    # Build a stable color mapping so colors don't shift when 'iso' is absent
    if isinstance(palette, dict):
        palette_map = palette
    else:
        base_colors = sns.color_palette(palette, n_colors=3)
        palette_map = {'ortho': base_colors[0], 'iso': base_colors[1], 'tere': base_colors[2]}
    if 'unknown' in set(present) and 'unknown' not in palette_map:
        palette_map['unknown'] = (0.5, 0.5, 0.5)  # neutral gray for unexpected labels

    if do_stat_tests:
        ax = axdict['box']
    else:
        _, ax = plt.subplots(figsize=(14, 6), dpi=120)

    sns.boxplot(
        data=df,
        # x='LongestCarbonBackbone',
        x='lcb_group',
        y='mean_activity',
        hue='IsomerName',
        order=order_labels,
        hue_order=isomer_order,
        palette=palette_map,
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

        # Map categorical positions to numeric centres (keys are string labels)
        pos_map = {lab: i for i, lab in enumerate(order_labels)}

        # Centered offsets for however many isomer levels are present (1, 2, or 3)
        offsets = np.linspace(-0.25, 0.25, num=len(isomer_order)) if len(isomer_order) > 1 else np.array([0.0])
        hue_offsets = dict(zip(isomer_order, offsets))

        xs = (
            df['lcb_group'].map(pos_map)
            + df['IsomerName'].map(hue_offsets)
            + np.random.uniform(-point_jitter, point_jitter, size=len(df))
        )
        ax.scatter(xs, df['mean_activity'], **default_pts)


    # ------------------------------------------------------------
    # 4. Cosmetics
    # ------------------------------------------------------------
    ax.set_xlabel("Longest Carbon Backbone (number of atoms)")
    ax.set_ylabel("Mean Activity Value")
    # ax.set_title("Mean Activity by LCB and Isomer (≤ C6)")

    # Ensure tick labels align with the explicit order
    ax.set_xticklabels(order_labels)

    # With named isomers, Seaborn legend labels are already human-friendly
    ax.legend(title="Isomer", frameon=False)


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
    else:
        plt.savefig(outdir / "lcb_isomer_boxplot.png")

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

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

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
    parser.add_argument('--cachedir', type=str, default='cache/entity_similarity2',
                        help='Directory to cache the activity matrix.')
    parser.add_argument('--outdir', type=str, default='cache/descriptors',
                        help='Directory to cache the descriptors.')
    parser.add_argument('--descriptor_plots', action='store_true',
                        help='Plot the activity features against the descriptors.')
    parser.add_argument('--generate_descriptors', action='store_true',
                        help='Generate descriptors for the molecules even if cache exists.')
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
    parser.add_argument('--zscore', action='store_true',
                        help='Calculate z-scores of the activity matrix by column.')
    args = parser.parse_args()

    cachedir = Path(args.cachedir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    activity_df = get_activity_df(cachedir)

    # calculate least and most active assays
    activity_df_stats = pd.DataFrame({
        "mean": activity_df.mean(axis=0, skipna=True),         # column-wise means
        "se":   activity_df.sem(axis=0, skipna=True, ddof=1),  # column-wise standard errors (s/√n, sample std)
    })
    # make sure the index clearly reflects assay titles
    activity_df_stats.index.name = activity_df.columns.name or "title"
    # sort and save the stats
    activity_df_stats = activity_df_stats.sort_values(by='mean')
    stats_file = outdir / 'activity_stats.csv'
    activity_df_stats.to_csv(stats_file)
    print(f"Activity stats saved to {stats_file}")

    if args.cluster_heatmap:
        g = styled_heatmap(activity_df, outdir=outdir, row_group_size = 1)        
        plt.show()

    # Convert the 'title' column to RDKit Mol objects
    mol_list = [AllChem.AddHs(AllChem.MolFromInchi(s)) for s in tqdm(activity_df.index, desc="Converting InChIs to RDKit Mol objects")]

    # Calculate descriptors for each molecule
    descriptor_parquet = outdir / 'descriptors.parquet'
    if descriptor_parquet.exists() and not args.generate_descriptors:
        print(f"Loading existing descriptors from {descriptor_parquet}")
        descriptor_df = pd.read_parquet(descriptor_parquet)
    else:
        descriptor_vectors = [get_descriptors(mol) for mol in tqdm(mol_list, desc="Calculating descriptors")]
        descriptor_df = pd.DataFrame(descriptor_vectors, index=activity_df.index)
        descriptor_df.to_parquet(descriptor_parquet)

    

    # manually dropping descriptors with high VIFs
    descriptor_df_cp = descriptor_df.copy()
    # descriptor_df = descriptor_df.drop(columns=[
    #     'MolWt',
    #     'MolMR',
    #     'Kappa2',
    #     'Kappa3',
    #     'cLogP',
    # ])
    descriptor_df = remove_high_vif_descriptors(descriptor_df, vif_threshold=10)

    # Data preprocessing
    X, _ = zscore_columns(descriptor_df)
    Y, _ = zscore_columns(activity_df)

    if args.lcb_plots:
        # plot the trend with longest carbon backbone
        if args.zscore:
            activity_Y = Y
        else:
            activity_Y = activity_df

        # plot_activity_boxplot_lcb_isomer(descriptor_df_cp, activity_Y, outdir=outdir)
        # plot_activity_boxplot_lcb_isomer(descriptor_df_cp, activity_Y, lcb_min=7, lcb_max=6, outdir=outdir, do_stat_tests=True)
        max_lcb = descriptor_df_cp['LongestCarbonBackbone'].max()
        plot_activity_boxplot_lcb_isomer(descriptor_df_cp, activity_Y, range_sets=[{1, 2, 3}, {4, 5, 6}, set(range(7, max_lcb + 1))], outdir=outdir, do_stat_tests=True)
        # show_C0_mols(descriptor_df_cp)

    # # Compute variance inflation factors (VIFs) to check for multicollinearity
    # vif_table = compute_vifs(X)
    # print("VIF Table:")
    # print(vif_table)
    
    # for descriptor in vif_table['descriptor']:
    #     if vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0] > 10:
    #         print(f"Warning: High VIF detected for descriptor '{descriptor}' (VIF={vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0]}). Consider removing it.")

    # # Remove descriptors with high VIFs
    # X = remove_high_vif_descriptors(X, vif_threshold=10)

    # Fit a linear regression model to the data
    # ols, marginal_r2 = get_linear_model(X, Y)
    ols = get_linear_model(X, Y)

    if args.descriptor_plots:
        # Plot the activity features against the descriptors
        # This will create a scatter plot for each descriptor against the mean activity
        # and a LOESS curve to show the trend.
        print("Plotting activity features against descriptors...")
        plot_activity_df = Y if args.zscore else activity_df

        if args.no_linear_descriptor:
            plot_activity_features(descriptor_df_cp, plot_activity_df, linear_model=None, outdir=outdir)
        else:
            y_mean = activity_df.mean(axis=1)  # mean activity across all assays
            # add linear model predictions as a new descriptor, unscaled
            descriptor_df_cp['LinearModel'] = ols.predict(sm.add_constant(X))*y_mean.std() + y_mean.mean() 
            plot_activity_features(descriptor_df_cp, plot_activity_df, linear_model=ols, outdir=outdir)

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

    
