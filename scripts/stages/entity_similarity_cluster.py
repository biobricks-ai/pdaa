import numpy as np
import re
import pathlib
import pandas as pd
import sqlite3
from tqdm import tqdm
import warnings

import rdkit, rdkit.Chem.rdMolDescriptors, rdkit.Chem.Crippen, rdkit.Chem.rdFingerprintGenerator, rdkit.DataStructs
from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex, to_rgba

from scipy.cluster import hierarchy

import sys
sys.path.append('./')
from scripts.utils.helpers import (
    get_descriptors, is_phthalate, is_diester_phthalate
)

resourcedir = pathlib.Path('resources')

# diverging colormap for heatmap
diverging_colormap = 'vlag'  # okay, too washed out
# diverging_colormap = 'coolwarm'  # looks terrible
# diverging_colormap = 'berlin'  # 

def _styled_heatmap(
        matrix, row_colors, *, dpi=600,
        fontcolor='white', linecolor='black', z_scale=False,
        xlabel = 'DART or ED Assays',
):
    """
    Wrapper around seaborn.clustermap with the same visual
    tweaks used in build_heatmap.py (_generate_heatmap).
    - matrix: rows = phthalates, cols = ICE assays
    - row_colors: list-like, same length as matrix.shape[0]
    """
    # Cluster only columns; we already ordered rows
    if z_scale:
        vscale = 3
        vmin = -vscale
        vmax = +vscale
    else:
        vmin=0
        vmax=1

    g = sns.clustermap(
        matrix,
        # square=True,       # ← force equal-sized cells
        cbar_kws={'drawedges': False},  # disable seaborn’s built-in bar
        cmap=diverging_colormap if z_scale else 'viridis',
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        xticklabels=False, yticklabels=False,
        linecolor=linecolor,
        # linewidths=0.5,
        figsize=(18, 9),
        # cbar_pos=(0.91, 0.3, 0.02, 0.4),
        cbar_pos=(0.95, 0.3, 0.02, 0.4),
        dendrogram_ratio=(0.10, 0.05),
        tree_kws={'linewidths': 0.5},
        vmin=vmin, vmax=vmax,
        # clip=True,
    )
    # Remove any stray colorbar
    if hasattr(g, 'cax') and g.cax:
        g.cax.remove()

    # Create a new colorbar on its own axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(g.ax_heatmap)
    cax = divider.append_axes("right", size="2%", pad=0.6)
    # expose the divider so callers can add more axes without destroying the layout
    g.divider = divider
    

    sm  = plt.cm.ScalarMappable(
        cmap=diverging_colormap if z_scale else 'viridis',
        norm=plt.Normalize(
            # vmin=matrix.min().min(),
            # vmax=matrix.max().max())
            vmin=vmin, vmax=vmax,
            clip=True,
        )
    )
    sm.set_array([])
    cb = g.figure.colorbar(sm, cax=cax)
    cb.set_label('Activity Score', fontsize=18, labelpad=10)
    cb.ax.tick_params(labelsize=14)

    # Hide col dendrogram but keep clustering
    show_dendrogram = True
    g.ax_col_dendrogram.set_visible(show_dendrogram)

    g.ax_heatmap.set_xlabel(xlabel, color=fontcolor, fontsize=20)

    # Label the colorbar
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.set_label('Activity Score', fontsize=20, labelpad=10)

    # Tidy up margins so nothing is clipped
    g.figure.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.05)

    return g  # caller can add arrows, bars, etc.
# --------------------------------------------------------------------

tqdm.pandas()

brickdir = pathlib.Path('brick')
cachedir = pathlib.Path('cache') / 'entity_similarity'
cachedir.mkdir(parents=True, exist_ok=True)

# region EXAMPLE PHTHALATES ====================================================================
example_phthalates_df = pd.read_csv(resourcedir / 'example_phthalates.csv')
# shorten the names
example_phthalates_df['name'] = example_phthalates_df['name'].str.replace('Dimethyl ', '')
example_phthalates_df.sort_values(by='name', inplace=True)

example_phthalates = [Chem.MolFromSmiles(smiles) for smiles in example_phthalates_df['smiles']]
example_names = example_phthalates_df['name'].tolist()
example_inchi = [Chem.MolToInchi(m) for m in example_phthalates]
example_weights = [rdkit.Chem.rdMolDescriptors.CalcExactMolWt(m) for m in example_phthalates]
example_inchi2name = {inchi: name for inchi, name in zip(example_inchi, example_names)}

# print the alias weights
for name, weight in zip(example_names, example_weights):
    print(f"{name}: {weight:.2f}")

# possible isomers to look for
isomers_list = [
    "ortho_phthalate",
    # "meta_phthalate",
    "para_phthalate"
]
phthalate_modes = tuple(isomers_list)

# which phthalate has the lowest mean ICE activity?
# region ICE ACTIVITY ===============================================================


def build_phthalate_ice_activity_df(use_cache=True):
    print("Building phthalate ICE activity dataframe...")
    
    ice_assays.to_csv(resourcedir / 'ice_assays.csv', index=False)

    ice_preds_parquet = cachedir / 'ice_preds.parquet'
    if use_cache and ice_preds_parquet.exists():
        print(f"Loading cached predictions from {ice_preds_parquet}...")
        ice_preds = pd.read_parquet(ice_preds_parquet)
    else:
        print("Fetching predictions from SQLite...")
        with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
            tokens = ','.join(map(str, ice_assays['token'].tolist()))
            query = f'SELECT * FROM predictions WHERE property_token IN ({tokens})'
            ice_preds = pd.read_sql(query, conn)
        ice_preds.to_parquet(ice_preds_parquet)

    try:
        assert all(inchi in ice_preds['inchi'].tolist() for inchi in example_inchi)
    except AssertionError:
        print(ice_preds['inchi'])
        bool_array = [inchi in ice_preds['inchi'].tolist() for inchi in example_inchi]
        print(bool_array)
        print("Some example InChIs are not present in the predictions data. Please check the SQLite database.")
        raise

    print("Processing predictions data...")
    df = ice_preds.sort_values('positive_prediction', ascending=True)[['inchi', 'property_token', 'positive_prediction']]
    df = df.groupby(['inchi','property_token'])['positive_prediction'].mean().reset_index()

    inchi_mol_pkl = cachedir / 'inchi_mol.pkl'
    if use_cache and inchi_mol_pkl.exists():
        print(f"Loading cached InChI to molecule mapping from {inchi_mol_pkl}...")
        inchi_mol_df = pd.read_pickle(inchi_mol_pkl)
    else:
        inchi_mol_df = df[['inchi']].drop_duplicates()
        print("Converting InChIs to molecules...")
        inchi_mol_df['mol'] = inchi_mol_df['inchi'].progress_apply(lambda x: Chem.MolFromInchi(x))
        inchi_mol_df.to_pickle(inchi_mol_pkl)

    df2 = df.merge(inchi_mol_df, on='inchi')

    try:
        assert all(is_diester_phthalate(m) for m in example_phthalates)
    except AssertionError as e:
        print("Some example phthalates are not diester phthalates. Please check the SMARTS pattern.")
        raise e

    print(f"Filtering for phthalates with modes = {phthalate_modes}...")
    phthalate_options = {
        'check_elements': True,
        'valid_num_rings': [1],
    }

    filtered_phthalates = inchi_mol_df[inchi_mol_df['mol'].progress_apply(
        # need to check that mol both has matching modes and is a diester 
        lambda m: \
            is_phthalate(
                m, modes=phthalate_modes, **phthalate_options, match_mode='one'
            ) and is_diester_phthalate(m, **phthalate_options)
    )]['inchi']
    df3 = df2[df2['inchi'].isin(filtered_phthalates)]
    # Ensure both columns are of the same type (int)
    df3.loc[:, 'property_token'] = df3['property_token'].astype(int)
    ice_assays.loc[:, 'token'] = ice_assays['token'].astype(int)
    df3 = df3.merge(ice_assays, left_on='property_token', right_on='token')[['uri','title','inchi','mol','positive_prediction']]
    print(f"Final dataset contains {len(df3)} rows")
    return df3

phthalate_df = build_phthalate_ice_activity_df()[['uri','title','inchi','mol','positive_prediction']]



# region HEATMAP & DENSITY OF PHTHALATE ACTIVITY ===============================
def cluster_rows_and_make_heatmap(
    *,
    z_scale=False,
    active_top=True,
):
    # save the filled activitiy matrix
    activity_matrix_filled = pd.read_parquet(cachedir / 'activity_matrix_filled.parquet')

    if z_scale:
        # Z-score normalization
        activity_matrix_filled = (activity_matrix_filled - activity_matrix_filled.mean(axis=0)) / activity_matrix_filled.std(axis=0)

    row_clusters = np.zeros(len(activity_matrix_filled), dtype=int)
    row_colors = None

    # Force NumPy array to avoid pandas Series positional-indexing deprecation.
    row_clusters = np.asarray(row_clusters, dtype=int)

    # Still use hierarchical clustering for column ordering
    col_linkage = hierarchy.linkage(activity_matrix_filled.T, method='average')
    col_order = hierarchy.leaves_list(col_linkage)
    # row_linkage = hierarchy.linkage(activity_matrix_filled, method='average')
    # cluster_order = hierarchy.leaves_list(row_linkage)

    mean_activities = activity_matrix_filled.mean(axis=1)
    cluster_order = np.argsort(mean_activities)

    # Normalize to plain ndarray so positional indexing is unambiguous.
    cluster_order = np.asarray(cluster_order, dtype=int)

    if active_top:
        # Reverse the order so that higher activity is at the top
        cluster_order = cluster_order[::-1]

    # Reorder the matrix
    reordered_matrix = activity_matrix_filled.iloc[cluster_order, col_order]
    # ─── Map each InChI to its new row index ─────────────────────────────
    reordered_indices = {
        inchi: pos
        for pos, inchi in enumerate(reordered_matrix.index)
    }

    mean_activity = mean_activities.iloc[cluster_order]
    # # Calculate mean activity per chemical across all assays
    # mean_activity = reordered_matrix.mean(axis=1)

    # Define specific colors for each cluster
    # isomer_colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blue, Red, Green
    base_colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blue, Red, Green
    # base_rgba = [tuple(int(c * 255) for c in plt.colors.to_rgba(color)) for color in base_colors]
    base_rgba = [np.array(to_rgba(color)) for color in base_colors]
    def mix_colors(colors_bool_list):
        """
        Mix colors based on a boolean list.
        """
        if not any(colors_bool_list):
            return '#000000'  # Return black if no colors are selected
        return to_hex(
            sum(
                [base_rgba[i] for i in range(len(base_rgba)) if colors_bool_list[i]]
            )/sum(colors_bool_list)
        )
    isomer_colors = {
        (i, j, k) : mix_colors([i, j, k])
        for i in [True, False]
        for j in [True, False]
        for k in [True, False]
    }

    xlabel = 'DART or ED Assays'

    g = _styled_heatmap(reordered_matrix, row_colors, fontcolor='black', z_scale=z_scale, xlabel=xlabel)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # ─── Attach a fresh bar axis ──────────────────────────────────────────
    # divider = make_axes_locatable(g.ax_heatmap)
    # ax_bar = divider.append_axes("right", size="15%", pad=0.8)
    divider = g.divider            # reuse the one that already holds the colour-bar
    ax_bar  = divider.append_axes("right", size="15%", pad=1.0)  # pad > 0.6 keeps some space

    # ─── Plot mean activity ──────────────────────────────────────────────
    cluster_colors = ["#ACACAD"]
    bar_colors = [cluster_colors[row_clusters[i]] for i in cluster_order]
    ax_bar.barh(range(len(mean_activity)), mean_activity, color=bar_colors)

    # ─── Collect & sort examples ──────────────────────────────────────────
    examples = sorted(
        [(reordered_indices[i], i, n)                       # (row-idx, InChI, name)
         for i, n in example_inchi2name.items()
         if i in reordered_indices],
        key=lambda t: t[0]
    )
    base_rows = np.array([p for p, _, _ in examples], dtype=float)

    # ─── Resolve collisions iteratively ──────────────────────────────────
    min_sep = 33.0                      # desired gap in row units
    shifts  = np.zeros_like(base_rows) # incremental y-offsets
    max_iter = 600
    for iter in range(max_iter):
        moved = False
        # walk down sorted list and push pairs that overlap
        for j in range(1, len(base_rows)):
            y_prev = base_rows[j-1] + shifts[j-1]
            y_curr = base_rows[j]   + shifts[j]
            gap = y_curr - y_prev
            if gap < min_sep:
                delta = 0.5*(min_sep - gap)
                shifts[j-1] -= delta   # push up
                shifts[j]   += delta   # push down
                moved = True
        if not moved:
            print(f"Resolved all overlaps in {iter+1} iterations")
            break   # no overlaps; done
    else:
        warnings.warn("label-spreading hit max_iter without fully resolving overlaps")

    # 3) render annotations

    # fraction of the longest bar use for horizontal text offset
    if z_scale:
        bar_tip_fraction = 0.25
    else:
        bar_tip_fraction = 0.10  

    x_offset = mean_activity.max()*bar_tip_fraction
    for (pos, inchi, name), y_shift in zip(examples, shifts):
        x = mean_activity.iloc[pos]
        ax_bar.annotate(
            name,
            xy=(x, pos),                       # arrow starts at bar tip
            xytext=(x + x_offset, pos + y_shift),
            ha='left', va='center',
            fontsize=11, color='black',
            arrowprops=dict(arrowstyle='-', lw=0.6),
            clip_on=False
        )


    # ─── Tidy up axis ────────────────────────────────────────────────────
    ax_bar.set_ylim(g.ax_heatmap.get_ylim())
    # ax_bar.set_xlabel('Mean Activity', fontsize=20)
    ax_bar.set_xlabel('MAV', fontsize=20)
    ax_bar.set_yticks([])

    # ─── Move the y-axis label (“Diester Phthalates”) to the left side ───────────
    g.ax_heatmap.yaxis.set_label_position('left')
    g.ax_heatmap.set_ylabel(
        'Diester Phthalates',
        color='black', fontsize=20,
    )
    g.ax_heatmap.yaxis.tick_left()
    # bump the left margin so the label isn’t cut off
    g.figure.subplots_adjust(
        left=0.03,
        right=0.9,
        top=1.0,
        bottom=0.06
    )    

    g.figure.savefig(cachedir / "phthalate_activity_heatmap.png", dpi=600)
    plt.close(g.figure)

    assay_activity_counts = phthalate_df.reset_index().groupby(['inchi','title'])['positive_prediction'].mean().reset_index()
    assay_activity = assay_activity_counts.groupby('title')['positive_prediction'].mean().reset_index()
    assay_activity.sort_values('positive_prediction', ascending=True)
    assay_activity.to_csv(cachedir / 'assay_activity.csv', index=False)

    # Create a mapping of cluster number to color
    cluster_color_map = {i: color for i, color in enumerate(cluster_colors)}

    # Add both cluster number and color to the dataframe
    inchi_clusters = []
    for inchi in reordered_matrix.index:
        # cluster = row_clusters[cluster_order[reordered_indices[inchi]]]
        pos = int(cluster_order[reordered_indices[inchi]])
        cluster = int(row_clusters[pos])
        inchi_clusters.append((inchi, cluster, cluster_color_map[cluster]))

    inchi_cluster_df = pd.DataFrame(inchi_clusters, columns=['inchi', 'cluster', 'cluster_color'])
    clustered_phthalate_df = phthalate_df.merge(inchi_cluster_df, on='inchi')

    clustered_phthalate_df = phthalate_df.merge(inchi_cluster_df, on='inchi')
    # add either 'None' or name of example in the 'example' column
    clustered_phthalate_df['example'] = clustered_phthalate_df['inchi'].progress_apply(lambda x: example_inchi2name.get(x, 'None'))
    return clustered_phthalate_df

clustered_phthalate_df = cluster_rows_and_make_heatmap(
    z_scale=True,
    group_clusters=False,
)[
    ['uri','title','inchi','mol','positive_prediction','cluster','cluster_color','example']
]

# save dataframes
phthalate_df.to_csv(cachedir / 'phthalate_df.csv', index=False)
clustered_phthalate_df.to_csv(cachedir / 'clustered_phthalate_df.csv', index=False)


# region CHARACTERIZE PRIORITY PHTHALATES ===============================================================
def plot_phthalate_activity_relationships(*, example_plot='legend'):
    df4 = clustered_phthalate_df.groupby(['inchi'])['positive_prediction'].mean().reset_index()
    df5 = clustered_phthalate_df[['inchi', 'mol', 'cluster','cluster_color', 'example']].drop_duplicates()
    df6 = df4.merge(df5, on='inchi')
    df6 = df6[['inchi', 'mol', 'positive_prediction', 'example', 'cluster', 'cluster_color']]

    # create a logistic regression model to predict activity from the metrics
    from scripts.utils.helpers import remove_high_vif_descriptors
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from adjustText import adjust_text  # auto-spread labels to avoid overlap

    try:
        X = pd.read_parquet('cache/descriptors/descriptors.parquet')
        df6[['MolWt', 'cLogP', 'RotB', 'BranchingRatio']] = X.loc[df6['inchi'], ['MolWt', 'cLogP', 'RotB', 'BranchingRatio']].values

        X = remove_high_vif_descriptors(X, vif_threshold=10)
        print(f"Loaded descriptors from cache: {X.shape[0]} molecules, {X.shape[1]} features")
    except FileNotFoundError:
        # Extract feature dicts for each molecule
        feats_list = []
        unique_inchis = df6['inchi'].unique()
        for inchi in unique_inchis:
            mol = df6.loc[df6['inchi'] == inchi, 'mol'].values[0]
            mol_H = Chem.AddHs(mol)  # Add hydrogens to the molecule
            feats = get_descriptors(mol_H)
            # limit the keys to the ones we want to use
            feats = {k: v for k, v in feats.items() if k in [
                'MolWt', 'cLogP', 'RotB', 'BranchingRatio'
            ]}
            feats_list.append(feats)

        # Create a DataFrame from the list of dicts
        features_df = pd.DataFrame(feats_list, index=unique_inchis)

        # Merge features_df with df6 on 'inchi'
        df6 = df6.merge(features_df, left_on='inchi', right_index=True, how='left')

        X = df6[['MolWt', 'cLogP', 'RotB', 'BranchingRatio']]

    # y = df6['positive_prediction']
    activity_matrix_filled = pd.read_parquet(cachedir / 'activity_matrix_filled.parquet')
    Z_activity = (activity_matrix_filled - activity_matrix_filled.mean(axis=0)) / activity_matrix_filled.std(axis=0)
    y = Z_activity.mean(axis=1).reindex(df6['inchi'])  # Z-scored mean activity value
    df6['positive_prediction'] = y.to_list()  # add the Z-scored mean activity to df6
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 between predicted and true activity: {r2:.3f}")
    # Print feature names and their corresponding weights
    print("\nModel coefficients:")
    for feature, weight in zip(['Molecular Weight', 'cLogP', 'Number of Rotatable Bonds', 'Branching Ratio'], model.coef_):
        print(f"{feature:25} {weight:>8.3f}")
    print(f"{'Intercept':25} {model.intercept_:>8.3f}")
    
    df6['activity_pred'] = model.predict(X)

    correlations = {}
    metrics = {
        'MolWt' : ('Molecular Weight', 'Molecular Weight (Da)'),
        'cLogP': ('cLogP', 'cLogP'),
        'RotB' : ('Number of Rotatable Bonds', 'Number of Rotatable Bonds'),
        'BranchingRatio' : ('Branching Ratio', 'Branching Ratio'),
        'activity_pred' : ('Predicted Activity', 'Predicted Activity'),
    }

    # Create bins for each metric
    for metric in metrics:
        df6[f'{metric}_bin'] = pd.qcut(df6[metric], q=5)

    # Calculate correlations
    for metric, (label, axis_label) in metrics.items():
        corr = df6[metric].corr(df6['positive_prediction'])
        correlations[metric] = (corr, label, axis_label, f'{metric}_bin')

    # Sort metrics by absolute correlation strength
    sorted_metrics = sorted(correlations.items(), key=lambda x: x[1][0], reverse=True)
    for metric, (corr, label, axis_label, bin_col) in sorted_metrics:
        print(f"{label}\t{corr:.3f}")

    if example_plot == 'legend':

        example_names = example_phthalates_df['name'].tolist()
        # # Distinct marker styles
        marker_styles = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "p", "8"]

        # Cycle through colors from seaborn or matplotlib
        palette = sns.color_palette(
            "Set2",
            n_colors=len(example_names)
        )

        example_markers = {
            name: (marker_styles[i % len(marker_styles)], palette[i % len(palette)])
            for i, name in enumerate(example_names)
        }


    cluster_colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blue, Red, Green
    plotdf = df6.copy()
    plotdf.query('example != "None"')[['cluster','cluster_color']]

    # Create individual high-resolution plots for each metric
    for idx, (metric, (corr, label, axis_label, bin_col)) in enumerate(sorted_metrics):
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(24, 6), dpi=600)  # Same width but height/4

        # Create scatter plot colored by cluster
        sns.scatterplot(
            x=metric, y='positive_prediction', data=plotdf[plotdf['example'] == 'None'],
            hue=None,
            palette=None,
            # alpha=0.7,
            alpha=0.3,
            ax=ax, s=150,
        )
        
        # Fit and draw an overall least-squares line for this metric
        # Uses all available points (including examples) for stability.
        valid = plotdf[[metric, 'positive_prediction']].dropna()
        if len(valid) >= 2:
            lr_X = valid[[metric]].values
            lr_y = valid['positive_prediction'].values
            lr = LinearRegression().fit(lr_X, lr_y)

            # Line spans observed x-range to avoid extrapolation artifacts.
            x_min = float(np.nanmin(valid[metric]))
            x_max = float(np.nanmax(valid[metric]))
            x_vals = np.linspace(x_min, x_max, 200).reshape(-1, 1)
            y_vals = lr.predict(x_vals)

            # Use a neutral line without adding a legend entry.
            ax.plot(x_vals.ravel(), y_vals, linewidth=2.5, alpha=0.9, color='black', linestyle='--', label='_nolegend_')

            

        if example_plot == 'legend':
            # Plot example phthalates with distinct symbols
            example_df = plotdf[plotdf['example'] != 'None'].sort_values(by='example')
            for _, row in example_df.iterrows():
                name = row['example']
                marker, color = example_markers.get(name, ("o", "black"))
                ax.scatter(
                    row[metric],
                    row['positive_prediction'],
                    marker=marker,
                    s=300,
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                    label=name
                )

            # Deduplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                title="Examples",
                fontsize=14,
                title_fontsize=16,
                loc="best"
            )

        elif example_plot == 'labels':
            # Plot example phthalates
            example_df = plotdf[plotdf['example'] != 'None']
            sns.scatterplot(x=metric, y='positive_prediction', data=example_df,
                        color=example_df['cluster_color'], alpha=1.0, ax=ax,
                        marker='s', s=300, legend=False)  # Increased marker size
        
            texts = []
            ex = plotdf[plotdf['example'] != 'None']
            for _, row in ex.iterrows():
                # start each label at the point; adjust_text will move it
                t = ax.text(
                    row[metric],
                    row['positive_prediction'],
                    row['example'],
                    fontsize=16,
                    alpha=0.9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                )
                texts.append(t)

            # Repel labels from each other and from background points; draw subtle leader lines
            bg = plotdf[plotdf['example'] == 'None']
            adjust_text(
                texts,
                x=bg[metric].to_numpy(),
                y=bg['positive_prediction'].to_numpy(),
                ax=ax,
                lim=400,
                expand=(1.20, 1.35),
                expand_points=(1.25, 1.45),
                force_text=0.6,
                force_points=0.4,
                only_move={'points': 'y', 'texts': 'xy'},
                arrowprops=None                      # turn off leader lines
            )

            # Give labels a bit more room near plot edges
            ax.margins(x=0.05, y=0.12)

        
        # Customize labels and title with larger fonts
        ax.set_xlabel(axis_label, fontsize=20, fontweight='bold')
        ax.set_ylabel('MAV', fontsize=20, fontweight='bold')
        ax.set_title(
            # f'{label} vs ICE Activity (Pearson r = {corr:.3f})',
            f'MAV vs. {label} (Pearson r = {corr:.3f})',
            fontsize=24, pad=20
        )
        
        # Enhance grid lines
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
            
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()
        # Save each plot separately with high resolution
        plt.savefig(cachedir / f"ice_activity_relationships_{metric}.png", 
                   dpi=1200, bbox_inches='tight',
                   facecolor='white')
        plt.close()

    # Extract molecular weight correlation info for caption
    mw_metric = next((m for m in sorted_metrics if 'MolWt' in m[0]), None)
    mw_corr = mw_metric[1][0] if mw_metric else None

    caption = f"""The point plots show relationships between molecular properties and mean ICE Assay Activity. 
    Molecular weight shows a correlation of {mw_corr:.2f}. Each point represents a phthalate, colored by its structural cluster. 
    Other notable correlations include: """

    other_metrics = [m for m in sorted_metrics if 'mw' not in m[0]]
    corr_descriptions = [f"{m[1][1]}: r={m[1][0]:.2f}" for m in other_metrics]
    caption += ", ".join(corr_descriptions) + "."
    print(caption)

