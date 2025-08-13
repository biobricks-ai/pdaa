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

import sys
sys.path.append('./')
import stages.utils.pdaa as pdaa
import stages.utils.sparql as sparql

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

    # g.ax_heatmap.set_xlabel('DART or ED Assays', color=fontcolor, fontsize=20)
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
cachedir = pathlib.Path('cache') / 'entity_similarity2'
cachedir.mkdir(parents=True, exist_ok=True)

# region EXAMPLE PHTHALATES ====================================================================
example_phthalates_df = pd.read_csv(resourcedir / 'example_phthalates.csv')
# shorten the names
example_phthalates_df['name'] = example_phthalates_df['name'].str.replace('Dimethyl ', '')

example_phthalates = [Chem.MolFromSmiles(smiles) for smiles in example_phthalates_df['smiles']]
example_names = example_phthalates_df['name'].tolist()
example_inchi = [Chem.MolToInchi(m) for m in example_phthalates]
example_weights = [rdkit.Chem.rdMolDescriptors.CalcExactMolWt(m) for m in example_phthalates]
example_inchi2name = {inchi: name for inchi, name in zip(example_inchi, example_names)}

# print the alias weights
for name, weight in zip(example_names, example_weights):
    print(f"{name}: {weight:.2f}")

# # run the model on the example phthalates (nly needs to be done once)
# pdaa.predict_all_properties_with_sqlite_cache(example_inchi)

# possible isomers to look for
isomers_list = ["ortho_phthalate", "meta_phthalate", "para_phthalate"]

# which phthalate has the lowest mean ICE activity?
# region ICE ACTIVITY ===============================================================
def build_phthalate_ice_activity_df(mask_method='prediction', use_dart = True, use_ed = True):
    print("Building phthalate ICE activity dataframe...")
    
    print("Querying PDAA graph for URI, title, and token mappings...")
    uri_title_token = sparql.Query(pdaa.pdaa_graph) \
        .select_typed({'uri': str, 'pp': str, 'title': str, 'token': int}) \
        .where('?pp a toxindex:predicted_property') \
        .where('?pp <http://purl.org/dc/elements/1.1/title> ?title') \
        .where('?pp rdf:value ?token') \
        .where('?pp <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
        .execute().groupby('uri').first().reset_index()

    if mask_method == 'list':
        dart_path = pathlib.Path(resourcedir / 'DART_endpoints.txt')

        with open(dart_path) as f:
            # strip whitespace, drop empties, remove punctuation, lowercase
            dart_clean = [
                re.sub(r'[^A-Za-z0-9]', '', line).lower()
                for line in f
                if line.strip()
            ]

        uri_title_token['clean_title'] = (
            uri_title_token['title']
            .str.strip()
            .str.lower()
            .apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s))
        )

        # build a boolean mask: True if any dart_clean entry is a substring
        mask = uri_title_token['clean_title'].apply(
            lambda ct: any(d in ct for d in dart_clean)
        )
    elif mask_method == 'prediction':

        

        # ----------------------------------------------------------------------
        # 1. Collect every file we should read this run
        flag_files = []
        if use_dart:
            flag_files.append(resourcedir / "assay_flags2_dart.txt")
        if use_ed:
            flag_files.append(resourcedir / "assay_flags2_ed.txt")

        # Nothing selected?  Short-circuit early.
        if not flag_files:
            mask = np.zeros(len(uri_title_token), dtype=bool)
        else:
            # ------------------------------------------------------------------
            # 2. Load, filter, and concatenate
            dfs = [
                pd.read_csv(p, sep="\t", header=None,
                            names=["title", "flag"])
                .loc[lambda d: d["flag"].astype(bool)]      # keep only positives
                .assign(title=lambda d: d["title"].str.lower().str.strip())
                for p in flag_files
            ]
            
            combined_titles = (
                pd.concat(dfs, ignore_index=True)
                .drop_duplicates("title")                   # OR-semantics (“any” match)
                ["title"]
                .tolist()
            )
            title_set = set(combined_titles)

            # ------------------------------------------------------------------
            # 3. Vectorised membership check on the current DataFrame
            mask = uri_title_token["title"].str.lower().str.strip().isin(title_set)

        print(mask.sum(), "DART/ED assays found")

    elif mask_method == 'categories':
        from scripts.utils.helpers import clean_title
        df = pd.read_csv(resourcedir / 'assay_strength.csv')

        mask = uri_title_token['title'].apply(clean_title).isin(df['title'].tolist())

    print(f"Using mask to filter {len(uri_title_token)} assays to {mask.sum()} relevant assays")
    ice_assays = uri_title_token[mask]
    ice_assays.to_csv(resourcedir / 'ice_assays.csv', index=False)
    # ice_assays = uri_title_token
    print(f"Found {len(ice_assays)} DART-filtered ICE assays")


    print("Fetching predictions from SQLite...")
    with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
        tokens = ','.join(map(str, ice_assays['token'].tolist()))
        query = f'SELECT * FROM predictions WHERE property_token IN ({tokens})'
        ice_preds = pd.read_sql(query, conn)

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

    inchi_mol_df = df[['inchi']].drop_duplicates()
    print("Converting InChIs to molecules...")
    inchi_mol_df['mol'] = inchi_mol_df['inchi'].progress_apply(lambda x: Chem.MolFromInchi(x))
    df2 = df.merge(inchi_mol_df, on='inchi')

    try:
        assert all(pdaa.is_diester_phthalate(m) for m in example_phthalates)
    except AssertionError as e:
        print("Some example phthalates are not diester phthalates. Please check the SMARTS pattern.")
        raise e

    print("Filtering for diester phthalates...")
    filtered_phthalates = inchi_mol_df[inchi_mol_df['mol'].progress_apply(pdaa.is_diester_phthalate)]['inchi']
    df3 = df2[df2['inchi'].isin(filtered_phthalates)]
    # Ensure both columns are of the same type (int)
    df3.loc[:, 'property_token'] = df3['property_token'].astype(int)
    ice_assays.loc[:, 'token'] = ice_assays['token'].astype(int)
    df3 = df3.merge(ice_assays, left_on='property_token', right_on='token')[['uri','title','inchi','mol','positive_prediction']]
    print(f"Final dataset contains {len(df3)} rows")
    return df3

phthalate_df = build_phthalate_ice_activity_df(
    mask_method='categories',
)[['uri','title','inchi','mol','positive_prediction']]

# endregion

# region HEATMAP & DENSITY OF PHTHALATE ACTIVITY ===============================
def cluster_rows_and_make_heatmap(
    z_scale=False,
    group_clusters=True,
    color_by = 'isomer',
    PCA_components=50,
    weighted_mean=False,
    PCA_keep_upto=1.0,
    active_top=True,
):
    activity_matrix = phthalate_df.groupby(['inchi','title'])['positive_prediction'].mean().reset_index()
    activity_matrix = activity_matrix.pivot(index='inchi', columns='title', values='positive_prediction')

    inchi_activity_counts = phthalate_df.reset_index().groupby(['inchi','title'])['positive_prediction'].mean().reset_index()
    inchi_activity_counts['active'] = inchi_activity_counts['positive_prediction'] > 0.6
    inchi_activity = inchi_activity_counts.groupby('inchi')['active'].mean().reset_index()
    inchi_activity.sort_values('active', ascending=True)
    inchi_activity.to_csv(cachedir / 'inchi_activity.csv', index=False)

    # Cluster the data using KMeans instead of hierarchical clustering
    from sklearn.cluster import KMeans
    from scipy.cluster import hierarchy

    # Fill any NaN values with 0 for clustering
    activity_matrix_filled = activity_matrix.fillna(0)
    # # flip the matrix so that higher activity is at the top
    # activity_matrix_filled = activity_matrix_filled.iloc[::-1]
    # save the filled activitiy matrix
    activity_matrix_filled.to_parquet(cachedir / 'activity_matrix_filled.parquet')

    if PCA_components is not None:
        from sklearn.decomposition import PCA
        # use the PCA matrix instead of the original activity matrix
        pca = PCA(n_components=PCA_components)
        if PCA_keep_upto < 1.0:
            # keep only the components that explain at least PCA_keep_upto of the variance
            pca.fit(activity_matrix_filled)
            explained_variance = np.cumsum(pca.explained_variance_ratio_)
            # use np.argmax to find the first index where explained_variance >= PCA_keep_upto
            PCA_components = np.argmax(explained_variance >= PCA_keep_upto) + 1
            print(f"Keeping {PCA_components} PCA components to explain at least {PCA_keep_upto*100:.1f}% of the variance")
            pca = PCA(n_components=PCA_components)
        activity_matrix_filled = pd.DataFrame(
            pca.fit_transform(activity_matrix_filled),
            index=activity_matrix_filled.index,
            columns=[f'PC{i+1}' for i in range(PCA_components)]
        )

    if z_scale:
        # Z-score normalization
        activity_matrix_filled = (activity_matrix_filled - activity_matrix_filled.mean(axis=0)) / activity_matrix_filled.std(axis=0)

    # Apply KMeans clustering
    # n_clusters = 3
    if color_by is None:
        # No color coding
        n_clusters = 1
        row_clusters = np.zeros(len(activity_matrix_filled), dtype=int)
        row_colors = None
    elif color_by == 'isomer':
        # ignore clustering for now
        n_clusters = len(isomers_list)
        row_clusters = np.zeros(len(activity_matrix_filled), dtype=int)
    elif color_by == 'cluster':
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        row_clusters = kmeans.fit_predict(activity_matrix_filled)

    # Still use hierarchical clustering for column ordering
    col_linkage = hierarchy.linkage(activity_matrix_filled.T, method='average')
    col_order = hierarchy.leaves_list(col_linkage)
    # row_linkage = hierarchy.linkage(activity_matrix_filled, method='average')
    # cluster_order = hierarchy.leaves_list(row_linkage)

    if (PCA_components is not None) and weighted_mean:
        # quick plot of pca explained variance ratio for debugging
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(pca.explained_variance_ratio_)), np.cumsum(pca.explained_variance_ratio_))
        plt.ylim(0, 1)
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.show()
        sys.exit(0)  # while debugging
        # weighted mean based on the percent variance explained by each PCA component
        mean_activities = (activity_matrix_filled * pca.explained_variance_ratio_).sum(axis=1)/ pca.explained_variance_ratio_.sum()
    else:
        # simple mean
        mean_activities = activity_matrix_filled.mean(axis=1)

    if group_clusters:
        # Sort by cluster first, then by mean activity
        cluster_order = []
        for i in range(n_clusters):
            cluster_indices = np.where(row_clusters == i)[0]
            sorted_indices = cluster_indices[np.argsort(mean_activities.iloc[cluster_indices])]
            if active_top:
                # Reverse the order so that higher activity is at the top
                sorted_indices = sorted_indices[::-1]
            cluster_order.extend(sorted_indices)
    else:
        # Sort by mean activity only
        cluster_order = np.argsort(mean_activities)
        if active_top:
            # Reverse the order so that higher activity is at the top
            cluster_order = cluster_order[::-1]

    # Reorder the matrix
    reordered_matrix = activity_matrix_filled.iloc[cluster_order, col_order]
    # reordered_matrix = activity_matrix_filled.iloc[cluster_order[::-1], col_order]
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
    # row_colors = [cluster_colors[row_clusters[i]] for i in cluster_order]
    # Instead, color based on ortho, iso, or tere phthalate
    
    if color_by == 'isomer':
        isomer_matches = np.array([0 for _ in isomers_list])
        row_colors = []
        # n_non_ortho = 0
        for i, inchi in enumerate(reordered_matrix.index):
            if (mol := Chem.MolFromInchi(inchi)) is None:
                continue  # skip invalid InChIs
            match_list = [pdaa.is_phthalate(mol, modes=(isomer,)) for isomer in isomers_list]
            isomer_matches += match_list
            row_clusters[i] = np.argmax(match_list)  # assign the cluster based on the first match
            row_colors.append(
                isomer_colors[tuple(match_list)]
            )

            # for i in range(len(isomers_list)):
            #     if pdaa.is_phthalate(mol, modes=(isomers_list[i],)):
            #         row_colors.append(isomer_colors[i])
            #         if i > 0:
            #             n_non_ortho += 1
            #             # print(f"non-ortho phthalate: {Chem.MolToSmiles(mol)} ({isomers_list[i]})")
            #             # # save image of the molecule
            #             # img = Draw.MolToImage(mol, size=(300, 300))
            #             # img.save("mol.png")
            #             # raise ValueError("Testing: non-ortho phthalate detected")
            #         break
        # keep the single colour-bar that _styled_heatmap makes
        # print(f"Number of non-ortho phthalates: {n_non_ortho}")
        print("isomer matches found:")
        print(isomer_matches)
    elif color_by == 'cluster':
        # Use the cluster colors instead
        row_colors = [base_colors[row_clusters[i]] for i in cluster_order]

    # g = sns.clustermap(
    #     reordered_matrix,
    #     # square=True,       # ← force equal-sized cells
    #     cbar_kws={'drawedges': False},  # disable seaborn’s built-in bar
    #     cmap=diverging_colormap if z_scale else 'viridis',
    #     row_cluster=True, col_cluster=False,
    #     row_colors=row_colors,
    #     xticklabels=False, yticklabels=False,
    #     figsize=(18, 9),
    #     cbar_pos=(0.95, 0.3, 0.02, 0.4),
    #     dendrogram_ratio=(0.10, 0.05),
    #     tree_kws={'linewidths': 0.5},
    #     vmin=-3, vmax=+3,
    # )
    # plt.show()
    # return
    xlabel = 'Principal Components' if PCA_components else 'DART or ED Assays'
    g = _styled_heatmap(reordered_matrix, row_colors, fontcolor='black', z_scale=z_scale, xlabel=xlabel)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # ─── Attach a fresh bar axis ──────────────────────────────────────────
    # divider = make_axes_locatable(g.ax_heatmap)
    # ax_bar = divider.append_axes("right", size="15%", pad=0.8)
    divider = g.divider            # reuse the one that already holds the colour-bar
    ax_bar  = divider.append_axes("right", size="15%", pad=1.0)  # pad > 0.6 keeps some space

    # ─── Plot mean activity ──────────────────────────────────────────────
    if color_by is None:
        # No color coding, use a single color
        cluster_colors = ["#ACACAD"]
    elif color_by == 'isomer':
        # cluster_colors = ['#1f77b4', '#d62728', '#2ca02c']  # Darker Blue, Darker Red, Darker Green
        cluster_colors = base_colors  # Darker Blue, Darker Red, Darker Green
    elif color_by == 'cluster':
        # Use the same colors as the clusters
        cluster_colors = base_colors
    bar_colors = [cluster_colors[row_clusters[i]] for i in cluster_order]
    ax_bar.barh(range(len(mean_activity)), mean_activity, color=bar_colors)
    # ax_bar.barh(range(len(mean_activity)), mean_activity[::-1], color=bar_colors[::-1])

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
    max_iter = 300
    for _ in range(max_iter):
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
    ax_bar.set_xlabel('Mean Activity', fontsize=20)
    ax_bar.set_yticks([])

    # ─── Cluster legend ─────────────────────────────────────────────────
    # Use three related but darker colors for cluster identification (to distinguish from isomer_colors)
    isomer_labels = ['Ortho', 'Iso', 'Tere']
    if color_by == 'isomer':
        legend_elements = [
            plt.Rectangle(
                (0,0),1,1,
                facecolor=cluster_colors[i],
                label=f'{isomer_labels[i]}\n(n={isomer_matches[i]})')
            for i in range(n_clusters)
        ]
        ax_bar.legend(
            handles=legend_elements,
            loc='upper left',
            # loc='center left',
            bbox_to_anchor=(1.05, 1.0),
            borderaxespad=0.0,
            title='Isomers',
            fontsize=12
        )
    elif color_by == 'cluster':
        legend_elements = [
            plt.Rectangle(
                (0,0),1,1,
                facecolor=cluster_colors[i],
                label=f'Cluster {i+1}\n(n={np.sum(row_clusters==i)})')
            for i in range(n_clusters)
        ]
        ax_bar.legend(
            handles=legend_elements,
            loc='upper left',
            # loc='center left',
            bbox_to_anchor=(1.05, 1.0),
            borderaxespad=0.0,
            title='Clusters',
            fontsize=12
        )

    # ─── Move the y-axis label (“Diester Phthalates”) to the left side ───────────
    g.ax_heatmap.yaxis.set_label_position('left')
    g.ax_heatmap.set_ylabel(
        'Diester Phthalates',
        # 'Ortho-Phthalates',
        # 'Terephthalates',
        # 'Isophthalates',
        color='black', fontsize=20,
        labelpad=35*(color_by is not None)
    )
    g.ax_heatmap.yaxis.tick_left()
    # bump the left margin so the label isn’t cut off
    g.figure.subplots_adjust(
        left=0.03,
        right=0.9,
        top=1.0,
        bottom=0.06
    )    
    # g.figure.subplots_adjust(
    #     left=0.02,   # plenty of space for row colors / dendrogram
    #     right=0.9,  # leave room for bar & legend
    #     top=0.98,
    #     bottom=0.06
    # )
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
        cluster = row_clusters[cluster_order[reordered_indices[inchi]]]
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
    color_by=None,  # 'isomer', 'cluster', or None
    # PCA_components=200,  # None to skip PCA
    PCA_components=None,  # None to skip PCA
    weighted_mean=False,  # use weighted mean based on PCA components
    # weighted_mean=True,  # use weighted mean based on PCA components
    PCA_keep_upto=0.95,  # keep enough components to explain at least 95% of the variance
)[
    ['uri','title','inchi','mol','positive_prediction','cluster','cluster_color','example']
]

sys.exit(0)  # stop here while iterating

# save dataframes
phthalate_df.to_csv(cachedir / 'phthalate_df.csv', index=False)
clustered_phthalate_df.to_csv(cachedir / 'clustered_phthalate_df.csv', index=False)
# endregion

# region CHARACTERIZE PRIORITY PHTHALATES ===============================================================
def plot_phthalate_activity_relationships():
    df4 = clustered_phthalate_df.groupby(['inchi'])['positive_prediction'].mean().reset_index()
    df5 = clustered_phthalate_df[['inchi', 'mol', 'cluster','cluster_color', 'example']].drop_duplicates()
    df6 = df4.merge(df5, on='inchi')
    df6 = df6[['inchi', 'mol', 'positive_prediction', 'example', 'cluster', 'cluster_color']]

    df6['mw'] = df6['mol'].progress_apply(lambda m: rdkit.Chem.rdMolDescriptors.CalcExactMolWt(m))
    df6['logp'] = df6['mol'].progress_apply(lambda m: Chem.Crippen.MolLogP(m))
    df6['num_rotatable_bonds'] = df6['mol'].progress_apply(lambda m: rdkit.Chem.rdMolDescriptors.CalcNumRotatableBonds(m))
    df6['branching_ratio'] = df6['mol'].progress_apply(lambda m: rdkit.Chem.rdMolDescriptors.CalcFractionCSP3(m))

    # create a logistic regression model to predict activity from the metrics
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X = df6[['mw', 'logp', 'num_rotatable_bonds', 'branching_ratio']]
    y = df6['positive_prediction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 between predicted and true activity: {r2:.3f}")
    # Print feature names and their corresponding weights
    print("\nModel coefficients:")
    for feature, weight in zip(['Molecular Weight', 'LogP', 'Number of Rotatable Bonds', 'Branching Ratio'], model.coef_):
        print(f"{feature:25} {weight:>8.3f}")
    print(f"{'Intercept':25} {model.intercept_:>8.3f}")
    
    df6['activity_pred'] = model.predict(df6[['mw', 'logp', 'num_rotatable_bonds', 'branching_ratio']])

    correlations = {}
    metrics = {
        'logp': ('LogP', 'LogP'),
        'num_rotatable_bonds' : ('Number of Rotatable Bonds', 'Number of Rotatable Bonds'),
        'mw' : ('Molecular Weight', 'Molecular Weight (Da)'),
        'branching_ratio' : ('Branching Ratio', 'Branching Ratio'),
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

    cluster_colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blue, Red, Green
    plotdf = df6.copy()
    plotdf.query('example != "None"')[['cluster','cluster_color']]

    # Create individual high-resolution plots for each metric
    for idx, (metric, (corr, label, axis_label, bin_col)) in enumerate(sorted_metrics):
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(24, 6), dpi=600)  # Same width but height/4

        # Create scatter plot colored by cluster
        sns.scatterplot(x=metric, y='positive_prediction', data=plotdf[plotdf['example'] == 'None'],
                    hue='cluster', palette=cluster_colors, alpha=0.7, ax=ax, s=150)  # Increased point size
        
        # Plot example phthalates
        example_df = plotdf[plotdf['example'] != 'None']
        sns.scatterplot(x=metric, y='positive_prediction', data=example_df,
                    color=example_df['cluster_color'], alpha=1.0, ax=ax,
                    marker='s', s=300, legend=False)  # Increased marker size
        
        # Add text annotations for example phthalates
        for _, row in plotdf[plotdf['example'] != 'None'].iterrows():
            ax.annotate(row['example'], 
                    xy=(row[metric], row['positive_prediction']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=16, alpha=0.8,  # Increased font size
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Customize labels and title with larger fonts
        ax.set_xlabel(axis_label, fontsize=20, fontweight='bold')
        ax.set_ylabel('Mean ICE Assay Activity', fontsize=20, fontweight='bold')
        ax.set_title(f'{label} vs ICE Activity (Pearson r = {corr:.3f})',
                    fontsize=24, pad=20)
        
        # Enhance grid lines
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Update legend with larger font
        legend = ax.legend(title='Cluster', labels=[f'Cluster {i+1}' for i in range(3)])
        legend.get_title().set_fontsize(18)
        for t in legend.get_texts():
            t.set_fontsize(16)
            
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()
        # Save each plot separately with high resolution
        plt.savefig(cachedir / f"ice_activity_relationships_{metric}.png", 
                   dpi=1200, bbox_inches='tight',
                   facecolor='white')
        plt.close()

    # Extract molecular weight correlation info for caption
    mw_metric = next((m for m in sorted_metrics if 'mw' in m[0]), None)
    mw_corr = mw_metric[1][0] if mw_metric else None

    caption = f"""The point plots show relationships between molecular properties and mean ICE Assay Activity. 
    Molecular weight shows a correlation of {mw_corr:.2f}. Each point represents a phthalate, colored by its structural cluster. 
    Other notable correlations include: """

    other_metrics = [m for m in sorted_metrics if 'mw' not in m[0]]
    corr_descriptions = [f"{m[1][1]}: r={m[1][0]:.2f}" for m in other_metrics]
    caption += ", ".join(corr_descriptions) + "."
    print(caption)

plot_phthalate_activity_relationships()
# endregion


# region SAMPLE PHTHALATES IN EACH ACTIVITY PERCENTILE ===============================================================
from PIL import Image, ImageDraw, ImageFont
def mkimage():
    # Get mean activity per compound and cluster
    df4 = clustered_phthalate_df.groupby(['inchi','cluster','cluster_color'])['positive_prediction'].mean().reset_index()
    df4['mol'] = df4['inchi'].progress_apply(lambda x: Chem.MolFromInchi(x))
    
    # Add flag for example compounds
    df4['is_example'] = df4['inchi'].isin(example_inchi)
    df4['name'] = df4['inchi'].map(example_inchi2name)
    
    # Get samples per cluster
    samples_list = []
    # for cluster in [2,0,1]:
    # order clusters by mean activity programmatically
    cluster_means = df4.groupby('cluster')['positive_prediction'].mean().sort_values()
    cluster_order = cluster_means.index.tolist()
    for cluster in cluster_order:
        cluster_df = df4[df4['cluster'] == cluster]
        
        # Get 5 compounds nearest to median activity
        non_examples = cluster_df[~cluster_df['is_example']]
        median_activity = non_examples['positive_prediction'].median()
        nearest_to_median = non_examples.iloc[(non_examples['positive_prediction'] - median_activity).abs().argsort()[:6]]
        
        samples_list.append(nearest_to_median)
    
    # Add all example compounds at the end
    examples = df4[df4['is_example']]
    samples = pd.concat(samples_list + [examples])
    
    mols = samples['mol'].tolist()
    # legends = [f"{row['name'] if row['is_example'] else ''} (Cluster {row['cluster'] + 1}, Activity: {row['positive_prediction']:.3f})" 
    legends = [f"{row['name'] if row['is_example'] else ''} (Cluster {row['cluster'] + 1}, MAV: {row['positive_prediction']:.3f})" 
              for _, row in samples.iterrows()]

    # Create individual images with colored backgrounds
    mol_imgs = []
    for i, (_, row) in enumerate(samples.iterrows()):
        # Draw molecule
        img = Draw.MolToImage(row['mol'], size=(400,400))
        
        # Convert to RGBA if not already
        img = img.convert('RGBA')
        
        # Create colored overlay
        color = df4[df4['cluster'] == row['cluster']]['cluster_color'].iloc[0]
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        overlay = Image.new('RGBA', img.size, (r,g,b,51))  # alpha=51 is 20% opacity
        
        # Composite the images
        img = Image.alpha_composite(img, overlay)
        mol_imgs.append(img)

    # Create grid layout
    n_rows = (len(mol_imgs) + 5) // 6  # Ceiling division for 6 mols per row
    grid_img = Image.new('RGBA', (400*6, 400*n_rows), (255,255,255,0))
    
    # Paste images into grid
    for idx, img in enumerate(mol_imgs):
        row = idx // 6
        col = idx % 6
        grid_img.paste(img, (col*400, row*400))

    # Add legends
    draw = ImageDraw.Draw(grid_img)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Replace with a valid font path
    font_size = 20  # Adjust font size as needed
    font = ImageFont.truetype(font_path, font_size)
    
    for idx, legend in enumerate(legends):
        row = idx // 6
        col = idx % 6
        draw.text((col*400 + 10, row*400 + 378), legend, font=font, fill=(0,0,0,255))

    # Save the image
    grid_img.save(cachedir / "ice_activity_clusters.png")
mkimage()
# save dpp 
