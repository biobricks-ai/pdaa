import numpy as np
import re
import pathlib
import pandas as pd
import sqlite3
from tqdm import tqdm

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

def _styled_heatmap(matrix, row_colors, *, dpi=600,
                    fontcolor='white', linecolor='black'):
    """
    Wrapper around seaborn.clustermap with the same visual
    tweaks used in build_heatmap.py (_generate_heatmap).
    - matrix: rows = phthalates, cols = ICE assays
    - row_colors: list-like, same length as matrix.shape[0]
    """
    # Cluster only columns; we already ordered rows
    g = sns.clustermap(matrix,
                    #    square=True,       # ← force equal-sized cells
                       cbar_kws={'drawedges': False},  # disable seaborn’s built-in bar
                       cmap='viridis',
                       row_cluster=False, col_cluster=True,
                       row_colors=row_colors,
                       xticklabels=False, yticklabels=False,
                       linecolor=linecolor,
                    #    linewidths=0.5,
                       figsize=(18, 9),
                    #    cbar_pos=(0.91, 0.3, 0.02, 0.4),
                       cbar_pos=(0.95, 0.3, 0.02, 0.4),
                       dendrogram_ratio=(0.10, 0.05),
                       tree_kws={'linewidths': 0.5})
    # Remove any stray colorbar
    if hasattr(g, 'cax') and g.cax:
        g.cax.remove()

    # Create a new colorbar on its own axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(g.ax_heatmap)
    cax = divider.append_axes("right", size="2%", pad=0.6)
    # expose the divider so callers can add more axes without destroying the layout
    g.divider = divider
    sm  = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=matrix.min().min(),
                                                                vmax=matrix.max().max()))
    sm.set_array([])
    cb = g.figure.colorbar(sm, cax=cax)
    cb.set_label('Activity Score', fontsize=18, labelpad=10)
    cb.ax.tick_params(labelsize=14)

    # Hide col dendrogram but keep clustering
    g.ax_col_dendrogram.set_visible(False)

    g.ax_heatmap.set_xlabel('DART or ED Assays', color=fontcolor, fontsize=20)

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

# run the model on the example phthalates
pdaa.predict_all_properties_with_sqlite_cache(example_inchi)

# possible isomers to look for
isomers_list = ["ortho_phthalate", "meta_phthalate", "para_phthalate"]

# which phthalate has the lowest mean ICE activity?
# region ICE ACTIVITY ===============================================================
def build_phthalate_ice_activity_df(mask_method='prediction', use_dart = True, use_ed = True):
    print("Building phthalate ICE activity dataframe...")
    
    print("Querying PDAA graph for URI, title and token mappings...")
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

    ice_assays = uri_title_token[mask]
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
    df = ice_preds.sort_values('positive_prediction', ascending=False)[['inchi', 'property_token', 'positive_prediction']]
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

phthalate_df = build_phthalate_ice_activity_df()[['uri','title','inchi','mol','positive_prediction']]

# endregion

# region HEATMAP & DENSITY OF PHTHALATE ACTIVITY ===============================
def cluster_rows_and_make_heatmap():
    activity_matrix = phthalate_df.groupby(['inchi','title'])['positive_prediction'].mean().reset_index()
    activity_matrix = activity_matrix.pivot(index='inchi', columns='title', values='positive_prediction')

    inchi_activity_counts = phthalate_df.reset_index().groupby(['inchi','title'])['positive_prediction'].mean().reset_index()
    inchi_activity_counts['active'] = inchi_activity_counts['positive_prediction'] > 0.6
    inchi_activity = inchi_activity_counts.groupby('inchi')['active'].mean().reset_index()
    inchi_activity.sort_values('active', ascending=False)
    inchi_activity.to_csv(cachedir / 'inchi_activity.csv', index=False)

    # Cluster the data using KMeans instead of hierarchical clustering
    from sklearn.cluster import KMeans
    from scipy.cluster import hierarchy

    # Fill any NaN values with 0 for clustering
    activity_matrix_filled = activity_matrix.fillna(0)
    # save the filled activitiy matrix
    activity_matrix_filled.to_parquet(cachedir / 'activity_matrix_filled.parquet')

    # Apply KMeans clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    row_clusters = kmeans.fit_predict(activity_matrix_filled)

    # Still use hierarchical clustering for column ordering
    col_linkage = hierarchy.linkage(activity_matrix_filled.T, method='average')
    col_order = hierarchy.leaves_list(col_linkage)

    # Sort rows by cluster and then by mean activity within clusters
    mean_activities = activity_matrix_filled.mean(axis=1)
    cluster_order = []
    for i in range(n_clusters):
        cluster_indices = np.where(row_clusters == i)[0]
        sorted_indices = cluster_indices[np.argsort(mean_activities.iloc[cluster_indices])]
        cluster_order.extend(sorted_indices)

    # Reorder the matrix
    reordered_matrix = activity_matrix_filled.iloc[cluster_order, col_order]
    # ─── Map each InChI to its new row index ─────────────────────────────
    reordered_indices = {
        inchi: pos
        for pos, inchi in enumerate(reordered_matrix.index)
    }

    # Calculate mean activity per chemical across all assays
    mean_activity = reordered_matrix.mean(axis=1)

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
    
    # color_by = 'isomer'
    color_by = 'cluster'

    if color_by == 'isomer':
        isomer_matches = np.array([0 for _ in isomers_list])
        row_colors = []
        # n_non_ortho = 0
        for inchi in reordered_matrix.index:
            if (mol := Chem.MolFromInchi(inchi)) is None:
                continue  # skip invalid InChIs
            match_list = [pdaa.is_phthalate(mol, modes=(isomer,)) for isomer in isomers_list]
            isomer_matches += match_list
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
    g = _styled_heatmap(reordered_matrix, row_colors, fontcolor='black')

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # ─── Attach a fresh bar axis ──────────────────────────────────────────
    # divider = make_axes_locatable(g.ax_heatmap)
    # ax_bar = divider.append_axes("right", size="15%", pad=0.8)
    divider = g.divider            # reuse the one that already holds the colour-bar
    ax_bar  = divider.append_axes("right", size="15%", pad=1.0)  # pad > 0.6 keeps some space

    # ─── Plot mean activity ──────────────────────────────────────────────
    if color_by == 'isomer':
        cluster_colors = ['#1f77b4', '#d62728', '#2ca02c']  # Darker Blue, Darker Red, Darker Green
    elif color_by == 'cluster':
        # Use the same colors as the clusters
        cluster_colors = base_colors
    bar_colors = [cluster_colors[row_clusters[i]] for i in cluster_order]
    ax_bar.barh(range(len(mean_activity)), mean_activity, color=bar_colors)

    # ─── Annotate the example phthalates ────────────────────────────────
    for inchi, name in example_inchi2name.items():
        pos = reordered_indices.get(inchi)
        if pos is not None:
            ax_bar.text(mean_activity.iloc[pos], pos,
                        f' {name}',
                        va='center', fontsize=10, color='black')

    # ─── Tidy up axis ────────────────────────────────────────────────────
    ax_bar.set_ylim(g.ax_heatmap.get_ylim())
    ax_bar.set_xlabel('Mean Activity', fontsize=20)
    ax_bar.set_yticks([])

    # ─── Cluster legend ─────────────────────────────────────────────────
    # Use three related but darker colors for cluster identification (to distinguish from isomer_colors)
    
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=cluster_colors[i],
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
        color='black', fontsize=20, labelpad=35
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
    assay_activity.sort_values('positive_prediction', ascending=False)
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

clustered_phthalate_df = cluster_rows_and_make_heatmap()[['uri','title','inchi','mol','positive_prediction','cluster','cluster_color','example']]

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
# sys.exit()
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
    for cluster in [0,2,1]:  # Order clusters by mean activity
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
