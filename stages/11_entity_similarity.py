import os
import numpy as np
import requests
import sys
sys.path.append('./')
import stages.utils.openai as openai_utils
# import stages.utils.chemprop as chemprop
import stages.utils.pdaa as pdaa
import stages.utils.pubchem as pubchem
import stages.utils.sparql as sparql
import stages.utils.toxindex as toxindex

import biobricks as bb
import pathlib
import pandas as pd
import faiss
import sqlite3
from rdflib import URIRef
from tqdm import tqdm

import rdkit, rdkit.Chem.rdMolDescriptors, rdkit.Chem.Crippen, rdkit.Chem.rdFingerprintGenerator, rdkit.DataStructs
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import threading

import seaborn as sns
import matplotlib.pyplot as plt

# import rdflib
# from rdflib.plugins.stores.sparqlstore import SPARQLStore

# brickdir = pathlib.Path('brick')
# sqlite_lock = threading.Lock()

# cachedir = pathlib.Path('cache') / 'util' / 'pdaa'
# cachedir.mkdir(parents=True, exist_ok=True)

# # Create a SPARQL store pointing to the Blazegraph endpoint
# pdaa_graph = rdflib.Graph(store=SPARQLStore('http://localhost:9999/blazegraph/namespace/pdaa/sparql'))
# pdaa_graph.namespace_manager.bind('aop', rdflib.Namespace('http://aopkb.org/aop_ontology#'))
# pdaa_graph.namespace_manager.bind('toxindex', rdflib.Namespace('http://toxindex.com/ontology/'))
# pdaa_graph.namespace_manager.bind('dcterms', rdflib.Namespace('http://purl.org/dc/elements/1.1/'))
# pdaa_graph_cache = cachedir / 'pdaa_graph'

# sqlite_lock = threading.Lock()
# # Fetch URIs linked to property tokens
# # TODO some predicted_properties have multiple tokens
# proptoken_uris = sparql.Query(pdaa_graph, pdaa_graph_cache) \
#     .select_typed({'uri': str, 'proptoken': str, 'token': int, 'title': str}) \
#     .where('?proptoken <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
#     .where('?proptoken a <http://toxindex.com/ontology/predicted_property>') \
#     .where('?proptoken rdf:value ?token') \
#     .where('?proptoken <http://purl.org/dc/elements/1.1/title> ?title') \
#     .cache_execute() \
#     .groupby('uri').first().reset_index()

# def add_predictions(predictions, lock):
#     with lock:
#         with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
#             for inchi, property_token, positive_prediction in predictions:
#                 conn.execute('INSERT INTO predictions (inchi, property_token, positive_prediction) VALUES (?, ?, ?)', (inchi, property_token, positive_prediction))

# def is_missing(inchi_list):
#     inchi_tok_pairs = [(inchi, tok) for inchi in inchi_list for tok in proptoken_uris['token']]
#     missing_inchi = set()
#     with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
#         for inchi, property_token in inchi_tok_pairs:
#             if inchi in missing_inchi:
#                 continue
#             cursor = conn.execute('SELECT * FROM predictions WHERE inchi = ? AND property_token = ?', (inchi, property_token))
#             exists = cursor.fetchone() is not None
#             if not exists:
#                 missing_inchi.add(inchi)
#     return missing_inchi

# def lookup_predictions(inchi_tok_pairs):
#     with sqlite_lock:
#         with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
#             results = []
#             for inchi, property_token in inchi_tok_pairs:
#                 cursor = conn.execute("""SELECT inchi, CAST(property_token AS INTEGER) as property_token, positive_prediction FROM predictions 
#                                       WHERE inchi = ? AND property_token = ?""", (inchi, property_token))
#                 result = cursor.fetchone()
#                 if result is not None:
#                     results.append((inchi, property_token, result[2]))
#             return results

# def predict_all_properties_with_sqlite_cache(inchi_list):
#     missing_inchi = is_missing(inchi_list)
#     preds = []
#     for inchi in missing_inchi:
#         preds.extend(chemprop.chemprop_predict_all(inchi))
    
#     preds = [(fullpred['inchi'],int(fullpred['property_token']),fullpred['value']) for fullpred in preds]
#     add_predictions(preds, sqlite_lock)

#     non_missing_inchi = [inchi for inchi in inchi_list if inchi not in missing_inchi]
#     for tok in proptoken_uris['token']:
#         preds.extend(lookup_predictions([(inchi, tok) for inchi in non_missing_inchi]))

#     return preds

tqdm.pandas()

brickdir = pathlib.Path('brick')
cachedir = pathlib.Path('cache') / 'entity_similarity'
cachedir.mkdir(parents=True, exist_ok=True)

# region EXAMPLE PHTHALATES ====================================================================
dehp = rdkit.Chem.MolFromSmiles('CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC')
diup = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCC(C)C')
dtdp = rdkit.Chem.MolFromSmiles('CCCCCCCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCCCCCC')
didp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCC(C)C')
dinp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCC(C)C')

# dpp = rdkit.Chem.MolFromSmiles('C1=CC=C(C=C1)OC(=O)C2=CC=CC=C2C(=O)OC3=CC=CC=C3') # diphenyl phthalate
# bbp = rdkit.Chem.MolFromSmiles('CCCCOC(=O)c1ccccc1C(=O)OCc2ccccc2') # Benzyl butyl phthalate
dbp = rdkit.Chem.MolFromInchi('InChI=1S/C16H22O4/c1-3-5-11-19-15(17)13-9-7-8-10-14(13)16(18)20-12-6-4-2/h7-10H,3-6,11-12H2,1-2H3')

example_phthalates = [dehp, diup, dtdp, didp, dinp, dbp]
example_names = ['DEHP', 'DIUP', 'DTDP', 'DIDP', 'DINP', 'DBP']
example_inchi = [Chem.MolToInchi(m) for m in example_phthalates]
example_weights = [rdkit.Chem.rdMolDescriptors.CalcExactMolWt(m) for m in example_phthalates]
example_inchi2name = {inchi: name for inchi, name in zip(example_inchi, example_names)}

# print the alias weights
for name, weight in zip(example_names, example_weights):
    print(f"{name}: {weight:.2f}")

# run the model on the example phthalates
_ = pdaa.predict_all_properties_with_sqlite_cache(example_inchi)
# _ = predict_all_properties_with_sqlite_cache(example_inchi)

# which phthalate has the lowest mean ICE activity?
# region ICE ACTIVITY ===============================================================
def build_phthalate_ice_activity_df():
    print("Building phthalate ICE activity dataframe...")
    
    print("Querying PDAA graph for URI, title and token mappings...")
    uri_title_token = sparql.Query(pdaa.pdaa_graph) \
        .select_typed({'uri': str, 'pp': str, 'title': str, 'token': int}) \
        .where('?pp a toxindex:predicted_property') \
        .where('?pp <http://purl.org/dc/elements/1.1/title> ?title') \
        .where('?pp rdf:value ?token') \
        .where('?pp <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
        .execute().groupby('uri').first().reset_index()

    ice_assays = uri_title_token[uri_title_token['uri'].str.contains('ice.ntp')]
    print(f"Found {len(ice_assays)} ICE assays")

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

    def is_diester_phthalate(m):
        diester_phthalate = Chem.MolFromSmarts("[cH][cH]c(C(=O)OC[CH2,CH,C])c(C(=O)OC[CH2,CH,C])[cH][cH]") # 542
        dp = Chem.AddHs(m).HasSubstructMatch(diester_phthalate)
        only_coh = all(atom.GetSymbol() in ['C', 'H', 'O'] for atom in m.GetAtoms())
        only_one_ring = m.GetRingInfo().NumRings() == 1
        return dp and only_coh and only_one_ring

    try:
        assert all(is_diester_phthalate(m) for m in example_phthalates)
    except AssertionError:
        print("Some example phthalates are not diester phthalates. Please check the SMARTS pattern.")
        raise

    print("Filtering for diester phthalates...")
    filtered_phthalates = inchi_mol_df[inchi_mol_df['mol'].progress_apply(is_diester_phthalate)]['inchi']
    df3 = df2[df2['inchi'].isin(filtered_phthalates)]
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
        sorted_indices = cluster_indices[np.argsort(mean_activities[cluster_indices])]
        cluster_order.extend(sorted_indices)

    # Reorder the matrix
    reordered_matrix = activity_matrix_filled.iloc[cluster_order, col_order]

    # Calculate mean activity per chemical across all assays
    mean_activity = reordered_matrix.mean(axis=1)

    # Define specific colors for each cluster
    cluster_colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blue, Red, Green
    row_colors = [cluster_colors[row_clusters[i]] for i in cluster_order]

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(35, 10), gridspec_kw={'width_ratios': [4, 1]})

    # Create heatmap
    im = ax1.imshow(reordered_matrix, 
            cmap='RdYlBu_r',
            aspect='auto',
            vmin=0,
            vmax=1)

    plt.colorbar(im, ax=ax1, label='Activity Score')

    # Add labels for heatmap
    ax1.set_xlabel('ICE assays', fontsize=24)
    ax1.set_ylabel('Selected Phthalates', fontsize=24)

    # Keep ticks hidden since there are too many to show clearly
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Map to reordered positions
    reordered_indices = {inchi: i for i, inchi in enumerate(reordered_matrix.index)}
    
    
    for i, inchi in enumerate(example_inchi2name):
        if inchi in reordered_indices:
            name = example_inchi2name[inchi]
            pos = reordered_indices[inchi]
            cluster = row_clusters[cluster_order[pos]]  # Get cluster for this ordered position
            color = cluster_colors[cluster]  # Get color for this cluster
            lbl = f'← {name}' if name == 'DEHP' else f'←'
            ax1.text(reordered_matrix.shape[1], pos, 
                    lbl, ha='left', va='center', 
                    fontsize=26, color=color)

    ax1.set_title('Clustered Heatmap of Selected Diester Phthalate Activity Across ICE Assays',
            fontsize=28, pad=20)

    # Create horizontal bar chart with cluster colors
    ax2.barh(range(len(mean_activity)), mean_activity, color=row_colors)
    # Add annotations for example phthalates on the bar chart
    for i, (idx, row) in enumerate(reordered_matrix.iterrows()):
        if idx in example_inchi2name:
            name = example_inchi2name[idx]
            ax2.text(mean_activity[i], i, f' {name}', va='center', fontsize=10)
            
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlabel('Mean Activity', fontsize=24)
    ax2.set_yticks([])

    # Add legend for clusters
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=cluster_colors[i], 
                                label=f'Cluster {i+1}\n(n={np.sum(row_clusters == i)})') for i in range(n_clusters)]
    ax2.legend(handles=legend_elements, loc='upper right', title='Clusters')

    plt.tight_layout()
    plt.savefig(cachedir / "phthalate_activity_heatmap.png", dpi=600, bbox_inches='tight')
    plt.close()

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
    for cluster in [2,0,1]:
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
    legends = [f"{row['name'] if row['is_example'] else ''} (Cluster {row['cluster']}, Activity: {row['positive_prediction']:.3f})" 
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
