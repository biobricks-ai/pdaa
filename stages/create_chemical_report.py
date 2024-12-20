import os
import numpy as np
import requests
import sys
sys.path.append('./')
import stages.utils.openai as openai_utils
import stages.utils.chemprop as chemprop
import stages.utils.pdaa as pdaa

import json
import sqlite3
import biobricks as bb
import rdflib
import pathlib
import pandas as pd
from tqdm import tqdm
import asyncio
import faiss
from rdflib import URIRef

cachedir = pathlib.Path('cache') / 'create_chemical_report'
cachedir.mkdir(parents=True, exist_ok=True)

# Load chemprop properties
with sqlite3.connect(bb.assets('chemprop-transformer').cvae_sqlite) as con:
    properties_df = pd.read_sql_query("SELECT distinct property_id,property_token, data FROM property p", con)
    propcat = pd.read_sql_query("SELECT distinct property_id, category_id FROM property_category p", con)
    propcat = propcat.merge(pd.read_sql_query("SELECT distinct category_id, category FROM category c", con), on="category_id", how="inner")
    propcat = properties_df.merge(propcat, on='property_id', how='inner')[['property_token','category']]
    properties_df['data'] = properties_df['data'].map(json.loads)

def lookup_chemical_inchi(chemical_name):
    # First get the PubChem CID by searching the name
    search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chemical_name}/cids/JSON"
    response = requests.get(search_url)
    response.raise_for_status()
    cid = response.json()['IdentifierList']['CID'][0]
        
    # Then get the InChI using the CID
    inchi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChI/JSON"
    response = requests.get(inchi_url)
    response.raise_for_status()
    return response.json()['PropertyTable']['Properties'][0]['InChI']

def get_all_property_predictions(inchi):
    inchi_tok_pairs = [(inchi, tok) for tok in properties_df['property_token']]
    predictions = pdaa.get_predictions_with_sqlite_cache(inchi_tok_pairs)
    return predictions

def important_pathways(chemical_name, prompt):
    # get all the chemprop predictions for the chemical
    predictions = get_all_property_predictions(inchi)
    prompt_embedding = np.array(openai_utils.embed(prompt))[np.newaxis, :]
    
    sims, indices = pdaa.faiss_index.search(prompt_embedding, 1000)
    faiss_uris = [pdaa.faiss_index_to_uri(i) for i in indices[0]]

    faiss_query = f"""
    SELECT ?uri ?o
    WHERE {{ 
        ?uri EDAM:has_identifier ?o 
    }}"""
    faiss_data = pdaa.pdaa_query(faiss_query)[['uri','o']]
    faiss_data = faiss_data[faiss_data['o'].isin([str(uri) for uri in faiss_uris])]
    faiss_data['uri'] = faiss_data['uri'].map(lambda x: URIRef(x))

    return predictions


chemical_name = 'dehp'
inchi = lookup_chemical_inchi(chemical_name)
predictions = get_all_property_predictions(inchi)
prompt = "endocrine disruption"

# region faiss cosine matrix ===============================================================
# lookup all faissindex uris and get their embeddings
# get faiss indexes of uris
uri_faissindex = pdaa.pdaa_query("""
SELECT ?uri ?faissindex ?index WHERE {
    ?uri EDAM:has_identifier ?faissindex .
    ?faissindex a toxindex:faiss_index .
    ?faissindex rdfs:label ?index .
}
""")

# get aops [ uri, faissindex]
aops = pdaa.aopwiki_query("SELECT ?uri WHERE { ?uri a aop:AdverseOutcomePathway . }")
aop_faissindex = aops.merge(uri_faissindex, on='uri', how='inner')
aop_faissindex['index'] = aop_faissindex['index'].astype(int)
aop_faissindex['embedding'] = [pdaa.faiss_index.reconstruct(int(i)) for i in aop_faissindex['index'].values]

# find uris that has_identifier ?x where ?x is a toxindex:property_token
proptoken_uris = pdaa.pdaa_query("""
    SELECT ?uri ?proptoken ?token WHERE { 
        ?uri EDAM:has_identifier ?proptoken . 
        ?proptoken a <http://toxindex.com/ontology/property> . 
        ?proptoken rdfs:label ?token .
    }""")
proptoken_faissindex = proptoken_uris.merge(uri_faissindex, on='uri', how='inner')[['uri','proptoken','index','token']]

proptoken_embeddings = np.vstack([pdaa.faiss_index.reconstruct(int(i)) for i in proptoken_faissindex['index'].values])
proptoken_faiss = faiss.IndexFlatIP(proptoken_embeddings.shape[1])
proptoken_faiss.add(proptoken_embeddings)

# Get similarities between property tokens and AOPs
D, I = proptoken_faiss.search(np.vstack(aop_faissindex['embedding'].values), k=len(proptoken_faissindex))

# Create similarity matrix with property tokens and AOPs
simtable = []
for aop_idx, (distances, indices) in enumerate(zip(D, I)):
    aop_uri = aop_faissindex.iloc[aop_idx]['uri']
    for dist, prop_idx in zip(distances, indices):
        if dist >= 0.4:  # Only keep similarities >= 40%
            prop_uri = proptoken_faissindex.iloc[prop_idx]['uri']
            prop_token = int(proptoken_faissindex.iloc[prop_idx]['token'])
            simtable.append({
                'aop': aop_uri,
                'property_uri': prop_uri,
                'property_token': prop_token,
                'similarity': dist
            })

aop_simtable = pd.DataFrame(simtable)
aop_simtable.to_csv(cachedir / 'aop_simtable.csv')

# endregion

# region AOPWIKI PREDICTIONS ===============================================================
# a function that takes a chemical name and returns of relevant adverse outcomes pathways
def get_aopwiki_predictions(chemical_name):

    predictions : dict[int,float] = get_all_property_predictions(inchi)
    pos_predictions = {tok: pred for tok, pred in predictions.items() if pred > 0.8}
    pos_indices = [idx for idx in pos_predictions.keys()]
    simtable = aop_simtable[aop_simtable['property_token'].isin(pos_indices)]
    aop_weight = simtable.groupby('aop')['similarity'].sum().reset_index()
    aop_weight = aop_weight.sort_values('similarity', ascending=False)
    aop_weight
    res = pdaa.aopwiki_query("SELECT ?p ?o WHERE { <https://identifiers.org/aop/43> ?p ?o }")[['o','p']]
    

# region AOP SPECIFIC PREDICTIONS ==========================================================
# a function that takes a chemical name and an AOP and returns relevant predictions
def get_aop_specific_predictions(chemical_name, aop):
    predictions = pdaa.get_predictions_with_sqlite_cache(inchi)

# region PRIORITY PHTALATE HEATMAP =======================================================
import rdkit, rdkit.Chem, rdkit.Chem.AllChem, rdkit.DataStructs, rdkit.Chem.rdFingerprintGenerator
import itertools as it
dehp = rdkit.Chem.MolFromSmiles('CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC')
diup = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCC(C)C')
dtdp = rdkit.Chem.MolFromSmiles('CCCCCCCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCCCCCC')
didp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCC(C)C')
dinp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCC(C)C')

phthalate_inchi_names = ['dehp', 'diup', 'dtdp', 'didp', 'dinp']
priority_phthalates = [dehp, diup, dtdp, didp, dinp]
phthalates_inchi = [rdkit.Chem.inchi.MolToInchi(p) for p in priority_phthalates]

# get all predictions 
inchi_tok_pairs = list(it.product(phthalates_inchi, properties_df['property_token'].astype(int).values))
inchi_tok_pairs = [(i, int(t)) for i,t in inchi_tok_pairs]
predictions : list[tuple[str,int,float]] = pdaa.get_predictions_with_sqlite_cache(inchi_tok_pairs)

# create a heatmap with phthalates on the x axis and properties on the y axis
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import math


# Convert predictions list to matrix form
pred_matrix = pd.DataFrame(predictions, columns=['inchi', 'property_token', 'value'])
pred_matrix = pred_matrix.merge(propcat, on='property_token', how='inner')

# create one heatmap
# Pivot the data to create a matrix for each category
categories = list(['neurotoxicity','carcinogenicity','chronic toxicity','mutagenicity','endocrine disruption','hepatotoxicity','reproductive toxicity',
                  'developmental toxicity','genotoxicity','eye irritation','immunotoxicity','aquatic toxicity',
                  'environmental toxicity','sub-chronic toxicity','acute oral toxicity',
                  'nephrotoxicity','ecotoxicity','acute inhalation toxicity','skin irritation','dermal absorption'])
pred_matrix = pred_matrix[pred_matrix['category'].isin(categories)]


# there should be at most 3 columns
# Create individual clustermaps for each category and save them
clustermap_figs = []
image_names = []
for cat in categories:
    cat_data = pred_matrix[pred_matrix['category'] == cat].pivot(
        index='property_token', 
        columns='inchi', 
        values='value'
    )
    
    # Skip if no data for this category
    if cat_data.empty:
        continue
        
    # Create clustermap
    g = sns.clustermap(
        cat_data,
        cmap='viridis',
        dendrogram_ratio=(.2, .1),
        figsize=(6, 4),
        yticklabels=False,
        xticklabels=False,
        cbar=False
    )
    
    # Add title
    g.ax_heatmap.set_title(cat)
    
    # Save figure
    path = f'heatmap_{cat.lower().replace(" ", "_")}.png'
    image_names.append(path)
    plt.savefig(path)
    plt.close()

images = []
for cat in categories:
    try:
        img_path = f'heatmap_{cat.lower().replace(" ", "_")}.png'
        images.append(Image.open(img_path))
    except:
        continue

# Calculate grid dimensions
n_images = len(images)
n_cols = 3
n_rows = math.ceil(n_images / n_cols)

# Create blank canvas
cell_width = images[0].width
cell_height = images[0].height
canvas = Image.new('RGB', (cell_width * n_cols, cell_height * n_rows))

# Paste images into grid
for idx, img in enumerate(images):
    row = idx // n_cols
    col = idx % n_cols
    canvas.paste(img, (col * cell_width, row * cell_height))

# Save final composite
canvas.save('category_heatmaps.png')

# remove the temporary files
for img in image_names:
    os.remove(img)

# CREATE GLOBAL HEATMAP ===============================================================
# For the main clustered heatmap, pivot the full matrix
# Create a mapping of InChI to phthalate names
phthalate_names = {
    rdkit.Chem.inchi.MolToInchi(dehp): 'DEHP',
    rdkit.Chem.inchi.MolToInchi(diup): 'DIUP', 
    rdkit.Chem.inchi.MolToInchi(dtdp): 'DTDP',
    rdkit.Chem.inchi.MolToInchi(didp): 'DIDP',
    rdkit.Chem.inchi.MolToInchi(dinp): 'DINP'
}

piv_pred_matrix = pred_matrix[['property_token','inchi','value']].drop_duplicates().pivot(
    index='property_token',
    columns='inchi', 
    values='value'
)

# Rename the columns using the phthalate names
piv_pred_matrix = piv_pred_matrix.rename(columns=phthalate_names)

g = sns.clustermap(
    piv_pred_matrix,
    cmap='viridis', 
    dendrogram_ratio=(.2, .1),
    cbar_pos=(0.02, .32, .03, .2),
    figsize=(6, 12),  # Increased height from 4 to 12
    yticklabels=False,
    xticklabels=True  # Show the phthalate name labels
)

plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.savefig('test.png', bbox_inches='tight')

# region MORE INCHI HEATMAP ===============================================================
import rdkit, rdkit.Chem, rdkit.Chem.AllChem, rdkit.DataStructs, rdkit.Chem.rdFingerprintGenerator
import itertools as it
dehp = rdkit.Chem.MolFromSmiles('CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC')
diup = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCC(C)C')
dtdp = rdkit.Chem.MolFromSmiles('CCCCCCCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCCCCCC')
didp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCC(C)C')
dinp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCC(C)C')

raw_df = pd.read_parquet('cache/priority_phthalates/priority_phthalates.parquet')
top_df = raw_df.sort_values(by='max_similarity', ascending=False)[['inchi', 'max_similarity']].drop_duplicates()
inchi_list = top_df['inchi'][:100].unique().tolist()

priority_phthalates = [dehp, diup, dtdp, didp, dinp]
phthalates_inchi = [rdkit.Chem.inchi.MolToInchi(p) for p in priority_phthalates]

# get all predictions 
inchi_tok_pairs = list(it.product(phthalates_inchi, properties_df['property_token'].astype(int).values))
inchi_tok_pairs = [(i, int(t)) for i,t in inchi_tok_pairs]
predictions : list[tuple[str,int,float]] = pdaa.get_predictions_with_sqlite_cache(inchi_tok_pairs)

# create a heatmap with phthalates on the x axis and properties on the y axis
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import math


# Convert predictions list to matrix form
pred_matrix = pd.DataFrame(predictions, columns=['inchi', 'property_token', 'value'])
pred_matrix = pred_matrix.merge(propcat, on='property_token', how='inner')

# create one heatmap
# Pivot the data to create a matrix for each category
categories = list(['neurotoxicity','carcinogenicity','chronic toxicity','mutagenicity','endocrine disruption','hepatotoxicity','reproductive toxicity',
                  'developmental toxicity','genotoxicity','eye irritation','immunotoxicity','aquatic toxicity',
                  'environmental toxicity','sub-chronic toxicity','acute oral toxicity',
                  'nephrotoxicity','ecotoxicity','acute inhalation toxicity','skin irritation','dermal absorption'])
pred_matrix = pred_matrix[pred_matrix['category'].isin(categories)]


# there should be at most 3 columns
# Create individual clustermaps for each category and save them
clustermap_figs = []
image_names = []
for cat in categories:
    cat_data = pred_matrix[pred_matrix['category'] == cat].pivot(
        index='property_token', 
        columns='inchi', 
        values='value'
    )
    
    # Skip if no data for this category
    if cat_data.empty:
        continue
        
    # Create clustermap
    g = sns.clustermap(
        cat_data,
        cmap='viridis',
        dendrogram_ratio=(.2, .1),
        figsize=(6, 4),
        yticklabels=False,
        xticklabels=False,
        cbar=False
    )
    
    # Add title
    g.ax_heatmap.set_title(cat)
    
    # Save figure
    path = f'heatmap_{cat.lower().replace(" ", "_")}.png'
    image_names.append(path)
    plt.savefig(path)
    plt.close()

images = []
for cat in categories:
    try:
        img_path = f'heatmap_{cat.lower().replace(" ", "_")}.png'
        images.append(Image.open(img_path))
    except:
        continue

# Calculate grid dimensions
n_images = len(images)
n_cols = 3
n_rows = math.ceil(n_images / n_cols)

# Create blank canvas
cell_width = images[0].width
cell_height = images[0].height
canvas = Image.new('RGB', (cell_width * n_cols, cell_height * n_rows))

# Paste images into grid
for idx, img in enumerate(images):
    row = idx // n_cols
    col = idx % n_cols
    canvas.paste(img, (col * cell_width, row * cell_height))

# Save final composite
canvas.save('category_heatmaps.png')

# remove the temporary files
for img in image_names:
    os.remove(img)

# For the main clustered heatmap, pivot the full matrix
piv_pred_matrix = pred_matrix[['property_token','inchi','value']].drop_duplicates().pivot(
    index='property_token',
    columns='inchi',
    values='value'
)

g = sns.clustermap(
    piv_pred_matrix, # Transpose the matrix to flip axes
    cmap='viridis',
    dendrogram_ratio=(.2, .1),
    cbar_pos=(0.02, .32, .03, .2),
    figsize=(12, 4), # Adjust figure size to make squares more square
    yticklabels=True, # Keep phthalate labels
    xticklabels=False # Remove property token labels
)

plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig('test.png')