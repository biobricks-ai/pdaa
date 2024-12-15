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

# Load chemprop properties
with sqlite3.connect(bb.assets('chemprop-transformer').cvae_sqlite) as con:
    properties_df = pd.read_sql_query("SELECT distinct property_token, data FROM property p", con)
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

def create_chemical_report(chemical_name, endpoint=None):
    pass

import importlib
importlib.reload(pdaa)

chemical_name = 'dehp'
inchi = lookup_chemical_inchi(chemical_name)
predictions = get_all_property_predictions(inchi)
prompt = "endocrine disruption"
create_chemical_report('acetaminophen')
