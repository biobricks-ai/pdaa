import asyncio
import faiss
import numpy as np
import pandas as pd
import pathlib
import rdflib
from rdflib import URIRef
from rdflib.plugins.stores.sparqlstore import SPARQLStore
import sqlite3
import threading
from tqdm import tqdm

import stages.utils.chemprop as chemprop
import stages.utils.openai as openai_utils
import stages.utils.sparql as sparql

from rdkit import Chem
from collections.abc import Iterable

brickdir = pathlib.Path('brick')
sqlite_lock = threading.Lock()

cachedir = pathlib.Path('cache') / 'util' / 'pdaa'
cachedir.mkdir(parents=True, exist_ok=True)

# uri faiss index
faiss_index = faiss.read_index('cache/associate_properties_with_aopwiki/faiss_index_cosine.index')

# Create a SPARQL store pointing to the Blazegraph endpoint
# pdaa_graph = rdflib.Graph(store=SPARQLStore('http://localhost:9999/blazegraph/namespace/pdaa/sparql'))
pdaa_graph = rdflib.Graph(store=SPARQLStore('http://localhost:9999/bigdata/namespace/pdaa/sparql'))

pdaa_graph.namespace_manager.bind('aop', rdflib.Namespace('http://aopkb.org/aop_ontology#'))
pdaa_graph.namespace_manager.bind('toxindex', rdflib.Namespace('http://toxindex.com/ontology/'))
pdaa_graph.namespace_manager.bind('dcterms', rdflib.Namespace('http://purl.org/dc/elements/1.1/'))
pdaa_graph_cache = cachedir / 'pdaa_graph'

uri_faissindex = sparql.Query(pdaa_graph, pdaa_graph_cache) \
    .select_typed({'uri': str, 'index': int}) \
    .where('?faissindex <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
    .where('?faissindex a <http://toxindex.com/ontology/faiss_index>') \
    .where('?faissindex rdf:value ?index') \
    .cache_execute()

uri_faissindex['embedding'] = uri_faissindex['index'].map(lambda i: faiss_index.reconstruct(i))

# Load mappings for molecular initiating events (MIE) and adverse outcomes (AO)
aop_mie_ao = sparql.Query(pdaa_graph, pdaa_graph_cache) \
    .select('aop', 'mie', 'ao') \
    .where('?aop aop:has_adverse_outcome ?ao') \
    .where('?aop aop:has_molecular_initiating_event ?mie') \
    .cache_execute()

# Fetch URIs linked to property tokens
# TODO some predicted_properties have multiple tokens
proptoken_uris = sparql.Query(pdaa_graph, pdaa_graph_cache) \
    .select_typed({'uri': str, 'proptoken': str, 'token': int, 'title': str}) \
    .where('?proptoken <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
    .where('?proptoken a <http://toxindex.com/ontology/predicted_property>') \
    .where('?proptoken rdf:value ?token') \
    .where('?proptoken <http://purl.org/dc/elements/1.1/title> ?title') \
    .cache_execute() \
    .groupby('uri').first().reset_index()

proptoken_uris[proptoken_uris['uri'].str.contains('ice.ntp')]

def lookup_predictions(inchi_tok_pairs):
    with sqlite_lock:
        with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
            results = []
            for inchi, property_token in inchi_tok_pairs:
                cursor = conn.execute("""SELECT inchi, CAST(property_token AS INTEGER) as property_token, positive_prediction FROM predictions 
                                      WHERE inchi = ? AND property_token = ?""", (inchi, property_token))
                result = cursor.fetchone()
                if result is not None:
                    results.append((inchi, property_token, result[2]))
            return results

def add_predictions(predictions, lock):
    with lock:
        with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
            for inchi, property_token, positive_prediction in predictions:
                conn.execute('INSERT INTO predictions (inchi, property_token, positive_prediction) VALUES (?, ?, ?)', (inchi, property_token, positive_prediction))

async def async_predict_all(inchi,semaphore):
    async with semaphore:
        for attempt in range(5):
            try:
                return await chemprop.chemprop_predict_all_async(inchi=inchi)
            except Exception as e:
                if attempt < 4:  # Don't sleep on last attempt
                    await asyncio.sleep(60)
                else:
                    raise e

def is_missing(inchi_list):
    inchi_tok_pairs = [(inchi, tok) for inchi in inchi_list for tok in proptoken_uris['token']]
    missing_inchi = set()
    with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
        for inchi, property_token in inchi_tok_pairs:
            if inchi in missing_inchi:
                continue
            cursor = conn.execute('SELECT * FROM predictions WHERE inchi = ? AND property_token = ?', (inchi, property_token))
            exists = cursor.fetchone() is not None
            if not exists:
                missing_inchi.add(inchi)
    return missing_inchi

def predict_all_properties_with_sqlite_cache(inchi_list):
    missing_inchi = is_missing(inchi_list)
    preds = []
    for inchi in tqdm(missing_inchi, desc="Predicting missing InChIs"):
        preds.extend(chemprop.chemprop_predict_all(inchi))
    
    preds = [(fullpred['inchi'],int(fullpred['property_token']),fullpred['value']) for fullpred in preds]
    add_predictions(preds, sqlite_lock)

    non_missing_inchi = [inchi for inchi in inchi_list if inchi not in missing_inchi]
    for tok in tqdm(proptoken_uris['token'], desc="Predicting non-missing InChIs"):
        preds.extend(lookup_predictions([(inchi, tok) for inchi in non_missing_inchi]))

    return preds

async def async_predict(inchi,tok,semaphore):
    async with semaphore:
        result = await chemprop.get_chemprop_prediction_async(inchi=inchi, property_token=tok)
        if result['error'] is not None: raise Exception(f"Error: {result['error']}")
        return result['result']

def get_predictions_with_sqlite_cache(inchi_tok_pairs, with_progress_bar=False) -> dict:
    predictions = lookup_predictions(inchi_tok_pairs)
    inchi_tok_pred_tuples = [(p[0], p[1]) for p in predictions]
    remaining_inchi_tok_pairs = [(i, t) for i, t in inchi_tok_pairs if (i, t) not in inchi_tok_pred_tuples]
    
    if len(remaining_inchi_tok_pairs) == 0:
        return predictions
    
    for inchi, tok in tqdm(remaining_inchi_tok_pairs):
        result = chemprop.get_chemprop_prediction(inchi=inchi, property_token=tok)
        add_predictions([(inchi, tok, result['value'])], sqlite_lock)
        predictions.append((inchi, tok, result['value']))
    return predictions

faissclass = URIRef('http://toxindex.com/ontology/faiss_index')
def faiss_index_to_uri(index):
    return URIRef(f"{faissclass}/faiss_index{index}")

predicted_property_class = URIRef('http://toxindex.com/ontology/predicted_property')
def property_token_to_uri(property_token):
    return URIRef(f"{predicted_property_class}/predicted_property{property_token}")

# PREDICTION UTILITIES ===============================================================
  
def get_all_property_predictions(inchi):
    inchi_tok_pairs = [(inchi, tok) for tok in proptoken_uris['token'].unique()]
    predictions = get_predictions_with_sqlite_cache(inchi_tok_pairs)
    return predictions

# SIMILARITY UTILITIES ===============================================================
def get_prompt_similars(prompt, target_uris, top_k_to_search=20000):
    """Find most similar target URIs to a text prompt using FAISS embeddings.
    
    Args:
        prompt (str): Text prompt to compare against
        target_uris (list): List of URIs to search within
        top_k_to_search (int): Number of nearest neighbors to search before filtering to target URIs
        
    Returns:
        DataFrame with columns ['uri', 'similarity'] containing matches
    """
    # Get embeddings and search FAISS index
    prompt_embedding = np.array(openai_utils.embed(prompt))[np.newaxis, :] 
    prompt_embedding = prompt_embedding / np.linalg.norm(prompt_embedding)
    distances, indices = faiss_index.search(prompt_embedding, min(top_k_to_search, uri_faissindex.shape[0]))
    
    # Filter to target URIs and format results
    target_indices = set(uri_faissindex[uri_faissindex['uri'].isin(target_uris)]['index'])
    matches = [(d, i) for d, i in zip(distances[0], indices[0]) if i in target_indices]
    
    if not matches:
        return pd.DataFrame(columns=['uri', 'similarity'])
        
    results = pd.DataFrame(matches, columns=['similarity', 'index'])
    res = results.merge(uri_faissindex, on='index').groupby('uri')['similarity'].max().reset_index()
    res = res[['uri', 'similarity']].sort_values('similarity', ascending=False)

    return res

def get_uri_similars(source_uris, target_uris):
    """Find most similar target URIs to source URIs using FAISS embeddings.
    
    Args:
        source_uris (list): List of source URIs to compare from
        target_uris (list): List of target URIs to search within
        top_k_to_search (int): Number of nearest neighbors to search before filtering
        
    Returns:
        DataFrame with columns ['source_uri', 'target_uri', 'similarity'] containing matches
    """
    # Get source embeddings from uri_faissindex
    source_faissindex = uri_faissindex[uri_faissindex['uri'].isin(source_uris)].reset_index()
    source_embeddings = np.vstack(source_faissindex['embedding'])
    
    # Get target embeddings and create temporary FAISS index
    target_faissindex = uri_faissindex[uri_faissindex['uri'].isin(target_uris)].reset_index()
    target_embeddings = np.vstack(target_faissindex['embedding'])
    target_faiss = faiss.IndexFlatIP(target_embeddings.shape[1])
    target_faiss.add(target_embeddings)
    
    # Search for nearest neighbors
    distances, indices = target_faiss.search(source_embeddings, target_embeddings.shape[0])
    
    # Convert to dataframe and expand pairs
    results = pd.DataFrame({
        'source_uri': np.repeat(source_faissindex['uri'].values, indices.shape[1]),
        'target_uri': target_faissindex.iloc[indices.ravel()]['uri'].values,
        'similarity': distances.ravel()
    })
    return results.groupby(['source_uri','target_uri'])['similarity'].max().reset_index()

def predict_predicted_property_uris(chemical_inchi, predicted_property_identifiers):
    tmp_proptoken = proptoken_uris[proptoken_uris['uri'].isin(predicted_property_identifiers)][['uri','token']]
    tmp_proptoken['int_token'] = tmp_proptoken['token'].astype(int)

    inchi_tok_pairs = set([(chemical_inchi, int(t)) for t in tmp_proptoken['int_token'].tolist()])
    predictions = get_predictions_with_sqlite_cache(inchi_tok_pairs)
    prediction_df = pd.DataFrame(predictions, columns=['inchi', 'int_token', 'prediction'])

    result = prediction_df.merge(tmp_proptoken, on='int_token', how='inner')
    return result[['uri','prediction']]

adverse_outcomes = set(aop_mie_ao['ao'].tolist())

def retrieve_adverse_outcomes(prompt, chemical_inchi):
    # get relevant adverse outcomes and their MIEs
    prompt_adverse_outcomes = get_prompt_similars(prompt, adverse_outcomes, top_k_to_search=2000).query('similarity > 0.4')
    prompt_ao_mie = aop_mie_ao[aop_mie_ao['ao'].isin(prompt_adverse_outcomes['uri'])][['ao','mie']].drop_duplicates()

    # get relevant predicted properties for given MIEs
    relevant_mie = prompt_ao_mie['mie'].unique()
    
    # Use get_predicted_properties_for_mies to get weighted predictions
    mie_predictions = get_predicted_properties_for_mies(chemical_inchi, relevant_mie)
    
    # Aggregate weights by MIE
    mie_weights = mie_predictions.groupby('mie_uri').agg({'weight': ['sum', 'count']}).reset_index()
    mie_weights.columns = ['mie', 'weight', 'count']

    # Calculate adverse outcome weights by summing MIE weights
    ao_mie_weight = prompt_ao_mie.merge(mie_weights, on='mie', how='inner')
    ao_mie_weight = ao_mie_weight.groupby('ao').agg({'weight': 'sum', 'count': 'sum'}).reset_index().sort_values('weight', ascending=False)
    ao_mie_weight.columns = ['ao','weight','relevant_predicted_properties']

    # Get URI titles
    title_pred = "<http://purl.org/dc/elements/1.1/title>"
    uri_titles = sparql.Query(pdaa_graph, pdaa_graph_cache).select('uri', 'title').where(f'?uri {title_pred} ?title').execute()

    results = ao_mie_weight.merge(uri_titles, left_on='ao', right_on='uri', how='inner')
    ao_results = results[['ao','title','weight','relevant_predicted_properties']]

    return ao_results

def get_predicted_properties_for_mies(chemical_inchi, mie_uris):
    predicted_property_identifiers = proptoken_uris['uri'].unique()
    mie_proptoken_id_simtable = get_uri_similars(mie_uris, predicted_property_identifiers).query('similarity > 0.4')
    mie_proptoken_id_simtable.columns = ['mie_uri','property_token_id_uri','similarity']
    
    property_token_ids = mie_proptoken_id_simtable['property_token_id_uri'].unique()
    predictions_df = predict_predicted_property_uris(chemical_inchi, property_token_ids)[['uri','prediction']]
    predictions_df = predictions_df.query('prediction > 0.8')[['uri','prediction']]
    predictions_df = predictions_df.merge(proptoken_uris, on='uri', how='inner')[['uri','proptoken','prediction']]
    predictions_df.columns = ['property_token_id_uri','predicted_property_uri','prediction']

    title_pred = "<http://purl.org/dc/elements/1.1/title>"
    uri_titles = sparql.Query(pdaa_graph, pdaa_graph_cache).select('uri', 'title').where(f'?uri {title_pred} ?title').execute()
    
    def mktitle_col(df,left,colname):
        df = df.merge(uri_titles, left_on=left, right_on='uri', how='inner')
        df = df.drop('uri', axis=1)
        return df.rename(columns={'title':colname})

    results_df = mie_proptoken_id_simtable.merge(predictions_df, on='property_token_id_uri', how='inner')
    results_df = mktitle_col(results_df, 'mie_uri', 'mie_title')
    results_df = mktitle_col(results_df, 'predicted_property_uri', 'predicted_property_title')
    
    results_df['weight'] = results_df['similarity'] * results_df['prediction']
    return results_df[['mie_uri','predicted_property_uri','mie_title','predicted_property_title','weight']]

def get_ao_mies(ao_uri):
    return sparql.Query(pdaa_graph, pdaa_graph_cache) \
        .select('mie') \
        .where(f'?aop aop:has_adverse_outcome <{ao_uri}>') \
        .where('?aop aop:has_molecular_initiating_event ?mie') \
        .cache_execute()['mie'].values

def get_uri_title(uri : URIRef):
    uri = URIRef(uri) if isinstance(uri, str) else uri
    title_pred = "<http://purl.org/dc/elements/1.1/title>"
    res = sparql.Query(pdaa_graph, pdaa_graph_cache).select('title').where(f'<{uri}> {title_pred} ?title').execute()
    return res['title'].values[0]

# this 
# 1. takes a chemical
# 2. takes an adverse outcome
# 3. gets all the ao mies
# 4. gets all the mie properties
# 5. gets the chemical properties
# 6. predicts the chemical properties
# 7. weights the mie properties by the sum of the predicted chemical properties
def get_mie_weights(mie_uris, chemical_inchi):    
    mie_properties = get_predicted_properties_for_mies(chemical_inchi, mie_uris)
    mie_weight = mie_properties.groupby(['mie_uri','mie_title'])['weight'].sum().reset_index()
    mie_weight.columns = ['mie_uri','mie_title','weight']
    
    # Add missing MIEs with weight 0
    missing_mies = set(mie_uris) - set(mie_weight['mie_uri'])
    if len(missing_mies) > 0:
        missing_rows = pd.DataFrame({
            'mie_uri': list(missing_mies),
            'mie_title': [get_uri_title(uri) for uri in missing_mies], 
            'weight': [0.0] * len(missing_mies)
        })
        mie_weight = pd.concat([mie_weight, missing_rows], ignore_index=True)
    
    # Create a mapping from mie_uri to its original position in mie_uris
    mie_order = {uri: idx for idx, uri in enumerate(mie_uris)}
    
    # Add a column for sorting based on original order
    mie_weight['sort_order'] = mie_weight['mie_uri'].map(mie_order)
    
    # Sort by the original order and drop the helper column
    mie_weight = mie_weight.sort_values('sort_order').drop('sort_order', axis=1)

    return mie_weight

# chemical_inchi = pubchem.lookup_chemical_inchi('dehp')
# ao_uri = URIRef("https://identifiers.org/aop.events/406")
# mie_uris = pdaa.aop_mie_ao[pdaa.aop_mie_ao['ao'].isin([ao_uri])]['mie'].values


# Central registry of phthalate SMARTS patterns
SMARTS_PATTERNS = {
    # Dialkyl/diaryl di-esters of ortho-phthalic acid
    "diester": "[cH][cH]c(C(=O)OC[CH2,CH,C])c(C(=O)OC[CH2,CH,C])[cH][cH]",
    # Any 1,2-phthalate (acid, mono-, or di-ester, R = H or any group)
    "ortho":   "[cH][cH]c(C(=O)O[*])c(C(=O)O[*])[cH][cH]",
}

# Pre-compile once at import time
COMPILED_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS_PATTERNS.items()}

def is_phthalate(mol, modes=("any",), check_elements=True, valid_num_rings=[1]):
    """
    Return True if *mol* matches any phthalate class named in *modes*.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
    modes : str | Iterable[str]
        Allowed keys: "diester", "ortho", "any".
        "any" is equivalent to {"diester", "ortho"}.
    check_elements : bool
        If True, check that all atoms are C, H, or O.
    valid_num_rings : list[int] | None
        If not None, check that the number of rings in the molecule is in this list.

    Notes
    -----
    • Only-C/H/O atoms and exactly one ring are required.
    • The molecule is H-added internally because the SMARTS use [cH].
    """
    # Empty or None?
    if (mol is None) or (not isinstance(mol, Chem.Mol)):
        return False

    # Normalize modes -> tuple
    if isinstance(modes, str) or not isinstance(modes, Iterable):
        modes = (modes,)

    # Structural guards
    if check_elements and any(a.GetSymbol() not in ("C", "H", "O") for a in mol.GetAtoms()):
        return False
    if (valid_num_rings is not None) and (mol.GetRingInfo().NumRings() not in valid_num_rings):
        return False

    # Evaluate each pattern once
    mol_h = Chem.AddHs(mol)
    matches = {
        name: mol_h.HasSubstructMatch(pat)
        for name, pat in COMPILED_PATTERNS.items()
    }
    matches["any"] = any(matches.values())

    # Decide by requested modes
    for key in modes:
        if key not in matches:
            raise ValueError(f"Unknown mode: {key!r}")
        if matches[key]:
            return True
    return False