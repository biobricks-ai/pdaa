import sys
import sqlite3 
import pathlib
import threading
sys.path.append('.')
import stages.utils.chemprop as chemprop
import faiss
import rdflib
import pandas as pd
from tqdm import tqdm
from rdflib import URIRef
import asyncio

brickdir = pathlib.Path('brick')
sqlite_lock = threading.Lock()

# uri faiss index
faiss_index = faiss.read_index('faiss_index_cosine.index')

# Load the similarity graph created in stage 10
simgraph = rdflib.Graph()
simgraph.parse('cache/associate_properties_with_aopwiki/simgraph.nt', format='nt')

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

async def async_predict(inchi,tok,semaphore):
    async with semaphore:
        result = await chemprop.get_chemprop_prediction_async(inchi=inchi, property_token=tok)
        if result['error'] is not None: raise Exception(f"Error: {result['error']}")
        return result['result']

def get_predictions_with_sqlite_cache(inchi_tok_pairs) -> dict:
    predictions = lookup_predictions(inchi_tok_pairs)
    inchi_tok_pred_tuples = [(p[0], p[1]) for p in predictions]
    remaining_inchi_tok_pairs = [(i, t) for i, t in inchi_tok_pairs if (i, t) not in inchi_tok_pred_tuples]
    for inchi, tok in tqdm(remaining_inchi_tok_pairs):
        result = chemprop.get_chemprop_prediction(inchi=inchi, property_token=tok)
        add_predictions([(inchi, tok, result['value'])], sqlite_lock)
        predictions.append((inchi, tok, result['value']))
    return predictions

def faiss_index_to_uri(index):
    return URIRef(f"toxindex:faiss_index{index}")

def property_token_to_uri(property_token):
    return URIRef(f"toxindex:property_token{property_token}")

# sparql queries

aop_graph = rdflib.Graph()
aop_graph.namespace_manager.bind('aop', rdflib.Namespace('http://aopkb.org/aop_ontology#'))
aop_graph.namespace_manager.bind('dcterms', rdflib.Namespace('http://purl.org/dc/terms/'))
aop_graph.parse('./hdtworkdir/out.nt', format='nt')
def aopwiki_query(query):
    graph_result = aop_graph.query(query)
    res = pd.DataFrame(graph_result.bindings).map(str)
    res.columns = [str(c) for c in res.columns]
    return res

pdaa_graph = rdflib.Graph()
pdaa_graph.parse('cache/associate_properties_with_aopwiki/simgraph.nt')
pdaa_graph.namespace_manager.bind('toxindex', rdflib.Namespace('http://toxindex.com/ontology/'))
pdaa_graph.namespace_manager.bind('EDAM', rdflib.Namespace('http://edamontology.org/'))
pdaa_graph.namespace_manager.bind('aop', rdflib.Namespace('http://aopkb.org/aop_ontology#'))
pdaa_graph.namespace_manager.bind('dcterms', rdflib.Namespace('http://purl.org/dc/terms/'))

def pdaa_query(query):
    graph_result = pdaa_graph.query(query)
    res = pd.DataFrame(graph_result.bindings).map(str)
    res.columns = [str(c) for c in res.columns]
    return res