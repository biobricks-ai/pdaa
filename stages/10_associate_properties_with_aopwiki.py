import sys
sys.path.append('./')
import stages.utils.simple_cache as simple_cache
import stages.utils.openai as openai_utils
import stages.utils.pdaa as pdaa

import json
import faiss
import shutil
import rdflib
import dotenv
import openai
import pathlib
import functools
import biobricks
import subprocess
import numpy as np
import pandas as pd
import itertools as it

from tqdm import tqdm

cachedir = pathlib.Path('cache/associate_properties_with_aopwiki')
cachedir.mkdir(parents=True, exist_ok=True)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 80)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 80)

tqdm.pandas()

# outs: [aop_titles, aop_descriptions, aop_abstracts, key_events, membership, graph]
# region GET AOPWIKI KEY EVENTS AND ADVERSE OUTCOME PATHWAYS ==========================================

## BUILD AOPWIKI RDF ========================================================
sr = functools.partial(subprocess.run, shell=True)
sr("git clone https://github.com/rdfhdt/hdt-cpp.git")
sr('docker build -t hdt hdt-cpp/.')

# start the container
hdtworkdir = pathlib.Path('./hdtworkdir')
hdtworkdir.exists() and shutil.rmtree(hdtworkdir)
hdtworkdir.mkdir(parents=True, exist_ok=True)

sr(f'docker rm -f hdt || true')
sr(f'docker run -d --name hdt -v $(pwd)/hdtworkdir:/workdir hdt tail -f /dev/null')

# ADD AOPWikiRDF-Genes.hdt AOPWikiRDF.hdt TO DOCKER CONTAINER
aopwiki = biobricks.Brick.Resolve('aopwikirdf-kg').path() / 'brick'
for hdt_file in aopwiki.glob('*.hdt'):
    sr(f'docker cp {hdt_file.resolve()} hdt:/workdir/{hdt_file.name}')

## QUERY AOPWIKI RDF ========================================================
hdtcmd = lambda cmd: sr(f'docker exec -it hdt /bin/bash -c "{cmd}"')
hdtcmd(f"hdt2rdf /workdir/AOPWikiRDF.hdt /workdir/out.nt")

graph = rdflib.Graph()
graph.namespace_manager.bind('aop', rdflib.Namespace('http://aopkb.org/aop_ontology#'))
graph.namespace_manager.bind('dcterms', rdflib.Namespace('http://purl.org/dc/terms/'))
graph.parse('./hdtworkdir/out.nt', format='nt')
def gquery(q):
    res = pd.DataFrame(graph.query(q).bindings).map(str)
    res.columns = [str(c) for c in res.columns]
    return res

# get KeyEvent labels and descriptions
key_events = gquery("""
SELECT ?key_event ?variable ?text WHERE {
    ?key_event a aop:KeyEvent .
    ?key_event ?variable ?text .
    FILTER (?variable = dc:title || ?variable = rdfs:label || ?variable = dc:description)
}
""")
key_events['variable'].value_counts()

aops = gquery("""
SELECT ?aop ?variable ?text WHERE {
    ?aop a aop:AdverseOutcomePathway .
    ?aop ?variable ?text .
    FILTER (?variable = dc:title || ?variable = dc:description || ?variable = dcterms:abstract)
}
""")

membership = gquery("""
SELECT ?key_event ?aop WHERE {
    ?key_event dcterms:isPartOf ?aop .
    ?key_event a aop:KeyEvent .
    ?aop a aop:AdverseOutcomePathway .
} limit 5
""")
# endregion

# TODO: we need better URIs for bindingdb
# outs: 
#   - propvars - property_token, title, data, source
#   - proptokens - uri, property_token
# region GET CHEMPROP-TRANSFORMER PROPERTIES ========================================================
import sqlite3
with sqlite3.connect(biobricks.assets('chemprop-transformer').cvae_sqlite) as con:
    rawprops = pd.read_sql_query("SELECT property_token, title, data, s.source FROM property p INNER JOIN source s ON p.source_id = s.source_id", con)
    rawprops['property_token'] = rawprops['property_token'].astype(int)
    rawprops['data'] = rawprops['data'].map(lambda x: json.loads(x))

rawprops['source'].value_counts()

ctbindingdb = rawprops[rawprops['source'] == 'bindingdb'].reset_index()
ctbindingdb['uri'] = ctbindingdb['data'].progress_apply(lambda x: x['Link to Target in BindingDB'])

ctpubchem = rawprops[rawprops['source'] == 'pubchem'].reset_index()
mkaid = lambda aid: f"https://identifiers.org/pubchem.bioassay:{aid}"
ctpubchem['uri'] = ctpubchem['data'].progress_apply(lambda x: mkaid(int(x['aid'])))

ctchembl = rawprops[rawprops['source'] == 'chembl'].reset_index()
mkassayid = lambda aid: f"https://identifiers.org/chembl.target:{aid}"
ctchembl['uri'] = ctchembl['data'].progress_apply(lambda x: mkassayid(x['assay_id']))

ctprops = pd.concat([ctbindingdb, ctpubchem, ctchembl], ignore_index=True)
ctprops['data'] = ctprops['data'].map(lambda x: json.dumps(x))

# associate uris with property_token
proptokens = ctprops[['uri','property_token']].drop_duplicates()

# build text values
propvars = ctprops[['uri','title','data']]
propvars = pd.melt(ctprops, id_vars=['uri'], value_vars=['title','data'], var_name='variable', value_name='value')
propvars = propvars.drop_duplicates()

# endregion

# region ASSOCIATE PROPERTIES WITH AOPWIKI PATHWAYS ========================================================
uri_aops = aops.rename(columns={'aop': 'uri', 'text': 'value'}) 
uri_key_events = key_events.rename(columns={'key_event': 'uri', 'text': 'value'})

embed_df = pd.concat([propvars, uri_aops, uri_key_events], ignore_index=True)
embed_df = embed_df[['uri','variable','value']]
embed_df = embed_df.dropna().drop_duplicates()

truncate = lambda text: text[:10000] if len(text) > 10000 else text
embed_df['embedding'] = embed_df['value'].apply(truncate).progress_apply(openai_utils.embed)

# Normalize embeddings for cosine similarity
embeddings = np.vstack(embed_df['embedding'].values)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Create and save FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss_index_cosine.index")

# endregion

# region CREATE RDF ========================================================
# write a new ntriples that adds:
# 1. ctprops and their property_token,source, title, and data
# 2. similarity entities that link ctprops and key events
from rdflib import Literal, URIRef, XSD, RDFS, RDF

EDAM = rdflib.Namespace('http://edamontology.org/')

simgraph = rdflib.Graph()
simgraph.namespace_manager.bind('aop', rdflib.Namespace('http://aopkb.org/aop_ontology#'))
simgraph.namespace_manager.bind('dcterms', rdflib.Namespace('http://purl.org/dc/terms/'))
simgraph.namespace_manager.bind('toxindex', rdflib.Namespace('http://toxindex.com/ontology/'))

faissclass = URIRef('http://toxindex.com/ontology/faiss_index')
tox_property_class = URIRef('http://toxindex.com/ontology/property')

# link uris to toxindex property tokens
for ind, uri, property_token in proptokens.itertuples():
    uri = URIRef(uri)
    token_uri = pdaa.property_token_to_uri(property_token)
    _ = simgraph.add((uri, EDAM.term('has_identifier'), token_uri))
    _ = simgraph.add((token_uri, RDFS.label, Literal(int(property_token), datatype=XSD.integer)))
    _ = simgraph.add((token_uri, RDF.type, tox_property_class))

print(f"there are {len(simgraph)} triples in the graph")

# link uris to faiss index
for i, uri in enumerate(embed_df['uri']):
    faiss_token = pdaa.faiss_index_to_uri(i)
    faiss_value = Literal(int(i),datatype=XSD.integer)
    _ = simgraph.add((URIRef(uri), EDAM.term('has_identifier'), faiss_token))
    _ = simgraph.add((faiss_token, RDFS.label, faiss_value))
    _ = simgraph.add((faiss_token, RDF.type, faissclass))

print(f"there are {len(simgraph)} triples in the graph")

# Commit the changes to the graph
simgraph.commit()
simgraph.serialize(destination=cachedir / 'simgraph.nt', format='nt')
# endregion