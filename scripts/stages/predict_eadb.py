import numpy as np
import pandas as pd

from pathlib import Path
from rdkit import Chem
import re
from tqdm import tqdm

import sys
sys.path.append('./')
import stages.utils.pdaa as pdaa

import stages.utils.sparql as sparql
import sqlite3

resourcedir = Path('resources')
cachedir = Path('cache/eadb')
eadb_parquet = cachedir / 'eadb.parquet'
eadb = pd.read_parquet(eadb_parquet)

# performs predictions and saves them to a SQLite database (no return value needed)
pdaa.predict_all_properties_with_sqlite_cache(eadb['inchi'].unique())

tqdm.pandas()

def build_substance_ice_activity_df(mask_method = 'prediction'):
    print("Building substance ICE activity dataframe...")
    
    print("Querying PDAA graph for URI, title and token mappings...")
    uri_title_token = sparql.Query(pdaa.pdaa_graph) \
        .select_typed({'uri': str, 'pp': str, 'title': str, 'token': int}) \
        .where('?pp a toxindex:predicted_property') \
        .where('?pp <http://purl.org/dc/elements/1.1/title> ?title') \
        .where('?pp rdf:value ?token') \
        .where('?pp <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
        .execute().groupby('uri').first().reset_index()

    if mask_method == 'list':
        dart_path = Path('resources/DART_endpoints.txt')

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

        use_dart = True
        use_ed = True

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
    brickdir = Path('brick')
    with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
        tokens = ','.join(map(str, ice_assays['token'].tolist()))
        query = f'SELECT * FROM predictions WHERE property_token IN ({tokens})'
        ice_preds = pd.read_sql(query, conn)

    print("Processing predictions data...")
    df = ice_preds.sort_values('positive_prediction', ascending=False)[['inchi', 'property_token', 'positive_prediction']]
    df = df.groupby(['inchi','property_token'])['positive_prediction'].mean().reset_index()

    inchi_mol_df = df[['inchi']].drop_duplicates()
    print("Converting InChIs to molecules...")
    inchi_mol_df['mol'] = inchi_mol_df['inchi'].progress_apply(lambda x: Chem.MolFromInchi(x))
    df2 = df.merge(inchi_mol_df, on='inchi')

    def in_eadb(m):
        return any(eadb.inchi.isin([Chem.MolToInchi(m)]))

    print("Filtering for EADB substances...")
    filtered_substances = inchi_mol_df[inchi_mol_df['mol'].progress_apply(in_eadb)]['inchi']
    df3 = df2[df2['inchi'].isin(filtered_substances)]
    # Ensure both columns are of the same type (int)
    df3.loc[:, 'property_token'] = df3['property_token'].astype(int)
    ice_assays.loc[:, 'token'] = ice_assays['token'].astype(int)
    df3 = df3.merge(ice_assays, left_on='property_token', right_on='token')[['uri','title','inchi','mol','positive_prediction']]
    print(f"Final dataset contains {len(df3)} rows")
    return df3


substance_df = build_substance_ice_activity_df()[['uri','title','inchi','mol','positive_prediction']]

def make_activity_matrix():
    activity_matrix = substance_df.groupby(['inchi','title'])['positive_prediction'].mean().reset_index()
    activity_matrix = activity_matrix.pivot(index='inchi', columns='title', values='positive_prediction')

    # Fill any NaN values with 0 for clustering
    activity_matrix_filled = activity_matrix.fillna(0)
    # save the filled activitiy matrix
    activity_matrix_filled.to_parquet(cachedir / 'activity_matrix_filled.parquet')
    return activity_matrix_filled

activity_matrix_filled = make_activity_matrix()