# from math import ceil
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

from tenacity import RetryError
import random

resourcedir = Path('resources')
cachedir = Path('cache/edkb')
edkb_parquet = resourcedir / 'edkb_full.parquet'
edkb = pd.read_parquet(edkb_parquet)

# performs predictions and saves them to a SQLite database (no return value needed)
# chunk size is set to 100 to output rows to the SQLite database more frequently
def get_chunks(iterable, chunk_size=100):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]

failed_inchis_file = cachedir / 'failed_inchis.txt'
if failed_inchis_file.exists():
    print(f"Found {failed_inchis_file} with failed InChIs. Loading them...")
    with open(failed_inchis_file) as f:
        inchi_list = [line.strip() for line in f if line.strip()]

    chunk_size = 1  # retry each InChI individually

else:
    chunk_size = 16
    inchi_list = edkb['inchi'].unique().tolist()

# remove any None values from the list
inchi_list = [inchi for inchi in inchi_list if inchi != 'None']
# shuffle the list to see if specific InChIs are causing issues
# this is useful for debugging, but can be removed in production
random.shuffle(inchi_list)

# retry loop because the first run may fail due to a timeout
# import time
# max_retries = 15
# while True:
#     random.shuffle(inchi_list)
#     try:
#         with tqdm(total=len(inchi_list), desc="Predicting properties") as pbar:
#             # for chunk in tqdm(get_chunks(inchi_list, chunk_size), desc="Predicting properties in chunks", total=ceil(len(inchi_list)/chunk_size)):
#             for chunk in get_chunks(inchi_list, chunk_size):
#                 # pdaa.predict_all_properties_with_sqlite_cache(chunk)
#                 pdaa.predict_all_properties_with_sqlite_cache_parallel(chunk)
#                 pbar.update(len(chunk))  # usually chunk_size, but can be less for the last chunk
#         break
#     except RetryError as e:
#         max_retries -= 1
#         if max_retries <= 0:
#             print("Max retries reached. Exiting.")
#             raise e
#         print(f"Retrying due to error: {e}\n\n\n\n")
#         time.sleep(5)  # wait before retrying


failed_inchis = []
with tqdm(total=len(inchi_list), desc="Predicting properties") as pbar:
    # for chunk in tqdm(get_chunks(inchi_list, chunk_size), desc="Predicting properties in chunks", total=ceil(len(inchi_list)/chunk_size)):
        for chunk in get_chunks(inchi_list, chunk_size):
            try:
                # pdaa.predict_all_properties_with_sqlite_cache_parallel(chunk)
                pdaa.predict_all_properties_with_sqlite_cache(chunk)
            except RetryError as e:
                print(f"RetryError: {e}. Skipping chunk.")
                failed_inchis.extend(chunk)
                
            pbar.update(len(chunk))  # usually chunk_size, but may be less for the last chunk
            
if failed_inchis:
    print(f"Failed to predict properties for {len(failed_inchis)} InChIs. Saving to 'failed_inchis.txt'.")
    with open(cachedir / 'failed_inchis.txt', 'w') as f:
        for inchi in failed_inchis:
            f.write(f"{inchi}\n")
elif failed_inchis_file.exists():
    print(f"Deleting {failed_inchis_file} as no failed InChIs were found.")
    failed_inchis_file.unlink()
1
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

    # def in_edkb(m):
    #     return any(edkb.inchi.isin([Chem.MolToInchi(m)]))
    def in_edkb(inchi):
        return any(edkb.inchi.isin([inchi]))

    print("Filtering for EDKB substances...")
    # filtered_substances = inchi_mol_df[inchi_mol_df['mol'].progress_apply(in_edkb)]['inchi']
    filtered_substances = inchi_mol_df[inchi_mol_df['inchi'].progress_apply(in_edkb)]['inchi']
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