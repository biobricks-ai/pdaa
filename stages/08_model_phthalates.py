import sys
import threading
import random
sys.path.append('./')
import stages.utils.chemprop as chemprop
import asyncio
import pandas as pd
import pathlib
import itertools as it
import biobricks as bb
import sqlite3
import random

from tqdm.asyncio import tqdm as tqdm_async, tqdm_asyncio
from tqdm import tqdm

from rdkit import Chem
from collections.abc import Iterable

from stages.utils.pdaa import is_phthalate

async def async_predict(inchi: str, tok: str, sem: asyncio.Semaphore):
    async with sem:
        resp = await chemprop.get_chemprop_prediction_async(
            inchi=inchi, property_token=tok
        )
        if resp['error']:
            raise RuntimeError(resp['error'])

        # chemprop returns {'value': float, ...}
        score = float(resp['result']['value'])
        return inchi, tok, score          # <- scalar, not dict
        
def add_predictions(predictions, lock):
    with lock:
        with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
            conn.executemany(
                'INSERT OR IGNORE INTO predictions '
                '(inchi, property_token, positive_prediction) '
                'VALUES (?, ?, ?)',
                predictions
            )

# SETUP PATHS AND CACHES ========================================================
brickdir = pathlib.Path('brick')
brickdir.mkdir(parents=True, exist_ok=True)
DB_PATH  = brickdir / "predictions.sqlite"
cachedir = pathlib.Path('cache/model_phthalates')
cachedir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DATABASE INITIALISATION (run once per process)
# ---------------------------------------------------------------------------
def init_db():
    brickdir.mkdir(parents=True, exist_ok=True)          # Folder must exist
    db_path = brickdir / 'predictions.sqlite'
    with sqlite3.connect(db_path) as conn:
        conn.execute('PRAGMA journal_mode=WAL;')          # Better concurrency
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                inchi            TEXT NOT NULL,
                property_token   TEXT NOT NULL,
                positive_prediction REAL,
                PRIMARY KEY (inchi, property_token)       -- prevent duplicates
            );
        ''')
init_db()

# LOAD CHEMPROP-TRANSFORMER PROPERTY TOKENS =====================================
with sqlite3.connect(bb.assets('chemprop-transformer').cvae_sqlite) as con:
    property_tokens = pd.read_sql_query("SELECT property_token FROM property", con)['property_token'].tolist()
    property_tokens = sorted(list(set(property_tokens)))

# LOAD ZINC PHTALATES DATA =====================================================
raw_df = pd.read_parquet('cache/zinc_phthalates/zinc_phthalates.parquet')
# raw_df = pd.read_parquet('cache/priority_phthalates/priority_phthalates.parquet')
# top_df = raw_df.sort_values(by='max_similarity', ascending=False)[['inchi', 'max_similarity']].drop_duplicates()
# inchi_list = top_df['inchi'].unique().tolist()
inchi_list = [
    inch for inch in raw_df['inchi'].unique()
    if (mol := Chem.MolFromInchi(inch)) is not None
    and is_phthalate(mol, modes=("ortho_phthalate", "meta_phthalate", "para_phthalate",), check_elements=True)  # isomer filter
]

# save the InChIs to a file for later use
inchi_file = cachedir / 'phthalates_inchi.txt'
with open(inchi_file, 'w') as f:
    for inchi in inchi_list:
        f.write(f"{inchi}\n")

# save the SMILES to a file for later use
smiles_file = cachedir / 'phthalates_smiles.txt'
with open(smiles_file, 'w') as f:
    for inchi in inchi_list:
        # Find SMILES corresponding to InChI in top_df (or raw_df)
        smiles = raw_df.loc[raw_df['inchi'] == inchi, 'smiles'].values
        if len(smiles) > 0:
            f.write(f"{smiles[0]}\n")
        else:
            raise ValueError(f"No SMILES found for InChI: {inchi}")

# BATCH RUN ==============================================================
def get_missing(inchi_tok_pairs):
    with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
        results = []
        for inchi, property_token in inchi_tok_pairs:
            cursor = conn.execute('SELECT * FROM predictions WHERE inchi = ? AND property_token = ?', (inchi, property_token))
            exists = cursor.fetchone() is not None
            if not exists:
                results.append((inchi, property_token))
    return results

# NEW --------------- helper to write a whole dict for one inchi
def add_prediction_dict(inchi, predictions_dict, lock):
    rows = [
        (inchi, tok, score)
        for tok, score in predictions_dict.items()
    ]
    with lock, sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            'INSERT OR REPLACE INTO predictions VALUES (?,?,?)',
            rows
        )
        conn.commit()

# NEW --------------- async wrapper around /predict_all
async def async_predict_all(inchi, semaphore):
    async with semaphore:
        resp = await chemprop.chemprop_predict_all_async(inchi)
        # The endpoint returns a list of {"property_token": "...", "positive_prediction": float}
        # return inchi, {d["property_token"]: d["positive_prediction"] for d in resp}
        return inchi, {d["property_token"]: d["value"] for d in resp}

# MAIN -------------- outer loop now one call per inchi
async def process():
    semaphore    = asyncio.Semaphore(40)
    lock         = threading.Lock()
    inchi_queue  = inchi_list[:]          # shuffle for load-balancing
    random.shuffle(inchi_queue)

    for inchi in tqdm(inchi_queue, desc="InChI processed", unit="inchi"):
        # skip if we already have every token for this inchi
        with sqlite3.connect(DB_PATH) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE inchi = ?", (inchi,)
            ).fetchone()[0]
        if count == len(property_tokens):
            continue                        # nothing missing → next inchi

        # launch async prediction
        inchi_, pred_dict = await async_predict_all(inchi, semaphore)
        add_prediction_dict(inchi_, pred_dict, lock)

asyncio.run(process())
