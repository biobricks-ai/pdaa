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
# raw_df = pd.read_parquet('cache/zinc_phthalates/zinc_phthalates.parquet')
raw_df = pd.read_parquet('cache/priority_phthalates/priority_phthalates.parquet')
top_df = raw_df.sort_values(by='max_similarity', ascending=False)[['inchi', 'max_similarity']].drop_duplicates()
inchi_list = top_df['inchi'].unique().tolist()

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

# BUILD PREDICTION FUNCTION =====================================================

# async def process_batches():
#     BATCH_SIZE = 100000
#     num_combinations = len(inchi_list) * len(property_tokens)
#     rand_inchi, rand_tok = random.sample(inchi_list, len(inchi_list)), random.sample(property_tokens, len(property_tokens))
#     tuple_generator = it.product(rand_inchi, rand_tok)
#     batch_generator = it.batched(tuple_generator, BATCH_SIZE)
#     num_batches = num_combinations // BATCH_SIZE
    
#     sqlite_lock = threading.Lock()
#     semaphore = asyncio.Semaphore(40)

#     for batch in tqdm(batch_generator, total=num_batches, desc="Processing batches"):
#         new_inchi_tok_pairs = get_missing(batch)
#         print(f"new inchi-tok pairs: {len(new_inchi_tok_pairs)}")
#         # predictions = [pdaa.async_predict(inchi, tok, semaphore) for inchi, tok in new_inchi_tok_pairs]
#         predictions = [async_predict(inchi, tok, semaphore) for inchi, tok in new_inchi_tok_pairs]
#         results = await tqdm_asyncio.gather(*predictions, desc="predicting...")
#         # pdaa.add_predictions(results, sqlite_lock)
#         add_predictions(results, sqlite_lock)
#         print(f"Processed batch: {len(results)}")
    
# asyncio.run(process_batches())

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
