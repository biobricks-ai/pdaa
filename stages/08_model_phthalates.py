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
# inchi_list = top_df['inchi'].unique().tolist()
inchi_list = [
    inch for inch in top_df['inchi'].unique()
    if (mol := Chem.MolFromInchi(inch)) is not None
    and is_phthalate(mol, modes=("ortho",), check_elements=False)  # structure filter
]

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
