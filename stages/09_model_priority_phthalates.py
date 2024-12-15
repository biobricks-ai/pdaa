import joblib as jl
import json
import time
import asyncio
import pandas as pd
import pathlib
import rdkit, rdkit.Chem
from itertools import product, islice
from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import biobricks as bb
import sqlite3

import sys
sys.path.append('./')
import stages.utils.chemprop as chemprop

tqdm.pandas()

# Setup paths and caches
cachedir = pathlib.Path('cache/model_priority_phthalates')
cachedir.mkdir(parents=True, exist_ok=True)

savedir = cachedir / 'tmp' / 'batches'
savedir.mkdir(parents=True, exist_ok=True)

cptransformer = bb.assets('chemprop-transformer').cvae_sqlite
cpsqlite = sqlite3.connect(cptransformer)
property_tokens = pd.read_sql_query("SELECT property_token FROM property", cpsqlite)['property_token'].tolist()
property_tokens = sorted(property_tokens)

raw_df = pd.read_parquet('cache/priority_phthalates/priority_phthalates.parquet')
top_df = raw_df.sort_values(by='max_similarity', ascending=False).head(100)
inchi_list = top_df['inchi'].tolist()

async def get_prediction(inchi,tok,semaphore):
    async with semaphore:
        return await chemprop.get_chemprop_prediction_async(inchi=inchi, property_token=tok)

async def run_model(combination_generator, num_combinations=100, semaphore_size=3):
    semaphore = asyncio.BoundedSemaphore(3)
    tasks = (get_prediction(inchi,tok,semaphore) for inchi, tok in combination_generator)
    results = []
    async for i, result in tqdm_async(enumerate(asyncio.as_completed(tasks)), total=num_combinations, desc="Processing"):
        results.append(await result)
    
    return results

num_combinations = len(inchi_list) * len(property_tokens)
total_batches = (num_combinations + 99999) // 100000
print(f"Processing {num_combinations} combinations in {total_batches} batches")

combination_generator = islice(product(inchi_list, property_tokens), 0, 5)
res = asyncio.run(run_model(combination_generator, 5))
print(res)

start_time = time.time()
for batch_idx, batch in enumerate(range(0, num_combinations, 100000)):
    batch_size = min(100000, num_combinations - batch)
    res = asyncio.run(run_model(islice(product(inchi_list, property_tokens), batch, batch + 100000), batch_size, semaphore_size=9))
    
    timestamp = int(time.time())
    with open(savedir / f'model_phthalates_{timestamp}.json', 'w') as f:
        json.dump(res, f)
    eta = (time.time() - start_time) / (batch + batch_size) * (num_combinations - batch - batch_size) / 3600
    print(f"Batch {batch//100000 + 1}/{total_batches}, ETA: {eta:.1f}h")


# Load previously processed results
def load(batch_file):
    with open(batch_file) as f:
        return json.load(f)

# count the number of errors
bf = list(savedir.glob('model_phthalates_*.json'))
res = [r['result'] for r in load(bf[0])]
errors = [r for r in load(bf[0]) if r['error'] is not None]
print(f"Number of errors: {len(errors)}")