import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support
import os
import glob
import time

tqdm.pandas()

def formatTimeElapsed(elapsed):
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = "{:02} hours {:02} minutes {:02} seconds".format(int(hours), int(minutes), int(seconds))
    return formatted_time

def isPhthalate(smiles):
    phthalatePattern = 'OC(=O)C1=CC=CC=C1C(=O)O'
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmiles(phthalatePattern)
    match = mol.HasSubstructMatch(pattern)
    return match

def processFile(path):
    start_time = time.time()
    print('filtering {}'.format(path))
    df = pd.read_parquet(path)
    df['phthalate'] = df['smiles'].apply(isPhthalate)
    df_filtered = df[df['phthalate']]
    output_path = path.replace('raw/zinc.parquet/', 'raw/zinc_phthalates/')
    
    nPhthalates = len(list(df_filtered['smiles']))
    
    df_filtered.to_parquet(output_path, index=False)
    
    elapsed_time = time.time() - start_time
    elapsed_time = formatTimeElapsed(elapsed_time)
    
    print('saved {} phthalates to {} in {}'.format(nPhthalates, output_path, elapsed_time))
    
def processInParallel(file_paths, n=16):
    print('stating pooling over {} cores.'.format(n))
    with Pool(n) as pool:
        list(tqdm(pool.imap(processFile, file_paths), total=len(file_paths)))
    print('parallel processing done.')
    
if __name__ == "__main__":
    freeze_support()
    
    os.makedirs('raw/zinc_phthalates', exist_ok=True)
    file_paths = glob.glob('raw/zinc.parquet/*.parquet')
    
    processInParallel(file_paths)
