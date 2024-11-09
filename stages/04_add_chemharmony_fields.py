import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support
import os
import glob
import time

import biobricks as bb

tqdm.pandas()

def formatTimeElapsed(elapsed):
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = "{:02} hours {:02} minutes {:02} seconds".format(int(hours), int(minutes), int(seconds))
    return formatted_time

def harmonizeFile(args):
    path, chemHarmony = args
    
    start_time = time.time()
    print('harmonizing {}'.format(path))
    
    df = pd.read_parquet(path)
    harmonized = pd.merge(chemHarmony, df, on='smiles')
    output_path = path.replace('raw/zinc_phthalates/', 'raw/harmonized_phthalates/')
    harmonized.to_parquet(output_path, index=False)
    
    elapsed_time = time.time() - start_time
    elapsed_time = formatTimeElapsed(elapsed_time)
    
    print('saved harmonized phthalates to {} in {}'.format(output_path, elapsed_time))
    

def processInParallel(args, n=16):
    print('stating pooling over {} cores.'.format(n))
    with Pool(n) as pool:
        list(tqdm(pool.imap(harmonizeFile, args), total=len(file_paths)))
    print('parallel processing done.')

if __name__ == "__main__":
    freeze_support()
    # bb.install('chemharmony')
    chemharmony = bb.assets('chemharmony')
    chemharmony = pd.read_parquet(chemharmony.activities_parquet)
    
    os.makedirs('raw/harmonized_phthalates', exist_ok=True)
    file_paths = glob.glob('raw/zinc_phthalates/*.parquet')
    
    args = [(path, chemharmony) for path in file_paths]
    processInParallel(args)