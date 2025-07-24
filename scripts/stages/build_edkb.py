import sys
from pathlib import Path
import shutil
from pyspark.sql import SparkSession
import biobricks as bb
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
sys.path.append('./')

outdir = Path('cache/zinc_phthalates')
tempdir = outdir / 'temp'
temp_edkb = tempdir / 'temp_edkb'

# process each file from partitioned zinc 
# output to `temp_edkb`
def process_file_with_index(index_parquet_file_tuple):
    index, parquet_file = index_parquet_file_tuple
    outfile = temp_edkb / f'phthalates_{index}.parquet'

    try:
        df = pd.read_parquet(parquet_file)
        # substructure = Chem.MolFromSmiles('OC(=O)C1=CC=CC=C1C(=O)O')

        # Filter for phthalates
        df['phthalate'] = df['smiles'].apply(
            lambda x: smiles_is_phthalate(x)
        )
        phthalates_df = df[df['phthalate'] == True]  # Explicit boolean comparison
        
        # Save intermediate results if we found any phthalates
        if len(phthalates_df) > 0:
            phthalates_df.to_parquet(outfile)
    except Exception as e:
        print(f"Error processing file {parquet_file}: {str(e)}")


files = list((tempdir / 'zinc_partitioned').glob('*.parquet'))
with ProcessPoolExecutor(max_workers=30) as executor:
    file_tuples = list(enumerate(files))
    list(tqdm(
        executor.map(process_file_with_index, file_tuples),
        total=len(files),
        desc='Processing files'
    ))

# Combine all results
phthalates_files = list(temp_edkb.glob('*.parquet'))
final_df = pd.concat([pd.read_parquet(f) for f in phthalates_files])

# generate inchi for all phthalates
from tqdm.notebook import tqdm
tqdm.pandas()
final_df['inchi'] = final_df['smiles'].progress_apply(lambda x: Chem.MolToInchi(Chem.MolFromSmiles(x)) if pd.notnull(x) else None)

final_df.to_parquet(outdir / 'zinc_edkb.parquet')
