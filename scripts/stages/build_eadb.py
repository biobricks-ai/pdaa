import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rdkit.Chem import AllChem
import cirpy

import sys
sys.path.append('./')
# from scripts.utils.helpers import smiles_to_inchi

tqdm.pandas()

resourcedir = Path('resources')
df = pd.read_csv(resourcedir / 'EADB.csv')

# have to get an InChI for each chemical name
unique_names = df['Name'].unique()
name_to_inchi = dict()
for name in tqdm(unique_names, desc="Converting names to InChI"):
    # skipping duplicates for subsequent calls
    if (name in name_to_inchi.keys()) and (name_to_inchi[name] is not None):
        continue
    try:
        name_to_inchi[name] = cirpy.resolve(name, 'inchi')
    except Exception as e:
        print(f"Error fetching InChI for {name}: {e}")
        # name_to_inchi[name] = None
        # try using the Cas Decimal instead
        cas = df.loc[df['Name'] == name, 'Cas Decimal'].values[0]
        if pd.notnull(cas):
            try:
                name_to_inchi[name] = cirpy.resolve(cas, 'inchi')
            except Exception as e:
                print(f"Error fetching InChI for CAS {cas}: {e}")
                name_to_inchi[name] = None

# map the InChI back to the original DataFrame
df['inchi'] = df['Name'].map(name_to_inchi)

# save the DataFrame with InChI to a new parquet file
outdir = Path('cache/eadb')
outdir.mkdir(parents=True, exist_ok=True)
df.to_parquet(outdir / 'eadb.parquet')