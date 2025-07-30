import numpy as np
import pandas as pd
import biobricks as bb
from pathlib import Path
from tqdm import tqdm
from rdkit.Chem import AllChem

import sys
sys.path.append('./')
# from scripts.utils.helpers import smiles_to_inchi

tqdm.pandas()

bindingdb_parquet = bb.assets('bindingdb').full_tsv_dump_parquet
df = pd.read_parquet(bindingdb_parquet)
subset = df[[
    'Ligand InChI',
    'Ligand SMILES',
    'Ki (nM)',
    'IC50 (nM)',
    'Kd (nM)',
    'EC50 (nM)',
]].copy()
# rename columns
subset.rename(columns={
    'Ligand InChI': 'inchi',
    'Ligand SMILES': 'smiles',
    'Ki (nM)': 'pKi',
    'IC50 (nM)': 'pIC50',
    'Kd (nM)': 'pKd',
    'EC50 (nM)': 'pEC50',
}, inplace=True)

def nM_to_p(nM):
    """Convert nM to pX value."""
    if pd.isna(nM) or nM <= 0:
        return None
    return -np.log10(1e-9*nM)

# Convert all activity values to pX values
for col in ['pKi', 'pIC50', 'pKd', 'pEC50']:
    subset[col] = pd.to_numeric(subset[col], errors='coerce').apply(nM_to_p)

# set index as InChI
subset.set_index('inchi', inplace=True)

# save the subset with InChI to a new parquet file
outdir = Path('cache/bindingdb')
outdir.mkdir(parents=True, exist_ok=True)
subset.to_parquet(outdir / 'bindingdb_subset.parquet')