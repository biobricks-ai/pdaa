"""
Add a set of phthalates to the SQLite database.
This script:
    1) reads a CSV file containing phthalate SMILES,
    2) extracts InChI strings and performs ToxTransformer predictions,
"""

import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem

sys.path.append('./')
from scripts.utils.helpers import smiles_to_inchi
from stages.utils.pdaa import predict_all_properties_with_sqlite_cache

# Path to the input file with phthalate InChIs
# fname = "resources/example_phthalates.csv"
fname = "resources/long_chain_phthalates.csv"
infile = Path(fname)  

# Read the SMILES from the input file
# smiles_list = infile.read_text().splitlines()
df = pd.read_csv(infile)
smiles_list = df['smiles'].tolist()
# Convert SMILES to InChI
# inchi_list = [smiles_to_inchi(smiles) for smiles in smiles_list]
inchi_list = df['inchi'].to_list()

predict_all_properties_with_sqlite_cache(inchi_list)