"""
Add a set of phthalates to the SQLite database.
This script:
    1) reads a text file containing phthalate SMILES,
    2) extracts InChI strings and performs ToxTransformer predictions,
"""

import sys
from pathlib import Path
# import pandas as pd
from rdkit import Chem

sys.path.append('./')
from scripts.utils.helpers import smiles_to_inchi
from stages.utils.pdaa import predict_all_properties_with_sqlite_cache



infile = Path("resources/new_phthalates.txt")  # Path to the input file with phthalate SMILES
# Read the SMILES from the input file
smiles_list = infile.read_text().splitlines()
# Convert SMILES to InChI
inchi_list = [smiles_to_inchi(smiles) for smiles in smiles_list]

predict_all_properties_with_sqlite_cache(inchi_list)