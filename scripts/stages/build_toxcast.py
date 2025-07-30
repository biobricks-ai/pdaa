import pandas as pd
import biobricks as bb
from pathlib import Path
from tqdm import tqdm
from rdkit.Chem import AllChem

import sys
sys.path.append('./')
# from scripts.utils.helpers import smiles_to_inchi

tqdm.pandas()

toxcast_parquet = bb.assets('toxcast').invitrodb_parquet
toxcast = pd.read_parquet(toxcast_parquet)
subset = toxcast[['casn', 'modl_ga']].dropna()
# filter out entries with no CAS number (has NOCAS in casn)
subset = subset[~subset['casn'].str.contains('NOCAS')]
## filter out entries with None as the CAS number
subset = subset[subset['casn'].notnull()]

# check if the 'inchi' column exists, if not, create it
if 'inchi' not in toxcast.columns:
    # import pubchempy as pcp
    import cirpy
    def cas_to_inchi(cas):
        try:
            # compound = pcp.get_compounds(cas, 'name')[0]
            # return compound.inchi

            # this works but takes a long time
            return cirpy.resolve(cas, 'inchi')
        except Exception as e:
            print(f"Error fetching InChI for CAS {cas}: {e}")
            return None
        
    print("Creating 'inchi' column from 'cas'...")
    
    subset['inchi'] = toxcast['casn'].progress_apply(
        lambda x: cas_to_inchi(x)
    )

# save the subset with InChI to a new parquet file
outdir = Path('cache/toxcast')
outdir.mkdir(parents=True, exist_ok=True)
subset.to_parquet(outdir / 'toxcast_subset.parquet')