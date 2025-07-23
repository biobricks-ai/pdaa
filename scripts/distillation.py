import argparse
import pandas as pd

from pathlib import Path
from rdkit import Chem

import sys
sys.path.append('./')
from scripts.utils.helpers import get_activity_df

def load_edkb(path: Path, activity_field: str):
    # rename *.txt to *.sdf* if you prefer
    suppl = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    records = []
    for mol in suppl:
        if mol is None:              # skip bad parses
            continue
        log_rba = float(mol.GetProp(activity_field))
        if log_rba <= -5_000:        # sentinel for “inactive”
            continue                 # or keep and label 0 for classification
        records.append({
            "smiles": Chem.MolToSmiles(mol),
            "log_rba": log_rba
        })
    return pd.DataFrame(records)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process activity matrix for entity similarity.")
    parser.add_argument('--cachedir', type=str, default='cache/entity_similarity',
                        help='Directory to cache the activity matrix.')
    parser.add_argument('--outdir', type=str, default='cache/descriptors',
                        help='Directory to cache the descriptors.')
    args = parser.parse_args()

    # define the important directories
    cachedir = Path(args.cachedir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load the DataFrames
    activity_df = get_activity_df(cachedir)
    descriptor_parquet = outdir / 'descriptors.parquet'
    descriptor_df = pd.read_parquet(descriptor_parquet)

    def get_endpoint(dataset_name: str):
        endpoint = 'NCTRlogRBA'
        if dataset_name == 'Androgen':
            endpoint = 'ar' + endpoint

        return endpoint
    
    resourcedir = Path('resources')
    dataset_names = ['Androgen', 'Estrogen']
    datasets = {
        name: load_edkb(resourcedir / f'EDKB_{name}.sdf', get_endpoint(name))
        for name in dataset_names
    }

    # add InChI column to the datasets
    def smiles_to_inchi(smiles):
        mol = Chem.MolFromSmiles(smiles)
        inchi = Chem.MolToInchi(mol)
        return inchi
    
    for df in datasets.values():
        df['inchi'] = df['smiles'].apply(smiles_to_inchi)

    # concatenate the dataframes
    edkb = pd.concat(datasets.values()).reset_index(drop=True)
