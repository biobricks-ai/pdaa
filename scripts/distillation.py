import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from rdkit import Chem

from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import sys
sys.path.append('./')
from scripts.utils.helpers import get_activity_df, smiles_to_inchi

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

def get_combined_edkb(resourcedir: Path):
    edkb_file = resourcedir / 'edkb.parquet'

    if edkb_file.exists():
        edkb = pd.read_parquet(edkb_file)
    else:
        def get_endpoint(dataset_name: str):
            endpoint = 'NCTRlogRBA'
            if dataset_name == 'Androgen':
                endpoint = 'ar' + endpoint

            return endpoint
        
        dataset_names = ['Androgen', 'Estrogen']
        datasets = {
            name: load_edkb(resourcedir / f'EDKB_{name}.sdf', get_endpoint(name))
            for name in dataset_names
        }

        # add InChI column to the datasets      
        for df in datasets.values():
            df['inchi'] = df['smiles'].apply(smiles_to_inchi)

        # concatenate the dataframes
        edkb = pd.concat(datasets.values()).reset_index(drop=True)
        edkb.to_parquet(edkb_file)

    return edkb

def get_edkb_log_rba(resourcedir: Path):
    edkb_file = resourcedir / 'edkb_log_rba.parquet'
    edkb = pd.read_parquet(edkb_file)
    return edkb

def load_dataframes(cachedir: Path, outdir: Path):
    activity_df = get_activity_df(cachedir)  # chemicals × assays
    descriptor_parquet = outdir / 'descriptors.parquet'
    descriptor_df = pd.read_parquet(descriptor_parquet)  # chemicals × descriptors

    return activity_df, descriptor_df

def fit_lin_nonlin(X, y):
    # 1. split (random example)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. linear baseline
    enet = GridSearchCV(
        ElasticNet(max_iter=10_000),
        param_grid={'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
        cv=5
    ).fit(X_train, y_train)

    print("ElasticNet MAE:", mean_absolute_error(y_test, enet.predict(X_test)))

    # 3. boosted trees
    gbr = GridSearchCV(
        HistGradientBoostingRegressor(),
        param_grid={'learning_rate': [0.03, 0.1],
                    'max_depth': [None, 6, 10],
                    'l2_regularization': [0.0, 1.0]},
        cv=5
    ).fit(X_train, y_train)

    print("GradientBoosting MAE:", mean_absolute_error(y_test, gbr.predict(X_test)))

    return enet, gbr

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

    activity_df, descriptor_df = load_dataframes(cachedir, outdir)

    # edkb = get_combined_edkb(Path('resources'))
    edkb = get_edkb_log_rba(Path('resources'))

    # for testing purposes, make up some data for the chemicals not included in EDKB
    # random_data = []
    # for inchi in activity_df.index:
    #     if inchi in edkb.inchi:
    #         continue  # entry already exists
    #     random_data.append({
    #         'inchi': inchi,
    #         'smiles': '',
    #         'log_rba': np.random.randn()
    #     }) 
    # edkb = pd.concat([edkb, pd.DataFrame(random_data)])

    # Select rows from activity_df whose index is in edkb['inchi']
    X = activity_df.loc[activity_df.index.intersection(edkb['inchi'])]
    y = edkb[['inchi', 'log_rba']]
    y = y[y['inchi'].isin(X.index)].drop_duplicates(subset=['inchi'])
    y = y.set_index('inchi').loc[:, 'log_rba']
    print(f'Predicting using {len(X)} chemicals.')

    print(X)
    # print(activity_df)

    # enet, gbr = fit_lin_nonlin(X, y)
