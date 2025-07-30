import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import statsmodels.api as sm

from rdkit.Chem import AllChem, Descriptors, Descriptors3D

import sys
sys.path.append('./')  # so utility scripts can be found
from scripts.utils.helpers import (
    z_scale_df,
    get_linear_model,
    get_descriptors,
    compute_vifs,
    PCA_plot,
    plot_activity_features,
    pca_variance_ratio,
    remove_high_vif_descriptors,
)

cachedir = Path('cache/eadb')
activity_df = pd.read_parquet(cachedir / 'activity_matrix_filled.parquet')

def characterize_descriptors():
    # Convert the 'title' column to RDKit Mol objects
    mol_list = [AllChem.AddHs(AllChem.MolFromInchi(s)) for s in tqdm(activity_df.index, desc="Converting InChIs to RDKit Mol objects")]

    # Calculate descriptors for each molecule
    descriptor_parquet = cachedir / 'descriptors.parquet'
    use_cache = False
    if descriptor_parquet.exists() and use_cache:
        print(f"Loading existing descriptors from {descriptor_parquet}")
        descriptor_df = pd.read_parquet(descriptor_parquet)
    else:
        descriptor_vectors = [get_descriptors(
            mol,
            use_phthalate_set=False,
            use_general_set=True,
            # use_vectors=True,
        ) for mol in tqdm(mol_list, desc="Calculating descriptors")]
        descriptor_df = pd.DataFrame(descriptor_vectors, index=activity_df.index)
        descriptor_df.to_parquet(descriptor_parquet)

    # Data preprocessing: scale the descriptors
    X = z_scale_df(descriptor_df)
    Y = z_scale_df(activity_df)

    X = remove_high_vif_descriptors(X, vif_threshold=10)

    # Compute variance inflation factors (VIFs) to check for multicollinearity
    vif_table = compute_vifs(X)
    print("VIF Table:")
    print(vif_table)

    for descriptor in vif_table['descriptor']:
        if vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0] > 10:
            print(f"Warning: High VIF detected for descriptor '{descriptor}' (VIF={vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0]}). Consider removing it.")

# SECTION: Make a predictive model for EADB endpoints
def comma_remove(s):
    return s.replace(',', '')

def get_log_rba(eadb):
    # filter for logRBA endpoint
    log_rba = eadb.loc[eadb['EndpointName'] == 'logRBA', ['inchi', 'EndpointValue']]
    log_rba.rename(columns={'EndpointValue': 'logRBA'}, inplace=True)
    # set the InChI as index
    log_rba.set_index('inchi', inplace=True)
    log_rba = log_rba[log_rba.index.notna()]
    # clean the logRBA column
    log_rba['logRBA'] = log_rba['logRBA'].apply(comma_remove)
    # convert logRBA to numeric
    log_rba['logRBA'] = pd.to_numeric(log_rba['logRBA'], errors='coerce')
    # set sentinel values to NaN
    log_rba[log_rba < -10] = pd.NA
    # drop rows with NaN in logRBA
    log_rba = log_rba.dropna()
    # for duplicate InChIs, take the mean of logRBA values
    log_rba = log_rba.groupby(log_rba.index).mean()

    return log_rba

eadb = pd.read_parquet(cachedir / 'eadb.parquet')
log_rba = get_log_rba(eadb)

# make predictor and target DataFrames
index_intersection = activity_df.index.intersection(log_rba.index)
X = activity_df.loc[index_intersection]
y = log_rba.loc[index_intersection, 'logRBA']

# Fit a linear regression model
print("Fitting linear regression model...")
ols = get_linear_model(X, y)

# Use k-means clustering, attempting to distinguish between high and low RBA