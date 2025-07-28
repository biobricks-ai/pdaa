import pandas as pd
from pathlib import Path
from tqdm import tqdm

from rdkit.Chem import AllChem, Descriptors, Descriptors3D

import sys
sys.path.append('./')  # so utility scripts can be found
from scripts.utils.helpers import (
    z_scale_df, get_linear_model, get_descriptors, compute_vifs
)

cachedir = Path('cache/edkb')
activity_df = pd.read_parquet(cachedir / 'activity_matrix_filled.parquet')


# Convert the 'title' column to RDKit Mol objects
mol_list = [AllChem.AddHs(AllChem.MolFromInchi(s)) for s in tqdm(activity_df.index, desc="Converting InChIs to RDKit Mol objects")]

# Calculate descriptors for each molecule
descriptor_parquet = cachedir / 'descriptors.parquet'
if descriptor_parquet.exists():
    print(f"Loading existing descriptors from {descriptor_parquet}")
    descriptor_df = pd.read_parquet(descriptor_parquet)
else:
    descriptor_vectors = [get_descriptors(mol, use_phthalate_set=False) for mol in tqdm(mol_list, desc="Calculating descriptors")]
    descriptor_df = pd.DataFrame(descriptor_vectors, index=activity_df.index)
    descriptor_df.to_parquet(descriptor_parquet)

# Remove descriptors with high multicollinearity or low variance
# Note: Adjust the list of descriptors based on your analysis
descriptor_df = descriptor_df.drop(columns=[
    'MolWt',
    'Kappa2',
])

# Data preprocessing: scale the descriptors
X = z_scale_df(descriptor_df)
Y = z_scale_df(activity_df)

# Compute variance inflation factors (VIFs) to check for multicollinearity
vif_table = compute_vifs(X)
print("VIF Table:")
print(vif_table)

for descriptor in vif_table['descriptor']:
    if vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0] > 10:
        print(f"Warning: High VIF detected for descriptor '{descriptor}' (VIF={vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0]}). Consider removing it.")


# Fit a linear regression model to the data
ols = get_linear_model(X, Y)