import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

import sys
sys.path.append('./')
from stages.utils.pdaa import is_phthalate, longest_carbon_backbone

def z_scale_df(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize the dataframe by columns."""
    # mean = np.mean(matrix, axis=0)
    # std_dev = np.std(matrix, axis=0)
    return (df - df.mean())/df.std()

def classify_isomer(mol: AllChem.Mol) -> int:
    """
    Classify the isomer type of a phthalate molecule based on its structure.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule already validated as a phthalate.

    Returns
    -------
    int
        0 for ortho, 1 for iso (meta), 2 for tere (para) phthalate.
    """
    possible_modes = ('ortho_phthalate', 'meta_phthalate', 'para_phthalate')
    for i, mode in enumerate(possible_modes):
        if is_phthalate(mol, modes=(mode,)):
            return i

def get_descriptors(mol):
    """Calculate descriptors for a given molecule."""
    feats = {
        # 'MolWt'            : Descriptors.MolWt(mol),
        'cLogP'            : Descriptors.MolLogP(mol),
        'TPSA'             : Descriptors.TPSA(mol),
        'RotB'             : Descriptors.NumRotatableBonds(mol),
        # 'MolMR'            : Descriptors.MolMR(mol),
        'Fsp3'             : Descriptors.FractionCSP3(mol),
        'Kappa1'           : Descriptors.Kappa1(mol),
        # 'Kappa2'           : Descriptors.Kappa2(mol),
        # 'Kappa3'           : Descriptors.Kappa3(mol),
        'LongestCarbonBackbone': longest_carbon_backbone(mol),
    }
    # 3-D shape (needs conformer)
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    feats['Rgyr'] = Descriptors3D.RadiusOfGyration(mol)
    # custom: side-chain length & branching (sketch)
    feats['Isomer'] = classify_isomer(mol)  # 0=ortho,1=iso,2=tere
    return feats

def get_activity_df(cachedir: str | Path) -> pd.DataFrame:
    """Load the activity matrix from a parquet file."""
    # Define the path to the parquet file
    cachedir = Path('cache/entity_similarity')
    activity_df_path = cachedir / 'activity_matrix_filled.parquet'

    # Load the activity matrix from the parquet file
    activity_df = pd.read_parquet(activity_df_path)

    return activity_df

def compute_vifs(X: pd.DataFrame, *, add_intercept: bool = False) -> pd.DataFrame:
    """
    Calculate VIF for each column in a descriptor matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Columns are descriptors, rows are compounds (already z-scaled is ideal).
    add_intercept : bool, default False
        - If True, appends a constant column before computing VIFs.
        - Most chem-descriptor sets don't need the intercept; set to True only
          if you plan to include one in later regression models.

    Returns
    -------
    pd.DataFrame
        Two columns:
        - 'descriptor': original column names
        - 'VIF': variance inflation factor (≥ 1; > 10 is a red flag)
    """
    # - Guard against perfectly constant descriptors (std == 0)
    constant_cols = X.columns[X.std() == 0]
    if len(constant_cols):
        raise ValueError(f"Constant descriptors detected: {list(constant_cols)}")

    XX = X.copy()
    if add_intercept:
        XX = XX.assign(_intercept_=1.0)

    # - Compute VIF for each column; statsmodels needs ndarray input
    vifs = [
        variance_inflation_factor(XX.values, idx)
        for idx in range(XX.shape[1])
    ]

    res = pd.DataFrame({
        'descriptor': XX.columns,
        'VIF': vifs
    })

    # - If an intercept was added, drop it from the result
    if add_intercept:
        res = res.query("descriptor != '_intercept_'").reset_index(drop=True)

    return res.sort_values('VIF', ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process activity matrix for entity similarity.")
    parser.add_argument('--cachedir', type=str, default='cache/entity_similarity',
                        help='Directory to cache the activity matrix.')
    parser.add_argument('--outdir', type=str, default='cache/descriptors',
                        help='Directory to cache the descriptors.')
    # parser.add_argument('--normalize', action='store_true',
    #                     help='Normalize the activity matrix.')
    # parser.add_argument('--z_score', action='store_true',
    #                     help='Calculate z-scores of the activity matrix by column.')
    args = parser.parse_args()

    cachedir = Path(args.cachedir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    activity_df = get_activity_df(cachedir)

    # # Convert the DataFrame to a NumPy array
    # activity_array = activity_df.to_numpy()

    # # subtract the mean by column
    # activity_array -= np.mean(activity_array, axis=0)
    # # optionally calculate z-scores of the activity matrix by column
    # if args.z_score:
    #     activity_array = activity_array/np.std(activity_array, axis=0)
    #     if args.normalize:
    #         # give a warning that only one of normalize or z-score should be used
    #         print("Warning: Both --normalize and --z_score are set. Only z-score will be applied.")
    # optionally normalize the activity matrix by row
    # elif args.normalize:
    #     activity_array /= np.linalg.norm(activity_array, axis=1, keepdims=True)

    # Convert the 'title' column to RDKit Mol objects
    mol_list = [AllChem.AddHs(AllChem.MolFromInchi(s)) for s in tqdm(activity_df.index, desc="Converting InChIs to RDKit Mol objects")]

    # Calculate descriptors for each molecule
    descriptor_parquet = outdir / 'descriptors.parquet'
    if descriptor_parquet.exists():
        print(f"Loading existing descriptors from {descriptor_parquet}")
        descriptor_df = pd.read_parquet(descriptor_parquet)
    else:
        descriptor_vectors = [get_descriptors(mol) for mol in tqdm(mol_list, desc="Calculating descriptors")]
        descriptor_df = pd.DataFrame(descriptor_vectors, index=activity_df.index)
        descriptor_df.to_parquet(descriptor_parquet)

    # descriptor_matrix = descriptor_df.to_numpy()
    # manually dropping descriptors with high VIFs
    descriptor_df = descriptor_df.drop(columns=[
        'MolWt', 'MolMR', 'Kappa2', 'Kappa3',
    ])

    # Data preprocessing
    X = z_scale_df(descriptor_df)
    Y = z_scale_df(activity_df)

    # Compute variance inflation factors (VIFs) to check for multicollinearity
    vif_table = compute_vifs(X)
    print("VIF Table:")
    print(vif_table)

    # Quick Pearson/Spearman heat-map
    # rho = X.corrwith(
    #     Y,
    #     # axis = 1,
    #     method='spearman'
    # )  # p × d matrix
    xy   = pd.concat([X, Y], axis=1)                 # same index, side-by-side
    rho  = xy.corr(method='spearman')                # full (p+d) × (p+d) matrix
    rho  = rho.loc[X.columns, Y.columns]             # slice to p × d block
    print("rho (Spearman correlation):")
    print(rho)
    # correct p-values → q-values (Benjamini–Hochberg)
    
    for descriptor in vif_table['descriptor']:
        if vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0] > 10:
            print(f"Warning: High VIF detected for descriptor '{descriptor}' (VIF={vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0]}). Consider removing it.")
    # TODO: PLS regression to find the most predictive descriptors
    from sklearn.ensemble import RandomForestRegressor
    import shap
    rf = RandomForestRegressor(n_estimators=500, oob_score=True, n_jobs=-1)
    rf.fit(X, Y)
    print(f"Random Forest OOB Score: {rf.oob_score_}")
    # explainer = shap.TreeExplainer(rf)
    # shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X, plot_type="bar", max_display=20)
    # TODO: random forest regression for nonlinear relationships 
    # TODO: summarize the results in a report or visualization

    
