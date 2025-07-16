import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors, Descriptors3D

import sys
sys.path.append('./')
from stages.utils.pdaa import is_phthalate

def longest_carbon_backbone(mol: AllChem.Mol) -> int:
    """
    Given an RDKit Mol that is already known to be a phthalate, return
    the number of carbon atoms in the longest un-branched alkyl segment
    (the “backbone”) of its ester side-chains.

    Strategy
    --------
    1.  Locate each ester linkage with the SMARTS pattern 'C(=O)O'.
        - Index 0 is the carbonyl carbon.
        - Index 2 is the single-bonded oxygen that connects to the side-chain.
    2.  For each of those oxygens, identify the first carbon in the side-chain
        (the oxygen's neighbour that is not the carbonyl carbon).
    3.  Depth-first search outward **only through carbon atoms** to find the
        maximum path length.  At every branch we explore all possibilities
        and keep the longest.
    4.  Track the maximum over both ester arms and return it.

    The function ignores non-carbon atoms and avoids cycles by passing a
    `prev_idx` argument during recursion.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule already validated as a phthalate.

    Returns
    -------
    int
        Length of the longest linear carbon segment (backbone).
    """
    def _dfs(atom, visited = set()) -> int:
        """Depth-first search returning longest carbon chain length from `atom`."""
        if atom.GetIdx() not in visited:
            visited.add(atom.GetIdx())

        max_len = 0
        for nbr in atom.GetNeighbors():
            if (nbr.GetIdx() in visited) or nbr.GetSymbol() != 'C':
                continue
            branch_len = _dfs(nbr, atom.GetIdx())
            max_len = max(max_len, branch_len)

        # if all neighbors for atom checked, remove it from visited
        visited.remove(atom.GetIdx())
        return 1 + max_len  # count this carbon

    ester_pattern = AllChem.MolFromSmarts('C(=O)O')
    longest = 0

    for match in mol.GetSubstructMatches(ester_pattern):
        carbonyl_c_idx, _, single_o_idx = match
        single_o = mol.GetAtomWithIdx(single_o_idx)

        # Identify the first carbon in the side-chain (O-C).
        side_c = next(
            (nbr for nbr in single_o.GetNeighbors()
             if nbr.GetIdx() != carbonyl_c_idx and nbr.GetSymbol() == 'C'),
            None
        )
        if side_c is None:
            continue  # malformed ester; skip

        chain_len = _dfs(side_c, single_o_idx)
        longest = max(longest, chain_len)

    return longest

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
        'MolWt'            : Descriptors.MolWt(mol),
        'cLogP'            : Descriptors.MolLogP(mol),
        'TPSA'             : Descriptors.TPSA(mol),
        'RotB'             : Descriptors.NumRotatableBonds(mol),
        'MolMR'            : Descriptors.MolMR(mol),
        'Fsp3'             : Descriptors.FractionCSP3(mol),
        'Kappa1'           : Descriptors.Kappa1(mol),
        'Kappa2'           : Descriptors.Kappa2(mol),
        'Kappa3'           : Descriptors.Kappa3(mol),
        'LongestCarbonBackbone': longest_carbon_backbone(mol),
    }
    # 3-D shape (needs conformer)
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    feats['Rgyr'] = Descriptors3D.RadiusOfGyration(mol)
    # custom: side-chain length & branching (sketch)
    feats['Isomer'] = classify_isomer(mol)  # 0=ortho,1=iso,2=tere
    return feats

def get_activity_matrix(cachedir: str | Path) -> pd.DataFrame:
    """Load the activity matrix from a parquet file."""
    # Define the path to the parquet file
    cachedir = Path('cache/entity_similarity')
    activity_matrix_path = cachedir / 'activity_matrix_filled.parquet'

    # Load the activity matrix from the parquet file
    activity_matrix = pd.read_parquet(activity_matrix_path)

    return activity_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process activity matrix for entity similarity.")
    parser.add_argument('--cachedir', type=str, default='cache/entity_similarity',
                        help='Directory to cache the activity matrix.')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize the activity matrix.')
    parser.add_argument('--z_score', action='store_true',
                        help='Calculate z-scores of the activity matrix by column.')
    args = parser.parse_args()

    cachedir = Path(args.cachedir)
    activity_matrix = get_activity_matrix(cachedir)

    # Convert the DataFrame to a NumPy array
    activity_array = activity_matrix.to_numpy()
    # subtract the mean by column
    activity_array -= np.mean(activity_array, axis=0)

    # optionally calculate z-scores of the activity matrix by column
    if args.z_score:
        activity_array = activity_array/np.std(activity_array, axis=0)
        if args.normalize:
            # give a warning that only one of normalize or z-score should be used
            print("Warning: Both --normalize and --z_score are set. Only z-score will be applied.")
    # optionally normalize the activity matrix by row
    elif args.normalize:
        activity_array /= np.linalg.norm(activity_array, axis=1, keepdims=True)

    # Convert the 'title' column to RDKit Mol objects
    mol_list = [AllChem.MolFromInchi(s) for s in activity_matrix.index]

    
