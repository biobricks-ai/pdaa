import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import DataStructs
from rdkit.Chem import AllChem

def get_activity_matrix(cachedir: str | Path) -> pd.DataFrame:
    """Load the activity matrix from a parquet file."""
    # Define the path to the parquet file
    cachedir = Path('cache/entity_similarity')
    activity_matrix_path = cachedir / 'activity_matrix_filled.parquet'

    # Load the activity matrix from the parquet file
    activity_matrix = pd.read_parquet(activity_matrix_path)

    return activity_matrix

def calculate_similarity(fingerprints: list) -> np.ndarray:
    """Calculate the similarity between fingerprints."""
    # Create an empty matrix to store the similarity values
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n), dtype=float)

    # Calculate the similarity between each pair of fingerprints
    for i in range(n):
        similarity_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]

    return similarity_matrix

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

    # Get the Morgan fingerprints for each molecule
    MorganGenerator = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints = [
        MorganGenerator.GetSparseCountFingerprint(mol) for mol in mol_list if mol is not None
    ]

    similarity_matrix = calculate_similarity(fingerprints)
    print(similarity_matrix)
    
